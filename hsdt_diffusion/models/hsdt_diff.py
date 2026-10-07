"""Time-conditioned HSDT backbone for conditional hyperspectral diffusion.

The diffusion wrapper remains 4-D (B, bands, H, W).  Internally, the noisy
diffusion state and the observed condition are converted into the two feature
channels expected by a 3-D HSDT: (B, 2, bands, H, W).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, requested: int) -> int:
    groups = min(channels, requested)
    while channels % groups != 0:
        groups -= 1
    return groups


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 64):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t.float()[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(t, self.frequency_embedding_size))


class TimeFiLM(nn.Module):
    """Feature-wise scale/shift modulation broadcast over band and space."""

    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels * 2))

    def zero_init(self) -> None:
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(time_embedding).chunk(2, dim=1)
        shift = shift[:, :, None, None, None]
        scale = scale[:, :, None, None, None]
        return x * (1.0 + scale) + shift


class S3Conv(nn.Module):
    """Parallel spatial and spectral convolutions from the original HSDT."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(inplace=True),
        )
        self.spectral = nn.Conv3d(
            in_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x) + self.spectral(x)


class GuidedSpectralSelfAttention(nn.Module):
    """Band-wise GSSA with a configurable, explicit number of bands."""

    def __init__(
        self,
        channels: int,
        num_bands: int,
        mode: str = "hybrid",
        content_probability: float = 0.5,
    ):
        super().__init__()
        if mode not in {"hybrid", "learned", "content"}:
            raise ValueError(f"unsupported GSSA mode: {mode}")
        self.channels = channels
        self.num_bands = num_bands
        self.mode = mode
        self.content_probability = content_probability
        self.attn_proj = nn.Linear(channels, num_bands)
        self.value_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def _logits(self, summary: torch.Tensor) -> torch.Tensor:
        if self.mode == "content":
            return summary @ summary.transpose(1, 2)
        if self.mode == "learned":
            return self.attn_proj(summary)
        if self.training and torch.rand((), device=summary.device) < self.content_probability:
            return summary @ summary.transpose(1, 2)
        return self.attn_proj(summary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, bands, _, _ = x.shape
        if bands != self.num_bands:
            raise ValueError(f"expected {self.num_bands} bands, got {bands}")

        residual = x
        summary = x.mean(dim=(-1, -2)).transpose(1, 2)  # B, bands, channels
        attention = self._logits(summary).reshape(batch, bands, bands)
        attention = F.softmax(attention, dim=-1)

        values = x.permute(0, 3, 4, 2, 1)  # B, H, W, bands, channels
        values = self.value_proj(values)
        output = torch.matmul(attention[:, None, None], values)
        output = self.out_proj(output).permute(0, 4, 3, 1, 2).contiguous()
        return output + residual


class SelfModulatedFeedForward(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels * 2
        self.main_in = nn.Linear(channels, hidden, bias=True)
        self.main_out = nn.Linear(hidden, channels, bias=True)
        self.gate = nn.Linear(channels, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_last = x.permute(0, 2, 3, 4, 1)
        main = self.main_out(F.gelu(self.main_in(channel_last)))
        value, gate = self.gate(channel_last).chunk(2, dim=-1)
        output = main + value * torch.sigmoid(gate)
        return output.permute(0, 4, 1, 2, 3).contiguous()


class HSDTTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_bands: int,
        gssa_mode: str,
        gssa_content_probability: float,
    ):
        super().__init__()
        self.attention = GuidedSpectralSelfAttention(
            channels,
            num_bands,
            mode=gssa_mode,
            content_probability=gssa_content_probability,
        )
        self.ffn = SelfModulatedFeedForward(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attention(x))


class TimeConditionedBlock(nn.Module):
    def __init__(
        self,
        convolution: nn.Module,
        out_channels: int,
        num_bands: int,
        time_dim: int,
        norm_groups: int,
        gssa_mode: str,
        gssa_content_probability: float,
        upsample: bool = False,
    ):
        super().__init__()
        self.upsample = upsample
        self.convolution = convolution
        self.norm = nn.GroupNorm(_group_count(out_channels, norm_groups), out_channels)
        self.time_film = TimeFiLM(time_dim, out_channels)
        self.transformer = HSDTTransformerBlock(
            out_channels,
            num_bands,
            gssa_mode,
            gssa_content_probability,
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        x = self.convolution(x)
        x = self.norm(x)
        x = self.time_film(x, time_embedding)
        return self.transformer(x)


def _plain_block(
    in_channels: int,
    out_channels: int,
    **kwargs,
) -> TimeConditionedBlock:
    return TimeConditionedBlock(S3Conv(in_channels, out_channels), out_channels, **kwargs)


def _down_block(
    in_channels: int,
    out_channels: int,
    **kwargs,
) -> TimeConditionedBlock:
    convolution = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=(1, 2, 2),
        padding=1,
        bias=False,
    )
    return TimeConditionedBlock(convolution, out_channels, **kwargs)


def _up_block(
    in_channels: int,
    out_channels: int,
    **kwargs,
) -> TimeConditionedBlock:
    convolution = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    return TimeConditionedBlock(convolution, out_channels, upsample=True, **kwargs)


class HSDTDiffusionBackbone(nn.Module):
    """HSDT denoiser with GroupNorm, full-depth time FiLM, and condition residual."""

    def __init__(
        self,
        num_bands: int = 31,
        base_channels: int = 16,
        num_half_layer: int = 5,
        sample_indices: Sequence[int] = (1, 3),
        time_dim: int = 128,
        norm_groups: int = 8,
        gssa_mode: str = "hybrid",
        gssa_content_probability: float = 0.5,
    ):
        super().__init__()
        if not sample_indices:
            raise ValueError("sample_indices cannot be empty")
        if min(sample_indices) < 0 or max(sample_indices) >= num_half_layer:
            raise ValueError("sample_indices must refer to encoder layer indices")

        self.num_bands = num_bands
        self.base_channels = base_channels
        self.num_half_layer = num_half_layer
        self.sample_indices = tuple(sorted(set(sample_indices)))
        self.spatial_multiple = 2 ** len(self.sample_indices)
        self.time_embedder = TimestepEmbedder(time_dim)

        block_kwargs = dict(
            num_bands=num_bands,
            time_dim=time_dim,
            norm_groups=norm_groups,
            gssa_mode=gssa_mode,
            gssa_content_probability=gssa_content_probability,
        )

        self.head = _plain_block(2, base_channels, **block_kwargs)

        encoder = []
        channels = base_channels
        for index in range(num_half_layer):
            if index in self.sample_indices:
                encoder.append(_down_block(channels, channels * 2, **block_kwargs))
                channels *= 2
            else:
                encoder.append(_plain_block(channels, channels, **block_kwargs))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for index in reversed(range(num_half_layer)):
            if index in self.sample_indices:
                decoder.append(_up_block(channels, channels // 2, **block_kwargs))
                channels //= 2
            else:
                decoder.append(_plain_block(channels, channels, **block_kwargs))
        self.decoder = nn.ModuleList(decoder)
        self.tail = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.zero_init_time_modulation()

    def zero_init_time_modulation(self) -> None:
        for module in self.modules():
            if isinstance(module, TimeFiLM):
                module.zero_init()

    def forward(self, model_input: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if model_input.ndim != 4:
            raise ValueError(f"expected 4-D input, got shape {tuple(model_input.shape)}")
        if model_input.shape[1] != self.num_bands * 2:
            raise ValueError(
                f"expected {self.num_bands * 2} channels ([x_t, condition]), "
                f"got {model_input.shape[1]}"
            )
        height, width = model_input.shape[-2:]
        if height % self.spatial_multiple or width % self.spatial_multiple:
            raise ValueError(
                f"H and W must be divisible by {self.spatial_multiple}; got {height}x{width}"
            )

        noisy_state, condition = model_input.chunk(2, dim=1)
        # The condition is deliberately first: HSDT predicts a residual over Y, not over x_t.
        x = torch.stack((condition, noisy_state), dim=1)
        time_embedding = self.time_embedder(timesteps)

        head = self.head(x, time_embedding)
        skips = [head]
        x = head
        for index, layer in enumerate(self.encoder):
            x = layer(x, time_embedding)
            if index < len(self.encoder) - 1:
                skips.append(x)

        x = self.decoder[0](x, time_embedding)
        for layer in self.decoder[1:]:
            skip = skips.pop()
            if x.shape != skip.shape:
                raise RuntimeError(f"decoder/skip mismatch: {tuple(x.shape)} vs {tuple(skip.shape)}")
            x = layer(x + skip, time_embedding)

        x = x + skips.pop()
        delta = self.tail(x)
        return (delta + condition.unsqueeze(1)).squeeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
