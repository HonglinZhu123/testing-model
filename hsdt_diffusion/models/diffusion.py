"""Conditional Gaussian diffusion with a pred-x0 objective and DDIM sampling."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def _linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float64)


def _cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_bands: int = 31,
        timesteps: int = 1000,
        sampling_timesteps: int = 10,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        p2_gamma: float = 1.0,
        p2_k: float = 1.0,
        ddim_eta: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__()
        if sampling_timesteps > timesteps:
            raise ValueError("sampling_timesteps cannot exceed timesteps")
        if loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        if beta_schedule == "linear":
            betas = _linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unsupported beta schedule: {beta_schedule}")

        self.model = model
        self.num_bands = num_bands
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.loss_type = loss_type
        self.ddim_eta = ddim_eta
        self.clip_min = clip_min
        self.clip_max = clip_max

        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", cumulative.float())
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt().float())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / cumulative).sqrt().float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", (1.0 / cumulative - 1.0).sqrt().float())
        snr = cumulative / (1.0 - cumulative)
        self.register_buffer("p2_loss_weight", (p2_k + snr).pow(-p2_gamma).float())

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape) * noise
        )

    def predict_noise_from_start(
        self,
        noisy_state: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_start: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy_state.shape) * noisy_state
            - predicted_start
        ) / _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy_state.shape)

    def model_predictions(
        self,
        noisy_state: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        clip: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = torch.cat((noisy_state, condition), dim=1)
        predicted_start = self.model(model_input, timesteps)
        if clip:
            predicted_start = predicted_start.clamp(self.clip_min, self.clip_max)
        predicted_noise = self.predict_noise_from_start(noisy_state, timesteps, predicted_start)
        return predicted_noise, predicted_start

    def p_losses(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if clean.shape != condition.shape:
            raise ValueError(f"clean and condition shapes differ: {clean.shape} vs {condition.shape}")
        if clean.ndim != 4 or clean.shape[1] != self.num_bands:
            raise ValueError(f"expected [B,{self.num_bands},H,W], got {tuple(clean.shape)}")
        batch = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (batch,), device=clean.device)
        noise = torch.randn_like(clean) if noise is None else noise
        noisy_state = self.q_sample(clean, timesteps, noise)
        predicted_start = self.model(torch.cat((noisy_state, condition), dim=1), timesteps)
        if self.loss_type == "l1":
            element_loss = F.l1_loss(predicted_start, clean, reduction="none")
        else:
            element_loss = F.mse_loss(predicted_start, clean, reduction="none")
        per_sample = element_loss.flatten(1).mean(dim=1)
        return (per_sample * _extract(self.p2_loss_weight, timesteps, per_sample.shape)).mean()

    def forward(self, clean: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.p_losses(clean, condition)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        sampling_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        if condition.ndim != 4 or condition.shape[1] != self.num_bands:
            raise ValueError(f"expected condition [B,{self.num_bands},H,W], got {tuple(condition.shape)}")
        steps = self.sampling_timesteps if sampling_timesteps is None else sampling_timesteps
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("sampling steps must be between 1 and num_timesteps")

        times = torch.linspace(-1, self.num_timesteps - 1, steps=steps + 1, device=condition.device)
        times = list(reversed(times.long().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        image = torch.randn_like(condition)

        for time, next_time in time_pairs:
            batch_times = torch.full(
                (condition.shape[0],), time, device=condition.device, dtype=torch.long
            )
            predicted_noise, predicted_start = self.model_predictions(
                image, condition, batch_times, clip=True
            )
            if next_time < 0:
                image = predicted_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[next_time]
            sigma = self.ddim_eta * torch.sqrt(
                ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).clamp(min=0)
            )
            direction = torch.sqrt((1 - alpha_next - sigma.square()).clamp(min=0))
            image = (
                predicted_start * alpha_next.sqrt()
                + direction * predicted_noise
                + sigma * torch.randn_like(image)
            )

        return image.clamp(self.clip_min, self.clip_max)
