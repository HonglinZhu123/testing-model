from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from .data import build_dataset
from .factory import build_diffusion
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.optimizer import build_optimizer


def create_training_objects(config: Mapping[str, Any], device: torch.device):
    diffusion = build_diffusion(config["diffusion"], config["model"]).to(device)
    dataset = build_dataset(config["data"], config["model"])
    train_config = config["train"]
    loader = DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_workers(config)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    optimizer = build_optimizer(diffusion.model.parameters(), train_config)
    return diffusion, loader, optimizer


def data_workers(config: Mapping[str, Any]) -> int:
    return int(config["data"].get("num_workers", 0))


def train_steps(
    diffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: Optional[int],
    grad_clip: float,
    use_amp: bool,
) -> tuple[float, int]:
    diffusion.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    total_loss = 0.0
    completed = 0
    for condition, clean in loader:
        condition = condition.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            loss = diffusion(clean, condition)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
        completed += 1
        if max_steps is not None and completed >= max_steps:
            break
    if completed == 0:
        raise RuntimeError("training loader produced no batches")
    return total_loss / completed, completed


def checkpoint_roundtrip(
    path: str | Path,
    diffusion,
    optimizer,
    config: Mapping[str, Any],
    epoch: int,
    step: int,
    device: torch.device,
) -> None:
    save_checkpoint(path, diffusion, optimizer, epoch, step, config)
    load_checkpoint(path, diffusion, optimizer, map_location=device)
