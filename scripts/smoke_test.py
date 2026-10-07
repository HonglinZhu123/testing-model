from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsdt_diffusion.engine import checkpoint_roundtrip, create_training_objects, train_steps
from hsdt_diffusion.utils import choose_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end HSDT-Diffusion smoke test")
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["train"]["seed"]))
    device = choose_device(args.device)
    diffusion, loader, optimizer = create_training_objects(config, device)

    loss, steps = train_steps(
        diffusion,
        loader,
        optimizer,
        device,
        max_steps=1,
        grad_clip=float(config["train"]["grad_clip"]),
        use_amp=False,
    )
    if not torch.isfinite(torch.tensor(loss)):
        raise RuntimeError(f"non-finite training loss: {loss}")

    condition, _ = next(iter(loader))
    condition = condition.to(device)
    with tempfile.TemporaryDirectory(prefix="hsdt_diffusion_") as directory:
        checkpoint = Path(directory) / "smoke.pt"
        checkpoint_roundtrip(checkpoint, diffusion, optimizer, config, epoch=0, step=steps, device=device)
        diffusion.eval()
        restored = diffusion.sample(condition)

    if restored.shape != condition.shape:
        raise RuntimeError(f"sample shape mismatch: {restored.shape} vs {condition.shape}")
    if not torch.isfinite(restored).all():
        raise RuntimeError("sampling produced non-finite values")

    parameters = diffusion.model.parameter_count()
    print(
        "SMOKE TEST PASSED | "
        f"device={device} loss={loss:.6f} shape={tuple(restored.shape)} params={parameters:,}"
    )


if __name__ == "__main__":
    main()
