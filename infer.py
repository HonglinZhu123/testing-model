from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hsdt_diffusion.data import load_hsi_file
from hsdt_diffusion.factory import build_diffusion
from hsdt_diffusion.utils import choose_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSDT-Diffusion DDIM inference")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npy, .npz, or .mat HSI")
    parser.add_argument("--output", required=True, help="output .npy path (CHW)")
    parser.add_argument("--key", default=None, help="array key for .npz/.mat")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    diffusion = build_diffusion(config["diffusion"], config["model"]).to(device)
    load_checkpoint(args.checkpoint, diffusion, map_location=device)
    diffusion.eval()

    condition = load_hsi_file(args.input, int(config["model"]["num_bands"]), args.key)
    condition = condition.unsqueeze(0).to(device)
    height, width = condition.shape[-2:]
    multiple = diffusion.model.spatial_multiple
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    condition = F.pad(condition, (0, pad_width, 0, pad_height), mode="reflect")
    restored = diffusion.sample(condition, sampling_timesteps=args.steps)[..., :height, :width]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, restored.squeeze(0).cpu().numpy())
    print(f"saved {tuple(restored.shape)} to {output}")


if __name__ == "__main__":
    main()
