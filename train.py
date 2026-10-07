from __future__ import annotations

import argparse
from pathlib import Path

from hsdt_diffusion.engine import create_training_objects, train_steps
from hsdt_diffusion.utils import choose_device, load_checkpoint, load_config, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HSDT-Diffusion")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="limit each epoch for diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["train"].get("seed", 2024)))
    device = choose_device(args.device)
    output = Path(args.output or config["train"].get("output_dir", "runs/default"))
    diffusion, loader, optimizer = create_training_objects(config, device)

    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = load_checkpoint(args.resume, diffusion, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("step", 0))

    epochs = int(config["train"].get("epochs", 1))
    for epoch in range(start_epoch, epochs):
        loss, steps = train_steps(
            diffusion,
            loader,
            optimizer,
            device,
            max_steps=args.max_steps,
            grad_clip=float(config["train"].get("grad_clip", 1.0)),
            use_amp=bool(config["train"].get("amp", True)),
        )
        global_step += steps
        checkpoint = output / f"epoch_{epoch:04d}.pt"
        save_checkpoint(checkpoint, diffusion, optimizer, epoch, global_step, config)
        save_checkpoint(output / "latest.pt", diffusion, optimizer, epoch, global_step, config)
        print(f"epoch={epoch} steps={steps} loss={loss:.6f} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
