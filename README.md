# HSDT-Diffusion

Conditional hyperspectral diffusion using a time-conditioned HSDT denoising backbone.

This repository is an experimental integration of two ideas:

- HSDT: 3-D spectral-spatial convolution, guided spectral self-attention, and an encoder-decoder topology.
- S2TDM-style conditional diffusion: `concat(x_t, noisy_condition) -> predict x_0`, trained with P2 weighting and sampled with DDIM.

The initial implementation targets 31-band HSI. Final training hyperparameters and complex-noise synthesis are intentionally left open until the baseline pipeline is validated on the target server.

## Architecture

The public diffusion interface stays 4-D:

```text
x_t, Y: [B, 31, H, W]
concat: [B, 62, H, W]
```

Inside the backbone it becomes:

```text
[Y, x_t] -> [B, 2, 31, H, W]
          -> time-conditioned HSDT
          -> residual delta [B, 1, 31, H, W]
          -> Y + delta
          -> predicted x_0 [B, 31, H, W]
```

Key design choices:

- GroupNorm replaces BatchNorm3d so batches can contain different diffusion timesteps.
- Every head, encoder, and decoder block receives zero-initialized scale/shift FiLM conditioning.
- The condition is the first internal HSDT channel, and the network predicts a residual over the condition rather than over `x_t`.
- GSSA uses an explicit `num_bands`; the current default is 31.
- The original HSDT spatial downsampling indices `[1, 3]` and base width 16 are retained.

## Project layout

```text
configs/default.json             proposed 31-band experiment configuration
configs/smoke.json               tiny deterministic integration test
hsdt_diffusion/models/           HSDT backbone and Gaussian diffusion
hsdt_diffusion/data/             .npy/.npz/.mat loading and Gaussian synthesis
hsdt_diffusion/engine.py         shared training primitives
train.py                         server training entry point
infer.py                         DDIM inference entry point
scripts/smoke_test.py            train/checkpoint/sample end-to-end test
tests/test_smoke.py              unit-level shape, backward, and sampling tests
```

## Installation

Use the PyTorch build appropriate for the server CUDA version, then install this project:

```bash
pip install -e .
```

`scipy` is only required for MATLAB `.mat` files. Core model execution uses only PyTorch.

## Local smoke test

```bash
python scripts/smoke_test.py --device auto
python -m unittest discover -s tests -v
```

The smoke test performs one optimizer step, saves and reloads a checkpoint, and completes DDIM sampling.
It uses the bundled `portable_adamw` diagnostic optimizer so the test does not depend on optional
PyTorch compiler/ONNX components. The default server configuration uses `torch.optim.AdamW`.

## Dataset format

The initial training loader accepts `.npy`, `.npz`, and MATLAB `.mat` cubes. Arrays may be either:

```text
[bands, height, width]
[height, width, bands]
```

For `.npz` and `.mat`, set `data.key` in the configuration if automatic array selection is ambiguous. Data normalization is explicit: set `data.normalize` to `none`, `clamp`, or `minmax` after confirming the server dataset scale.

The initial loader synthesizes Gaussian noise from clean images. Complex-noise generation should be added after its exact experimental definition is fixed.

## Training

Edit at least these fields in `configs/default.json`:

```json
{
  "data": {
    "train_root": "/path/to/training/cubes",
    "key": null,
    "normalize": "none"
  }
}
```

Then run:

```bash
python train.py --config configs/default.json --device cuda
```

For a short server diagnostic:

```bash
python train.py --config configs/default.json --device cuda --max-steps 2
```

Resume training with:

```bash
python train.py --config configs/default.json --device cuda --resume runs/hsdt_diff_icvl/latest.pt
```

## Inference

```bash
python infer.py \
  --config configs/default.json \
  --checkpoint runs/hsdt_diff_icvl/latest.pt \
  --input /path/to/noisy.npy \
  --output outputs/restored.npy \
  --device cuda
```

Inference pads spatial dimensions to the HSDT downsampling multiple and crops the result back to the original size.

## Configuration status

The values in `configs/default.json` are starting points copied from the design assessment, not claimed final training settings. In particular, the following still require controlled experiments:

- training epochs and batch size;
- L1 versus L2 `pred_x0` loss;
- linear versus cosine beta schedule;
- stochastic (`eta=1`) versus deterministic (`eta=0`) DDIM;
- number of DDIM sampling steps;
- HSDT checkpoint initialization;
- complex-noise synthesis and real-noise datasets.

## Attribution

The architecture is derived conceptually from the official HSDT and S2TDM research implementations. When publishing results, cite the corresponding HSDT and S2TDM papers and comply with the licenses of any copied datasets or pretrained weights.
