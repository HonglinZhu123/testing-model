from __future__ import annotations

import unittest

import torch

from hsdt_diffusion.factory import build_diffusion
from hsdt_diffusion.utils import load_config


class HSDTDiffusionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/smoke.json")

    def test_backbone_shape(self):
        diffusion = build_diffusion(self.config["diffusion"], self.config["model"])
        bands = self.config["model"]["num_bands"]
        model_input = torch.randn(1, bands * 2, 8, 8)
        output = diffusion.model(model_input, torch.tensor([3]))
        self.assertEqual(output.shape, (1, bands, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_loss_backward_and_ddim(self):
        diffusion = build_diffusion(self.config["diffusion"], self.config["model"])
        bands = self.config["model"]["num_bands"]
        clean = torch.rand(1, bands, 8, 8)
        condition = (clean + 0.05 * torch.randn_like(clean)).clamp(0, 1)
        loss = diffusion(clean, condition)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        diffusion.eval()
        restored = diffusion.sample(condition)
        self.assertEqual(restored.shape, clean.shape)
        self.assertTrue(torch.isfinite(restored).all())


if __name__ == "__main__":
    unittest.main()
