from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .models import GaussianDiffusion, HSDTDiffusionBackbone


def build_model(config: Mapping[str, Any]) -> HSDTDiffusionBackbone:
    return HSDTDiffusionBackbone(**deepcopy(dict(config)))


def build_diffusion(config: Mapping[str, Any], model_config: Mapping[str, Any]) -> GaussianDiffusion:
    model = build_model(model_config)
    diffusion_config = deepcopy(dict(config))
    diffusion_config.setdefault("num_bands", model_config["num_bands"])
    return GaussianDiffusion(model=model, **diffusion_config)
