"""A small AdamW fallback for diagnostics in minimal or broken PyTorch installs.

Production training should use torch.optim.AdamW.  This implementation exists so
the complete pipeline can still be smoke-tested when the local torch.optim import
is broken by an unrelated torch/torchvision/ONNX package mismatch.
"""

from __future__ import annotations

from typing import Iterable

import torch


class PortableAdamW:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = [
            {
                "step": 0,
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
            for parameter in self.parameters
        ]
        self.param_groups = [{"params": self.parameters, "lr": lr}]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.betas
        for parameter, state in zip(self.parameters, self.state):
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("PortableAdamW does not support sparse gradients")
            state["step"] += 1
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denominator = state["exp_avg_sq"].sqrt().div_(bias_correction2 ** 0.5).add_(self.eps)
            parameter.addcdiv_(
                state["exp_avg"],
                denominator,
                value=-self.lr / bias_correction1,
            )

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "state": self.state,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.lr = float(payload["lr"])
        self.weight_decay = float(payload["weight_decay"])
        self.betas = tuple(payload["betas"])
        self.eps = float(payload["eps"])
        source_state = payload["state"]
        if len(source_state) != len(self.state):
            raise ValueError("optimizer parameter count differs from checkpoint")
        for parameter, target, source in zip(self.parameters, self.state, source_state):
            target["step"] = int(source["step"])
            target["exp_avg"] = source["exp_avg"].to(parameter.device)
            target["exp_avg_sq"] = source["exp_avg_sq"].to(parameter.device)


def build_optimizer(parameters, train_config):
    name = train_config.get("optimizer", "adamw")
    kwargs = dict(
        lr=float(train_config.get("learning_rate", 1e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    if name == "portable_adamw":
        return PortableAdamW(parameters, **kwargs)
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {name}")
    return torch.optim.AdamW(parameters, **kwargs)
