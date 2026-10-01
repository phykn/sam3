from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT = "sam3.finetune.v1"


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("finetune checkpoint must be a dictionary")
    if not {"format", "model", "config"}.issubset(checkpoint):
        raise ValueError("finetune checkpoint fields are incomplete")
    if checkpoint["format"] != FORMAT:
        raise ValueError(
            f"unsupported finetune checkpoint format: {checkpoint['format']}"
        )
    for name in ("model", "config"):
        if not isinstance(checkpoint[name], dict):
            raise ValueError(f"finetune checkpoint {name} must be a dictionary")
    return checkpoint


def unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }


def load_trainable_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint model state must be a dictionary")
    expected = {
        name: param
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }
    if set(state) != set(expected):
        raise ValueError("checkpoint trainable keys do not match model")

    for name, param in expected.items():
        if not isinstance(state[name], torch.Tensor):
            raise ValueError(f"checkpoint value must be a tensor: {name}")
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"checkpoint shape mismatch: {name}")

    with torch.no_grad():
        for name, param in expected.items():
            value = state[name]
            param.copy_(value.to(device=param.device, dtype=param.dtype))
