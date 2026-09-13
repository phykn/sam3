import torch
from torch import nn

FORMAT = "sam3.finetune.v1"


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
    expected = {
        name: param
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }
    if set(state) != set(expected):
        raise ValueError("checkpoint trainable keys do not match model")

    for name, param in expected.items():
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"checkpoint shape mismatch: {name}")

    with torch.no_grad():
        for name, param in expected.items():
            value = state[name]
            param.copy_(value.to(device=param.device, dtype=param.dtype))
