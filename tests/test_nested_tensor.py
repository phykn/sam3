import torch

from src.ml.structures import NestedTensor


def test_dtype_conversion_preserves_boolean_padding_mask():
    mask = torch.tensor([[[False, True]]])
    value = NestedTensor(torch.ones(1, 3, 1, 2), mask)
    converted = value.to(dtype=torch.float16)
    assert converted.tensors.dtype == torch.float16
    assert converted.mask.dtype == torch.bool
    torch.testing.assert_close(converted.mask, mask)


def test_pin_memory_returns_batch_container(monkeypatch):
    calls = []

    def pin(tensor, device=None):
        calls.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    value = NestedTensor(torch.ones(1, 3, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool))
    assert value.pin_memory() is value
    assert len(calls) == 2
