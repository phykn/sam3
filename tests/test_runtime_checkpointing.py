import pytest
import torch

from src.ml.runtime.checkpointing import activation_checkpoint


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_preserves_keyword_tensors_and_gradients(enabled):
    class Layer(torch.nn.Module):
        def forward(self, x, *, gate, **kwargs):
            return torch.sin(x * gate) * kwargs["weight"]

    layer = Layer()
    values = [torch.randn(3, requires_grad=True) for _ in range(3)]
    expected_values = [value.detach().clone().requires_grad_() for value in values]
    actual = activation_checkpoint(
        layer,
        values[0],
        gate=values[1],
        weight=values[2],
        enabled=enabled,
    )
    expected = layer(
        expected_values[0], gate=expected_values[1], weight=expected_values[2]
    )
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    for first, second in zip(values, expected_values):
        torch.testing.assert_close(first.grad, second.grad)
