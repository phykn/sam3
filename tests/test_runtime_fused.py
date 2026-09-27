import pytest
import torch
from src.ml.components.backbone.vit import Mlp
from src.ml.runtime.fused import apply_addmm_activation


def test_apply_addmm_activation_keeps_float32_on_cpu():
    linear = torch.nn.Linear(4, 8)
    tensor = torch.randn(2, 3, 4)

    with torch.inference_mode():
        out = apply_addmm_activation(torch.nn.GELU, linear, tensor)

    assert out.shape == (2, 3, 8)
    assert out.dtype == torch.float32


def test_mlp_runs_float32_inference_on_cpu():
    model = Mlp(4, 8).eval()

    with torch.inference_mode():
        output = model(torch.ones(2, 4))

    assert output.shape == (2, 4)
    assert output.dtype == torch.float32


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_fused_linear_without_bias_matches_reference(device):
    linear = torch.nn.Linear(4, 8, bias=False, device=device)
    tensor = torch.randn(2, 3, 4, device=device)
    with torch.inference_mode():
        expected = torch.nn.functional.gelu(linear(tensor))
        actual = apply_addmm_activation(torch.nn.GELU, linear, tensor)
    torch.testing.assert_close(actual, expected)
