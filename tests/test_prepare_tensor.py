import pytest
import torch

from src.prepare import ground, prompt


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["image", "grounding"])
def test_mask_prompt_accepts_tensor_without_numpy_conversion(device, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mask = torch.ones(2, 3, device=device, requires_grad=True)

    if kind == "image":
        out = prompt.build_mask(mask, (4, 6), device)
        assert out.shape == (1, 1, 4, 6)
    else:
        out, labels = ground.build_masks(mask, device)
        assert out.shape == (1, 1, 1, 2, 3)
        assert labels.tolist() == [[1]]

    assert out.device.type == device
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.ones_like(out))
    out.sum().backward()
    assert mask.grad is not None
    assert torch.all(mask.grad > 0)
