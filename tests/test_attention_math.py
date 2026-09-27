import pytest
import torch

from src.ml.components.nn.attention import MultiheadAttention


def test_custom_attention_matches_torch_attention():
    torch.manual_seed(0)
    custom = MultiheadAttention(8, 2, dropout=0.0).eval()
    reference = torch.nn.MultiheadAttention(8, 2, dropout=0.0).eval()
    reference.load_state_dict(custom.state_dict())
    query = torch.randn(5, 2, 8)
    padding = torch.tensor([[False, False, False, True, True]]).expand(2, -1)

    actual, _ = custom(query, query, query, key_padding_mask=padding)
    expected, _ = reference(
        query,
        query,
        query,
        key_padding_mask=padding,
        need_weights=False,
    )

    torch.testing.assert_close(actual, expected)


def test_attention_backward_is_finite():
    torch.manual_seed(0)
    attention = MultiheadAttention(8, 2, dropout=0.0)
    query = torch.randn(5, 2, 8, requires_grad=True)

    output, _ = attention(query, query, query)
    output.square().mean().backward()

    assert torch.isfinite(query.grad).all()


@pytest.mark.parametrize("batch_first", [False, True])
def test_unbatched_attention_matches_torch(batch_first):
    custom = MultiheadAttention(8, 2, batch_first=batch_first).eval()
    reference = torch.nn.MultiheadAttention(8, 2, batch_first=batch_first).eval()
    reference.load_state_dict(custom.state_dict())
    query = torch.randn(3, 8)
    padding = torch.tensor([False, False, True])
    actual = custom(query, query, query, key_padding_mask=padding, need_weights=True)
    expected = reference(query, query, query, key_padding_mask=padding)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("floating_padding", [False, True])
@pytest.mark.parametrize("need_weights", [False, True])
def test_attention_combines_bias_and_masks(floating_padding, need_weights):
    custom = MultiheadAttention(8, 2).eval()
    reference = torch.nn.MultiheadAttention(8, 2).eval()
    reference.load_state_dict(custom.state_dict())
    query = torch.randn(3, 2, 8)
    bias = torch.randn(2, 2, 3, 3)
    mask = torch.zeros(3, 3)
    mask[0, 1] = -float("inf")
    padding = torch.tensor([[False, False, True], [True, False, False]])
    if floating_padding:
        padding = torch.zeros_like(padding, dtype=torch.float32).masked_fill(
            padding, -10
        )
    actual = custom(
        query,
        query,
        query,
        key_padding_mask=padding,
        attn_mask=mask,
        attn_bias=bias,
        need_weights=need_weights,
        average_attn_weights=False,
    )
    expected = reference(
        query,
        query,
        query,
        key_padding_mask=padding,
        attn_mask=(bias + mask).flatten(0, 1),
        need_weights=need_weights,
        average_attn_weights=False,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_attention_dropout_and_backward_match_torch(separate, checkpoint):
    dims = {"kdim": 4, "vdim": 6} if separate else {}
    custom = MultiheadAttention(
        8, 2, dropout=0.25, use_act_checkpoint=checkpoint, **dims
    )
    reference = torch.nn.MultiheadAttention(8, 2, dropout=0.25, **dims)
    reference.load_state_dict(custom.state_dict())
    inputs = [
        torch.randn(3, 2, dim, requires_grad=True)
        for dim in (8, dims.get("kdim", 8), dims.get("vdim", 8))
    ]
    expected_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    torch.manual_seed(17)
    actual = custom(*inputs, need_weights=True)
    torch.manual_seed(17)
    expected = reference(*expected_inputs, need_weights=True)
    torch.testing.assert_close(actual, expected)
    actual[0].square().sum().backward()
    expected[0].square().sum().backward()
    for first, second in zip(inputs, expected_inputs):
        torch.testing.assert_close(first.grad, second.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_attention_accepts_float32_bias(dtype):
    torch.manual_seed(5)
    attention = MultiheadAttention(8, 2).to(dtype=dtype).eval()
    query = torch.randn(3, 2, 8, dtype=dtype)
    bias = torch.randn(2, 2, 3, 3, dtype=torch.float32)
    actual, weights = attention(query, query, query, attn_bias=bias, need_weights=True)
    expected, _ = attention(query, query, query, attn_bias=bias, need_weights=False)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.004)
    assert torch.isfinite(weights).all()


def test_fully_masked_queries_match_sdpa_and_have_finite_gradients():
    attention = MultiheadAttention(8, 2).eval()
    query = torch.randn(3, 2, 8, requires_grad=True)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0] = True
    with torch.no_grad():
        attention.out_proj.bias.fill_(0.3)
        expected, _ = attention(query, query, query, attn_mask=mask, need_weights=False)
    actual, weights = attention(query, query, query, attn_mask=mask, need_weights=True)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(weights[:, 0]) == 0
    actual.sum().backward()
    assert torch.isfinite(query.grad).all()
