import pytest
import torch
from torch import nn

from src.ml.components.transformer.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)


def make_decoder(resolution=8, mode="log"):
    layer = TransformerDecoderLayer(
        activation="relu",
        d_model=8,
        dim_feedforward=16,
        dropout=0.0,
        cross_attention=nn.MultiheadAttention(8, 2),
        n_heads=2,
    )
    return TransformerDecoder(
        d_model=8,
        frozen=False,
        interaction_layer=None,
        layer=layer,
        num_layers=1,
        num_queries=2,
        return_intermediate=True,
        box_refine=True,
        boxRPB=mode,
        resolution=resolution,
        stride=2,
    ).eval()


@pytest.mark.parametrize("mode", ["linear", "log", "both"])
def test_coordinate_cache_preserves_bias_across_sizes(mode):
    decoder = make_decoder(mode=mode)
    boxes = torch.tensor([[[0.5, 0.4, 0.2, 0.3]], [[0.2, 0.1, 0.1, 0.1]]])
    fresh = make_decoder(resolution=None, mode=mode)
    fresh.load_state_dict(decoder.state_dict(), strict=True)

    for size in ((4, 4), (2, 3), (4, 4)):
        fresh.coord_cache = None
        expected = fresh._get_rpb_matrix(boxes, size)
        actual = decoder._get_rpb_matrix(boxes, tuple(torch.tensor(size)))
        assert actual.shape == (1, 2, 2, size[0] * size[1])
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()


def test_coordinate_cache_follows_model_device_without_checkpoint_keys():
    decoder = make_decoder()
    boxes = torch.full((2, 1, 4), 0.5)
    keys = set(decoder.state_dict())
    decoder._get_rpb_matrix(boxes, (4, 4))
    decoder.to("meta")
    result = decoder._get_rpb_matrix(boxes.to("meta"), (4, 4))

    assert result.device.type == "meta"
    assert set(decoder.state_dict()) == keys
    assert all(coord.device.type == "meta" for coord in decoder.coord_cache)


def test_warmed_coordinate_bias_compiles_without_graph_breaks():
    decoder = make_decoder()
    boxes = torch.full((2, 1, 4), 0.5)
    size = tuple(torch.tensor([4, 4]))
    expected = decoder._get_rpb_matrix(boxes, size)
    compiled = torch.compile(decoder._get_rpb_matrix, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(boxes, size), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_coordinate_bias_survives_cpu_cuda_cpu_transfer():
    decoder = make_decoder()
    boxes = torch.full((2, 1, 4), 0.5)
    expected = decoder._get_rpb_matrix(boxes, (4, 4))
    decoder.cuda()
    actual = decoder._get_rpb_matrix(boxes.cuda(), (4, 4))
    torch.testing.assert_close(actual.cpu(), expected)
    decoder.cpu()
    torch.testing.assert_close(decoder._get_rpb_matrix(boxes, (4, 4)), expected)
