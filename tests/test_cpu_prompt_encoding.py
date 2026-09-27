from types import SimpleNamespace

import torch
from torch import nn

from src.ml.components.grounding.geometry import SequenceGeometryEncoder
from src.ml.components.video.tracker.memory.context import (
    encode_temporal_positions,
    get_1d_sine_pe,
)


def test_box_pooling_runs_without_an_accelerator(monkeypatch):
    def no_pin(*args, **kwargs):
        raise RuntimeError("no accelerator available")

    monkeypatch.setattr(torch.Tensor, "pin_memory", no_pin)
    encoder = SequenceGeometryEncoder(
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=False,
        points_pos_enc=False,
        boxes_direct_project=False,
        boxes_pool=True,
        boxes_pos_enc=False,
        d_model=4,
        pos_enc=None,
        num_layers=0,
        layer=None,
        roi_size=2,
    ).eval()
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    mask = torch.zeros(1, 1, dtype=torch.bool)
    labels = torch.ones(1, 1, dtype=torch.long)
    features = torch.ones(1, 4, 4, 6)

    embeds, actual_mask = encoder._encode_boxes(boxes, mask, labels, features)
    expected = encoder.boxes_pool_project(torch.ones(1, 4, 2, 2)).view(1, 1, 4)
    expected = expected + encoder.label_embed(labels)
    torch.testing.assert_close(embeds, expected)
    torch.testing.assert_close(actual_mask, mask)


def test_temporal_positions_run_without_an_accelerator(monkeypatch):
    def no_pin(*args, **kwargs):
        raise RuntimeError("no accelerator available")

    monkeypatch.setattr(torch.Tensor, "pin_memory", no_pin)
    model = SimpleNamespace(
        mem_dim=8, proj_tpos_enc_in_obj_ptrs=False, obj_ptr_tpos_proj=nn.Identity()
    )

    actual = encode_temporal_positions(model, [1, 3], torch.device("cpu"), max_abs_pos=5)
    expected = get_1d_sine_pe(torch.tensor([0.25, 0.75]), dim=8)
    torch.testing.assert_close(actual, expected)
