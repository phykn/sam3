from pathlib import Path

import torch

from src.ml.components.backbone.neck import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from src.ml.components.backbone.vit import PatchEmbed, ViT
from src.ml.structures import NestedTensor


def test_vit_is_the_backbone_vit_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "backbone" / "vit.py").is_file()
    assert not (root / "src" / "ml" / "backbone").exists()
    assert not (root / "src" / "backbone").exists()
    assert not (root / "src" / "vit.py").exists()
    assert ViT.__module__ == "src.ml.components.backbone.vit"
    assert PatchEmbed.__module__ == "src.ml.components.backbone.vit"


def test_neck_is_the_backbone_neck_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "backbone" / "neck.py").is_file()
    assert not (root / "src" / "ml" / "backbone").exists()
    assert not (root / "src" / "backbone").exists()
    assert not (root / "src" / "neck.py").exists()
    assert Sam3DualViTDetNeck.__module__ == "src.ml.components.backbone.neck"
    assert Sam3TriViTDetNeck.__module__ == "src.ml.components.backbone.neck"


def test_backbone_package_does_not_reexport_internal_modules():
    import src.ml.components.backbone as backbone

    for name in (
        "Sam3DualViTDetNeck",
        "Sam3TriViTDetNeck",
        "ViT",
    ):
        assert not hasattr(backbone, name)


def test_multiscale_necks_resize_padding_masks_without_changing_source():
    class Trunk(torch.nn.Module):
        channel_list = [8]

        def forward(self, value):
            return [value]

    neck = Sam3TriViTDetNeck(Trunk(), torch.nn.Identity(), 8)
    mask = torch.tensor([[[False, True], [False, True]]])
    source = NestedTensor(torch.randn(1, 8, 2, 2), mask)
    outputs = neck(source)
    for features in outputs[::2]:
        for item in features:
            assert item.mask.shape == (1, *item.tensors.shape[-2:])
            expected = torch.nn.functional.interpolate(
                mask[:, None].float(), item.tensors.shape[-2:], mode="nearest"
            )[:, 0].bool()
            torch.testing.assert_close(item.mask, expected)
    assert source.mask is mask
    assert mask.shape == (1, 2, 2)


def test_vit_class_embedding_uses_input_layer_decay():
    vit = ViT.__new__(ViT)
    torch.nn.Module.__init__(vit)
    vit.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    assert vit.get_layer_id("class_embedding") == 0
    assert vit.get_layer_id("blocks.1.attn.qkv.weight") == 2
