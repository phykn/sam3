import numpy as np
import pytest
import torch
from PIL import Image

from src.ops.mask import restore_mask
from src.finetune.checkpoint import FORMAT
from src.predict.single import SinglePredictor
from src.prepare.batch import build_prompts


class FakePromptEncoder:
    mask_input_size = (288, 288)

    def __init__(self):
        self.pe_calls = 0

    def get_dense_pe(self):
        self.pe_calls += 1
        return torch.zeros(1, 256, 72, 72)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = FakePromptEncoder()
        self.images = []
        self.image_grad_enabled = []
        self.prompts = []
        self.decodes = []

    @property
    def mask_input_size(self):
        return self.prompt_encoder.mask_input_size

    def get_image_position_encoding(self, device=None):
        pe = self.prompt_encoder.get_dense_pe()
        return pe if device is None else pe.to(device)

    def encode_image(self, images):
        self.images.append(tuple(images.shape))
        self.image_grad_enabled.append(torch.is_grad_enabled())
        batch = images.shape[0]
        return {
            "image_embed": torch.zeros(batch, 256, 72, 72),
            "high_res_features": (
                torch.zeros(batch, 32, 288, 288),
                torch.zeros(batch, 64, 144, 144),
            ),
        }

    def encode_prompt(self, points=None, boxes=None, masks=None):
        self.prompts.append((points, boxes, masks))
        batch = points[0].shape[0] if points is not None else masks.shape[0]
        return torch.zeros(batch, 3, 256), torch.zeros(batch, 256, 72, 72)

    def decode_masks(
        self,
        image_embed,
        high_res_features,
        prompt,
        image_pe,
        multimask=True,
        repeat_image=False,
        cond=None,
        prompt_type=None,
    ):
        self.decodes.append(
            {
                "image_embed": tuple(image_embed.shape),
                "high_res": [tuple(x.shape) for x in high_res_features],
                "image_pe": tuple(image_pe.shape),
                "multimask": multimask,
                "repeat_image": repeat_image,
                "cond": cond,
                "prompt_type": prompt_type,
            }
        )
        batch = prompt[0].shape[0]
        return (
            torch.ones(batch, 1, 288, 288),
            torch.full((batch, 1), 0.75),
            torch.zeros(batch, 1, 256),
            torch.ones(batch, 1),
        )


class FakeModelWithClasses(FakeModel):
    def decode_masks(self, *args, **kwargs):
        out = super().decode_masks(*args, **kwargs)
        batch = out[0].shape[0]
        classes = torch.tensor([[[2.0, -2.0]]]).expand(batch, -1, -1)
        return (*out, classes)


def test_single_predictor_predicts_from_box():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")

    objects = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        box=np.array([2, 1, 18, 9], dtype=np.float32),
        multimask=False,
    )

    assert len(objects) == 1
    item = objects[0]
    assert item["prompt_index"] == 0
    assert item["candidate_index"] == 0
    assert item["logit"].shape == (288, 288)
    assert item["metrics"]["score"] == pytest.approx(0.75)
    mask = restore_mask((10, 20), item["box"], item["roi"])
    assert mask.shape == (10, 20)
    assert mask.all()
    coords, labels = model.prompts[0][0]
    assert coords.shape == (1, 2, 2)
    assert labels.tolist() == [[2, 3]]
    assert model.decodes[0]["repeat_image"] is True
    assert model.decodes[0]["multimask"] is False


def test_finetune_prediction_adds_per_mask_class_scores():
    predictor = SinglePredictor(FakeModelWithClasses(), device="cpu")

    objects = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        point_coords=np.array([[10.0, 5.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask=False,
    )

    metrics = objects[0]["metrics"]
    assert objects[0]["class_id"] is None
    assert metrics["class_logits"] == [2.0, -2.0]
    np.testing.assert_allclose(
        metrics["class_scores"],
        1 / (1 + np.exp(-np.array(metrics["class_logits"]))),
        rtol=1e-6,
    )


def test_plain_prediction_does_not_add_class_keys():
    predictor = SinglePredictor(FakeModel(), device="cpu")

    objects = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        point_coords=np.array([[10.0, 5.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask=False,
    )

    assert "class_logits" not in objects[0]["metrics"]
    assert "class_scores" not in objects[0]["metrics"]


def test_single_predictor_uses_dummy_point_for_mask_only_prompt():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")

    predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        mask=np.ones((288, 288), dtype=np.float32),
    )

    coords, labels = model.prompts[0][0]
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[-1]]


def test_single_predictor_refines_from_logit():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    logit = np.ones((288, 288), dtype=np.float32)

    objects = predictor.refine(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        logit,
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )

    assert len(objects) == 1
    assert restore_mask((10, 20), objects[0]["box"], objects[0]["roi"]).all()
    assert model.prompts[0][2].dtype == torch.float32
    assert model.prompts[0][0][1].tolist() == [[1]]
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_encode_disables_gradients():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")

    with torch.enable_grad():
        predictor.encode(Image.new("RGB", (20, 10)))

    assert model.image_grad_enabled == [False]


def test_single_predictor_refines_embed_from_logit():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    embed = predictor.encode(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    objects = predictor.refine_embed(
        embed,
        np.ones((288, 288), dtype=np.float32),
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )

    assert len(objects) == 1
    assert model.prompts[0][2].dtype == torch.float32
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_can_keep_low_res_masks():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    embed = predictor.encode(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    out = predictor.predict_low(
        embed,
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
        multimask=False,
    )

    assert out["masks"].shape == (1, 1, 288, 288)
    assert out["logits"].shape == (1, 1, 288, 288)
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_passes_condition_and_prompt_type():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu", cond=2)
    embed = predictor.encode(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    predictor.predict_low(
        embed,
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )
    predictor.predict_low(
        embed,
        box=np.array([2, 1, 18, 9], dtype=np.float32),
        cond=1,
    )
    predictor.refine_embed(
        embed,
        np.ones((288, 288), dtype=np.float32),
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )

    assert model.decodes[0]["cond"] == 2
    assert model.decodes[0]["prompt_type"] == "point"
    assert model.decodes[1]["cond"] == 1
    assert model.decodes[1]["prompt_type"] == "box"
    assert model.decodes[2]["cond"] == 2
    assert model.decodes[2]["prompt_type"] == "mask"


def test_single_predictor_rejects_empty_prompt():
    predictor = SinglePredictor(FakeModel(), device="cpu")

    try:
        predictor.predict(Image.new("RGB", (20, 10), color=(0, 0, 0)))
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_single_predictor_keeps_fixed_image_options():
    predictor = SinglePredictor(FakeModel(), device="cpu")
    assert not hasattr(predictor, "mask_threshold")

    for kwargs in ({"image_size": 512}, {"mask_threshold": 0.5}):
        try:
            SinglePredictor(FakeModel(), device="cpu", **kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError(f"Expected TypeError for {kwargs}")


def test_single_predictor_exposes_one_low_level_method():
    predictor = SinglePredictor(FakeModel(), device="cpu")

    assert hasattr(predictor, "predict_low")
    assert not hasattr(predictor, "_predict_low")
    assert not hasattr(predictor, "predict_embed_low")
    assert not hasattr(predictor, "refine_low")


def test_single_predictor_from_path_uses_explicit_options(monkeypatch):
    monkeypatch.setattr("src.predict.single.Sam3ImageModel", lambda path: FakeModel())

    predictor = SinglePredictor.from_path("unused.pt", device="cpu", cond=3)

    assert predictor.device == torch.device("cpu")
    assert predictor.cond == 3
    with pytest.raises(TypeError):
        SinglePredictor.from_path("unused.pt", {"device": "cpu"})


def test_single_predictor_from_finetune_uses_saved_model_config(
    monkeypatch,
    tmp_path,
):
    captured = []
    checkpoint_path = tmp_path / "last.pt"
    torch.save(
        {
            "format": FORMAT,
            "model": {},
            "config": {
                "model": {
                    "path": "stale-base.pt",
                    "num_conditions": 2,
                    "num_experts": 3,
                    "num_classes": 4,
                    "lora_rank": 5,
                    "feature_rank": 6,
                }
            },
        },
        checkpoint_path,
    )

    def build(config):
        captured.append(config)
        return FakeModel()

    monkeypatch.setattr("src.predict.single.build_finetune_model", build)
    predictor = SinglePredictor.from_finetune(
        "new-base.pt",
        checkpoint_path,
        device="cpu",
        cond=1,
    )

    assert predictor.cond == 1
    assert captured == [
        {
            "path": "new-base.pt",
            "device": "cpu",
            "num_conditions": 2,
            "num_experts": 3,
            "num_classes": 4,
            "lora_rank": 5,
            "feature_rank": 6,
        }
    ]


def test_finetune_prediction_rejects_invalid_state_before_build(monkeypatch, tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format": FORMAT, "model": None, "config": {"model": {}}}, path)

    def build(config):
        raise AssertionError("invalid checkpoint must not construct a model")

    monkeypatch.setattr("src.predict.single.build_finetune_model", build)
    with pytest.raises(ValueError, match="model"):
        SinglePredictor.from_finetune("base.pt", path, device="cpu")


@pytest.mark.parametrize("kind", ["point", "box", "mixed", "mask"])
def test_training_and_prediction_build_equivalent_prompt_batches(kind):
    coords = np.array([[[10, 20]], [[30, 40]]], dtype=np.float32)
    labels = np.array([[1], [0]], dtype=np.int32)
    boxes = np.array([[1, 2, 50, 60], [3, 4, 70, 80]], dtype=np.float32)
    masks = np.stack([np.zeros((288, 288)), np.ones((288, 288))]).astype(np.float32)
    coords = coords if kind in ("point", "mixed") else None
    labels = labels if coords is not None else None
    boxes = boxes if kind in ("box", "mixed") else None
    masks = masks if kind == "mask" else None
    items = [
        {
            "points": None if coords is None else coords[idx],
            "point_labels": None if labels is None else labels[idx],
            "box": None if boxes is None else boxes[idx],
            "mask": None if masks is None else masks[idx],
        }
        for idx in range(2)
    ]
    train_points, train_masks = build_prompts(items, 1008, (288, 288), "cpu")
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    predictor.predict_low(
        {
            "orig_hw": (1008, 1008),
            "image_embed": torch.zeros(1, 256, 2, 2),
            "high_res": (torch.zeros(1, 32, 8, 8), torch.zeros(1, 64, 4, 4)),
        },
        point_coords=coords,
        point_labels=labels,
        box=boxes,
        mask=masks,
    )
    predict_points, _, predict_masks = model.prompts[0]
    for train, predict in zip(train_points, predict_points, strict=True):
        torch.testing.assert_close(train, predict, rtol=0, atol=0)
    if train_masks is None:
        assert predict_masks is None
    else:
        torch.testing.assert_close(train_masks, predict_masks, rtol=0, atol=0)
