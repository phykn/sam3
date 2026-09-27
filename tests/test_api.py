from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from backend import app as api
from backend import images
from starlette.datastructures import Headers


class FakePredictor:
    def __init__(self):
        self.refine_calls = 0

    def start(self, image):
        return {"boxes": [], "box_labels": [], "points": [], "size": image.size}

    def add_prompt(self, state, box, positive=True):
        state["boxes"].append(box)
        state["box_labels"].append(int(positive))
        state["points"].append(None)
        return self.add_prompt_result()

    def add_point(self, state, point, positive=True):
        state["boxes"].append([0, 0, 2, 2])
        state["box_labels"].append(int(positive))
        state["points"].append(point)
        return self.add_prompt_result()

    def update_point(self, state, index, point):
        state["points"][index] = point
        return self.add_prompt_result()

    @staticmethod
    def add_prompt_result():
        return [
            {
                "object_id": 1,
                "box": (1, 1, 3, 3),
                "roi": np.ones((2, 2), dtype=bool),
                "metrics": {"score": 0.9, "similarity": 0.8},
            }
        ]

    def remove_prompt(self, state):
        if state["boxes"]:
            state["boxes"].pop()
            state["box_labels"].pop()
            state["points"].pop()
        return []

    def update_prompt(self, state, index, box):
        state["boxes"][index] = box
        return self.add_prompt_result()

    def remove_prompt_at(self, state, index):
        state["boxes"].pop(index)
        state["box_labels"].pop(index)
        state["points"].pop(index)
        return []

    def remove_prompts_at(self, state, indices):
        for index in sorted(set(indices), reverse=True):
            state["boxes"].pop(index)
            state["box_labels"].pop(index)
            state["points"].pop(index)
        return self.add_prompt_result() if state["points"] else []

    def refine_objects(self, state, objects):
        assert state["boxes"]
        self.refine_calls += 1
        return [
            {
                **item,
                "metrics": {**item["metrics"], "refine_score": 0.95},
            }
            for item in objects
        ]


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 6), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def setup(monkeypatch):
    monkeypatch.setattr(api, "_predictor", FakePredictor())
    monkeypatch.setattr(api, "_sessions", {})


def upload(data, content_type):
    return UploadFile(
        BytesIO(data),
        filename="image.png",
        headers=Headers({"content-type": content_type}),
    )


def test_backend_resolves_weights_from_repository_root():
    assert api.ROOT == Path(__file__).resolve().parents[1]
    assert (api.ROOT / "weight" / "visual_token.pt").is_file()


def test_session_prompt_and_undo(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))

    session_id = created["session_id"]
    assert created["width"] == 8
    assert created["height"] == 6

    prompted = api.add_prompt(
        session_id,
        api.Prompt(box=(1, 1, 3, 3), positive=True),
    )
    assert prompted["prompt_count"] == 1
    assert prompted["objects"][0]["mask"].startswith("data:image/png;base64,")

    undone = api.remove_prompt(session_id)
    assert undone["prompt_count"] == 0


def test_session_updates_and_deletes_selected_prompt(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    api.add_prompt(session_id, api.Prompt(box=(1, 1, 3, 3), positive=True))
    api.add_prompt(session_id, api.Prompt(box=(4, 1, 6, 3), positive=True))

    updated = api.update_prompt(
        session_id,
        0,
        api.PromptUpdate(box=(0, 0, 2, 2)),
    )
    deleted = api.delete_prompt(session_id, 0)

    assert updated["prompt_count"] == 2
    assert deleted["prompt_count"] == 1


def test_session_adds_moves_and_deletes_point_prompt(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]

    added = api.add_point(
        session_id,
        api.PointPrompt(point=(3, 2), positive=True),
    )
    updated = api.update_point(
        session_id,
        0,
        api.PointUpdate(point=(5, 4)),
    )
    deleted = api.delete_point(session_id, 0)

    assert added["prompt_count"] == 1
    assert updated["prompt_count"] == 1
    assert deleted["prompt_count"] == 0


def test_session_clicks_result_to_add_exclude_box(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    api.add_prompt(session_id, api.Prompt(box=(0, 0, 4, 4), positive=True))

    excluded = api.exclude_object(session_id, 1)

    assert excluded["prompt_count"] == 2
    assert api._sessions[session_id]["state"]["box_labels"] == [1, 0]
    assert api._sessions[session_id]["state"]["boxes"][-1] == (1, 1, 3, 3)


def test_session_refines_current_results(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    api.add_prompt(session_id, api.Prompt(box=(0, 0, 4, 4), positive=True))

    refined = api.refine_results(session_id)
    repeated = api.refine_results(session_id)

    assert refined["objects"][0]["metrics"]["refine_score"] == pytest.approx(0.95)
    assert repeated == refined
    assert api._predictor.refine_calls == 2


def test_session_deletes_multiple_selected_points(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    for point in ((1, 1), (3, 2), (5, 4)):
        api.add_point(session_id, api.PointPrompt(point=point, positive=True))

    deleted = api.delete_points(
        session_id,
        api.PointDelete(indices=[0, 2]),
    )

    assert deleted["prompt_count"] == 1


def test_uploading_another_image_keeps_existing_session(monkeypatch):
    setup(monkeypatch)
    first = api.create_session(upload(image_bytes(), "image/png"))
    api.create_session(upload(image_bytes(), "image/png"))

    prompted = api.add_point(
        first["session_id"],
        api.PointPrompt(point=(3, 2), positive=True),
    )

    assert prompted["prompt_count"] == 1


def test_session_rejects_non_image_upload(monkeypatch):
    setup(monkeypatch)

    with pytest.raises(HTTPException) as error:
        api.create_session(upload(b"hello", "text/plain"))

    assert error.value.status_code == 415


def test_oversized_image_is_rejected_before_decoding(monkeypatch):
    setup(monkeypatch)
    raw = image_bytes()
    monkeypatch.setattr(images, "MAX_PIXELS", 10)

    def reject_decode(*args, **kwargs):
        pytest.fail("image was decoded before its dimensions were checked")

    monkeypatch.setattr(Image.Image, "convert", reject_decode)
    with pytest.raises(HTTPException) as error:
        api.create_session(upload(raw, "image/png"))

    assert error.value.status_code == 413
    assert api._sessions == {}


def test_pillow_decompression_bomb_is_rejected(monkeypatch):
    setup(monkeypatch)
    raw = image_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(HTTPException) as error:
        api.create_session(upload(raw, "image/png"))

    assert error.value.status_code == 413
    assert api._sessions == {}


def test_upload_byte_limit_precedes_image_parsing(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(images, "MAX_BYTES", 2)
    file = upload(b"not an image", "image/png")

    with pytest.raises(HTTPException) as error:
        api.create_session(file)

    assert error.value.status_code == 413
    assert file.file.tell() == 3


def test_invalid_image_returns_decode_error(monkeypatch):
    setup(monkeypatch)
    with pytest.raises(HTTPException) as error:
        api.create_session(upload(b"not an image", "image/png"))
    assert error.value.status_code == 400


def test_mask_response_preserves_pixels_and_fields():
    import base64

    roi = np.array([[True, False], [False, True]])
    metrics = {"score": 0.75}
    result = images.pack_objects(
        [{"object_id": 7, "box": (1, 2, 3, 4), "roi": roi, "metrics": metrics}]
    )[0]
    assert set(result) == {"object_id", "box", "mask", "color", "metrics"}
    assert result["object_id"] == 7
    assert result["box"] == [1, 2, 3, 4]
    assert result["metrics"] == metrics
    rgba = np.array(Image.open(BytesIO(base64.b64decode(result["mask"].split(",")[1]))))
    np.testing.assert_array_equal(rgba[..., 3], roi.astype(np.uint8) * 132)
    np.testing.assert_array_equal(rgba[..., :3], np.tile([100, 217, 194], (2, 2, 1)))
