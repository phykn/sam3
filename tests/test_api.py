from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from backend.app import create_app
from backend.runtime import InferenceRuntime, ROOT
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


@pytest.fixture
def runtime():
    return InferenceRuntime(FakePredictor())


@pytest.fixture
def client(runtime):
    with TestClient(create_app(runtime)) as client:
        yield client


def create_session(client):
    response = client.post(
        "/api/sessions", files={"file": ("image.png", image_bytes(), "image/png")}
    )
    assert response.status_code == 200, response.text
    return "/api/sessions/" + response.json()["session_id"]


def test_backend_resolves_weights_from_repository_root(monkeypatch):
    monkeypatch.delenv("SAM3_WEIGHT", raising=False)
    monkeypatch.delenv("SAM3_VISUAL", raising=False)
    runtime = InferenceRuntime()
    assert ROOT == Path(__file__).resolve().parents[1]
    assert runtime.weight == ROOT / "weight/sam3.1_multiplex.pt"
    assert runtime.visual == ROOT / "weight/visual_token.pt"


def test_session_prompt_and_undo(client):
    path = create_session(client)
    prompted = client.post(path + "/prompts", json={"box": [1, 1, 3, 3]}).json()
    assert prompted["width"] == 8 and prompted["height"] == 6
    assert prompted["prompt_count"] == 1
    assert prompted["objects"][0]["mask"].startswith("data:image/png;base64,")
    undone = client.delete(path + "/prompts/last").json()
    assert undone["prompt_count"] == 0


def test_undo_validation_error_returns_400(client, runtime, monkeypatch):
    path = create_session(client)

    def invalid_undo(state):
        raise ValueError("at least one positive point is required")

    monkeypatch.setattr(runtime._model, "remove_prompt", invalid_undo)
    response = client.delete(path + "/prompts/last")
    assert response.status_code == 400
    assert response.json()["detail"] == "at least one positive point is required"


def test_session_updates_and_deletes_selected_prompt(client):
    path = create_session(client)
    for box in ([1, 1, 3, 3], [4, 1, 6, 3]):
        assert client.post(path + "/prompts", json={"box": box}).status_code == 200
    updated = client.put(path + "/prompts/0", json={"box": [0, 0, 2, 2]}).json()
    deleted = client.delete(path + "/prompts/0").json()
    assert updated["prompt_count"] == 2
    assert deleted["prompt_count"] == 1


def test_session_adds_moves_and_deletes_point_prompt(client):
    path = create_session(client)
    added = client.post(path + "/points", json={"point": [3, 2]}).json()
    updated = client.put(path + "/points/0", json={"point": [5, 4]}).json()
    deleted = client.delete(path + "/points/0").json()
    assert added["prompt_count"] == updated["prompt_count"] == 1
    assert deleted["prompt_count"] == 0


def test_session_clicks_result_to_add_exclude_box(client, runtime):
    path = create_session(client)
    client.post(path + "/prompts", json={"box": [0, 0, 4, 4]})
    excluded = client.post(path + "/objects/1/exclude").json()
    assert excluded["prompt_count"] == 2
    state = runtime._sessions[excluded["session_id"]].state
    assert state["box_labels"] == [1, 0]
    assert tuple(state["boxes"][-1]) == (1, 1, 3, 3)


def test_session_refines_current_results(client, runtime):
    path = create_session(client)
    client.post(path + "/prompts", json={"box": [0, 0, 4, 4]})
    refined = client.post(path + "/refine").json()
    repeated = client.post(path + "/refine").json()
    assert refined["objects"][0]["metrics"]["refine_score"] == pytest.approx(0.95)
    assert repeated == refined
    assert runtime._model.refine_calls == 2


def test_session_deletes_multiple_selected_points(client):
    path = create_session(client)
    for point in ([1, 1], [3, 2], [5, 4]):
        client.post(path + "/points", json={"point": point})
    deleted = client.post(path + "/points/delete", json={"indices": [0, 2]}).json()
    assert deleted["prompt_count"] == 1


def test_uploading_another_image_keeps_existing_session(client):
    first = create_session(client)
    create_session(client)
    prompted = client.post(first + "/points", json={"point": [3, 2]}).json()
    assert prompted["prompt_count"] == 1


def test_session_limit_evicts_oldest_image(client):
    paths = [create_session(client) for _ in range(9)]
    assert client.delete(paths[0] + "/prompts/last").status_code == 404
    assert client.delete(paths[1] + "/prompts/last").status_code == 200


def test_app_instances_have_separate_sessions_and_release_runtime():
    first, second = InferenceRuntime(FakePredictor()), InferenceRuntime(FakePredictor())
    with TestClient(create_app(first)) as a, TestClient(create_app(second)) as b:
        path = create_session(a)
        assert a.get("/api/health").json()["model_loaded"] is True
        assert b.delete(path + "/prompts/last").status_code == 404
    assert not first.model_loaded and not second.model_loaded
    assert not first._sessions and not second._sessions


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.setenv("SAM3_WEIGHT", "does-not-exist.pt")
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/sessions", files={"file": ("image.png", image_bytes(), "image/png")}
        )
    assert response.status_code == 503


def test_unknown_session_and_object_return_404(client):
    assert (
        client.post("/api/sessions/absent/points", json={"point": [1, 1]}).status_code
        == 404
    )
    path = create_session(client)
    assert client.post(path + "/objects/500/exclude").status_code == 404


def test_schema_and_model_errors_are_http_errors(client):
    path = create_session(client)
    assert client.post(path + "/points", json={"point": [1]}).status_code == 422
    assert client.put(path + "/points/9", json={"point": [1, 1]}).status_code == 400


def test_session_rejects_non_image_upload(client):
    response = client.post(
        "/api/sessions", files={"file": ("text.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_oversized_image_is_rejected_before_decoding(client, monkeypatch):
    raw = image_bytes()
    monkeypatch.setattr(images, "MAX_PIXELS", 10)

    def reject_decode(*args, **kwargs):
        pytest.fail("image decoded before dimensions were checked")

    monkeypatch.setattr(Image.Image, "convert", reject_decode)
    response = client.post(
        "/api/sessions", files={"file": ("image.png", raw, "image/png")}
    )
    assert response.status_code == 413


def test_pillow_decompression_bomb_is_rejected(client, monkeypatch):
    raw = image_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    response = client.post(
        "/api/sessions", files={"file": ("image.png", raw, "image/png")}
    )
    assert response.status_code == 413


def test_upload_byte_limit_precedes_image_parsing(monkeypatch):
    monkeypatch.setattr(images, "MAX_BYTES", 2)
    file = UploadFile(
        BytesIO(b"not an image"),
        filename="image.png",
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(HTTPException) as error:
        images.read_image(file)
    assert error.value.status_code == 413
    assert file.file.tell() == 3


def test_invalid_image_returns_decode_error(client):
    response = client.post(
        "/api/sessions", files={"file": ("image.png", b"not an image", "image/png")}
    )
    assert response.status_code == 400


def test_mask_response_preserves_pixels_and_fields():
    import base64

    roi = np.array([[True, False], [False, True]])
    metrics = {"score": 0.75}
    result = images.pack_objects(
        [{"object_id": 7, "box": (1, 2, 3, 4), "roi": roi, "metrics": metrics}]
    )[0]
    assert set(result) == {"object_id", "box", "mask", "color", "metrics"}
    assert result["object_id"] == 7 and result["box"] == [1, 2, 3, 4]
    assert result["metrics"] == metrics
    rgba = np.array(Image.open(BytesIO(base64.b64decode(result["mask"].split(",")[1]))))
    np.testing.assert_array_equal(rgba[..., 3], roi.astype(np.uint8) * 132)
    np.testing.assert_array_equal(rgba[..., :3], np.tile([100, 217, 194], (2, 2, 1)))
