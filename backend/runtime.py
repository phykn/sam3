import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

import torch

from src.predict import GroundPredictor

from .images import pack_objects

ROOT = Path(__file__).resolve().parents[1]


class ResourceNotFound(LookupError):
    pass


@dataclass
class Session:
    id: str
    state: dict
    width: int
    height: int
    objects: list = field(default_factory=list)

    def object_box(self, object_id):
        for item in self.objects:
            if item["object_id"] == object_id:
                return item["box"]
        raise ResourceNotFound("result object was not found")

    def result(self, objects):
        packed = pack_objects(objects)
        self.objects = objects
        return {
            "session_id": self.id,
            "width": self.width,
            "height": self.height,
            "prompt_count": len(self.state["boxes"]),
            "objects": packed,
        }


class InferenceRuntime:
    def __init__(self, model=None):
        self.weight = Path(
            os.getenv("SAM3_WEIGHT", ROOT / "weight/sam3.1_multiplex.pt")
        )
        self.visual = Path(os.getenv("SAM3_VISUAL", ROOT / "weight/visual_token.pt"))
        self.device = os.getenv(
            "SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = model
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    @property
    def model_loaded(self):
        return self._model is not None

    def _predictor(self):
        if self._model is None:
            if not self.weight.is_file() or not self.visual.is_file():
                raise RuntimeError("local grounding weights are missing")
            self._model = GroundPredictor.from_path(
                self.weight,
                self.visual,
                device=self.device,
                score_thr=0.45,
                sim_thr=0.45,
                negative_margin=0.05,
                top_k=None,
            )
        return self._model

    def create(self, image):
        with self._lock:
            state = self._predictor().start(image)
            session = Session(uuid4().hex, state, image.width, image.height)
            self._sessions[session.id] = session
            while len(self._sessions) > 8:
                self._sessions.pop(next(iter(self._sessions)))
            return session.result([])

    def change(self, session_id, operation):
        with self._lock:
            if session_id not in self._sessions:
                raise ResourceNotFound("image session was not found")
            session = self._sessions[session_id]
            objects = operation(self._predictor(), session)
            return session.result(objects)

    def close(self):
        with self._lock:
            self._sessions.clear()
            self._model = None
