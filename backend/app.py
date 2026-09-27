from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .images import read_image
from .runtime import InferenceRuntime, ResourceNotFound


class Prompt(BaseModel):
    box: tuple[float, float, float, float]
    positive: bool = True


class PromptUpdate(BaseModel):
    box: tuple[float, float, float, float]


class PointPrompt(BaseModel):
    point: tuple[float, float]
    positive: bool = True


class PointUpdate(BaseModel):
    point: tuple[float, float]


class PointDelete(BaseModel):
    indices: list[int]


def create_app(runtime=None):
    runtime = InferenceRuntime() if runtime is None else runtime

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="SAM 3 Similar Object API", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ResourceNotFound)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    def change(session_id, operation, errors=(TypeError, ValueError, IndexError)):
        try:
            return runtime.change(session_id, operation)
        except errors as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "device": runtime.device,
            "model_loaded": runtime.model_loaded,
        }

    @app.post("/api/sessions")
    def create_session(file: UploadFile = File(...)):
        image = read_image(file)
        try:
            return runtime.create(image)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    @app.post("/api/sessions/{session_id}/prompts")
    def add_prompt(session_id: str, prompt: Prompt):
        return change(
            session_id,
            lambda model, data: model.add_prompt(
                data.state, prompt.box, positive=prompt.positive
            ),
            errors=(TypeError, ValueError),
        )

    @app.post("/api/sessions/{session_id}/points")
    def add_point(session_id: str, prompt: PointPrompt):
        return change(
            session_id,
            lambda model, data: model.add_point(
                data.state, prompt.point, positive=prompt.positive
            ),
            errors=(TypeError, ValueError, RuntimeError),
        )

    @app.post("/api/sessions/{session_id}/objects/{object_id}/exclude")
    def exclude_object(session_id: str, object_id: int):
        return change(
            session_id,
            lambda model, data: model.add_prompt(
                data.state, data.object_box(object_id), positive=False
            ),
            errors=(TypeError, ValueError),
        )

    @app.post("/api/sessions/{session_id}/refine")
    def refine_results(session_id: str):
        return change(
            session_id,
            lambda model, data: model.refine_objects(data.state, data.objects),
            errors=(TypeError, ValueError, RuntimeError),
        )

    @app.post("/api/sessions/{session_id}/points/delete")
    def delete_points(session_id: str, request: PointDelete):
        return change(
            session_id,
            lambda model, data: model.remove_prompts_at(data.state, request.indices),
        )

    @app.delete("/api/sessions/{session_id}/prompts/last")
    def remove_prompt(session_id: str):
        return change(session_id, lambda model, data: model.remove_prompt(data.state))

    @app.put("/api/sessions/{session_id}/prompts/{prompt_index}")
    def update_prompt(session_id: str, prompt_index: int, prompt: PromptUpdate):
        return change(
            session_id,
            lambda model, data: model.update_prompt(
                data.state, prompt_index, prompt.box
            ),
        )

    @app.delete("/api/sessions/{session_id}/prompts/{prompt_index}")
    @app.delete(
        "/api/sessions/{session_id}/points/{prompt_index}",
        operation_id="delete_point_api_sessions__session_id__points__prompt_index__delete",
        summary="Delete Point",
    )
    def delete_prompt(session_id: str, prompt_index: int):
        return change(
            session_id,
            lambda model, data: model.remove_prompt_at(data.state, prompt_index),
        )

    @app.put("/api/sessions/{session_id}/points/{prompt_index}")
    def update_point(session_id: str, prompt_index: int, prompt: PointUpdate):
        return change(
            session_id,
            lambda model, data: model.update_point(
                data.state, prompt_index, prompt.point
            ),
            errors=(TypeError, ValueError, IndexError, RuntimeError),
        )

    return app


app = create_app()
