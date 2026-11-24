"""HTTP and WebSocket entry points for the inspection workspace."""

import json
import logging
import os
from contextlib import asynccontextmanager
from math import isfinite
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .batch import MAX_VIDEO_BYTES, analyze_video
from .evidence import (
    CLASSES, DEFAULT_MODEL, ROOT, annotations, figure_path, model_registry, project_data, training_run, validation_case,
)
from .inference import MAX_IMAGE_BYTES, InspectionEngine, InvalidInput
from .samples import FPS, FRAMES, sample_annotations, sample_clip, sample_image

logger = logging.getLogger(__name__)


class StreamFrame(BaseModel):
    id: int = Field(ge=0)
    source_time: float = Field(ge=0, allow_inf_nan=False)
    model: str = DEFAULT_MODEL


async def read_payload(request: Request, limit: int, detail: str) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(413, detail)
        data.extend(chunk)
    return bytes(data)


def create_app(device: str | None = None) -> FastAPI:
    engine = InspectionEngine(device or os.environ.get("ATDL_DEVICE", "auto"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.engine_error = None
        try:
            await run_in_threadpool(engine.start)
        except Exception as error:
            logger.exception("Inspection engine startup failed")
            application.state.engine_error = str(error)
        yield

    app = FastAPI(title="ATDL Steel Inspection Lab", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {
            "ready": app.state.engine_error is None,
            "error": app.state.engine_error,
            **engine.status(),
        }

    @app.get("/api/project")
    def project():
        return project_data()

    @app.get("/api/training/{model}/{seed}")
    def training(model: str, seed: int):
        record = training_run(model, seed)
        if record is None:
            raise HTTPException(404, "Training log unavailable.")
        return record

    @app.get("/api/figures/{name}")
    def figure(name: str):
        path = figure_path(name)
        if path is None:
            raise HTTPException(404, "Figure unavailable.")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/cases/{name}/image")
    def case_image(name: str):
        case = validation_case(name)
        if case is None:
            raise HTTPException(404, "Restore the local validation images and labels to open this case.")
        return FileResponse(case["image"], media_type="image/jpeg")

    @app.get("/api/cases/{name}/annotation")
    def case_annotation(name: str):
        case = validation_case(name)
        if case is None:
            raise HTTPException(404, "Validation annotation unavailable.")
        return {"boxes": annotations(case["label"]), "kind": "NEU-DET validation"}

    @app.get("/api/sample/image")
    def demo_image(
        position: int = Query(0, ge=0, lt=FRAMES),
        transform: Literal["original", "flip", "rotate"] = "original",
    ):
        return Response(sample_image(position, transform), media_type="image/png")

    @app.get("/api/sample/annotation")
    def demo_annotation(position: int = Query(0, ge=0, lt=FRAMES)):
        return {"boxes": sample_annotations(position), "kind": "Synthetic reference", "fps": FPS}

    @app.get("/api/sample/clip")
    def demo_clip(request: Request):
        data = sample_clip()
        size = len(data)
        headers = {"Accept-Ranges": "bytes"}
        byte_range = request.headers.get("range")
        if byte_range:
            try:
                unit, _, bounds = byte_range.partition("=")
                left, _, right = bounds.partition("-")
                if unit != "bytes" or "-" not in bounds or "," in bounds:
                    raise ValueError
                start = int(left) if left else max(0, size - int(right))
                end = min(size - 1, int(right)) if left and right else size - 1
                if not 0 <= start <= end < size:
                    raise ValueError
            except ValueError:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            return Response(data[start:end + 1], media_type="video/webm", status_code=206, headers=headers)
        return Response(data, media_type="video/webm", headers=headers)

    @app.post("/api/predict")
    async def predict(request: Request, model: str = DEFAULT_MODEL, trace: bool = False, activations: bool = False):
        if app.state.engine_error:
            raise HTTPException(503, app.state.engine_error)
        payload = await read_payload(request, MAX_IMAGE_BYTES, "Choose an image under 12 MiB.")
        try:
            return await run_in_threadpool(engine.predict, payload, model, trace, activations)
        except InvalidInput as error:
            raise HTTPException(422, str(error)) from error
        except (OSError, RuntimeError) as error:
            logger.exception("Prediction failed")
            raise HTTPException(503, f"Prediction failed: {error}") from error

    @app.post("/api/batch")
    async def batch(request: Request, model: str, thresholds: str = Query(max_length=1024)):
        if app.state.engine_error:
            raise HTTPException(503, app.state.engine_error)
        try:
            cutoffs = json.loads(thresholds)
            valid = isinstance(cutoffs, dict) and set(cutoffs) == set(CLASSES) and all(
                type(value) in (int, float) and isfinite(value) and 0.01 <= value <= 1
                for value in cutoffs.values()
            )
            if model not in model_registry() or not valid:
                raise ValueError
        except (ValueError, OverflowError):
            raise HTTPException(422, "Choose a registered checkpoint and one cutoff per defect class.")
        payload = await read_payload(request, MAX_VIDEO_BYTES, "Choose a video under 64 MiB.")

        async def events():
            iterator = analyze_video(engine, payload, model, cutoffs)
            try:
                while (line := await run_in_threadpool(next, iterator, None)) is not None:
                    yield line
            except (InvalidInput, OSError, RuntimeError) as error:
                yield json.dumps({"kind": "error", "error": str(error)}) + "\n"
            finally:
                iterator.close()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.websocket("/api/stream")
    async def stream(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                options = message.get("text")
                if options is None:
                    await socket.close(code=1003, reason="Send frame metadata as JSON text.")
                    return
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is None:
                    await socket.close(code=1003, reason="Send the image as a binary message.")
                    return
                frame_id = None
                try:
                    frame = StreamFrame.model_validate_json(options)
                    frame_id = frame.id
                    if app.state.engine_error:
                        raise RuntimeError(app.state.engine_error)
                    result = await run_in_threadpool(engine.predict, data, frame.model)
                except ValidationError:
                    await socket.send_json({"id": frame_id, "error": "Invalid frame metadata."})
                except (InvalidInput, OSError, RuntimeError) as error:
                    await socket.send_json({"id": frame_id, "error": str(error)})
                else:
                    await socket.send_json({"id": frame.id, "source_time": frame.source_time, "result": result})
        except WebSocketDisconnect:
            return

    dist = ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:
        @app.get("/")
        def setup():
            raise HTTPException(503, "Build the frontend with npm --prefix web run build.")

    return app


app = create_app()
