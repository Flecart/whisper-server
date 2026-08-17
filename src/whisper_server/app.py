from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from .config import Settings
from .protocol import ListenOptions, ProtocolError, error_message, new_request_id
from .runtime import FasterWhisperRuntime
from .scheduler import InferenceScheduler
from .session import StreamingSession, StreamRegistry, authorized


def create_app(
    settings: Settings | None = None,
    *,
    runtime_factory: Callable[
        [str, str, str], FasterWhisperRuntime
    ] = FasterWhisperRuntime,
) -> FastAPI:
    settings = (settings or Settings.from_env()).validated()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = runtime_factory(
            settings.model, settings.device, settings.compute_type
        )
        scheduler = InferenceScheduler(
            runtime, concurrency=settings.inference_concurrency
        )
        await scheduler.start()
        app.state.runtime = runtime
        app.state.scheduler = scheduler
        app.state.registry = StreamRegistry(settings.max_streams)
        try:
            yield
        finally:
            await scheduler.close()

    app = FastAPI(
        title="whisper-server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        runtime = app.state.runtime
        scheduler = app.state.scheduler
        registry = app.state.registry
        return JSONResponse(
            {
                "ready": True,
                "model": runtime.model_name,
                "device": runtime.device,
                "compute_type": runtime.compute_type,
                "active_streams": registry.active,
                "capacity": settings.max_streams,
                "inference_concurrency": settings.inference_concurrency,
                "inference_queue_depth": scheduler.queue_depth,
            }
        )

    @app.websocket("/v1/listen")
    async def listen(websocket: WebSocket) -> None:
        await websocket.accept()
        if not authorized(websocket.headers.get("authorization"), settings.api_token):
            await websocket.send_json(error_message("invalid API token", "AUTH_FAILED"))
            await websocket.close(code=1008)
            return
        try:
            options = ListenOptions.parse(websocket.query_params.multi_items())
        except ProtocolError as error:
            await websocket.send_json(error_message(str(error)))
            await websocket.close(code=1008)
            return
        registry = app.state.registry
        if not await registry.acquire():
            await websocket.send_json(
                error_message("server is at stream capacity", "CAPACITY")
            )
            await websocket.close(code=1013)
            return
        try:
            session = StreamingSession(
                websocket,
                request_id=new_request_id(),
                options=options,
                settings=settings,
                scheduler=app.state.scheduler,
                runtime=app.state.runtime,
            )
            await session.run()
        finally:
            await registry.release()

    return app
