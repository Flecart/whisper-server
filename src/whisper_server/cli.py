from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import uvicorn

from .app import create_app
from .config import Settings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="GPU-backed Deepgram-compatible faster-whisper server"
    )
    result.add_argument("--host")
    result.add_argument("--port", type=int)
    result.add_argument("--model")
    result.add_argument("--device", choices=("cpu", "cuda"))
    result.add_argument("--compute-type")
    result.add_argument(
        "--unsafe-allow-unauthenticated", action="store_true", default=None
    )
    return result


def main() -> None:
    args = parser().parse_args()
    settings = Settings.from_env()
    overrides = {key: value for key, value in vars(args).items() if value is not None}
    settings = replace(settings, **overrides).validated()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        ws_max_size=settings.max_frame_bytes,
        timeout_graceful_shutdown=15,
    )
