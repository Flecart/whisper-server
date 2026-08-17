from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    model: str = "turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    max_streams: int = 2
    inference_concurrency: int = 1
    partial_interval_seconds: float = 1.25
    active_window_seconds: float = 15.0
    overlap_seconds: float = 1.5
    context_characters: int = 500
    max_session_seconds: float = 300.0
    idle_timeout_seconds: float = 10.0
    max_frame_bytes: int = 2**20
    max_buffer_seconds: float = 20.0
    api_token: str | None = None
    unsafe_allow_unauthenticated: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8765")),
            model=os.environ.get("WHISPER_MODEL", "turbo"),
            device=os.environ.get("WHISPER_DEVICE", "cuda"),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16"),
            max_streams=int(os.environ.get("MAX_STREAMS", "2")),
            inference_concurrency=int(os.environ.get("INFERENCE_CONCURRENCY", "1")),
            partial_interval_seconds=float(
                os.environ.get("PARTIAL_INTERVAL_SECONDS", "1.25")
            ),
            active_window_seconds=float(os.environ.get("ACTIVE_WINDOW_SECONDS", "15")),
            overlap_seconds=float(os.environ.get("OVERLAP_SECONDS", "1.5")),
            context_characters=int(os.environ.get("CONTEXT_CHARACTERS", "500")),
            max_session_seconds=float(os.environ.get("MAX_SESSION_SECONDS", "300")),
            idle_timeout_seconds=float(os.environ.get("IDLE_TIMEOUT_SECONDS", "10")),
            max_frame_bytes=int(os.environ.get("MAX_FRAME_BYTES", str(2**20))),
            max_buffer_seconds=float(os.environ.get("MAX_BUFFER_SECONDS", "20")),
            api_token=os.environ.get("WHISPER_API_TOKEN") or None,
            unsafe_allow_unauthenticated=_env_bool(
                "UNSAFE_ALLOW_UNAUTHENTICATED", False
            ),
        ).validated()

    def validated(self) -> Settings:
        if self.max_streams < 1 or self.inference_concurrency < 1:
            raise ValueError("MAX_STREAMS and INFERENCE_CONCURRENCY must be positive")
        if self.max_frame_bytes < 2 or self.max_buffer_seconds <= 0:
            raise ValueError("frame and buffer limits must be positive")
        if (
            not is_loopback(self.host)
            and not self.api_token
            and not self.unsafe_allow_unauthenticated
        ):
            raise ValueError(
                "WHISPER_API_TOKEN is required when HOST is not loopback; "
                "set UNSAFE_ALLOW_UNAUTHENTICATED=true only for development"
            )
        return self
