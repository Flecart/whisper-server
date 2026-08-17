from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .agreement import FloatArray, Hypothesis, LocalAgreement
from .config import Settings
from .protocol import (
    ListenOptions,
    error_message,
    metadata_message,
    results_message,
    speech_started_message,
    utterance_end_message,
)
from .runtime import FasterWhisperRuntime
from .scheduler import InferenceScheduler

LOG = logging.getLogger(__name__)


def authorized(header: str | None, token: str | None) -> bool:
    if token is None:
        return True
    if header is None or not header.startswith("Token "):
        return False
    return hmac.compare_digest(header[6:], token)


@dataclass(slots=True)
class StreamRegistry:
    maximum: int
    active: int = 0
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self.active >= self.maximum:
                return False
            self.active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)


class StreamingSession:
    def __init__(
        self,
        websocket: WebSocket,
        *,
        request_id: str,
        options: ListenOptions,
        settings: Settings,
        scheduler: InferenceScheduler,
        runtime: FasterWhisperRuntime,
    ) -> None:
        self.websocket = websocket
        self.request_id = request_id
        self.options = options
        self.settings = settings
        self.scheduler = scheduler
        self.runtime = runtime
        self.state = LocalAgreement(
            active_window_seconds=settings.active_window_seconds,
            overlap_seconds=settings.overlap_seconds,
            max_buffer_seconds=settings.max_buffer_seconds,
            context_characters=settings.context_characters,
        )
        self._started = time.monotonic()
        self._last_contact = self._started
        self._last_inference_audio = 0.0
        self._inference: asyncio.Future[Hypothesis] | None = None
        self._inference_final = False
        self._inference_reason = "partial"
        self._pending_final: str | None = None
        self._close_requested = False
        self._speech_active = False
        self._last_speech_audio = 0.0
        self._pending_utterance_end: float | None = None
        self._vad_audio: FloatArray = np.empty(0, dtype=np.float32)

    async def send(self, message: dict[str, Any]) -> None:
        await self.websocket.send_text(json.dumps(message, ensure_ascii=False))

    async def run(self) -> None:
        try:
            while True:
                now = time.monotonic()
                if now - self._started >= self.settings.max_session_seconds:
                    await self.send(
                        error_message("maximum session duration exceeded", "TIMEOUT")
                    )
                    await self.websocket.close(code=1000)
                    return
                if now - self._last_contact >= self.settings.idle_timeout_seconds:
                    await self.send(error_message("audio/keepalive timeout", "TIMEOUT"))
                    await self.websocket.close(code=1001)
                    return

                await self._complete_inference_if_ready()
                await self._maybe_schedule()
                await self._maybe_send_utterance_end()
                if self._close_requested and self._inference is None:
                    if self._pending_final is None:
                        await self._finish_close()
                        return

                try:
                    event = await asyncio.wait_for(
                        self.websocket.receive(), timeout=0.05
                    )
                except TimeoutError:
                    continue
                event_type = event.get("type")
                if event_type == "websocket.disconnect":
                    return
                if event.get("bytes") is not None:
                    await self._audio(event["bytes"])
                elif event.get("text") is not None:
                    if not await self._control(event["text"]):
                        return
        except WebSocketDisconnect:
            return
        finally:
            await self.scheduler.cancel(self.request_id)
            self.state.audio = self.state.audio[:0]
            self._vad_audio = self._vad_audio[:0]

    async def _audio(self, pcm: bytes) -> None:
        self._last_contact = time.monotonic()
        if len(pcm) > self.settings.max_frame_bytes:
            await self.send(
                error_message("binary frame exceeds 1 MiB", "FRAME_TOO_LARGE")
            )
            await self.websocket.close(code=1009)
            raise WebSocketDisconnect(code=1009)
        try:
            self.state.append_pcm(pcm)
        except ValueError as error:
            await self.send(error_message(str(error), "INVALID_AUDIO"))
            await self.websocket.close(code=1003)
            raise WebSocketDisconnect(code=1003) from error
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._vad_audio = np.concatenate((self._vad_audio, samples))
        if len(self._vad_audio) >= 5120:
            block = self._vad_audio
            self._vad_audio = self._vad_audio[:0]
            speech = await asyncio.to_thread(self.runtime.speech_timestamps, block)
            if speech:
                if not self._speech_active:
                    self._speech_active = True
                    if self.options.vad_events:
                        timestamp = max(
                            0.0,
                            self.state.total_audio_seconds - len(block) / 16_000,
                        )
                        await self.send(speech_started_message(round(timestamp, 3)))
                self._last_speech_audio = self.state.total_audio_seconds
            elif (
                self._speech_active
                and self.state.total_audio_seconds - self._last_speech_audio
                >= self.options.endpointing_ms / 1000
                and self._pending_final is None
            ):
                self._pending_final = "endpoint"

    async def _control(self, payload: str) -> bool:
        self._last_contact = time.monotonic()
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            await self.send(error_message("text frames must contain JSON"))
            await self.websocket.close(code=1003)
            return False
        message_type = message.get("type") if isinstance(message, dict) else None
        if message_type == "KeepAlive":
            return True
        if message_type == "Finalize":
            self._pending_final = "finalize"
            return True
        if message_type == "CloseStream":
            self._close_requested = True
            if self.state.has_uncommitted_audio:
                self._pending_final = "close"
            return True
        await self.send(error_message("unsupported control message"))
        await self.websocket.close(code=1003)
        return False

    async def _maybe_schedule(self) -> None:
        if self._inference is not None:
            return
        if self._pending_final is not None:
            reason = self._pending_final
            self._pending_final = None
            request = self.state.request(self.options.language, final=True)
            self._inference = await self.scheduler.submit(
                self.request_id, request, final=True
            )
            self._inference_final = True
            self._inference_reason = reason
            self._last_inference_audio = self.state.total_audio_seconds
            return
        enough_audio = (
            self.state.total_audio_seconds - self._last_inference_audio
            >= self.settings.partial_interval_seconds
        )
        if enough_audio and self._speech_active and not self._close_requested:
            request = self.state.request(self.options.language, final=False)
            self._inference = await self.scheduler.submit(
                self.request_id, request, final=False
            )
            self._inference_final = False
            self._inference_reason = "partial"
            self._last_inference_audio = self.state.total_audio_seconds

    async def _complete_inference_if_ready(self) -> None:
        if self._inference is None or not self._inference.done():
            return
        future = self._inference
        final = self._inference_final
        reason = self._inference_reason
        self._inference = None
        try:
            hypothesis = future.result()
        except asyncio.CancelledError:
            return
        except Exception as error:
            LOG.exception("Inference failed for stream %s", self.request_id)
            await self.send(
                error_message("Whisper inference failed", "INFERENCE_ERROR")
            )
            await self.websocket.close(code=1011)
            raise WebSocketDisconnect(code=1011) from error

        update = self.state.accept(hypothesis, final=final)
        duration = self.state.total_audio_seconds
        if update.committed:
            await self.send(
                results_message(
                    list(update.committed),
                    is_final=True,
                    speech_final=final,
                    from_finalize=reason == "finalize",
                    duration=duration,
                    request_id=self.request_id,
                )
            )
        elif final and reason == "finalize":
            await self.send(
                results_message(
                    [],
                    is_final=True,
                    speech_final=True,
                    from_finalize=True,
                    duration=duration,
                    request_id=self.request_id,
                )
            )
        if update.interim and self.options.interim_results:
            await self.send(
                results_message(
                    list(update.interim),
                    is_final=False,
                    duration=duration,
                    request_id=self.request_id,
                )
            )
        if final:
            self.state.reset_utterance()
            self._speech_active = False
            if reason == "endpoint" and update.committed:
                self._pending_utterance_end = update.committed[-1].end

    async def _maybe_send_utterance_end(self) -> None:
        if self._pending_utterance_end is None:
            return
        if (
            self.state.total_audio_seconds - self._pending_utterance_end
            >= self.options.utterance_end_ms / 1000
        ):
            await self.send(utterance_end_message(self._pending_utterance_end))
            self._pending_utterance_end = None

    async def _finish_close(self) -> None:
        await self.send(
            metadata_message(
                self.request_id,
                duration=self.state.total_audio_seconds,
                model=self.settings.model,
            )
        )
        await self.websocket.close(code=1000)
