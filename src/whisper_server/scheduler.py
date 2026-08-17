from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from .agreement import Hypothesis, TranscriptionRequest


class Runtime(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> Hypothesis: ...


@dataclass(slots=True)
class _Job:
    session_id: str
    request: TranscriptionRequest
    future: asyncio.Future[Hypothesis]
    final: bool


class InferenceScheduler:
    """Fair FIFO scheduler with coalesced partials and prioritized final jobs."""

    def __init__(self, runtime: Runtime, concurrency: int = 1) -> None:
        self.runtime = runtime
        self.concurrency = concurrency
        self._finals: deque[_Job] = deque()
        self._partials: deque[_Job] = deque()
        self._queued: dict[str, _Job] = {}
        self._running: dict[str, asyncio.Future[Hypothesis]] = {}
        self._cancelled: set[str] = set()
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._closing = False

    async def start(self) -> None:
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._worker(), name=f"inference-{index}")
                for index in range(self.concurrency)
            ]

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            self._condition.notify_all()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(
        self, session_id: str, request: TranscriptionRequest, *, final: bool
    ) -> asyncio.Future[Hypothesis]:
        async with self._condition:
            self._cancelled.discard(session_id)
            existing = self._queued.get(session_id)
            if existing is not None:
                existing.request = request
                if final and not existing.final:
                    existing.final = True
                    try:
                        self._partials.remove(existing)
                    except ValueError:
                        pass
                    self._finals.append(existing)
                return existing.future
            if session_id in self._running:
                return self._running[session_id]
            future = asyncio.get_running_loop().create_future()
            job = _Job(session_id, request, future, final)
            self._queued[session_id] = job
            (self._finals if final else self._partials).append(job)
            self._condition.notify()
            return future

    async def cancel(self, session_id: str) -> None:
        async with self._condition:
            self._cancelled.add(session_id)
            job = self._queued.pop(session_id, None)
            if job is not None:
                for queue in (self._finals, self._partials):
                    try:
                        queue.remove(job)
                    except ValueError:
                        pass
                job.request = TranscriptionRequest(
                    audio=job.request.audio[:0],
                    audio_start=job.request.audio_start,
                    language=job.request.language,
                    context="",
                    final=job.request.final,
                )
                job.future.cancel()
            running = self._running.get(session_id)
            if running is not None:
                running.cancel()

    @property
    def queue_depth(self) -> int:
        return len(self._finals) + len(self._partials)

    async def _worker(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._closing or self._finals or self._partials
                )
                if self._closing and not self._finals and not self._partials:
                    return
                job = (
                    self._finals.popleft() if self._finals else self._partials.popleft()
                )
                self._queued.pop(job.session_id, None)
                self._running[job.session_id] = job.future
            try:
                result = await asyncio.to_thread(self.runtime.transcribe, job.request)
                if job.session_id not in self._cancelled and not job.future.done():
                    job.future.set_result(result)
            except Exception as error:
                if not job.future.done():
                    job.future.set_exception(error)
            finally:
                self._running.pop(job.session_id, None)
