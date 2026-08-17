import asyncio
import threading
import time

import numpy as np

from whisper_server.agreement import Hypothesis, TranscriptionRequest, Word
from whisper_server.scheduler import InferenceScheduler


def request(marker: int, *, final: bool = False) -> TranscriptionRequest:
    return TranscriptionRequest(
        np.array([marker], dtype=np.float32), 0, "en", "", final
    )


class RecordingRuntime:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def transcribe(self, item):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        marker = int(item.audio[0])
        self.calls.append(marker)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return Hypothesis((Word(f" {marker}", 0, 0.1),))


def test_fifo_final_priority_coalescing_and_single_model_access():
    async def run():
        runtime = RecordingRuntime()
        scheduler = InferenceScheduler(runtime, concurrency=1)
        first = await scheduler.submit("a", request(1), final=False)
        coalesced = await scheduler.submit("a", request(2), final=False)
        second = await scheduler.submit("b", request(3), final=False)
        final = await scheduler.submit("c", request(4, final=True), final=True)
        assert first is coalesced
        assert scheduler.queue_depth == 3
        await scheduler.start()
        await asyncio.gather(first, second, final)
        await scheduler.close()
        assert runtime.calls == [4, 2, 3]
        assert runtime.maximum_active == 1

    asyncio.run(run())


def test_final_supersedes_queued_partial_and_cancel_releases_job():
    async def run():
        runtime = RecordingRuntime()
        scheduler = InferenceScheduler(runtime)
        partial = await scheduler.submit("a", request(1), final=False)
        upgraded = await scheduler.submit("a", request(5, final=True), final=True)
        cancelled = await scheduler.submit("b", request(2), final=False)
        assert upgraded is partial
        await scheduler.cancel("b")
        assert cancelled.cancelled()
        assert scheduler.queue_depth == 1
        await scheduler.start()
        await upgraded
        await scheduler.close()
        assert runtime.calls == [5]

    asyncio.run(run())


def test_two_workers_are_configurable_but_not_the_default():
    async def run():
        runtime = RecordingRuntime()
        scheduler = InferenceScheduler(runtime, concurrency=2)
        one = await scheduler.submit("a", request(1), final=False)
        two = await scheduler.submit("b", request(2), final=False)
        await scheduler.start()
        await asyncio.gather(one, two)
        await scheduler.close()
        assert runtime.maximum_active == 2

    asyncio.run(run())
