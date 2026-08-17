import asyncio
import threading

import numpy as np

from whisper_server.agreement import Hypothesis
from whisper_server.config import Settings
from whisper_server.protocol import ListenOptions
from whisper_server.session import StreamingSession


class WebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, payload):
        self.messages.append(payload)


class Scheduler:
    def __init__(self):
        self.requests = []

    async def submit(self, session_id, request, *, final):
        self.requests.append((session_id, request, final))
        return asyncio.get_running_loop().create_future()


class Runtime:
    def __init__(self, release=None):
        self.release = release

    def speech_timestamps(self, audio):
        if self.release is not None:
            self.release.wait(timeout=1)
        return []

    def transcribe(self, request):
        return Hypothesis(())


def options():
    return ListenOptions(
        language="en",
        interim_results=True,
        vad_events=True,
        endpointing_ms=300,
        utterance_end_ms=1000,
        punctuate=True,
        smart_format=True,
        keyterms=(),
        mip_opt_out=False,
        requested_model="nova-3",
    )


def session(runtime, scheduler):
    return StreamingSession(
        WebSocket(),
        request_id="test-stream",
        options=options(),
        settings=Settings(device="cpu", compute_type="int8"),
        scheduler=scheduler,
        runtime=runtime,
    )


def test_slow_vad_does_not_block_finalize_control():
    async def run():
        release = threading.Event()
        scheduler = Scheduler()
        stream = session(Runtime(release), scheduler)
        pcm = np.zeros(5120, dtype="<i2").tobytes()

        await stream._audio(pcm)
        assert stream._vad_task is not None
        assert not stream._vad_task.done()

        assert await stream._control('{"type":"Finalize"}')
        await stream._maybe_schedule()
        assert scheduler.requests[0][2] is True

        release.set()
        await asyncio.wait_for(stream._vad_task, timeout=1)
        await stream._complete_vad_if_ready()
        stream._inference.cancel()

    asyncio.run(run())


def test_endpoint_is_not_queued_again_while_final_inference_runs():
    async def run():
        scheduler = Scheduler()
        stream = session(Runtime(), scheduler)
        stream._speech_active = True
        pcm = np.zeros(5120, dtype="<i2").tobytes()

        await stream._audio(pcm)
        await asyncio.wait_for(stream._vad_task, timeout=1)
        await stream._complete_vad_if_ready()
        assert stream._pending_final == "endpoint"
        await stream._maybe_schedule()
        assert stream._pending_final is None
        assert stream._inference_final is True

        await stream._audio(pcm)
        await asyncio.wait_for(stream._vad_task, timeout=1)
        await stream._complete_vad_if_ready()
        assert stream._pending_final is None
        assert len(scheduler.requests) == 1
        stream._inference.cancel()

    asyncio.run(run())
