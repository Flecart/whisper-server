from types import SimpleNamespace

import numpy as np

from whisper_server.agreement import TranscriptionRequest
from whisper_server.runtime import FasterWhisperRuntime


def test_runtime_uses_fast_partial_and_quality_final_settings():
    calls = []

    class Model:
        def transcribe(self, audio, **options):
            calls.append((audio, options))
            segment = SimpleNamespace(
                words=[
                    SimpleNamespace(word=" città", start=0.1, end=0.4, probability=0.9)
                ]
            )
            return iter([segment]), SimpleNamespace(
                language="it", language_probability=0.99
            )

    runtime = FasterWhisperRuntime.__new__(FasterWhisperRuntime)
    runtime.model = Model()
    audio = np.zeros(160, dtype=np.float32)
    partial = TranscriptionRequest(audio, 2.0, "it", "contesto", False)
    result = runtime.transcribe(partial)
    assert result.words[0].text == " città"
    assert result.words[0].start == 2.1
    assert calls[0][1]["beam_size"] == 1
    assert calls[0][1]["word_timestamps"] is True
    assert calls[0][1]["vad_filter"] is True
    assert calls[0][1]["condition_on_previous_text"] is False
    assert calls[0][1]["initial_prompt"] == "contesto"
    assert calls[0][1]["repetition_penalty"] == 1.1

    runtime.transcribe(TranscriptionRequest(audio, 0, "it", "", True))
    assert calls[1][1]["beam_size"] == 5
