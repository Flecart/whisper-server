import json

import numpy as np
from fastapi.testclient import TestClient

from whisper_server.agreement import Hypothesis, Word
from whisper_server.app import create_app
from whisper_server.config import Settings


class FakeRuntime:
    model_name = "mock-turbo"
    device = "cpu"
    compute_type = "int8"

    def transcribe(self, request):
        if not request.audio.any():
            return Hypothesis(())
        return Hypothesis(
            (Word(" città", request.audio_start, request.audio_start + 0.2, 0.9),)
        )

    def speech_timestamps(self, audio):
        return [{"start": 0, "end": len(audio)}] if audio.any() else []


def app(*, token=None, max_streams=2, idle=10):
    settings = Settings(
        device="cpu",
        compute_type="int8",
        api_token=token,
        max_streams=max_streams,
        idle_timeout_seconds=idle,
        partial_interval_seconds=10,
    )
    return create_app(settings, runtime_factory=lambda *_: FakeRuntime())


def listen_url(**changes):
    values = {
        "model": "nova-3",
        "language": "it-IT",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "interim_results": "true",
        "vad_events": "true",
        "endpointing": "300",
        "utterance_end_ms": "1000",
    }
    values.update(changes)
    return "/v1/listen?" + "&".join(f"{key}={value}" for key, value in values.items())


def test_health_ready_and_model_are_process_scoped():
    with TestClient(app()) as client:
        assert client.get("/health").json() == {"ok": True}
        ready = client.get("/ready").json()
        assert ready["model"] == "mock-turbo"
        assert ready["active_streams"] == 0
        assert ready["capacity"] == 2
        assert ready["inference_queue_depth"] == 0


def test_authentication_and_audio_shape_validation():
    with TestClient(app(token="secret")) as client:
        with client.websocket_connect(listen_url()) as websocket:
            assert websocket.receive_json()["variant"] == "AUTH_FAILED"
        with client.websocket_connect(
            listen_url(sample_rate="48000"),
            headers={"Authorization": "Token secret"},
        ) as websocket:
            message = websocket.receive_json()
            assert message["type"] == "Error"
            assert "16000" in message["description"]


def test_binary_pcm_finalize_close_and_deepgram_shapes():
    with TestClient(app()) as client:
        with client.websocket_connect(listen_url()) as websocket:
            websocket.send_bytes(np.ones(5120, dtype="<i2").tobytes())
            assert websocket.receive_json()["type"] == "SpeechStarted"
            websocket.send_text(json.dumps({"type": "KeepAlive"}))
            websocket.send_text(json.dumps({"type": "Finalize"}))
            result = websocket.receive_json()
            assert result["type"] == "Results"
            assert result["channel"]["alternatives"][0]["transcript"] == "città"
            assert result["is_final"] is True
            assert result["speech_final"] is True
            assert result["from_finalize"] is True
            websocket.send_text(json.dumps({"type": "CloseStream"}))
            assert websocket.receive_json()["type"] == "Metadata"


def test_finalize_without_speech_always_returns_empty_result():
    with TestClient(app()) as client:
        with client.websocket_connect(listen_url()) as websocket:
            websocket.send_text(json.dumps({"type": "Finalize"}))
            result = websocket.receive_json()
            assert result["type"] == "Results"
            assert result["from_finalize"] is True
            assert result["channel"]["alternatives"][0]["transcript"] == ""
            websocket.send_text(json.dumps({"type": "CloseStream"}))
            assert websocket.receive_json()["type"] == "Metadata"


def test_malformed_controls_and_odd_pcm_are_rejected():
    with TestClient(app()) as client:
        with client.websocket_connect(listen_url()) as websocket:
            websocket.send_text("not-json")
            assert websocket.receive_json()["type"] == "Error"
        with client.websocket_connect(listen_url()) as websocket:
            websocket.send_bytes(b"x")
            message = websocket.receive_json()
            assert message["variant"] == "INVALID_AUDIO"


def test_capacity_rejection_and_disconnect_cleanup():
    with TestClient(app(max_streams=1)) as client:
        with client.websocket_connect(listen_url()):
            with client.websocket_connect(listen_url()) as rejected:
                assert rejected.receive_json()["variant"] == "CAPACITY"
            assert client.get("/ready").json()["active_streams"] == 1
        assert client.get("/ready").json()["active_streams"] == 0
