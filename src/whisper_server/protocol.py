from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .agreement import Word

LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2,3}(?:[-_][a-zA-Z0-9]{2,8})*$")


class ProtocolError(ValueError):
    pass


def _boolean(name: str, value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ProtocolError(f"{name} must be true or false")


def _milliseconds(name: str, value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except ValueError as error:
        raise ProtocolError(f"{name} must be an integer") from error
    if result < 0 or result > 60_000:
        raise ProtocolError(f"{name} must be between 0 and 60000")
    return result


def normalize_language(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not LANGUAGE_RE.fullmatch(value):
        raise ProtocolError("language must be a BCP-47-style language tag")
    return value.replace("_", "-").split("-", 1)[0].lower()


@dataclass(frozen=True, slots=True)
class ListenOptions:
    language: str | None
    interim_results: bool
    vad_events: bool
    endpointing_ms: int
    utterance_end_ms: int
    punctuate: bool
    smart_format: bool
    keyterms: tuple[str, ...]
    mip_opt_out: bool
    requested_model: str

    @classmethod
    def parse(cls, items: Iterable[tuple[str, str]]) -> ListenOptions:
        pairs = list(items)
        values: dict[str, str] = {}
        keyterms: list[str] = []
        supported = {
            "language",
            "encoding",
            "sample_rate",
            "channels",
            "interim_results",
            "vad_events",
            "endpointing",
            "utterance_end_ms",
            "punctuate",
            "smart_format",
            "keyterm",
            "mip_opt_out",
            "model",
        }
        for key, value in pairs:
            if key not in supported:
                raise ProtocolError(f"unsupported query parameter: {key}")
            if key == "keyterm":
                if value.strip():
                    keyterms.append(value.strip())
            else:
                values[key] = value
        if values.get("encoding", "linear16").lower() != "linear16":
            raise ProtocolError("encoding must be linear16")
        if values.get("sample_rate", "16000") != "16000":
            raise ProtocolError("sample_rate must be 16000")
        if values.get("channels", "1") != "1":
            raise ProtocolError("channels must be 1")
        return cls(
            language=normalize_language(values.get("language")),
            interim_results=_boolean(
                "interim_results", values.get("interim_results"), True
            ),
            vad_events=_boolean("vad_events", values.get("vad_events"), False),
            endpointing_ms=_milliseconds("endpointing", values.get("endpointing"), 300),
            utterance_end_ms=_milliseconds(
                "utterance_end_ms", values.get("utterance_end_ms"), 1000
            ),
            punctuate=_boolean("punctuate", values.get("punctuate"), True),
            smart_format=_boolean("smart_format", values.get("smart_format"), True),
            keyterms=tuple(keyterms),
            mip_opt_out=_boolean("mip_opt_out", values.get("mip_opt_out"), False),
            requested_model=values.get("model", "nova-3"),
        )


def error_message(description: str, code: str = "BAD_REQUEST") -> dict[str, Any]:
    return {
        "type": "Error",
        "variant": code,
        "description": description,
    }


def results_message(
    words: list[Word],
    *,
    is_final: bool,
    speech_final: bool = False,
    from_finalize: bool = False,
    duration: float = 0.0,
    request_id: str,
) -> dict[str, Any]:
    transcript = "".join(word.text for word in words).strip()
    confidence = sum(word.confidence for word in words) / len(words) if words else 0.0
    return {
        "type": "Results",
        "channel_index": [0, 1],
        "duration": round(duration, 3),
        "start": round(words[0].start if words else duration, 3),
        "is_final": is_final,
        "speech_final": speech_final,
        "from_finalize": from_finalize,
        "channel": {
            "alternatives": [
                {
                    "transcript": transcript,
                    "confidence": round(confidence, 4),
                    "words": [word.as_deepgram() for word in words],
                }
            ]
        },
        "metadata": {"request_id": request_id},
    }


def speech_started_message(timestamp: float) -> dict[str, Any]:
    return {"type": "SpeechStarted", "channel": [0, 1], "timestamp": timestamp}


def utterance_end_message(last_word_end: float) -> dict[str, Any]:
    return {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": last_word_end}


def metadata_message(request_id: str, *, duration: float, model: str) -> dict[str, Any]:
    return {
        "type": "Metadata",
        "request_id": request_id,
        "duration": round(duration, 3),
        "channels": 1,
        "models": [model],
        "model_info": {model: {"name": model}},
    }


def new_request_id() -> str:
    return str(uuid.uuid4())
