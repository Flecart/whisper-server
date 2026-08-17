from __future__ import annotations

import logging
from typing import cast

from .agreement import (
    FloatArray,
    Hypothesis,
    TranscriptionRequest,
    Word,
    normalize_punctuation,
    sanitize_hypothesis,
)

LOG = logging.getLogger(__name__)


class FasterWhisperRuntime:
    """One process-wide faster-whisper model shared by every stream."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        LOG.info("Loading Whisper %s on %s with %s", model_name, device, compute_type)
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        LOG.info("Whisper model is ready")

    def transcribe(self, request: TranscriptionRequest) -> Hypothesis:
        segments, info = self.model.transcribe(
            request.audio,
            language=request.language,
            task="transcribe",
            beam_size=5 if request.final else 1,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=request.context or None,
            repetition_penalty=1.1,
        )
        words: list[Word] = []
        for segment in segments:
            for word in segment.words or ():
                words.append(
                    Word(
                        text=normalize_punctuation(word.word),
                        start=request.audio_start + float(word.start),
                        end=request.audio_start + float(word.end),
                        confidence=float(word.probability),
                    )
                )
        safe_words = sanitize_hypothesis(
            tuple(words), audio_seconds=len(request.audio) / 16_000
        )
        if len(safe_words) != len(words):
            LOG.warning(
                "Suppressed runaway hypothesis words (raw=%d safe=%d audio=%.2fs)",
                len(words),
                len(safe_words),
                len(request.audio) / 16_000,
            )
        return Hypothesis(
            safe_words,
            language=getattr(info, "language", request.language),
            language_probability=float(getattr(info, "language_probability", 0.0)),
        )

    def speech_timestamps(self, audio: FloatArray) -> list[dict[str, int]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return cast(
            list[dict[str, int]],
            get_speech_timestamps(
                audio,
                VadOptions(
                    threshold=0.5,
                    min_speech_duration_ms=100,
                    min_silence_duration_ms=250,
                    speech_pad_ms=30,
                ),
                sampling_rate=16_000,
            ),
        )
