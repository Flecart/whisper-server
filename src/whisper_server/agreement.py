from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0

    def as_deepgram(self) -> dict[str, object]:
        written = self.text.strip()
        return {
            "word": written,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 4),
            "punctuated_word": written,
        }


@dataclass(frozen=True, slots=True)
class Hypothesis:
    words: tuple[Word, ...]
    language: str | None = None
    language_probability: float = 0.0


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio: FloatArray
    audio_start: float
    language: str | None
    context: str
    final: bool


@dataclass(frozen=True, slots=True)
class AgreementUpdate:
    committed: tuple[Word, ...]
    interim: tuple[Word, ...]


def _same_word(left: Word, right: Word) -> bool:
    # Preserve punctuation and Unicode exactly; normalize only Whisper's
    # inconsistent leading spaces between adjacent word tokens.
    return left.text.strip() == right.text.strip()


def _same_overlap_word(left: Word, right: Word) -> bool:
    return _lexical(left.text) == _lexical(right.text)


def _lexical(text: str) -> str:
    normalized = "".join(
        character.casefold()
        for character in text.strip()
        if unicodedata.category(character)[0] in {"L", "N"}
    )
    return normalized or text.strip().casefold()


def normalize_punctuation(text: str) -> str:
    """Turn unstable punctuation runs into one dictation-friendly mark."""

    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\.([,;:])", r"\1", text)
    text = re.sub(r",\.", ".", text)
    return text


def sanitize_hypothesis(
    words: tuple[Word, ...],
    *,
    audio_seconds: float,
    maximum_repeats: int = 3,
) -> tuple[Word, ...]:
    """Bound decoder loops before an unstable hypothesis reaches a live client."""

    accepted: list[Word] = []
    lexical: list[str] = []
    for word in words:
        accepted.append(word)
        lexical.append(_lexical(word.text))
        for width in range(1, min(8, len(lexical) // maximum_repeats) + 1):
            repeated_width = width * (maximum_repeats + 1)
            if len(lexical) < repeated_width:
                continue
            block = lexical[-width:]
            if all(
                lexical[-width * (offset + 1) : -width * offset or None] == block
                for offset in range(1, maximum_repeats + 1)
            ):
                del accepted[-width:]
                del lexical[-width:]
                break

    # Conversational speech rarely exceeds eight words per second. The extra
    # eight-word margin accommodates very short windows and timestamp jitter.
    maximum_words = max(16, ceil(max(0.0, audio_seconds) * 8) + 8)
    return tuple(accepted[:maximum_words])


class LocalAgreement:
    """LocalAgreement-2 plus bounded audio/context state for one stream."""

    def __init__(
        self,
        *,
        active_window_seconds: float = 15.0,
        overlap_seconds: float = 1.5,
        max_buffer_seconds: float = 20.0,
        context_characters: int = 500,
        commit_lag_seconds: float = 1.0,
    ) -> None:
        self.active_window_seconds = active_window_seconds
        self.overlap_seconds = overlap_seconds
        self.max_buffer_seconds = max_buffer_seconds
        self.context_characters = context_characters
        self.commit_lag_seconds = commit_lag_seconds
        self.audio: FloatArray = np.empty(0, dtype=np.float32)
        self.audio_start = 0.0
        self.total_audio_seconds = 0.0
        self.previous: tuple[Word, ...] = ()
        self.committed_words: list[Word] = []
        self.committed_until = 0.0

    def append_pcm(self, pcm: bytes) -> None:
        if not pcm or len(pcm) % 2:
            raise ValueError("binary audio frames must contain 16-bit PCM samples")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.audio = np.concatenate((self.audio, samples))
        self.total_audio_seconds += len(samples) / SAMPLE_RATE
        self._bound_buffer()

    def _bound_buffer(self) -> None:
        maximum = int(self.max_buffer_seconds * SAMPLE_RATE)
        if len(self.audio) <= maximum:
            return
        removed = len(self.audio) - maximum
        self.audio = self.audio[removed:]
        self.audio_start += removed / SAMPLE_RATE
        self.previous = tuple(
            word for word in self.previous if word.end > self.audio_start
        )

    def request(self, language: str | None, *, final: bool) -> TranscriptionRequest:
        maximum = int(self.active_window_seconds * SAMPLE_RATE)
        audio = self.audio[-maximum:].copy()
        start = self.audio_start + (len(self.audio) - len(audio)) / SAMPLE_RATE
        context = "".join(word.text for word in self.committed_words)
        return TranscriptionRequest(
            audio=audio,
            audio_start=start,
            language=language,
            context=context[-self.context_characters :],
            final=final,
        )

    def accept(self, hypothesis: Hypothesis, *, final: bool = False) -> AgreementUpdate:
        current = tuple(
            word for word in hypothesis.words if word.end > self.committed_until + 0.01
        )
        current = self._suppress_overlap_repetition(current)
        if final:
            stable = current
            interim: tuple[Word, ...] = ()
        else:
            common = 0
            for old, new in zip(self.previous, current, strict=False):
                if not _same_word(old, new):
                    break
                common += 1
            commit_before = self.total_audio_seconds - self.commit_lag_seconds
            committable = sum(
                1 for word in current[:common] if word.end <= commit_before
            )
            stable = current[:committable]
            interim = current[committable:]
        if stable:
            self.committed_words.extend(stable)
            self.committed_until = max(self.committed_until, stable[-1].end)
            self._trim_at_boundary(self.committed_until)
        self.previous = interim
        return AgreementUpdate(stable, interim)

    def _suppress_overlap_repetition(self, words: tuple[Word, ...]) -> tuple[Word, ...]:
        if not words or not self.committed_words:
            return words
        if words[0].start > self.committed_until + self.overlap_seconds + 0.25:
            return words
        tail = self.committed_words[-20:]
        maximum = min(len(tail), len(words))
        for count in range(maximum, 0, -1):
            if all(
                _same_overlap_word(old, new)
                for old, new in zip(tail[-count:], words[:count], strict=True)
            ):
                return words[count:]
        return words

    def _trim_at_boundary(self, boundary: float) -> None:
        target = max(self.audio_start, boundary - self.overlap_seconds)
        samples = min(len(self.audio), int((target - self.audio_start) * SAMPLE_RATE))
        if samples > 0:
            self.audio = self.audio[samples:]
            self.audio_start += samples / SAMPLE_RATE

    def reset_utterance(self) -> None:
        self.previous = ()

    @property
    def has_uncommitted_audio(self) -> bool:
        return self.total_audio_seconds > self.committed_until + 0.01
