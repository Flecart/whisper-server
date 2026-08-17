import pytest

from whisper_server.agreement import Word
from whisper_server.protocol import (
    ListenOptions,
    ProtocolError,
    normalize_language,
    results_message,
)


def test_language_normalizes_bcp47_without_losing_supported_whisper_codes():
    assert normalize_language("it-IT") == "it"
    assert normalize_language("EN_us") == "en"
    assert normalize_language("zh-Hant-TW") == "zh"


@pytest.mark.parametrize(
    ("key", "value", "detail"),
    [
        ("encoding", "opus", "linear16"),
        ("sample_rate", "48000", "16000"),
        ("channels", "2", "must be 1"),
        ("interim_results", "yes", "true or false"),
        ("extra", "x", "unsupported"),
    ],
)
def test_query_validation(key, value, detail):
    with pytest.raises(ProtocolError, match=detail):
        ListenOptions.parse([(key, value)])


def test_voxkey_options_and_repeated_keyterms_are_accepted():
    options = ListenOptions.parse(
        [
            ("model", "nova-3"),
            ("language", "it-IT"),
            ("encoding", "linear16"),
            ("sample_rate", "16000"),
            ("channels", "1"),
            ("interim_results", "true"),
            ("vad_events", "true"),
            ("endpointing", "300"),
            ("utterance_end_ms", "1000"),
            ("punctuate", "true"),
            ("smart_format", "true"),
            ("mip_opt_out", "false"),
            ("keyterm", "VoxKey"),
            ("keyterm", "Città"),
        ]
    )
    assert options.language == "it"
    assert options.requested_model == "nova-3"
    assert options.keyterms == ("VoxKey", "Città")


def test_results_shape_preserves_accents_words_and_flags():
    message = results_message(
        [Word(" città", 0.1, 0.6, 0.875)],
        is_final=True,
        speech_final=True,
        from_finalize=True,
        duration=0.8,
        request_id="request",
    )
    assert message["type"] == "Results"
    assert message["is_final"] is True
    assert message["speech_final"] is True
    assert message["from_finalize"] is True
    alternative = message["channel"]["alternatives"][0]
    assert alternative["transcript"] == "città"
    assert alternative["words"][0] == {
        "word": "città",
        "start": 0.1,
        "end": 0.6,
        "confidence": 0.875,
        "punctuated_word": "città",
    }
