import numpy as np

from whisper_server.agreement import (
    Hypothesis,
    LocalAgreement,
    Word,
    normalize_punctuation,
    sanitize_hypothesis,
)


def word(text: str, start: float, end: float) -> Word:
    return Word(text, start, end, 0.9)


def test_local_agreement_commits_exact_unicode_prefix_on_second_hypothesis():
    state = LocalAgreement(commit_lag_seconds=0)
    state.append_pcm(np.zeros(2 * 16_000, dtype="<i2").tobytes())
    first = Hypothesis(
        (word(" Questa", 0.0, 0.4), word(" è", 0.4, 0.6), word(" vita", 0.6, 1.0))
    )
    assert state.accept(first).committed == ()

    second = Hypothesis(
        (
            word(" Questa", 0.0, 0.4),
            word(" è", 0.4, 0.6),
            word(" vita.", 0.6, 1.0),
        )
    )
    update = state.accept(second)

    assert "".join(item.text for item in update.committed) == " Questa è"
    assert "".join(item.text for item in update.interim) == " vita."


def test_punctuation_revision_is_not_committed_as_equal():
    state = LocalAgreement(commit_lag_seconds=0)
    state.append_pcm(np.zeros(2 * 16_000, dtype="<i2").tobytes())
    state.accept(Hypothesis((word(" Ciao", 0, 0.4), word(" mondo", 0.4, 0.8))))
    update = state.accept(
        Hypothesis((word(" Ciao", 0, 0.4), word(" mondo.", 0.4, 0.8)))
    )
    assert [item.text for item in update.committed] == [" Ciao"]
    assert [item.text for item in update.interim] == [" mondo."]


def test_final_flush_commits_changing_suffix_and_endpoint_reset():
    state = LocalAgreement(commit_lag_seconds=0)
    hypothesis = Hypothesis((word(" perché", 0, 0.5), word(" sì.", 0.5, 0.9)))
    state.accept(hypothesis)
    update = state.accept(hypothesis, final=True)
    assert "".join(item.text for item in update.committed) == " perché sì."
    assert update.interim == ()
    state.reset_utterance()
    assert state.previous == ()


def test_overlap_repetition_is_suppressed_but_later_repetition_is_preserved():
    state = LocalAgreement(overlap_seconds=1.5, commit_lag_seconds=0)
    original = Hypothesis((word(" hello", 0.0, 0.4), word(" world", 0.4, 0.8)))
    state.accept(original, final=True)

    overlap = Hypothesis(
        (word(" hello", 0.2, 0.6), word(" world", 0.6, 1.0), word(" again", 1.0, 1.3))
    )
    assert [item.text for item in state.accept(overlap, final=True).committed] == [
        " again"
    ]

    later = Hypothesis((word(" world", 4.0, 4.4),))
    assert [item.text for item in state.accept(later, final=True).committed] == [
        " world"
    ]


def test_overlap_repetition_suppression_tolerates_punctuation_revision():
    state = LocalAgreement(overlap_seconds=1.5, commit_lag_seconds=0)
    state.accept(
        Hypothesis((word(" città,", 0.0, 0.5), word(" perché?", 0.5, 0.9))),
        final=True,
    )
    revised = Hypothesis(
        (word(" perché", 0.7, 1.0), word(" la", 1.0, 1.2), word(" vita", 1.2, 1.5))
    )
    assert [item.text for item in state.accept(revised, final=True).committed] == [
        " la",
        " vita",
    ]


def test_audio_window_and_buffer_are_bounded_and_trim_at_word_boundary():
    state = LocalAgreement(
        active_window_seconds=15,
        max_buffer_seconds=20,
        overlap_seconds=1.5,
        commit_lag_seconds=0,
    )
    state.append_pcm(np.zeros(25 * 16_000, dtype="<i2").tobytes())
    assert len(state.audio) == 20 * 16_000
    assert state.audio_start == 5.0
    request = state.request("it", final=False)
    assert len(request.audio) == 15 * 16_000
    assert request.audio_start == 10.0

    state.accept(Hypothesis((word(" fine", 18.0, 19.0),)), final=True)
    assert state.audio_start >= 17.49


def test_committed_context_is_bounded():
    state = LocalAgreement(context_characters=8, commit_lag_seconds=0)
    state.accept(
        Hypothesis((word(" abcdef", 0, 0.5), word(" ghij", 0.5, 1))), final=True
    )
    assert state.request("en", final=False).context == "def ghij"


def test_runaway_single_word_is_bounded_before_live_output():
    words = tuple(word(" dark", index / 10, (index + 1) / 10) for index in range(200))
    safe = sanitize_hypothesis(words, audio_seconds=5)
    assert [item.text for item in safe] == [" dark", " dark", " dark"]


def test_runaway_phrase_is_bounded_without_removing_normal_words():
    repeated = tuple(
        word(f" {text}", index / 10, (index + 1) / 10)
        for index, text in enumerate((["thank", "you"] * 20) + ["friend"])
    )
    safe = sanitize_hypothesis(repeated, audio_seconds=5)
    assert [item.text for item in safe] == [
        " thank",
        " you",
        " thank",
        " you",
        " thank",
        " you",
        " friend",
    ]


def test_implausibly_dense_nonrepeating_hypothesis_is_bounded():
    words = tuple(
        word(f" word{index}", index / 100, (index + 1) / 100) for index in range(100)
    )
    assert len(sanitize_hypothesis(words, audio_seconds=1)) == 16


def test_recent_agreed_words_remain_revisable_until_commit_lag_passes():
    state = LocalAgreement(commit_lag_seconds=1.0)
    state.append_pcm(np.zeros(2 * 16_000, dtype="<i2").tobytes())
    hypothesis = Hypothesis((word(" introduce", 0.1, 0.5), word(" yourself", 1.1, 1.5)))
    state.accept(hypothesis)
    update = state.accept(hypothesis)
    assert [item.text for item in update.committed] == [" introduce"]
    assert [item.text for item in update.interim] == [" yourself"]


def test_pathological_punctuation_runs_are_normalized():
    assert normalize_punctuation("......") == "."
    assert normalize_punctuation(" word.,") == " word,"
    assert normalize_punctuation(" word,.") == " word."
    assert normalize_punctuation("don't") == "don't"
