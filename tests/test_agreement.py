import numpy as np

from whisper_server.agreement import Hypothesis, LocalAgreement, Word


def word(text: str, start: float, end: float) -> Word:
    return Word(text, start, end, 0.9)


def test_local_agreement_commits_exact_unicode_prefix_on_second_hypothesis():
    state = LocalAgreement()
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
    state = LocalAgreement()
    state.accept(Hypothesis((word(" Ciao", 0, 0.4), word(" mondo", 0.4, 0.8))))
    update = state.accept(
        Hypothesis((word(" Ciao", 0, 0.4), word(" mondo.", 0.4, 0.8)))
    )
    assert [item.text for item in update.committed] == [" Ciao"]
    assert [item.text for item in update.interim] == [" mondo."]


def test_final_flush_commits_changing_suffix_and_endpoint_reset():
    state = LocalAgreement()
    hypothesis = Hypothesis((word(" perché", 0, 0.5), word(" sì.", 0.5, 0.9)))
    state.accept(hypothesis)
    update = state.accept(hypothesis, final=True)
    assert "".join(item.text for item in update.committed) == " perché sì."
    assert update.interim == ()
    state.reset_utterance()
    assert state.previous == ()


def test_overlap_repetition_is_suppressed_but_later_repetition_is_preserved():
    state = LocalAgreement(overlap_seconds=1.5)
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


def test_audio_window_and_buffer_are_bounded_and_trim_at_word_boundary():
    state = LocalAgreement(
        active_window_seconds=15, max_buffer_seconds=20, overlap_seconds=1.5
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
    state = LocalAgreement(context_characters=8)
    state.accept(
        Hypothesis((word(" abcdef", 0, 0.5), word(" ghij", 0.5, 1))), final=True
    )
    assert state.request("en", final=False).context == "def ghij"
