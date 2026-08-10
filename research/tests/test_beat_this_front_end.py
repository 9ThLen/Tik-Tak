"""Statistics used by the clean P0 Beat This!/BeatNet matched rerun."""

from __future__ import annotations

import pytest

from eval.beat_this_front_end import corpus_summary, paired_bootstrap


def _row(name: str, delta: float, corpus: str = "gtzan") -> dict:
    return {
        "name": name,
        "corpus": corpus,
        "delta_f": delta,
        "beatnet": {"f_measure": 0.5, "usable": False},
        "beat_this": {"f_measure": 0.5 + delta, "usable": True},
    }


def test_paired_bootstrap_is_recording_level_and_deterministic():
    rows = [_row("a", 0.1), _row("b", 0.2), _row("c", 0.3)]

    first = paired_bootstrap(rows, resamples=2_000, seed=17)
    second = paired_bootstrap(rows, resamples=2_000, seed=17)

    assert first == second
    assert first["unit"] == "recording/composition"
    assert first["mean_delta_f"] == pytest.approx(0.2)
    assert first["ci95"][0] <= first["mean_delta_f"] <= first["ci95"][1]


def test_corpus_summary_keeps_failures_out_of_the_paired_estimate():
    rows = [
        _row("a", 0.1),
        _row("b", 0.3),
        {"name": "failed", "corpus": "gtzan", "error": "decoder failed"},
    ]

    summary = corpus_summary(rows)

    assert summary["n"] == 2
    assert [row["name"] for row in summary["failures"]] == ["failed"]
    assert summary["mean_delta_f"] == pytest.approx(0.2)
    assert summary["paired_bootstrap"]["mean_delta_f"] == pytest.approx(0.2)


def test_empty_corpus_summary_fails_instead_of_writing_nan():
    with pytest.raises(ValueError, match="no scored recordings"):
        corpus_summary([])
