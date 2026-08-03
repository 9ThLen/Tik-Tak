"""The eval stand reports the causal tracker's tempo, not the offline tempo."""

import numpy as np
import pytest

from eval.analysis import Analyser, DEFAULT_BINARY, Estimate
from eval.live_corpus_benchmark import (
    verdict,
    load_reference_beats,
    local_reference_bpm,
    octave_statistics,
    summarize,
    tempo_state,
)
from tiktak.synth import make_clip


pytestmark = pytest.mark.skipif(
    not Analyser(DEFAULT_BINARY).available,
    reason="dump_analysis is not built — see eval/analysis.py",
)


def test_live_tempo_has_a_separate_final_value_and_time_series():
    clip = make_clip(duration_sec=12, bpm=120, seed=7)
    result = Analyser(DEFAULT_BINARY).analyse_live_audio(
        clip.audio, clip.sample_rate, manual_bpm=137.0)

    assert result.bpm == pytest.approx(120.0, abs=2.0)
    assert result.live_bpm == pytest.approx(137.0)
    assert result.live_tempo_spread_octaves == pytest.approx(0.0, abs=1e-6)

    columns = (
        result.live_times,
        result.live_bpms,
        result.live_confidences,
        result.live_tempo_spreads_octaves,
    )
    assert len(result.live_times) >= 10
    assert all(len(values) == len(result.live_times) for values in columns)
    assert np.all(np.diff(result.live_times) > 0.0)
    np.testing.assert_allclose(result.live_bpms, 137.0, atol=1e-6)
    assert np.all(np.isfinite(result.live_confidences))


def test_octave_statistics_separates_in_lock_and_reacquisition_switches():
    reference = np.arange(0.0, 20.0, 0.5)
    estimate = Estimate(
        beats=np.zeros(0),
        downbeats=np.zeros(0),
        beats_per_bar=0,
        downbeat_strength=0.0,
        downbeat_phase_margin=0.0,
        downbeat_meter_margin=0.0,
        live_bpm=60.0,
        live_times=np.arange(5.0, 10.0),
        live_bpms=np.array([120.0, 120.0, 240.0, 240.0, 60.0]),
        live_confidences=np.array([0.30, 0.20, 0.20, 0.01, 0.30]),
        live_tempo_spreads_octaves=np.zeros(5),
    )

    result = octave_statistics(estimate, reference)

    assert local_reference_bpm(reference, 8.0) == pytest.approx(120.0)
    assert tempo_state(60.0, 120.0) == "half"
    assert tempo_state(120.0, 120.0) == "same"
    assert tempo_state(240.0, 120.0) == "double"
    assert result["within_switches"] == 1
    assert result["reacquire_switches"] == 1
    assert result["switches"] == 2
    assert result["final_state"] == "half"


def test_normalized_manifest_annotation_header_is_accepted(tmp_path):
    annotation = tmp_path / "track.csv"
    annotation.write_text(
        "time_seconds,beat_position,is_downbeat\n"
        "0.25,1,1\n"
        "0.75,2,0\n",
        encoding="utf-8-sig",
    )

    np.testing.assert_allclose(load_reference_beats(annotation), [0.25, 0.75])


# ---------------------------------------------------------------- the verdict
#
# `verdict` decides what the headline number means, so its edge cases are worth
# pinning rather than trusting to a corpus run. The strict reading in particular
# was added after the loose one had already been quoted, and the whole point of
# it is that the two disagree in a specific direction — a test that only checked
# they agreed would have passed before it existed.


def _result(**overrides):
    """A recording that passes everything, so a test can break one thing."""
    base = {
        "acquired_at": 5.0, "settled_at": 6.0,
        "p70": 0.95, "r70": 0.95, "worst_wrong_octave_sec": 0.0,
        "p70_thinned": 0.5, "r70_thinned": 0.5,
        "p70_thinned_odd": 0.5, "r70_thinned_odd": 0.5,
        "p70_doubled": 0.5, "r70_doubled": 0.5,
    }
    base.update(overrides)
    return base


def test_a_clean_recording_passes_both_readings():
    got = verdict(_result())
    assert got["usable"] and got["usable_strict"]
    assert got["reasons"] == [] and got["reasons_strict"] == []


def test_a_lock_that_never_settles_passes_loosely_and_fails_strictly():
    # The case the strict reading exists for: confidence crosses the threshold
    # inside the limit, but the level it locked to was never the right one.
    got = verdict(_result(settled_at=None))
    assert got["usable"], "the loose reading only asks when confidence rose"
    assert not got["usable_strict"]
    assert "never_settled" in got["reasons_strict"]
    assert "never_settled" not in got["reasons"]


def test_settling_after_the_limit_fails_strictly_only():
    got = verdict(_result(acquired_at=2.0, settled_at=20.0))
    assert got["usable"]
    assert not got["usable_strict"]
    assert "slow_settle" in got["reasons_strict"]


def test_the_strict_reading_never_passes_where_the_loose_one_fails():
    # Strict is loose plus one more condition, so it can only ever be a subset.
    # Stated as a test because the two lists are built separately and a later
    # edit could easily make them diverge in the wrong direction.
    for broken in ({"p70": 0.1}, {"r70": 0.1}, {"worst_wrong_octave_sec": 30.0},
                   {"acquired_at": 40.0}, {"acquired_at": None}):
        got = verdict(_result(**broken))
        assert not got["usable"]
        assert not got["usable_strict"], broken


def test_a_recording_that_never_acquires_is_not_also_called_never_settled():
    # One failure, one reason. Listing both would double-count a single event in
    # the failure breakdown, which is read as shares of the corpus.
    got = verdict(_result(acquired_at=None, settled_at=None))
    assert "never_acquired" in got["reasons_strict"]
    assert "never_settled" not in got["reasons_strict"]


def _scored(corpus: str, name: str, **overrides):
    """One entry of `summarize`'s input, passing unless told otherwise."""
    result = _result(**overrides)
    result.update({
        "ok": True, "mode": "model", "corpus": corpus, "name": name,
        "annotated": True, "duration": 30.0, "wall": 1.0,
        "live_bpm": 120.0, "live_confidence": 0.5, "live_spread": 0.01,
        "beats": 60, "late": 0,
        "f_measure": 0.9, "cmlt": 0.8, "amlt": 0.85, "coverage": 1.0,
        "switches": 0, "within_switches": 0, "reacquire_switches": 0,
        "states": {"same": 30}, "active_samples": 30, "eligible_samples": 30,
        "final_state": "same", "final_ref_bpm": 120.0, "final_active": True,
    })
    result.update(verdict(result))
    return result


def test_the_strict_headline_is_macro_over_big_corpora_and_pooled_over_all():
    # Two corpora of deliberately different size, and a corpus below the macro
    # minimum that must be in the pooled figure and out of the macro one. The
    # strict aggregates were added after the loose ones and went unasserted;
    # a macro that silently pooled would read almost right on a real corpus mix
    # and be wrong in exactly the direction that flatters the bigger corpus.
    results = (
        [_scored("gtzan", f"g{i}") for i in range(40)]
        + [_scored("gtzan", f"gbad{i}", settled_at=None) for i in range(60)]
        + [_scored("smc", f"s{i}") for i in range(30)]
        + [_scored("tiny", f"t{i}", settled_at=None) for i in range(10)]
    )
    summary = summarize("model", results, wall=1.0)

    assert summary["by_corpus"]["gtzan"]["usable_rate_strict"] == pytest.approx(0.40)
    assert summary["by_corpus"]["smc"]["usable_rate_strict"] == pytest.approx(1.0)
    # Macro: the two corpora at n >= 30, each counted once. Not (40 + 30) / 140.
    assert summary["usable_rate_strict_macro"] == pytest.approx(0.70)
    # Pooled: every scored recording, `tiny` included.
    assert summary["usable_rate_strict_pooled"] == pytest.approx(70 / 140)
    # The loose reading is untouched by a lock that never settled, which is the
    # whole point of reporting the two side by side.
    assert summary["usable_rate_macro"] == pytest.approx(1.0)
    assert summary["usable_rate_pooled"] == pytest.approx(1.0)
