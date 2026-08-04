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


def _timeline(bpms, confidences, beat_bpm=120.0, duration=80.0):
    """A per-second live history and a matching annotated beat grid."""
    times = np.arange(len(bpms), dtype=np.float64)
    return (
        Estimate.from_json({
            "live_times": times.tolist(),
            "live_bpms": list(bpms),
            "live_confidences": list(confidences),
            "live_tempo_spreads_octaves": [0.01] * len(bpms),
            "live_bpm": float(bpms[-1]),
        }),
        np.arange(0.0, duration, 60.0 / beat_bpm),
    )


def test_correct_time_counts_silence_against_the_tracker():
    # Forty seconds after warm-up: right for twenty, then silent for twenty.
    # The share over *active* time is 100% — the tracker was never wrong when
    # it spoke. The share a user would recognise is 50%, because for half the
    # time it showed nothing. That difference is the whole reason this field
    # exists, so it is asserted rather than assumed.
    bpms = [120.0] * 65
    confidences = [0.9] * 45 + [0.0] * 20
    estimate, beats = _timeline(bpms, confidences)
    got = octave_statistics(estimate, beats)

    active_share = got["states"]["same"] / got["active_samples"]
    assert active_share == pytest.approx(1.0)
    assert got["correct_share_of_eligible"] == pytest.approx(40 / 60, abs=0.02)
    assert got["correct_share_of_eligible"] < active_share


def test_the_longest_correct_run_is_not_the_total_correct_time():
    # Right for ten seconds, wrong for ten, right for ten again. Thirty seconds
    # of correct time in total, but never more than ten in a row — and a
    # metronome that resets twice is not one that held for thirty.
    bpms = [120.0] * 15 + [60.0] * 10 + [120.0] * 10
    estimate, beats = _timeline(bpms, [0.9] * len(bpms))
    got = octave_statistics(estimate, beats)

    assert got["states"]["same"] == 20
    assert got["longest_correct_run_sec"] == pytest.approx(10.0, abs=1.5)


def test_a_tracker_that_has_not_locked_reads_as_silent_not_as_wrong():
    # The fixed-moment snapshots must distinguish "showing nothing" from
    # "showing the wrong tempo": they are different failures with different
    # fixes, and collapsing them would hide slow acquisition inside the octave
    # numbers.
    bpms = [120.0] * 65
    confidences = [0.0] * 20 + [0.9] * 45
    estimate, beats = _timeline(bpms, confidences)
    got = octave_statistics(estimate, beats)

    assert got["state_at_10s"] == "silent"
    assert got["state_at_30s"] == "same"


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
        "eligible_sec": 25.0,
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


def test_switch_rate_divides_by_the_time_switches_could_have_happened_in():
    # Switches are only counted after warm-up, so the rate has to divide by
    # eligible time. The first version divided by whole audio duration, which on
    # a thirty-second excerpt is a sixth too large -- and understated the rate
    # by that much on exactly the corpora made of short excerpts. The fixture is
    # built so the two denominators give different answers.
    results = [_scored("gtzan", f"g{i}", ) for i in range(30)]
    for result in results:
        result["switches"] = 2
        result["duration"] = 30.0        # what the wrong denominator used
        result["eligible_sec"] = 25.0    # what the right one uses

    summary = summarize("model", results, wall=1.0)
    got = summary["by_corpus"]["gtzan"]["switches_per_five_minutes"]

    assert got == pytest.approx(60 / (750 / 300.0))     # 24.0, over eligible
    assert got != pytest.approx(60 / (900 / 300.0))     # 20.0, over duration


def test_the_episode_metric_agrees_with_the_verdict_it_was_promoted_from():
    # The headline is now "never slipped for more than four seconds", and the
    # verdict already had that clause. Two spellings of one threshold would
    # eventually disagree, so the test pins them together rather than pinning a
    # number: every recording the fraction counts as clean must also be one the
    # verdict did not fail for `wrong_octave`.
    results = (
        [_scored("gtzan", f"ok{i}", worst_wrong_octave_sec=0.0) for i in range(20)]
        + [_scored("gtzan", f"edge{i}", worst_wrong_octave_sec=4.0) for i in range(5)]
        + [_scored("gtzan", f"bad{i}", worst_wrong_octave_sec=9.0) for i in range(15)]
    )
    summary = summarize("model", results, wall=1.0)

    assert summary["by_corpus"]["gtzan"]["no_wrong_level_episode_fraction"] == (
        pytest.approx(25 / 40))
    assert summary["no_wrong_level_episode_pooled"] == pytest.approx(25 / 40)
    clean = sum("wrong_octave" not in result["reasons"] for result in results)
    assert clean == 25


def test_the_three_correct_time_denominators_are_reported_separately():
    # Right whenever it spoke, silent a third of the time. Over active time that
    # is 100%; over eligible time it is 67%. Reporting one number without saying
    # which denominator it used is how "64.6%" ended up in a plan meaning
    # something different from what the plan compared it against.
    results = [_scored("gtzan", f"g{i}") for i in range(30)]
    for result in results:
        result["states"] = {"same": 20}
        result["active_samples"] = 20
        result["eligible_samples"] = 30
        result["correct_share_of_eligible"] = 20 / 30

    corpus = summarize("model", results, wall=1.0)["by_corpus"]["gtzan"]

    assert corpus["correct_share_of_active"] == pytest.approx(1.0)
    assert corpus["correct_share_of_eligible_time_pooled"] == pytest.approx(2 / 3)
    assert corpus["mean_correct_share_of_eligible"] == pytest.approx(2 / 3)
