import pathlib

import numpy as np
import pytest

from eval.m0a_oracle import (
    _phase_with_null,
    best_phase_events,
    fixed_meter_label,
    measure_outcome,
    oracle_channel,
    reference_emit,
    summarise,
    visible_indices,
)


def test_oracle_channel_uses_earlier_frame_on_tie():
    times = np.asarray([0.0, 0.02, 0.04])
    got = oracle_channel(times, np.asarray([0.03]))
    assert np.array_equal(got, [0.0, 1.0, 0.0])


def test_reference_emit_is_first_block_inside_lookahead():
    beats = np.asarray([0.01, 0.5, 1.0])
    got = reference_emit(beats, 48000.0)
    assert np.array_equal(got, [1.0, 43.0, 90.0])


def test_visible_indices_keep_late_grid_beats_between_outputs():
    all_beats = np.asarray([0.0, 0.5, 1.0, 1.5])
    visible = np.asarray([0.0, 1.0, 1.5])
    assert np.array_equal(visible_indices(all_beats, visible), [0, 2, 3])


def test_fixed_meter_label_is_fail_closed():
    assert fixed_meter_label({"name": "ok", "meter": "4/4"}) == "4/4"
    with pytest.raises(RuntimeError, match="fixed meter"):
        fixed_meter_label({"name": "missing", "meter": None})


def test_best_phase_events_selects_planted_bar_line():
    beats = np.arange(0.0, 8.0, 0.5)
    downbeats = beats[2::4]
    assert np.array_equal(best_phase_events(beats, 4, downbeats), downbeats)


def test_hard_negative_uses_upper_bound_and_null_precondition():
    def record(corpus, a1, a4):
        arms = {
            arm: {"actual": {"f1": value}, "answered": True,
                  "random_phase_f1": 0.2}
            for arm, value in (("A1", a1), ("A2", 0.5),
                               ("A3", 0.5), ("A4", a4))
        }
        return {"corpus": corpus, "arms": arms,
                "A1_sensitivity": {"amplitude_0_5": a1,
                                   "alternating_jitter_20ms": a1}}

    got = summarise([record("gtzan", 0.6, 0.5),
                     record("harmonix", 0.6, 0.5)])
    assert got["decision"]["band"] == "band1_hard_negative"


def test_phase_diagnostics_report_common_cut_coverage():
    result = _phase_with_null(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([2.0, 2.0, 2.0, 2.0]),
        np.array([0.0, 2.0]),
        2.0,
    )
    assert result["first_decision_sec"] == 0.0
    assert result["coverage"] == {
        "decided_before_common": 2,
        "beats_before_common": 2,
        "decided_after_common": 2,
        "beats_after_common": 2,
    }


def test_one_corpus_cannot_produce_binding_m0a_verdict():
    arms = {
        arm: {"actual": {"f1": 1.0 if arm == "A1" else 0.0},
              "answered": True, "random_phase_f1": 0.0}
        for arm in ("A1", "A2", "A3", "A4")
    }
    summary = summarise([{
        "corpus": "gtzan", "arms": arms,
        "A1_sensitivity": {"amplitude_0_5": 1.0,
                           "alternating_jitter_20ms": 1.0},
    }])
    assert summary["decision"]["band"] == "band2_inconclusive"
    assert not summary["decision"]["required_corpora_present"]


def test_measure_outcome_keeps_pre_arm_exclusions_auditable(monkeypatch):
    def fail(*_args):
        raise RuntimeError("manifest does not identify a fixed meter")

    monkeypatch.setattr("eval.m0a_oracle.measure_one", fail)
    kind, payload = measure_outcome(
        {"name": "fixture", "corpus": "gtzan", "annotation": None},
        pathlib.Path("dump_analysis"), pathlib.Path("model.ttw"))
    assert kind == "exclusion"
    assert payload["error_type"] == "RuntimeError"
    assert "fixed meter" in payload["reason"]
