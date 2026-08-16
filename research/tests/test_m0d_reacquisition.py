import numpy as np

from eval.m0d_reacquisition import (
    ARMS,
    BASELINE,
    CHECKPOINT_SCHEMA,
    POSITIVE_CONTROL,
    _flags,
    checkpoint_identity,
    stable_exact_counts,
    summarise,
)


def _transition(acquired, *, observable=True, ordinal=0):
    return {
        "transition_id": f"transition-{ordinal}",
        "two_bar_latency_observable": observable,
        "acquired_within_two_bars": acquired,
        "acquired_first_bar": acquired,
        "acquisition_latency_bars": 0.0 if acquired else None,
        "outcome_class": ("right_censored" if not observable else
                          "acquired_within_two_bars" if acquired else
                          "acquired_late"),
    }


def _records(acquired_by_arm, stable_by_arm, count=34):
    records = []
    for index in range(count):
        transition_count = 3 if index < 33 else 24
        observable_count = 2 if index < 30 else 1 if index == 30 else 0
        stable_events = 505 if index < 33 else 510
        arms = {}
        for arm in ARMS:
            stable = stable_by_arm.get(arm, 1.0)
            arms[arm] = {
                "transitions": [
                    _transition(
                        acquired_by_arm.get(arm, False) and ordinal < observable_count,
                        observable=ordinal < observable_count, ordinal=ordinal)
                    for ordinal in range(transition_count)],
                "stable_exact": {"correct": int(stable_events * stable),
                                 "events": stable_events,
                                 "accuracy": stable},
                "diagnostics": {"resolver_path_state_changes": 0,
                                "held_state_changes": 0,
                                "path_changes_reflected_in_held": 0,
                                "path_held_disagreements": 0},
            }
        records.append({"work_id": f"work-{index}", "arms": arms})
    return records


def test_arm_flags_keep_readout_and_hysteresis_separate():
    assert "--bar-latest-path-phase" not in _flags(BASELINE)
    assert _flags(BASELINE) == ["--bar-phase-switch-cost", "64.0"]
    assert "--bar-latest-path-phase" in _flags("L64_latest")
    assert _flags("L8_latest")[:2] == ["--bar-phase-switch-cost", "8.0"]
    assert _flags(POSITIVE_CONTROL)[:2] == ["--bar-phase-switch-cost", "0.0"]


def test_stable_exact_excludes_registered_two_bar_adaptation_window():
    reference = {
        "times": np.arange(12, dtype=np.float64),
        "positions": np.asarray([1, 2, 3, 4] * 3),
        "groupings": np.asarray([4] * 12),
        "supported": np.ones(12, dtype=bool),
    }
    positions = reference["positions"].astype(np.float64) - 1
    positions[2:10] = (positions[2:10] + 1) % 4
    counts = stable_exact_counts(
        positions, reference["groupings"].astype(np.float64), reference, 0.0,
        [{"reference_index": 2, "next_change_or_end_index": 12,
          "new_grouping": 4}])
    assert counts == {"correct": 4, "events": 4, "accuracy": 1.0}


def test_summary_selects_readout_before_lower_cost_candidates():
    acquired = {BASELINE: False, "L64_latest": True,
                "L8_latest": True, "L2_latest": True,
                POSITIVE_CONTROL: True}
    summary = summarise(_records(acquired, {}), [])
    assert summary["positive_control_passed"] is True
    assert summary["selected_candidate"] == "L64_latest"
    assert summary["interpretation"] == "opening_phase_readout_bottleneck"


def test_summary_selects_hysteresis_when_readout_alone_is_null():
    acquired = {BASELINE: False, "L64_latest": False,
                "L8_latest": True, "L2_latest": True,
                POSITIVE_CONTROL: True}
    summary = summarise(_records(acquired, {}), [])
    assert summary["selected_candidate"] == "L8_latest"
    assert summary["interpretation"] == "phase_hysteresis_bottleneck"


def test_summary_withholds_candidate_that_damages_stable_accuracy():
    acquired = {BASELINE: False, "L64_latest": False,
                "L8_latest": True, "L2_latest": True,
                POSITIVE_CONTROL: True}
    stable = {"L8_latest": 0.90, "L2_latest": 0.90}
    summary = summarise(_records(acquired, stable), [])
    assert summary["selected_candidate"] is None
    assert summary["interpretation"] == "transition_gain_static_cost"


def test_summary_is_inconclusive_when_positive_control_is_inert():
    summary = summarise(_records({}, {}), [])
    assert summary["positive_control_passed"] is False
    assert summary["interpretation"] == "inconclusive"


def test_positive_control_gate_prevents_a_numerical_candidate_selection():
    acquired = {"L64_latest": True, "L8_latest": True, "L2_latest": True}
    summary = summarise(_records(acquired, {}), [])
    assert summary["paired_effects"]["L64_latest"]["effective"] is True
    assert summary["positive_control_passed"] is False
    assert summary["selected_candidate"] is None
    assert summary["interpretation"] == "inconclusive"


def test_checkpoint_identity_binds_decoder_arms_thresholds_and_population():
    provenance = {
        "commit": "commit", "binary": {"sha256": "binary"},
        "model": {"sha256": "model"}, "manifest": {"sha256": "manifest"},
        "source_m0c": {"sha256": "source"},
    }
    item = {
        "name": "track", "audio_sha256": "audio",
        "annotation_sha256": "annotation",
        "source_m0c": {"transitions": [{"transition_id": "fixed"}]},
    }
    identity = checkpoint_identity(
        provenance, [item], limit=0, skip_audio_verification=False)
    assert identity["artifact_schema"] == "tiktak.m0d_reacquisition/v1"
    assert identity["arms"] == ARMS
    assert identity["candidate_order"] == [
        "L64_latest", "L8_latest", "L2_latest"]
    assert identity["fixed_fully_observable"] == 61
    assert identity["fixed_stable_events"] == 17175
    assert CHECKPOINT_SCHEMA == "tiktak.m0d_checkpoint/v1"
