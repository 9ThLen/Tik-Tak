import pathlib

import numpy as np
import pytest

import eval.m0e_non_oracle as m0e


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    monkeypatch.setattr(m0e, "BOOTSTRAP_DRAWS", 100)


def _transition(acquired, *, observable=True, ordinal=0):
    return {
        "transition_id": f"transition-{ordinal}",
        "two_bar_latency_observable": observable,
        "acquired_within_two_bars": acquired,
        "acquired_first_bar": acquired,
        "acquisition_latency_bars": 1.0 if acquired else None,
        "outcome_class": ("right_censored" if not observable else
                          "acquired_within_two_bars" if acquired else
                          "acquired_late"),
    }


def _records(*, candidate_acquires=True, candidate_accuracy=1.0,
             candidate_switches=0, candidate_long_episodes=0):
    records = []
    for index in range(m0e.FIXED_RECORDS):
        work_index = index if index < m0e.FIXED_WORKS else index % m0e.FIXED_WORKS
        transition_cohort = index < m0e.FIXED_TRANSITION_RECORDS
        transition_count = (3 if index < 33 else 24) if transition_cohort else 0
        observable_count = (2 if index < 30 else 1 if index == 30 else 0)
        arms = {}
        for arm in m0e.ARMS:
            acquired = arm == m0e.CANDIDATE and candidate_acquires
            accuracy = candidate_accuracy if arm == m0e.CANDIDATE else 1.0
            switches = candidate_switches if arm == m0e.CANDIDATE else 0
            long_episodes = (
                candidate_long_episodes if arm == m0e.CANDIDATE else 0)
            arms[arm] = {
                "transitions": [
                    _transition(
                        acquired and ordinal < observable_count,
                        observable=ordinal < observable_count,
                        ordinal=ordinal)
                    for ordinal in range(transition_count)],
                "static": {
                    "correct": int(round(100 * accuracy)),
                    "events": 100,
                    "accuracy": accuracy,
                    "eligible_duration_sec": 300.0,
                    "false_switches": switches,
                    "wrong_episodes": long_episodes,
                    "long_wrong_episodes_1bar": long_episodes,
                    "long_wrong_episodes_2bar": 0,
                    "longest_wrong_episode_events": 4 if long_episodes else 0,
                    "longest_wrong_episode_sec": 2.0 if long_episodes else 0.0,
                },
                "score": {
                    "phase_f1": 0.2,
                    "grouping_balanced_accuracy": 0.2,
                    "position_accuracy": 0.1,
                    "coverage": 0.4,
                    "false_confident_share": 0.2,
                    "unnecessary_unknown_share": 0.6,
                },
                "diagnostics": {
                    "resolver_path_state_changes": 0,
                    "held_state_changes": switches,
                },
            }
        records.append({
            "name": f"record-{index}",
            "corpus": "rwc2",
            "work_id": f"work-{work_index}",
            "transition_cohort": transition_cohort,
            "arms": arms,
        })
    return records


def test_arm_flags_freeze_the_only_registered_pair():
    assert m0e._flags(m0e.BASELINE) == [
        "--bar-phase-switch-cost", "64.0"]
    assert m0e._flags(m0e.CANDIDATE) == [
        "--bar-phase-switch-cost", "2.0", "--bar-latest-path-phase"]
    assert tuple(m0e.ARMS) == ("B64_opening", "L2_latest")


def test_baseline_replay_requires_exact_live_output_parity():
    initial = {
        "live_bar_positions_all": [0.0, 1.0],
        "live_bar_meters_all": [4.0, 4.0],
        "live_bar_confident_all": [0.0, 1.0],
    }
    values = tuple(np.asarray(initial[key], dtype=np.float64) for key in (
        "live_bar_positions_all",
        "live_bar_meters_all",
        "live_bar_confident_all",
    ))
    m0e._assert_live_baseline_parity(initial, *values, "track")
    changed = values[0].copy()
    changed[1] = 2.0
    with pytest.raises(m0e.InvariantError, match="positions.*parity"):
        m0e._assert_live_baseline_parity(
            initial, changed, values[1], values[2], "track")


def test_static_safety_excludes_adaptation_and_counts_churn():
    reference = {
        "times": np.arange(20, dtype=np.float64),
        "positions": np.asarray(
            [1, 2, 3, 4, 1, 2, 3, 4,
             1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64),
        "groupings": np.asarray([4] * 8 + [3] * 12, dtype=np.int64),
        "supported": np.ones(20, dtype=bool),
        "segments": np.zeros(20, dtype=np.int64),
    }
    positions = reference["positions"] - 1
    positions[2:6] = (positions[2:6] + 1) % 4
    positions[8:14] = (positions[8:14] + 1) % 3
    mapped = {
        "positions": positions,
        "meters": reference["groupings"].copy(),
        "confident": np.ones(20, dtype=bool),
    }
    mask = m0e.adaptation_mask(reference, 0.0)
    assert np.flatnonzero(mask).tolist() == list(range(8, 14))
    result = m0e.static_safety(mapped, reference, 0.0)
    assert result["events"] == 14
    assert result["correct"] == 10
    assert result["eligible_duration_sec"] == 12.0
    assert result["false_switches"] == 2
    assert result["wrong_episodes"] == 1
    assert result["long_wrong_episodes_1bar"] == 1
    assert result["long_wrong_episodes_2bar"] == 0


def test_summary_passes_only_when_efficacy_and_every_safety_gate_pass():
    summary = m0e.summarise(_records(), [])
    assert summary["gates"] == {
        "complete": True,
        "efficacy": True,
        "stable_exact": True,
        "false_switch_rate": True,
        "long_wrong_episode_rate": True,
    }
    assert summary["interpretation"] == "non_oracle_decoder_candidate_pass"


def test_summary_reports_transition_gain_with_static_accuracy_cost():
    summary = m0e.summarise(_records(candidate_accuracy=0.90), [])
    assert summary["gates"]["efficacy"] is True
    assert summary["gates"]["stable_exact"] is False
    assert summary["interpretation"] == "non_oracle_gain_static_cost"


def test_summary_reports_oracle_gain_that_does_not_transfer():
    summary = m0e.summarise(_records(candidate_acquires=False), [])
    assert summary["gates"]["efficacy"] is False
    assert summary["gates"]["stable_exact"] is True
    assert summary["interpretation"] == "oracle_gain_does_not_transfer"


def test_summary_reports_candidate_regression_without_efficacy():
    summary = m0e.summarise(_records(
        candidate_acquires=False, candidate_accuracy=0.90), [])
    assert summary["gates"]["efficacy"] is False
    assert summary["gates"]["stable_exact"] is False
    assert summary["interpretation"] == "non_oracle_candidate_regression"


@pytest.mark.parametrize(
    ("kwargs", "gate"),
    [
        ({"candidate_switches": 2}, "false_switch_rate"),
        ({"candidate_long_episodes": 1}, "long_wrong_episode_rate"),
    ],
)
def test_operational_churn_vetoes_an_otherwise_effective_candidate(kwargs, gate):
    summary = m0e.summarise(_records(**kwargs), [])
    assert summary["gates"]["efficacy"] is True
    assert summary["gates"][gate] is False
    assert summary["interpretation"] == "non_oracle_gain_static_cost"


def test_any_new_exclusion_makes_the_binding_result_inconclusive():
    summary = m0e.summarise(_records(), [{"name": "failed"}])
    assert summary["gates"]["complete"] is False
    assert summary["interpretation"] == "inconclusive"


def test_checkpoint_identity_binds_sources_population_and_all_gates():
    provenance = {
        "commit": "commit",
        "binary": {"sha256": "binary"},
        "model": {"sha256": "model"},
        "manifest": {"sha256": "manifest"},
        "source_m0b": {"sha256": "m0b"},
        "source_m0c": {"sha256": "m0c"},
        "source_m0d": {"sha256": "m0d"},
    }
    item = {
        "corpus": "rwc2", "name": "track", "work_id": "work",
        "audio_sha256": "audio", "annotation_sha256": "annotation",
        "source_m0c": {"transitions": [{"transition_id": "fixed"}]},
    }
    identity = m0e.checkpoint_identity(
        provenance, [item], limit=0, skip_audio_verification=False)
    assert identity["artifact_schema"] == "tiktak.m0e_non_oracle/v1"
    assert identity["arms"] == m0e.ARMS
    assert identity["fixed_records"] == 980
    assert identity["fixed_works"] == 414
    assert identity["fixed_fully_observable"] == 61
    assert identity["efficacy_min_gain"] == 0.10
    assert identity["stable_max_loss"] == 0.03
    assert identity["false_switch_max_increase_per_5min"] == 1.0
    assert identity["long_episode_max_increase_per_5min"] == 0.25
    assert m0e.CHECKPOINT_SCHEMA == "tiktak.m0e_checkpoint/v1"


def test_registration_and_runner_thresholds_stay_synchronised():
    registration = (
        pathlib.Path(__file__).parents[1] / "eval" / "PREREGISTERED_M0e.md"
    ).read_text(encoding="utf-8")
    for text in ("+0.10", "-0.03", "+1.0 per five minutes",
                 "+0.25 per five minutes", "980", "414", "61"):
        assert text in registration
