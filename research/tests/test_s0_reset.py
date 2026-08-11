import json
import pathlib

import numpy as np

import pytest

from eval.s0_reset import (
    InvariantError,
    arm_name,
    expected_resets,
    measure_outcome,
    paired_bootstrap,
    require_replay_parity,
    summarise,
)


def test_arm_names_are_stable_artifact_keys():
    assert arm_name(2.0) == "R2"
    assert arm_name(None) == "Rinf"


def test_paired_bootstrap_is_deterministic_and_paired():
    values = np.asarray([0.1, 0.2, 0.3])
    assert paired_bootstrap(values) == paired_bootstrap(values)
    low, high = paired_bootstrap(values)
    assert 0.1 <= low <= high <= 0.3


def test_expected_resets_select_first_frame_at_or_after_each_boundary():
    times = np.array([0.0, 0.9, 1.1, 1.9, 2.0, 3.1])
    assert np.array_equal(expected_resets(times, 1.0), [1.1, 2.0, 3.1])


def test_rinf_replay_parity_is_fail_closed():
    baseline = {"beats": [1.0], "live_bar_positions": [0],
                "live_bar_meters": [4]}
    require_replay_parity(baseline, dict(baseline), "fixture")
    with pytest.raises(InvariantError, match="beats"):
        require_replay_parity(baseline, {**baseline, "beats": [1.1]}, "fixture")


def test_measure_outcome_keeps_technical_failures_auditable(monkeypatch):
    def fail(*_args):
        raise ValueError("bad annotation")

    monkeypatch.setattr("eval.s0_reset.measure_one", fail)
    kind, payload = measure_outcome(
        {"name": "fixture", "corpus": "gtzan", "annotation": None},
        pathlib.Path("dump_analysis"), pathlib.Path("model.ttw"))
    assert kind == "exclusion"
    assert payload["error_type"] == "ValueError"
    assert payload["reason"] == "bad annotation"


def test_summary_uses_zero_for_an_abstaining_arm():
    arms = {}
    for name, value in (("R2", 0.0), ("R4", 0.1), ("R8", 0.2),
                        ("R16", 0.3), ("R32", 0.4), ("Rinf", 0.5)):
        arms[name] = {"phase": {"f1": value}}
    got = summarise([{"corpus": "fixture", "arms": arms}])["fixture"]
    assert got["phase_f1"]["R2"] == 0.0
    assert got["Rinf-R2"]["mean"] == 0.5


def test_negative_verdict_uses_upper_confidence_bound():
    arms = {}
    for name, value in (("R2", 0.2), ("R4", 0.21), ("R8", 0.22),
                        ("R16", 0.23), ("R32", 0.24), ("Rinf", 0.24)):
        arms[name] = {"phase": {"f1": value}}
    got = summarise([{"corpus": "gtzan", "arms": arms},
                     {"corpus": "harmonix", "arms": arms}])
    assert got["decision"]["verdict"] == "negative"


def test_one_corpus_cannot_produce_binding_s0_verdict():
    arms = {
        name: {"phase": {"f1": 1.0 if name == "Rinf" else 0.0}}
        for name in ("R2", "R4", "R8", "R16", "R32", "Rinf")
    }
    summary = summarise([{"corpus": "gtzan", "arms": arms}])
    assert summary["decision"]["verdict"] == "inconclusive"
    assert not summary["decision"]["required_corpora_present"]


def test_secondary_summary_ignores_unscorable_values_and_counts_support():
    def record(downbeat_f):
        arms = {
            name: {"phase": {"f1": 0.5}, "beat_f": 0.75,
                   "downbeat_f": downbeat_f, "usable_strict": True}
            for name in ("R2", "R4", "R8", "R16", "R32", "Rinf")
        }
        return {"corpus": "fixture", "arms": arms}

    got = summarise([record(0.25), record(float("nan"))])["fixture"]
    assert got["secondary"]["Rinf"]["beat_f"] == 0.75
    assert got["secondary"]["Rinf"]["beat_f_n_scored"] == 2
    assert got["secondary"]["Rinf"]["downbeat_f"] == 0.25
    assert got["secondary"]["Rinf"]["downbeat_f_n_scored"] == 1
    json.dumps(got, allow_nan=False)
