import json
import pathlib

import numpy as np
import pytest

from eval.m0b_oracle import (
    _match,
    load_manifest,
    project_downbeats_to_grid,
    score_dynamic,
    summarise,
)


def test_project_downbeats_is_monotonic_unique_and_ties_go_earlier():
    grid = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(
        project_downbeats_to_grid(grid, np.asarray([0.5, 0.6, 2.6])),
        [0.0, 1.0, 3.0])


def test_match_is_one_to_one():
    got = _match(np.asarray([0.0, 0.03, 1.0]), np.asarray([0.01, 1.01]))
    assert np.array_equal(got, [0, -1, 1])


def test_dynamic_score_reports_grouping_change_acquisition():
    reference = {
        "times": np.arange(0.0, 7.0, 1.0),
        "positions": np.asarray([1, 2, 3, 1, 2, 1, 2]),
        "groupings": np.asarray([3, 3, 3, 2, 2, 2, 2]),
        "supported": np.ones(7, dtype=bool),
        "segments": np.asarray([0, 0, 0, 1, 1, 1, 1]),
    }
    result = score_dynamic(
        reference["times"], reference["positions"] - 1,
        reference["groupings"], reference, 0.0)
    assert result["phase_f1"] == 1.0
    assert result["grouping_balanced_accuracy"] == 1.0
    assert result["position_accuracy"] == 1.0
    assert result["changes"] == {
        "total": 1, "acquired": 1, "within_two_bars": 1,
        "latency_sec": [0.0],
    }


def test_summarise_uses_works_not_recordings_and_requires_all_groupings():
    def record(name, work, phase, corpus="fixture"):
        metrics = {
            "phase_f1": phase, "grouping_accuracy": phase,
            "grouping_balanced_accuracy": phase, "position_accuracy": phase,
            "coverage": 1.0, "false_confident_share": 0.0,
            "unnecessary_unknown_share": 0.0,
            "changes": {"total": 1, "acquired": 1, "within_two_bars": 1,
                        "latency_sec": [0.0]},
        }
        return {"name": name, "work_id": work, "corpus": corpus,
                "primary_eligible": True, "groupings": [2, 3, 4, 6],
                "arms": {arm: dict(metrics) for arm in ("A1", "A2", "A3", "A4")}}

    summary = summarise([
        record("performance-a", "same-work", 1.0),
        record("performance-b", "same-work", 0.0),
    ])
    assert summary["independent_works"] == 1
    assert summary["arms"]["A1"]["phase_f1"]["mean"] == 0.5
    assert summary["decision"]["verdict"] == "inconclusive"

    passing = [record(f"p{index}", f"work-{index}", 1.0,
                      corpus="left" if index % 2 else "right")
               for index in range(10)]
    assert summarise(passing)["decision"]["verdict"] == "decoder_not_falsified"


def test_manifest_fails_closed_on_incomplete_hashes_and_path_escape(tmp_path):
    root = pathlib.Path(tmp_path)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "tiktak.m0b_manifest/v1",
        "audio_hashes_complete": False,
        "records": [{"name": "escape", "corpus": "fixture",
                     "audio_relpath": "../audio.wav",
                     "annotation_relpath": "../annotation.csv"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="complete audio hashes"):
        load_manifest(manifest, root, verify_audio=True)
    with pytest.raises(ValueError, match="escapes its root"):
        load_manifest(manifest, root, verify_audio=False)
