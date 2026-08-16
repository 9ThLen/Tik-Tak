import json
import pathlib

import numpy as np
import pytest

from eval.m0b_oracle import (
    _match,
    _require_outside_repository,
    checkpoint_identity,
    load_manifest,
    prepare_checkpoint,
    profiled_oracle_channels,
    project_downbeats_to_grid,
    run_checkpointed,
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


def test_profiled_oracle_copies_model_shape_and_shifts_positive_control():
    frame_times = np.arange(0.0, 5.02, 0.02)
    predicted = np.zeros(len(frame_times))
    source = int(np.argmin(np.abs(frame_times - 2.0)))
    predicted[source - 1:source + 2] = [0.25, 0.8, 0.5]
    reference_times = np.arange(0.5, 4.51, 0.5)
    positions = np.asarray([1, 2, 3, 4, 1, 2, 3, 4, 1])

    profiled, shifted, metadata = profiled_oracle_channels(
        frame_times, predicted, reference_times, positions)

    first_downbeat = int(np.argmin(np.abs(frame_times - 0.5)))
    first_shift = int(np.argmin(np.abs(frame_times - 1.0)))
    assert np.allclose(profiled[first_downbeat - 1:first_downbeat + 2],
                       [0.25, 0.8, 0.5])
    assert np.allclose(shifted[first_shift - 1:first_shift + 2],
                       [0.25, 0.8, 0.5])
    assert profiled[first_shift] == 0.0
    assert shifted[first_downbeat] == 0.0
    assert metadata["template_half_width_sec"] >= 0.5
    assert metadata["overlap_rule"] == "maximum"


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


def test_change_acquisition_requires_a_complete_bar_starting_at_position_one():
    reference = {
        "times": np.arange(0.0, 11.0, 1.0),
        "positions": np.asarray([1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4]),
        "groupings": np.asarray([3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),
        "supported": np.ones(11, dtype=bool),
        "segments": np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
    }
    predicted_positions = np.asarray([
        0, 1, 2, -1, -1, 2, 3, 0, 1, -1, -1])
    predicted_meters = np.asarray([3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4])
    result = score_dynamic(
        reference["times"], predicted_positions, predicted_meters,
        reference, 0.0)
    assert result["changes"] == {
        "total": 1, "acquired": 0, "within_two_bars": 0,
        "latency_sec": [],
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
            "by_grouping": {str(grouping): {
                "matched_tactus_phase_f1": phase}
                for grouping in (2, 3, 4, 6)},
        }
        return {"name": name, "work_id": work, "corpus": corpus,
                "primary_eligible": True, "groupings": [2, 3, 4, 6],
                "arms": {arm: dict(metrics) for arm in ("A1", "A2", "A3", "A4")},
                "A1_sensitivity": {
                    "profiled_oracle": {
                        "phase_f1": phase,
                        "by_grouping": metrics["by_grouping"]},
                    "shifted_one_tactus": {
                        "phase_f1": max(0.0, phase - 0.7),
                        "by_grouping": {str(grouping): {
                            "matched_tactus_phase_f1": max(0.0, phase - 0.7)}
                            for grouping in (2, 3, 4, 6)}},
                }}

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
    assert set(summarise(passing)["A1_sensitivity"]["by_grouping"]) == {
        "2", "3", "4", "6"}

    inert = [record(f"i{index}", f"inert-{index}", 1.0,
                    corpus="left" if index % 2 else "right")
             for index in range(10)]
    for row in inert:
        row["A1_sensitivity"]["shifted_one_tactus"]["phase_f1"] = 0.31
    inert_summary = summarise(inert)
    assert inert_summary["A1_sensitivity"]["passed"] is False
    assert inert_summary["decision"]["verdict"] == "inconclusive"

    low_and_inert = [record(f"l{index}", f"low-{index}", 0.3,
                            corpus="left" if index % 2 else "right")
                     for index in range(10)]
    for row in low_and_inert:
        row["A1_sensitivity"]["shifted_one_tactus"]["phase_f1"] = 0.28
    low_summary = summarise(low_and_inert)
    assert low_summary["arms"]["A1"]["phase_f1"]["mean"] == pytest.approx(0.3)
    assert low_summary["A1_sensitivity"][
        "profiled_oracle_minus_shifted_one_tactus"]["mean"] == pytest.approx(0.02)
    assert low_summary["A1_sensitivity"]["passed"] is False
    assert low_summary["decision"]["verdict"] == "inconclusive"

    format_biased = [record(f"f{index}", f"format-{index}", 1.0,
                            corpus="left" if index % 2 else "right")
                     for index in range(10)]
    for row in format_biased:
        row["A1_sensitivity"]["profiled_oracle"]["phase_f1"] = 0.8
        row["A1_sensitivity"]["shifted_one_tactus"]["phase_f1"] = 0.0
    biased_summary = summarise(format_biased)
    assert biased_summary["A1_sensitivity"]["passed"] is False
    assert biased_summary["decision"]["verdict"] == "inconclusive"


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


def _checkpoint_fixture(tmp_path, count=4):
    items = [{"corpus": "fixture", "name": f"item-{index}",
              "audio_sha256": f"audio-{index}",
              "annotation_sha256": f"annotation-{index}"}
             for index in range(count)]
    provenance = {
        "utc": "2026-08-11T00:00:00Z", "commit": "abc123",
        "binary": {"sha256": "binary"},
        "model": {"sha256": "model"},
        "manifest": {"sha256": "manifest"},
    }
    identity = checkpoint_identity(
        provenance, items, limit=0,
        skip_audio_verification=False)
    return items, provenance, identity, pathlib.Path(tmp_path) / "checkpoint"


def test_checkpoint_pause_then_resume_skips_completed_items(tmp_path, monkeypatch):
    items, provenance, identity, checkpoint = _checkpoint_fixture(tmp_path)
    pause_file = pathlib.Path(tmp_path) / "pause"
    calls = []

    def fake_measure(item, _binary, _model):
        calls.append(item["name"])
        if item["name"] == "item-0":
            pause_file.write_text("pause", encoding="utf-8")
        return "exclusion", {"name": item["name"]}

    monkeypatch.setattr("eval.m0b_oracle.measure_outcome", fake_measure)
    ordered, state, completed = prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=False, workers=1)
    assert completed == 0
    ordered, paused = run_checkpointed(
        items, pathlib.Path("binary"), pathlib.Path("model"), workers=1,
        checkpoint=checkpoint, state=state, ordered=ordered,
        pause_file=pause_file)
    assert paused is True
    assert calls == ["item-0"]
    assert json.loads((checkpoint / "state.json").read_text(
        encoding="utf-8"))["status"] == "paused"

    pause_file.unlink()
    resumed, state, completed = prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=True, workers=8)
    assert completed == 1
    resumed, paused = run_checkpointed(
        items, pathlib.Path("binary"), pathlib.Path("model"), workers=1,
        checkpoint=checkpoint, state=state, ordered=resumed,
        pause_file=pause_file)
    assert paused is False
    assert calls == ["item-0", "item-1", "item-2", "item-3"]
    assert all(outcome is not None for outcome in resumed)
    assert state["sessions"][-1]["workers"] == 8


def test_checkpoint_resume_fails_closed_when_run_identity_changes(tmp_path):
    items, provenance, identity, checkpoint = _checkpoint_fixture(tmp_path)
    prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=False, workers=1)
    changed = dict(identity)
    changed["commit"] = "different"
    with pytest.raises(ValueError, match="run identity changed"):
        prepare_checkpoint(
            checkpoint, changed, provenance, items, resume=True, workers=8)


def test_checkpoint_refuses_implicit_overwrite(tmp_path):
    items, provenance, identity, checkpoint = _checkpoint_fixture(tmp_path)
    prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=False, workers=1)
    with pytest.raises(ValueError, match="already exists"):
        prepare_checkpoint(
            checkpoint, identity, provenance, items, resume=False, workers=1)


def test_checkpoint_and_output_must_stay_outside_repository(tmp_path):
    repository = pathlib.Path(tmp_path) / "repository"
    repository.mkdir()
    with pytest.raises(ValueError, match="outside the repository"):
        _require_outside_repository(
            repository / "results" / "checkpoint", repository, "checkpoint")
    _require_outside_repository(
        pathlib.Path(tmp_path) / "artifacts" / "checkpoint",
        repository, "checkpoint")
