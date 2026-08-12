import json
import pathlib

import numpy as np
import pytest

from eval.m0b_oracle import prepare_checkpoint, run_checkpointed
from eval.m0c_transition import (
    CHECKPOINT_SCHEMA,
    _assert_parity,
    select_population,
    summarise,
    transition_traces,
)
from eval.s0_reset import InvariantError


def _reference(new_bars=3):
    positions = [1, 2, 3] + [value for _ in range(new_bars)
                              for value in (1, 2)]
    groupings = [3, 3, 3] + [2] * (2 * new_bars)
    length = len(positions)
    return {
        "times": np.arange(length, dtype=np.float64),
        "positions": np.asarray(positions, dtype=np.int64),
        "groupings": np.asarray(groupings, dtype=np.int64),
        "supported": np.ones(length, dtype=bool),
        "segments": np.asarray([0] * 3 + [1] * (2 * new_bars),
                               dtype=np.int64),
    }


def _trace(positions_zero, groupings, *, new_bars=3):
    return transition_traces(
        np.asarray(positions_zero), np.asarray(groupings),
        _reference(new_bars), 0.0, name="recording", work_id="work")[0]


def test_transition_trace_separates_acquired_stale_phase_and_censoring():
    reference = _reference(3)
    acquired = _trace(reference["positions"] - 1, reference["groupings"])
    assert acquired["outcome_class"] == "acquired_within_two_bars"
    assert acquired["acquired_first_bar"] is True
    assert acquired["two_bar_latency_observable"] is True

    stale = _trace(
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [3] * 9)
    assert stale["outcome_class"] == "stale_previous_grouping"
    assert stale["first_two_bars_shares"]["previous_grouping"] == 1.0

    wrong_phase = _trace(
        [0, 1, 2, 1, 0, 1, 0, 1, 0],
        [3, 3, 3, 2, 2, 2, 2, 2, 2])
    assert wrong_phase["outcome_class"] == (
        "new_grouping_wrong_phase_or_unstable")
    assert wrong_phase["first_new_grouping_offset_tactus"] == 0
    assert wrong_phase["acquisition_offset_tactus"] is None

    censored_reference = _reference(1)
    censored = _trace(
        censored_reference["positions"] - 1,
        censored_reference["groupings"], new_bars=1)
    assert censored["available_complete_bars"] == 1
    assert censored["two_bar_latency_observable"] is False
    assert censored["outcome_class"] == "right_censored"


def test_summary_dominance_uses_only_fully_observable_failures():
    records = []
    for index in range(30):
        records.append({
            "work_id": f"work-{index}",
            "transitions": [{
                "work_id": f"work-{index}",
                "previous_grouping": 4,
                "new_grouping": 3,
                "one_bar_observable": True,
                "two_bar_latency_observable": True,
                "acquired_within_two_bars": False,
                "acquired_first_bar": False,
                "outcome_class": "stale_previous_grouping",
            }],
        })
    records.append({
        "work_id": "censored",
        "transitions": [{
            "work_id": "censored",
            "previous_grouping": 4,
            "new_grouping": 2,
            "one_bar_observable": True,
            "two_bar_latency_observable": False,
            "acquired_within_two_bars": False,
            "acquired_first_bar": False,
            "outcome_class": "right_censored",
        }],
    })
    summary = summarise(records, [])
    assert summary["fully_observable_transitions"] == 30
    assert summary["interpretation"] == "stale_previous_grouping_dominant"
    assert summary["failure_class_shares"][
        "stale_previous_grouping"]["mean"] == 1.0
    assert summary["outcome_class_shares"][
        "right_censored"]["mean"] == pytest.approx(1.0 / 31.0)
    assert summary["by_transition_pair"]["4->3"]["transitions"] == 30
    assert summary["by_transition_pair"]["4->2"][
        "raw_outcome_counts"] == {"right_censored": 1}
    assert summary["by_work"]["censored"]["fully_observable"] == 0


def test_select_population_is_bound_to_34_records_and_123_transitions():
    items = []
    records = []
    totals = [3] * 33 + [24]
    for index, total in enumerate(totals):
        name = f"track-{index}"
        items.append({
            "name": name, "corpus": "rwc2", "primary_eligible": True,
            "audio_sha256": f"audio-{index}",
            "annotation_sha256": f"annotation-{index}",
        })
        records.append({
            "name": name, "corpus": "rwc2", "primary_eligible": True,
            "common_start_sec": float(index),
            "arms": {"A1": {"changes": {"total": total}}},
        })
    selected = select_population(items, {"records": records})
    assert len(selected) == 34
    assert sum(row["source_a1"]["changes"]["total"]
               for row in selected) == 123

    records[0]["arms"]["A1"]["changes"]["total"] = 2
    with pytest.raises(ValueError, match="123"):
        select_population(items, {"records": records})


def test_parity_is_recursive_and_uses_fixed_absolute_tolerance():
    _assert_parity({"metric": [1.0]}, {"metric": [1.0 + 5e-13]})
    with pytest.raises(InvariantError, match="A1.metric"):
        _assert_parity({"metric": [1.0]}, {"metric": [1.0 + 2e-12]})


def test_checkpoint_helpers_accept_m0c_schema(tmp_path):
    items = [{"corpus": "rwc2", "name": "track"}]
    provenance = {"utc": "2026-08-12T00:00:00Z"}
    checkpoint = pathlib.Path(tmp_path) / "checkpoint"
    ordered, state, completed = prepare_checkpoint(
        checkpoint, {"identity": "m0c"}, provenance, items,
        resume=False, workers=1, schema=CHECKPOINT_SCHEMA)
    assert ordered == [None]
    assert completed == 0
    assert state["schema"] == CHECKPOINT_SCHEMA
    header = json.loads((checkpoint / "header.json").read_text(encoding="utf-8"))
    assert header["schema"] == CHECKPOINT_SCHEMA


def test_shared_scheduler_persists_and_resumes_m0c_outcomes(tmp_path):
    items = [{"corpus": "rwc2", "name": f"track-{index}"}
             for index in range(2)]
    provenance = {"utc": "2026-08-12T00:00:00Z"}
    checkpoint = pathlib.Path(tmp_path) / "checkpoint"
    pause_file = pathlib.Path(tmp_path) / "pause"
    ordered, state, _ = prepare_checkpoint(
        checkpoint, {"identity": "m0c"}, provenance, items,
        resume=False, workers=1, schema=CHECKPOINT_SCHEMA)

    def measure(item, _binary, _model):
        return "record", {"name": item["name"]}

    ordered, paused = run_checkpointed(
        items, pathlib.Path("binary"), pathlib.Path("model"), workers=1,
        checkpoint=checkpoint, state=state, ordered=ordered,
        pause_file=pause_file, measure=measure,
        checkpoint_schema=CHECKPOINT_SCHEMA)
    assert paused is False
    assert [row[1]["name"] for row in ordered] == ["track-0", "track-1"]
    saved = json.loads((checkpoint / "outcomes" / "000000.json").read_text(
        encoding="utf-8"))
    assert saved["schema"] == CHECKPOINT_SCHEMA

    resumed, _, completed = prepare_checkpoint(
        checkpoint, {"identity": "m0c"},
        {"utc": "2026-08-12T01:00:00Z"}, items,
        resume=True, workers=8, schema=CHECKPOINT_SCHEMA)
    assert completed == 2
    assert resumed == ordered
