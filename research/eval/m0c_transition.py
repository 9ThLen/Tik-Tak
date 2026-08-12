#!/usr/bin/env python3
"""Run the preregistered M0c A1 meter-transition trace diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np

from eval.m0a_oracle import (
    oracle_channel,
    reference_emit,
    replay_bar,
    require_reference_within_audio,
)
from eval.m0b_oracle import (
    PAUSED_EXIT_CODE,
    _atomic_write_json,
    _checkpoint_state,
    _require_outside_repository,
    load_annotation,
    load_manifest,
    prepare_checkpoint,
    run_checkpointed,
    score_dynamic,
)
from eval.live_corpus_benchmark import _without_local_paths
from eval.octave_veto_replay import run
from eval.prepare_m0b import sha256
from eval.provenance import digest, experiment_provenance
from eval.s0_reset import InvariantError, paired_bootstrap


SOURCE_M0B_SHA256 = (
    "142580478abfe0734bc91ac8fdd20c605a392f9ad7334cb63047d98e5135e921")
SOURCE_M0B_COMMIT = "1b0cb7c6ed71b70e714b208d14f188a0564f165c"
SOURCE_MANIFEST_SHA256 = (
    "484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd")
SOURCE_MODEL_SHA256 = (
    "812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1")
SOURCE_BINARY_SHA256 = (
    "e04881ec4344e451cbdbb44c56ffb7c4b98408ba0d1eff2fc129d1ded620b426")
CHECKPOINT_SCHEMA = "tiktak.m0c_checkpoint/v1"
ARTIFACT_SCHEMA = "tiktak.m0c_transition/v1"
BOOTSTRAP_DRAWS = 2000
PARITY_TOLERANCE = 1e-12
DOMINANT_MEAN = 0.60
DOMINANT_LOWER_CI = 0.50
MIN_FULLY_OBSERVABLE = 30
FAILURE_CLASSES = (
    "stale_previous_grouping",
    "new_grouping_wrong_phase_or_unstable",
    "unknown_or_other",
)
OUTCOME_CLASSES = (
    "acquired_within_two_bars",
    "acquired_late",
    *FAILURE_CLASSES,
    "right_censored",
)


def _file_sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _source_digest(block: object, label: str) -> str:
    if not isinstance(block, dict) or not isinstance(block.get("sha256"), str):
        raise ValueError(f"source M0b has no {label} digest")
    return block["sha256"].lower()


def validate_source_artifact(path: pathlib.Path) -> dict:
    if _file_sha256(path) != SOURCE_M0B_SHA256:
        raise ValueError("source M0b artifact digest changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {})
    if provenance.get("experiment") != "M0b":
        raise ValueError("source artifact is not M0b")
    if provenance.get("commit") != SOURCE_M0B_COMMIT:
        raise ValueError("source M0b commit changed")
    if provenance.get("tree_clean") is not True:
        raise ValueError("source M0b tree was not clean")
    expected = {
        "manifest": SOURCE_MANIFEST_SHA256,
        "model": SOURCE_MODEL_SHA256,
        "binary": SOURCE_BINARY_SHA256,
    }
    for label, expected_digest in expected.items():
        if _source_digest(provenance.get(label), label) != expected_digest:
            raise ValueError(f"source M0b {label} digest changed")
    if not isinstance(payload.get("records"), list):
        raise ValueError("source M0b has no records")
    return payload


def select_population(items: list[dict], source: dict) -> list[dict]:
    manifest_by_identity = {}
    for item in items:
        identity = (item["corpus"], item["name"])
        if identity in manifest_by_identity:
            raise ValueError(f"duplicate manifest identity: {identity}")
        manifest_by_identity[identity] = item

    selected = []
    source_names = set()
    for record in source["records"]:
        if (record.get("corpus") != "rwc2"
                or record.get("primary_eligible") is not True
                or record.get("arms", {}).get("A1", {}).get(
                    "changes", {}).get("total", 0) <= 0):
            continue
        name = record["name"]
        if name in source_names:
            raise ValueError(f"duplicate source M0b record: {name}")
        source_names.add(name)
        item = manifest_by_identity.get(("rwc2", name))
        if item is None:
            raise ValueError(f"source M0b record missing from manifest: {name}")
        if item.get("corpus") != "rwc2" or item.get("primary_eligible") is not True:
            raise ValueError(f"source/manifest population mismatch: {name}")
        common_start = record.get("common_start_sec")
        if not isinstance(common_start, (int, float)) or not math.isfinite(common_start):
            raise ValueError(f"invalid source common_start_sec: {name}")
        selected.append({
            **item,
            "source_common_start_sec": float(common_start),
            "source_a1": record["arms"]["A1"],
        })
    selected.sort(key=lambda row: (row["corpus"], row["name"]))
    if len(selected) != 34:
        raise ValueError(f"expected 34 registered RWC2 records, got {len(selected)}")
    if sum(item["source_a1"]["changes"]["total"] for item in selected) != 123:
        raise ValueError("expected 123 registered RWC2 transitions")
    return selected


def _assert_parity(actual: object, expected: object, path: str = "A1") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise InvariantError(f"{path}: parity keys changed")
        for key in expected:
            _assert_parity(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise InvariantError(f"{path}: parity list changed")
        for index, (got, wanted) in enumerate(zip(actual, expected, strict=True)):
            _assert_parity(got, wanted, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (not isinstance(actual, (int, float))
                or not math.isclose(float(actual), float(expected), rel_tol=0.0,
                                    abs_tol=PARITY_TOLERANCE)):
            raise InvariantError(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise InvariantError(f"{path}: {actual!r} != {expected!r}")


def _first_index(values: np.ndarray, predicate, start: int, stop: int) -> int | None:
    for index in range(start, stop):
        if predicate(index, values[index]):
            return index
    return None


def transition_traces(
    predicted_positions_zero: np.ndarray,
    predicted_groupings: np.ndarray,
    reference: dict[str, np.ndarray],
    start_sec: float,
    *,
    name: str,
    work_id: str,
) -> list[dict]:
    ref_times = reference["times"]
    ref_positions = reference["positions"]
    ref_groupings = reference["groupings"]
    supported = reference["supported"] & (ref_times >= start_sec)
    if (len(predicted_positions_zero) != len(ref_times)
            or len(predicted_groupings) != len(ref_times)):
        raise InvariantError(f"{name}: incomplete A1 transition replay")
    predicted_positions = predicted_positions_zero.astype(np.int64) + 1
    predicted_groupings = predicted_groupings.astype(np.int64)

    changes = []
    downbeats = np.flatnonzero((ref_positions == 1) & supported)
    for previous, current in zip(downbeats, downbeats[1:]):
        if ref_groupings[current] != ref_groupings[previous]:
            changes.append((int(current), int(ref_groupings[previous]),
                            int(ref_groupings[current])))

    traces = []
    for ordinal, (change, old_grouping, new_grouping) in enumerate(changes):
        stop = changes[ordinal + 1][0] if ordinal + 1 < len(changes) else len(ref_times)
        complete_starts = []
        for candidate in range(change, max(change, stop - new_grouping + 1)):
            span = np.arange(candidate, candidate + new_grouping)
            if (ref_positions[candidate] == 1
                    and np.all(ref_groupings[span] == new_grouping)
                    and np.array_equal(ref_positions[span],
                                       np.arange(1, new_grouping + 1))):
                complete_starts.append(candidate)

        acquired = None
        for candidate in complete_starts:
            span = np.arange(candidate, candidate + new_grouping)
            if (np.all(predicted_groupings[span] == new_grouping)
                    and np.array_equal(predicted_positions[span],
                                       ref_positions[span])):
                acquired = candidate
                break

        window_stop = min(stop, change + 2 * new_grouping)
        window = np.arange(change, window_stop)
        denominator = len(window)
        old_share = (float(np.mean(predicted_groupings[window] == old_grouping))
                     if denominator else 0.0)
        new_share = (float(np.mean(predicted_groupings[window] == new_grouping))
                     if denominator else 0.0)
        unknown_share = (float(np.mean(predicted_groupings[window] <= 0))
                         if denominator else 0.0)
        other_share = max(0.0, 1.0 - old_share - new_share - unknown_share)

        first_new = _first_index(
            predicted_groupings,
            lambda _index, value: value == new_grouping,
            change, stop)
        first_phased = _first_index(
            predicted_groupings,
            lambda index, value: (
                value == new_grouping
                and ref_positions[index] == 1
                and predicted_positions[index] == 1),
            change, stop)
        observable = len(complete_starts) >= 3
        offset = acquired - change if acquired is not None else None
        within_two = offset is not None and offset <= 2 * new_grouping
        if not observable:
            outcome = "right_censored"
        elif within_two:
            outcome = "acquired_within_two_bars"
        elif acquired is not None:
            outcome = "acquired_late"
        elif old_share >= 0.50:
            outcome = "stale_previous_grouping"
        elif first_new is not None and first_new < change + 2 * new_grouping:
            outcome = "new_grouping_wrong_phase_or_unstable"
        else:
            outcome = "unknown_or_other"

        context_start = max(0, change - old_grouping)
        context_stop = min(stop, change + 3 * new_grouping)
        context = [{
            "offset_tactus": index - change,
            "time_sec": float(ref_times[index]),
            "reference_grouping": int(ref_groupings[index]),
            "reference_position": int(ref_positions[index]),
            "predicted_grouping": int(predicted_groupings[index]),
            "predicted_position": int(predicted_positions[index]),
        } for index in range(context_start, context_stop)]
        traces.append({
            "transition_id": f"{name}:{ordinal}:{change}",
            "recording": name,
            "work_id": work_id,
            "ordinal": ordinal,
            "reference_index": change,
            "reference_time_sec": float(ref_times[change]),
            "previous_grouping": old_grouping,
            "new_grouping": new_grouping,
            "next_change_or_end_index": stop,
            "available_complete_bars": len(complete_starts),
            "one_bar_observable": len(complete_starts) >= 1,
            "two_bar_latency_observable": observable,
            "first_new_grouping_offset_tactus": (
                first_new - change if first_new is not None else None),
            "first_correctly_phased_downbeat_offset_tactus": (
                first_phased - change if first_phased is not None else None),
            "acquisition_offset_tactus": offset,
            "acquisition_latency_sec": (
                float(ref_times[acquired] - ref_times[change])
                if acquired is not None else None),
            "acquisition_latency_bars": (
                float(offset / new_grouping) if offset is not None else None),
            "acquired_first_bar": acquired == change,
            "acquired_within_two_bars": within_two,
            "first_two_bars_shares": {
                "previous_grouping": old_share,
                "new_grouping": new_share,
                "unknown": unknown_share,
                "other": other_share,
                "observed_tactus": denominator,
            },
            "outcome_class": outcome,
            "context": context,
        })
    return traces


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    reference = load_annotation(item["annotation"])
    ref_times = reference["times"]
    ref_downbeats = ref_times[reference["positions"] == 1]
    initial = run(binary, item["audio"], model, extra=["--live-bars"])
    require_reference_within_audio(
        ref_times, float(initial["duration_sec"]), item["name"])
    frame_times = np.asarray(initial["activation_times"], dtype=np.float64)
    frame_emit = np.asarray(initial["activation_emit"], dtype=np.float64)
    beat_activation = np.asarray(initial["activation_beat"], dtype=np.float64)
    ref_emit = reference_emit(ref_times, float(initial["sample_rate"]))
    payload = replay_bar(
        binary, item["audio"], beat_activation,
        oracle_channel(frame_times, ref_downbeats), frame_emit, frame_times,
        ref_times, ref_emit)
    positions = np.asarray(payload["bar_replay_positions"], dtype=np.float64)
    groupings = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
    common_start = item["source_common_start_sec"]
    score = score_dynamic(
        ref_times, positions, groupings, reference, common_start)
    _assert_parity(score, item["source_a1"])
    traces = transition_traces(
        positions, groupings, reference, common_start,
        name=item["name"], work_id=item["work_id"])
    if len(traces) != item["source_a1"]["changes"]["total"]:
        raise InvariantError(f"{item['name']}: transition count parity failed")
    return {
        "name": item["name"],
        "corpus": item["corpus"],
        "work_id": item["work_id"],
        "annotation": digest(item["annotation"]),
        "common_start_sec": common_start,
        "source_a1": item["source_a1"],
        "transitions": traces,
        "invariants": {"source_A1_parity": True,
                       "reference_grid_complete": True},
    }


def measure_outcome(item: dict, binary: pathlib.Path,
                    model: pathlib.Path) -> tuple[str, dict]:
    try:
        return "record", measure_one(item, binary, model)
    except InvariantError:
        raise
    except Exception as error:
        return "exclusion", {
            "name": item["name"], "corpus": item["corpus"],
            "error_type": type(error).__name__,
            "reason": _without_local_paths(str(error)),
            "annotation": digest(item.get("annotation")),
        }


def _metric_ci(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=np.float64)
    if len(finite) == 0:
        return {"mean": None, "ci": [None, None], "n": 0}
    return {"mean": float(np.mean(finite)),
            "ci": paired_bootstrap(finite, draws=BOOTSTRAP_DRAWS),
            "n": int(len(finite))}


def _work_proportions(transitions: list[dict], predicate,
                      eligible=lambda _row: True) -> list[float]:
    grouped = defaultdict(list)
    for row in transitions:
        if eligible(row):
            grouped[row["work_id"]].append(row)
    return [float(np.mean([predicate(row) for row in rows]))
            for rows in grouped.values()]


def summarise(records: list[dict], exclusions: list[dict]) -> dict:
    transitions = [transition for record in records
                   for transition in record["transitions"]]
    observable = [row for row in transitions
                  if row["two_bar_latency_observable"]]
    failures = [row for row in observable
                if row["outcome_class"] != "acquired_within_two_bars"]
    classes = Counter(row["outcome_class"] for row in transitions)

    metrics = {
        "one_bar_observable": _metric_ci(_work_proportions(
            transitions, lambda row: row["one_bar_observable"])),
        "two_bar_latency_observable": _metric_ci(_work_proportions(
            transitions, lambda row: row["two_bar_latency_observable"])),
        "m0b_intention_to_treat_within_two_bars": _metric_ci(
            _work_proportions(
                transitions, lambda row: row["acquired_within_two_bars"])),
        "fully_observable_within_two_bars": _metric_ci(_work_proportions(
            transitions, lambda row: row["acquired_within_two_bars"],
            lambda row: row["two_bar_latency_observable"])),
        "first_bar_acquisition": _metric_ci(_work_proportions(
            transitions, lambda row: row["acquired_first_bar"])),
    }
    failure_metrics = {
        name: _metric_ci(_work_proportions(
            failures, lambda row, value=name: row["outcome_class"] == value))
        for name in FAILURE_CLASSES
    }
    outcome_metrics = {
        name: _metric_ci(_work_proportions(
            transitions, lambda row, value=name: row["outcome_class"] == value))
        for name in OUTCOME_CLASSES
    }
    dominant = [name for name, metric in failure_metrics.items()
                if metric["mean"] is not None
                and metric["mean"] >= DOMINANT_MEAN
                and metric["ci"][0] >= DOMINANT_LOWER_CI]
    if exclusions or len(observable) < MIN_FULLY_OBSERVABLE:
        interpretation = "inconclusive"
    elif len(dominant) == 1:
        interpretation = f"{dominant[0]}_dominant"
    else:
        interpretation = "mixed"

    by_pair = {}
    for pair in sorted({
            (row["previous_grouping"], row["new_grouping"])
            for row in transitions}):
        key = f"{pair[0]}->{pair[1]}"
        rows = [row for row in transitions
                if (row["previous_grouping"], row["new_grouping"]) == pair]
        counts = Counter(row["outcome_class"] for row in rows)
        by_pair[key] = {
            "transitions": len(rows),
            "works": len({row["work_id"] for row in rows}),
            "raw_outcome_counts": dict(sorted(counts.items())),
            "work_level_outcome_shares": {
                name: _metric_ci(_work_proportions(
                    rows, lambda row, value=name: row["outcome_class"] == value))
                for name in OUTCOME_CLASSES
            },
        }
    by_work = {}
    for work_id in sorted({row["work_id"] for row in transitions}):
        rows = [row for row in transitions if row["work_id"] == work_id]
        counts = Counter(row["outcome_class"] for row in rows)
        by_work[work_id] = {
            "transitions": len(rows),
            "fully_observable": sum(
                row["two_bar_latency_observable"] for row in rows),
            "raw_outcome_counts": dict(sorted(counts.items())),
        }
    return {
        "records": len(records),
        "independent_works": len({row["work_id"] for row in transitions}),
        "transitions": len(transitions),
        "fully_observable_transitions": len(observable),
        "fully_observable_failures": len(failures),
        "raw_outcome_counts": dict(sorted(classes.items())),
        "metrics": metrics,
        "outcome_class_shares": outcome_metrics,
        "failure_class_shares": failure_metrics,
        "by_transition_pair": by_pair,
        "by_work": by_work,
        "interpretation": interpretation,
        "thresholds": {
            "dominant_mean": DOMINANT_MEAN,
            "dominant_lower_ci": DOMINANT_LOWER_CI,
            "min_fully_observable": MIN_FULLY_OBSERVABLE,
            "parity_tolerance": PARITY_TOLERANCE,
        },
    }


def checkpoint_identity(provenance: dict, items: list[dict], *,
                        limit: int, skip_audio_verification: bool) -> dict:
    selection = [{
        "name": item["name"],
        "audio_sha256": item.get("audio_sha256"),
        "annotation_sha256": item.get("annotation_sha256"),
        "common_start_sec": item["source_common_start_sec"],
        "source_changes": item["source_a1"]["changes"]["total"],
    } for item in items]
    selection_sha = hashlib.sha256(json.dumps(
        selection, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "commit": provenance["commit"],
        "binary": provenance["binary"],
        "model": provenance["model"],
        "manifest": provenance["manifest"],
        "source_m0b": provenance["source_m0b"],
        "artifact_schema": ARTIFACT_SCHEMA,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "parity_tolerance": PARITY_TOLERANCE,
        "dominant_mean": DOMINANT_MEAN,
        "dominant_lower_ci": DOMINANT_LOWER_CI,
        "min_fully_observable": MIN_FULLY_OBSERVABLE,
        "limit": limit,
        "skip_audio_verification": skip_audio_verification,
        "selected": len(items),
        "selection_sha256": selection_sha,
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-m0b", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-file", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-audio-verification", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    checkpoint = (args.checkpoint if args.checkpoint is not None else
                  pathlib.Path(f"{args.output}.checkpoint"))
    pause_file = (args.pause_file if args.pause_file is not None else
                  pathlib.Path(f"{checkpoint}.pause"))
    for path, label in ((args.output, "output"),
                        (checkpoint, "checkpoint"),
                        (pause_file, "pause file")):
        try:
            _require_outside_repository(path, repository, label)
        except ValueError as error:
            parser.error(str(error))
    if pause_file.exists():
        parser.error(f"remove the pause file before starting: {pause_file}")

    source = validate_source_artifact(args.source_m0b)
    items, source_manifest = load_manifest(
        args.manifest, args.music_root,
        verify_audio=not args.skip_audio_verification)
    items = select_population(items, source)
    if args.limit:
        items = items[:args.limit]
    provenance = experiment_provenance(
        repository,
        files={"binary": args.binary, "model": args.model,
               "manifest": args.manifest, "source_m0b": args.source_m0b},
        experiment="M0c", arms=["A1_transition_trace"],
        bootstrap_draws=BOOTSTRAP_DRAWS, workers=args.workers)
    identity = checkpoint_identity(
        provenance, items, limit=args.limit,
        skip_audio_verification=args.skip_audio_verification)
    ordered, state, resumed = prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=args.resume,
        workers=args.workers, schema=CHECKPOINT_SCHEMA)
    print(json.dumps({"event": "start", "recordings": len(items),
                      "workers": args.workers, "resumed": resumed}), flush=True)
    ordered, paused = run_checkpointed(
        items, args.binary, args.model, workers=args.workers,
        checkpoint=checkpoint, state=state, ordered=ordered,
        pause_file=pause_file, measure=measure_outcome,
        checkpoint_schema=CHECKPOINT_SCHEMA)
    if paused:
        return PAUSED_EXIT_CODE
    records = [payload for kind, payload in ordered if kind == "record"]
    exclusions = [payload for kind, payload in ordered if kind == "exclusion"]
    summary = summarise(records, exclusions)
    if args.limit or args.skip_audio_verification:
        summary["interpretation"] = "inconclusive"
        summary["diagnostic_only"] = True
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "provenance": provenance,
        "source_m0b": {
            "sha256": SOURCE_M0B_SHA256,
            "commit": SOURCE_M0B_COMMIT,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "source_profile": source_manifest.get("profile", {}),
        "selected": len(items),
        "scored": len(records),
        "technical_exclusions": exclusions,
        "records": records,
        "checkpoint": {
            "schema": CHECKPOINT_SCHEMA,
            "resumed_outcomes": resumed,
            "sessions": state.get("sessions", []),
        },
        "summary": summary,
    }
    _atomic_write_json(args.output, artifact)
    _checkpoint_state(
        checkpoint, state, status="artifact_written",
        completed=len(items), total=len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
