#!/usr/bin/env python3
"""Run preregistered M0d decoder path-state reacquisition counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
import wave
from collections import Counter, defaultdict

import numpy as np

from eval.m0a_oracle import (
    BLOCK_SAMPLES,
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
from eval.m0c_transition import _assert_parity, transition_traces
from eval.live_corpus_benchmark import _without_local_paths
from eval.octave_veto_replay import run
from eval.provenance import digest, experiment_provenance
from eval.s0_reset import InvariantError, paired_bootstrap


SOURCE_M0C_SHA256 = (
    "88d7ecc2e2ef655faf475081768218a0dc2467d29d048bfa4b1eafc5d23f74fa")
SOURCE_M0C_COMMIT = "c4c5b0c52cce12c835c7f5c626820701c7ff5579"
SOURCE_M0B_SHA256 = (
    "142580478abfe0734bc91ac8fdd20c605a392f9ad7334cb63047d98e5135e921")
SOURCE_MANIFEST_SHA256 = (
    "484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd")
SOURCE_MODEL_SHA256 = (
    "812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1")
SOURCE_BINARY_SHA256 = (
    "e04881ec4344e451cbdbb44c56ffb7c4b98408ba0d1eff2fc129d1ded620b426")

CHECKPOINT_SCHEMA = "tiktak.m0d_checkpoint/v1"
ARTIFACT_SCHEMA = "tiktak.m0d_reacquisition/v1"
BOOTSTRAP_DRAWS = 2000
PARITY_TOLERANCE = 1e-12
MIN_FULLY_OBSERVABLE = 30
FIXED_FULLY_OBSERVABLE = 61
FIXED_EFFICACY_WORKS = 31
FIXED_STABLE_EVENTS = 17175
FIXED_STABLE_WORKS = 34
EFFICACY_MIN_GAIN = 0.20
SAFETY_MAX_LOSS = 0.05
CONTROL_MIN_GAIN = 0.30

ARMS = {
    "B64_opening": {"latest": False, "switch_cost": 64.0,
                    "candidate": False},
    "L64_latest": {"latest": True, "switch_cost": 64.0,
                   "candidate": True},
    "L8_latest": {"latest": True, "switch_cost": 8.0,
                  "candidate": True},
    "L2_latest": {"latest": True, "switch_cost": 2.0,
                  "candidate": True},
    "L0_latest_control": {"latest": True, "switch_cost": 0.0,
                          "candidate": False},
}
CANDIDATES = ("L64_latest", "L8_latest", "L2_latest")
BASELINE = "B64_opening"
POSITIVE_CONTROL = "L0_latest_control"


def _file_sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _source_digest(block: object, label: str) -> str:
    if not isinstance(block, dict) or not isinstance(block.get("sha256"), str):
        raise ValueError(f"source M0c has no {label} digest")
    return block["sha256"].lower()


def validate_source_artifact(path: pathlib.Path) -> dict:
    if _file_sha256(path) != SOURCE_M0C_SHA256:
        raise ValueError("source M0c artifact digest changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tiktak.m0c_transition/v1":
        raise ValueError("source artifact is not M0c schema v1")
    provenance = payload.get("provenance", {})
    if provenance.get("experiment") != "M0c":
        raise ValueError("source artifact is not M0c")
    if provenance.get("commit") != SOURCE_M0C_COMMIT:
        raise ValueError("source M0c commit changed")
    if provenance.get("tree_clean") is not True:
        raise ValueError("source M0c tree was not clean")
    expected = {
        "manifest": SOURCE_MANIFEST_SHA256,
        "model": SOURCE_MODEL_SHA256,
        "binary": SOURCE_BINARY_SHA256,
        "source_m0b": SOURCE_M0B_SHA256,
    }
    for label, wanted in expected.items():
        if _source_digest(provenance.get(label), label) != wanted:
            raise ValueError(f"source M0c {label} digest changed")
    records = payload.get("records")
    if (payload.get("selected") != 34 or payload.get("scored") != 34
            or payload.get("technical_exclusions") not in ([], None)
            or not isinstance(records, list) or len(records) != 34):
        raise ValueError("source M0c population/accounting changed")
    transitions = [row for record in records
                   for row in record.get("transitions", [])]
    if len(transitions) != 123 or len({row["transition_id"]
                                      for row in transitions}) != 123:
        raise ValueError("source M0c transition population changed")
    if sum(row.get("two_bar_latency_observable") is True
           for row in transitions) != 61:
        raise ValueError("source M0c observability population changed")
    for record in records:
        invariants = record.get("invariants", {})
        if (invariants.get("source_A1_parity") is not True
                or invariants.get("reference_grid_complete") is not True):
            raise ValueError(f"source M0c invariant failed: {record.get('name')}")
    return payload


def select_population(items: list[dict], source: dict) -> list[dict]:
    by_identity = {}
    for item in items:
        identity = (item["corpus"], item["name"])
        if identity in by_identity:
            raise ValueError(f"duplicate manifest identity: {identity}")
        by_identity[identity] = item
    selected = []
    for record in source["records"]:
        item = by_identity.get((record["corpus"], record["name"]))
        if item is None:
            raise ValueError(f"source M0c record missing from manifest: "
                             f"{record['name']}")
        if item.get("work_id") != record.get("work_id"):
            raise ValueError(f"source/manifest work mismatch: {record['name']}")
        selected.append({**item, "source_m0c": record})
    selected.sort(key=lambda row: (row["corpus"], row["name"]))
    if len(selected) != 34:
        raise ValueError(f"expected 34 M0c records, got {len(selected)}")
    return selected


def _flags(arm: str) -> list[str]:
    spec = ARMS[arm]
    flags = ["--bar-phase-switch-cost", repr(spec["switch_cost"])]
    if spec["latest"]:
        flags.append("--bar-latest-path-phase")
    return flags


def _state_change_indices(positions: np.ndarray, groupings: np.ndarray,
                          supported: np.ndarray) -> set[int]:
    previous = None
    changes = set()
    for index in np.flatnonzero(supported):
        meter = int(groupings[index])
        position = int(positions[index])
        if meter <= 0 or position < 0:
            state = None
        else:
            state = (meter, (int(index) - position) % meter)
        if previous is not None and state is not None and state != previous:
            changes.add(int(index))
        previous = state
    return changes


def stable_exact_counts(positions: np.ndarray, groupings: np.ndarray,
                        reference: dict[str, np.ndarray], start_sec: float,
                        transitions: list[dict]) -> dict:
    eligible = reference["supported"] & (reference["times"] >= start_sec)
    adaptation = np.zeros(len(eligible), dtype=bool)
    for row in transitions:
        begin = int(row["reference_index"])
        end = min(int(row["next_change_or_end_index"]),
                  begin + 2 * int(row["new_grouping"]))
        adaptation[begin:end] = True
    stable = eligible & ~adaptation
    predicted_positions = positions.astype(np.int64) + 1
    exact = ((groupings.astype(np.int64) == reference["groupings"])
             & (predicted_positions == reference["positions"]))
    return {"correct": int(np.sum(exact & stable)),
            "events": int(np.sum(stable)),
            "accuracy": (float(np.mean(exact[stable]))
                         if np.any(stable) else None)}


def synthetic_preflight(binary: pathlib.Path) -> dict:
    sample_rate = 48000.0
    duration_sec = 48
    frame_times = np.arange(0.0, duration_sec, 0.02, dtype=np.float64)
    frame_emit = np.ceil(
        (frame_times + 0.064) * sample_rate / BLOCK_SAMPLES)
    frame_emit = np.maximum(frame_emit, 1.0)
    beat_activation = np.zeros(len(frame_times), dtype=np.float64)
    beats = np.arange(1.0, duration_sec - 1.0, 0.5, dtype=np.float64)
    beat_emit = reference_emit(beats, sample_rate)
    change = 32
    old = np.arange(0, change, 4)
    new = np.arange(change + 2, len(beats), 4)
    downbeats = beats[np.concatenate([old, new])]

    with tempfile.TemporaryDirectory() as directory:
        audio = pathlib.Path(directory) / "m0d_phase_shift.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(b"\0\0" * int(sample_rate * duration_sec))
        outputs = {}
        for arm in (BASELINE, POSITIVE_CONTROL):
            payload = replay_bar(
                binary, audio, beat_activation,
                oracle_channel(frame_times, downbeats), frame_emit, frame_times,
                beats, beat_emit, extra=_flags(arm))
            outputs[arm] = np.asarray(
                payload["bar_replay_positions"], dtype=np.int64)
        expected = (np.arange(len(beats)) - (change + 2)) % 4
        check = np.arange(change + 4, min(change + 12, len(beats)))
        control_accuracy = float(np.mean(
            outputs[POSITIVE_CONTROL][check] == expected[check]))
        changed = int(np.sum(outputs[BASELINE][check]
                             != outputs[POSITIVE_CONTROL][check]))
        if control_accuracy < 0.90 or changed == 0:
            raise InvariantError(
                "M0d synthetic path/readout preflight failed: "
                f"control_accuracy={control_accuracy}, changed={changed}")
    return {"passed": True, "meter": 4, "phase_shift_tactus": 2,
            "checked_events": int(len(check)),
            "control_exact_accuracy": control_accuracy,
            "baseline_control_differences": changed,
            "neural_state_used": False}


def _measure_arm(item: dict, arm: str, binary: pathlib.Path,
                 reference: dict[str, np.ndarray], initial: dict,
                 ref_emit: np.ndarray) -> dict:
    ref_times = reference["times"]
    frame_times = np.asarray(initial["activation_times"], dtype=np.float64)
    frame_emit = np.asarray(initial["activation_emit"], dtype=np.float64)
    beat_activation = np.asarray(initial["activation_beat"], dtype=np.float64)
    ref_downbeats = ref_times[reference["positions"] == 1]
    payload = replay_bar(
        binary, item["audio"], beat_activation,
        oracle_channel(frame_times, ref_downbeats), frame_emit, frame_times,
        ref_times, ref_emit, extra=_flags(arm))
    positions = np.asarray(payload["bar_replay_positions"], dtype=np.float64)
    path_positions = np.asarray(
        payload["bar_replay_path_positions"], dtype=np.float64)
    groupings = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
    if not (len(positions) == len(path_positions) == len(groupings)
            == len(ref_times)):
        raise InvariantError(f"{item['name']} {arm}: incomplete replay")
    common_start = float(item["source_m0c"]["common_start_sec"])
    score = score_dynamic(
        ref_times, positions, groupings, reference, common_start)
    traces = transition_traces(
        positions, groupings, reference, common_start,
        name=item["name"], work_id=item["work_id"])
    stable = stable_exact_counts(
        positions, groupings, reference, common_start, traces)
    supported = reference["supported"] & (ref_times >= common_start)
    path_changes = _state_change_indices(path_positions, groupings, supported)
    held_changes = _state_change_indices(positions, groupings, supported)
    return {
        "score": score,
        "transitions": traces,
        "stable_exact": stable,
        "diagnostics": {
            "resolver_path_state_changes": len(path_changes),
            "held_state_changes": len(held_changes),
            "path_changes_reflected_in_held": len(path_changes & held_changes),
            "path_held_disagreements": int(np.sum(
                supported & (path_positions != positions))),
        },
    }


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    reference = load_annotation(item["annotation"])
    ref_times = reference["times"]
    initial = run(binary, item["audio"], model, extra=["--live-bars"])
    require_reference_within_audio(
        ref_times, float(initial["duration_sec"]), item["name"])
    ref_emit = reference_emit(ref_times, float(initial["sample_rate"]))
    arms = {arm: _measure_arm(
        item, arm, binary, reference, initial, ref_emit) for arm in ARMS}

    source = item["source_m0c"]
    _assert_parity(arms[BASELINE]["score"], source["source_a1"])
    _assert_parity(arms[BASELINE]["transitions"], source["transitions"])
    source_ids = [row["transition_id"] for row in source["transitions"]]
    for arm, payload in arms.items():
        if [row["transition_id"] for row in payload["transitions"]] != source_ids:
            raise InvariantError(f"{item['name']} {arm}: transition IDs changed")
        for actual, fixed in zip(payload["transitions"], source["transitions"],
                                 strict=True):
            for field in ("reference_index", "next_change_or_end_index",
                          "two_bar_latency_observable", "previous_grouping",
                          "new_grouping"):
                if actual[field] != fixed[field]:
                    raise InvariantError(
                        f"{item['name']} {arm}: fixed {field} changed")
    return {
        "name": item["name"], "corpus": item["corpus"],
        "work_id": item["work_id"],
        "annotation": digest(item["annotation"]),
        "common_start_sec": source["common_start_sec"],
        "arms": arms,
        "invariants": {"source_M0c_baseline_parity": True,
                       "arm_evidence_identical": True,
                       "transition_population_fixed": True},
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


def _metric(values: list[float]) -> dict:
    array = np.asarray([value for value in values if math.isfinite(value)],
                       dtype=np.float64)
    if not len(array):
        return {"mean": None, "ci": [None, None], "n": 0}
    return {"mean": float(np.mean(array)),
            "ci": paired_bootstrap(array, draws=BOOTSTRAP_DRAWS),
            "n": int(len(array))}


def _transition_values(records: list[dict], arm: str, predicate,
                       eligible=lambda _row: True) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        for row in record["arms"][arm]["transitions"]:
            if eligible(row):
                grouped[record["work_id"]].append(float(predicate(row)))
    return {work: float(np.mean(values)) for work, values in grouped.items()}


def _stable_values(records: list[dict], arm: str) -> dict[str, float]:
    grouped = defaultdict(lambda: [0, 0])
    for record in records:
        value = record["arms"][arm]["stable_exact"]
        grouped[record["work_id"]][0] += int(value["correct"])
        grouped[record["work_id"]][1] += int(value["events"])
    return {work: correct / events for work, (correct, events) in grouped.items()
            if events}


def _paired(candidate: dict[str, float], baseline: dict[str, float]) -> dict:
    works = sorted(set(candidate) & set(baseline))
    return _metric([candidate[work] - baseline[work] for work in works])


def summarise(records: list[dict], exclusions: list[dict]) -> dict:
    arm_summaries = {}
    efficacy_values = {}
    stable_values = {}
    for arm in ARMS:
        observable = _transition_values(
            records, arm, lambda row: row["acquired_within_two_bars"],
            lambda row: row["two_bar_latency_observable"])
        intention = _transition_values(
            records, arm, lambda row: row["acquired_within_two_bars"])
        first_bar = _transition_values(
            records, arm, lambda row: row["acquired_first_bar"])
        stable = _stable_values(records, arm)
        transitions = [row for record in records
                       for row in record["arms"][arm]["transitions"]]
        latency = [float(row["acquisition_latency_bars"])
                   for row in transitions
                   if row["acquisition_latency_bars"] is not None]
        efficacy_values[arm] = observable
        stable_values[arm] = stable
        arm_summaries[arm] = {
            "fully_observable_within_two_bars": _metric(list(observable.values())),
            "intention_to_treat_within_two_bars": _metric(list(intention.values())),
            "first_bar_acquisition": _metric(list(first_bar.values())),
            "stable_exact_position": _metric(list(stable.values())),
            "raw_outcome_counts": dict(sorted(Counter(
                row["outcome_class"] for row in transitions).items())),
            "acquired_latency_bars": {
                "n": len(latency),
                "median": float(np.median(latency)) if latency else None,
                "iqr": ([float(value) for value in np.percentile(latency, [25, 75])]
                        if latency else [None, None]),
            },
            "resolver_path_state_changes": sum(
                record["arms"][arm]["diagnostics"][
                    "resolver_path_state_changes"] for record in records),
            "held_state_changes": sum(
                record["arms"][arm]["diagnostics"]["held_state_changes"]
                for record in records),
            "path_changes_reflected_in_held": sum(
                record["arms"][arm]["diagnostics"][
                    "path_changes_reflected_in_held"] for record in records),
            "path_held_disagreements": sum(
                record["arms"][arm]["diagnostics"]["path_held_disagreements"]
                for record in records),
        }

    effects = {}
    for arm in ARMS:
        if arm == BASELINE:
            continue
        efficacy = _paired(efficacy_values[arm], efficacy_values[BASELINE])
        safety = _paired(stable_values[arm], stable_values[BASELINE])
        effects[arm] = {
            "within_two_bars_gain": efficacy,
            "stable_exact_difference": safety,
            "effective": bool(
                efficacy["mean"] is not None
                and efficacy["mean"] >= EFFICACY_MIN_GAIN
                and efficacy["ci"][0] > 0.0),
            "safe": bool(
                safety["mean"] is not None
                and safety["ci"][0] >= -SAFETY_MAX_LOSS),
        }

    control = effects[POSITIVE_CONTROL]["within_two_bars_gain"]
    control_passed = bool(
        control["mean"] is not None
        and control["mean"] >= CONTROL_MIN_GAIN
        and control["ci"][0] > 0.0)
    transition_count = sum(
        len(row["arms"][BASELINE]["transitions"]) for row in records)
    fully_observable = sum(
        row["two_bar_latency_observable"]
        for record in records
        for row in record["arms"][BASELINE]["transitions"])
    stable_event_count = sum(
        record["arms"][BASELINE]["stable_exact"]["events"]
        for record in records)
    complete = (
        not exclusions and len(records) == 34 and transition_count == 123
        and fully_observable == FIXED_FULLY_OBSERVABLE
        and fully_observable >= MIN_FULLY_OBSERVABLE
        and len(efficacy_values[BASELINE]) == FIXED_EFFICACY_WORKS
        and stable_event_count == FIXED_STABLE_EVENTS
        and len(stable_values[BASELINE]) == FIXED_STABLE_WORKS)
    numerical_candidate = next((arm for arm in CANDIDATES
                                if effects[arm]["effective"]
                                and effects[arm]["safe"]), None)
    selected = numerical_candidate if complete and control_passed else None
    any_effective = any(effects[arm]["effective"] for arm in CANDIDATES)
    if not complete or not control_passed:
        interpretation = "inconclusive"
    elif selected == "L64_latest":
        interpretation = "opening_phase_readout_bottleneck"
    elif selected in ("L8_latest", "L2_latest"):
        interpretation = "phase_hysteresis_bottleneck"
    elif any_effective:
        interpretation = "transition_gain_static_cost"
    else:
        interpretation = "registered_decoder_ladder_negative"

    return {
        "records": len(records),
        "independent_works": len({row["work_id"] for row in records}),
        "transitions": int(transition_count),
        "fully_observable_transitions": int(fully_observable),
        "efficacy_works": len(efficacy_values[BASELINE]),
        "stable_events": int(stable_event_count),
        "stable_works": len(stable_values[BASELINE]),
        "arms": arm_summaries,
        "paired_effects": effects,
        "positive_control_passed": control_passed,
        "selected_candidate": selected,
        "interpretation": interpretation,
        "thresholds": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "parity_tolerance": PARITY_TOLERANCE,
            "min_fully_observable": MIN_FULLY_OBSERVABLE,
            "fixed_fully_observable": FIXED_FULLY_OBSERVABLE,
            "fixed_efficacy_works": FIXED_EFFICACY_WORKS,
            "fixed_stable_events": FIXED_STABLE_EVENTS,
            "fixed_stable_works": FIXED_STABLE_WORKS,
            "efficacy_min_gain": EFFICACY_MIN_GAIN,
            "safety_max_loss": SAFETY_MAX_LOSS,
            "positive_control_min_gain": CONTROL_MIN_GAIN,
        },
    }


def checkpoint_identity(provenance: dict, items: list[dict], *, limit: int,
                        skip_audio_verification: bool) -> dict:
    selection = [{
        "name": item["name"],
        "audio_sha256": item.get("audio_sha256"),
        "annotation_sha256": item.get("annotation_sha256"),
        "transition_ids": [row["transition_id"]
                           for row in item["source_m0c"]["transitions"]],
    } for item in items]
    selection_sha = hashlib.sha256(json.dumps(
        selection, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "commit": provenance["commit"],
        "binary": provenance["binary"], "model": provenance["model"],
        "manifest": provenance["manifest"],
        "source_m0c": provenance["source_m0c"],
        "artifact_schema": ARTIFACT_SCHEMA,
        "arms": ARMS,
        "candidate_order": list(CANDIDATES),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "parity_tolerance": PARITY_TOLERANCE,
        "min_fully_observable": MIN_FULLY_OBSERVABLE,
        "fixed_fully_observable": FIXED_FULLY_OBSERVABLE,
        "fixed_efficacy_works": FIXED_EFFICACY_WORKS,
        "fixed_stable_events": FIXED_STABLE_EVENTS,
        "fixed_stable_works": FIXED_STABLE_WORKS,
        "efficacy_min_gain": EFFICACY_MIN_GAIN,
        "safety_max_loss": SAFETY_MAX_LOSS,
        "positive_control_min_gain": CONTROL_MIN_GAIN,
        "limit": limit,
        "skip_audio_verification": skip_audio_verification,
        "selected": len(items), "selection_sha256": selection_sha,
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-m0c", type=pathlib.Path, required=True)
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

    source = validate_source_artifact(args.source_m0c)
    items, source_manifest = load_manifest(
        args.manifest, args.music_root,
        verify_audio=not args.skip_audio_verification)
    items = select_population(items, source)
    if args.limit:
        items = items[:args.limit]
    provenance = experiment_provenance(
        repository,
        files={"binary": args.binary, "model": args.model,
               "manifest": args.manifest, "source_m0c": args.source_m0c},
        experiment="M0d", arms=list(ARMS),
        bootstrap_draws=BOOTSTRAP_DRAWS, workers=args.workers)
    preflight = synthetic_preflight(args.binary)
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
        "source_m0c": {"sha256": SOURCE_M0C_SHA256,
                       "commit": SOURCE_M0C_COMMIT},
        "source_profile": source_manifest.get("profile", {}),
        "synthetic_preflight": preflight,
        "selected": len(items), "scored": len(records),
        "technical_exclusions": exclusions,
        "records": records,
        "checkpoint": {"schema": CHECKPOINT_SCHEMA,
                       "resumed_outcomes": resumed,
                       "sessions": state.get("sessions", [])},
        "summary": summary,
    }
    _atomic_write_json(args.output, artifact)
    _checkpoint_state(checkpoint, state, status="artifact_written",
                      completed=len(ordered), total=len(items))
    print(json.dumps({"event": "complete", "records": len(records),
                      "exclusions": len(exclusions),
                      "interpretation": summary["interpretation"],
                      "output": args.output.name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
