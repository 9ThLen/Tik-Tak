#!/usr/bin/env python3
"""Run the pre-registered M0b time-varying causal oracle ladder."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import pathlib
import tempfile
import time
import wave
from collections import defaultdict

import numpy as np

from eval.causal_metre import score_phase
from eval.m0a_oracle import (
    ARMS,
    BLOCK_SAMPLES,
    oracle_channel,
    reference_emit,
    replay_bar,
    require_reference_within_audio,
    visible_indices,
)
from eval.live_corpus_benchmark import _without_local_paths
from eval.octave_veto_replay import run
from eval.prepare_m0b import SCHEMA, SUPPORTED_GROUPINGS, sha256
from eval.provenance import digest, experiment_provenance
from eval.s0_reset import InvariantError, paired_bootstrap


MATCH_TOLERANCE_SEC = 0.070
MIN_WORKS_PER_GROUPING = 5
PHASE_PASS = 0.90
GROUPING_PASS = 0.90
LOWER_CI_PASS = 0.85
UPPER_CI_HARD_NEGATIVE = 0.80
CHANGE_WITHIN_TWO_BARS_PASS = 0.80


def load_manifest(path: pathlib.Path, music_root: pathlib.Path,
                  verify_audio: bool = True) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}, got {payload.get('schema')!r}")
    if verify_audio and not payload.get("audio_hashes_complete"):
        raise ValueError("binding M0b requires complete audio hashes")
    items = []
    identities = set()
    for raw in payload.get("records", []):
        identity = (raw["corpus"], raw["name"])
        if identity in identities:
            raise ValueError(f"duplicate M0b record: {identity}")
        identities.add(identity)
        audio_rel = pathlib.Path(raw["audio_relpath"])
        annotation_rel = pathlib.Path(raw["annotation_relpath"])
        if audio_rel.is_absolute() or annotation_rel.is_absolute():
            raise ValueError(f"{raw['name']}: manifest paths must be relative")
        audio = (music_root / audio_rel).resolve()
        annotation = (path.parent / annotation_rel).resolve()
        try:
            audio.relative_to(music_root.resolve())
            annotation.relative_to(path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"{raw['name']}: manifest path escapes its root") from error
        for name, candidate in (("audio", audio), ("annotation", annotation)):
            if not candidate.is_file():
                raise ValueError(f"{raw['name']}: missing {name}")
        if annotation.stat().st_size != raw["annotation_bytes"]:
            raise ValueError(f"{raw['name']}: annotation size changed")
        if sha256(annotation) != raw["annotation_sha256"]:
            raise ValueError(f"{raw['name']}: annotation digest changed")
        if verify_audio:
            if audio.stat().st_size != raw["audio_bytes"]:
                raise ValueError(f"{raw['name']}: audio size changed")
            if sha256(audio) != raw["audio_sha256"]:
                raise ValueError(f"{raw['name']}: audio digest changed")
        items.append({**raw, "audio": audio, "annotation": annotation})
    if not items:
        raise ValueError("M0b manifest contains no records")
    return items, payload


def load_annotation(path: pathlib.Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name}: empty canonical annotation")
    times = np.asarray([float(row["time_seconds"]) for row in rows])
    positions = np.asarray([int(row["position"]) for row in rows], dtype=np.int64)
    groupings = np.asarray([int(row["grouping"]) for row in rows], dtype=np.int64)
    supported = np.asarray([
        str(row["supported"]).strip().lower() in {"1", "true"} for row in rows
    ], dtype=bool)
    segments = np.asarray([int(row["segment_id"]) for row in rows], dtype=np.int64)
    if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path.name}: tactus times must be finite and increasing")
    if np.any(positions < 1) or np.any(positions > groupings):
        raise ValueError(f"{path.name}: position outside grouping")
    if np.any(supported & ~np.isin(groupings, sorted(SUPPORTED_GROUPINGS))):
        raise ValueError(f"{path.name}: unsupported grouping marked supported")
    return {"times": times, "positions": positions, "groupings": groupings,
            "supported": supported, "segments": segments}


def project_downbeats_to_grid(grid: np.ndarray,
                              downbeats: np.ndarray) -> np.ndarray:
    """Monotonic unique nearest-grid oracle used by A3."""
    if len(grid) == 0 or len(downbeats) == 0:
        return np.zeros(0, dtype=np.float64)
    chosen = []
    lower = 0
    for downbeat in downbeats:
        if lower >= len(grid):
            break
        right = int(np.searchsorted(grid, downbeat, side="left"))
        candidates = [index for index in (right - 1, right)
                      if lower <= index < len(grid)]
        if not candidates:
            index = lower
        else:
            index = min(candidates, key=lambda value: (abs(grid[value] - downbeat), value))
        chosen.append(index)
        lower = index + 1
    return grid[np.asarray(chosen, dtype=np.int64)]


def _match(reference: np.ndarray, predicted: np.ndarray,
           tolerance: float = MATCH_TOLERANCE_SEC) -> np.ndarray:
    """Greedy monotonic one-to-one reference-to-predicted match."""
    out = np.full(len(reference), -1, dtype=np.int64)
    left = 0
    for index, value in enumerate(reference):
        while left < len(predicted) and predicted[left] < value - tolerance:
            left += 1
        candidates = [candidate for candidate in (left, left + 1)
                      if candidate < len(predicted)
                      and abs(predicted[candidate] - value) <= tolerance]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda candidate: (
            abs(predicted[candidate] - value), candidate))
        out[index] = chosen
        left = chosen + 1
    return out


def _metric_ci(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=np.float64)
    if len(finite) == 0:
        return {"mean": None, "ci": [None, None], "n": 0}
    return {"mean": float(np.mean(finite)), "ci": paired_bootstrap(finite),
            "n": int(len(finite))}


def score_dynamic(
    beats: np.ndarray, positions: np.ndarray, meters: np.ndarray,
    reference: dict[str, np.ndarray], start_sec: float,
) -> dict:
    ref_times = reference["times"]
    ref_positions = reference["positions"]
    ref_groupings = reference["groupings"]
    supported = reference["supported"] & (ref_times >= start_sec)
    if not np.any(supported):
        return {"phase_f1": 0.0, "grouping_accuracy": 0.0,
                "grouping_balanced_accuracy": 0.0, "position_accuracy": 0.0,
                "coverage": 0.0, "false_confident_share": 0.0,
                "unnecessary_unknown_share": 1.0, "by_grouping": {},
                "changes": {"total": 0, "acquired": 0,
                            "within_two_bars": 0, "latency_sec": []}}

    match = _match(ref_times, beats)
    predicted_meter = np.zeros(len(ref_times), dtype=np.int64)
    predicted_position = np.full(len(ref_times), -1, dtype=np.int64)
    matched = match >= 0
    predicted_meter[matched] = meters[match[matched]].astype(np.int64)
    predicted_position[matched] = positions[match[matched]].astype(np.int64) + 1

    eligible = np.flatnonzero(supported)
    meter_correct = predicted_meter[eligible] == ref_groupings[eligible]
    position_correct = meter_correct & (
        predicted_position[eligible] == ref_positions[eligible])
    answered = predicted_meter[eligible] > 0
    by_grouping = {}
    for grouping in sorted(set(ref_groupings[eligible])):
        mask = ref_groupings[eligible] == grouping
        by_grouping[str(int(grouping))] = {
            "events": int(np.sum(mask)),
            "grouping_accuracy": float(np.mean(meter_correct[mask])),
            "position_accuracy": float(np.mean(position_correct[mask])),
        }
    balanced = float(np.mean([
        block["grouping_accuracy"] for block in by_grouping.values()
    ])) if by_grouping else 0.0

    downbeats = ref_times[(ref_positions == 1) & reference["supported"]]
    phase = score_phase(beats, positions, meters, downbeats, start_sec=start_sec)
    phase_f1 = float(phase["f1"] if phase["f1"] is not None else 0.0)

    change_indices = []
    downbeat_indices = np.flatnonzero((ref_positions == 1) & supported)
    for previous, current in zip(downbeat_indices, downbeat_indices[1:]):
        if ref_groupings[current] != ref_groupings[previous]:
            change_indices.append(int(current))
    latencies = []
    within = 0
    acquired = 0
    for change_number, change in enumerate(change_indices):
        grouping = int(ref_groupings[change])
        stop = (change_indices[change_number + 1]
                if change_number + 1 < len(change_indices) else len(ref_times))
        found = None
        for candidate in range(change, max(change, stop - grouping + 1)):
            span = np.arange(candidate, candidate + grouping)
            if (np.all(ref_groupings[span] == grouping)
                    and np.all(predicted_meter[span] == grouping)
                    and np.all(predicted_position[span] == ref_positions[span])):
                found = candidate
                break
        if found is not None:
            acquired += 1
            latencies.append(float(ref_times[found] - ref_times[change]))
            if found - change <= 2 * grouping:
                within += 1

    return {
        "phase_f1": phase_f1,
        "grouping_accuracy": float(np.mean(meter_correct)),
        "grouping_balanced_accuracy": balanced,
        "position_accuracy": float(np.mean(position_correct)),
        "coverage": float(np.mean(answered)),
        "false_confident_share": float(np.mean(answered & ~meter_correct)),
        "unnecessary_unknown_share": float(np.mean(~answered)),
        "by_grouping": by_grouping,
        "changes": {"total": len(change_indices), "acquired": acquired,
                    "within_two_bars": within, "latency_sec": latencies},
    }


def synthetic_preflight(binary: pathlib.Path) -> dict:
    """Prove the replay seam can reacquire planted 3->4->6->2 changes."""
    sample_rate = 48000.0
    beat_period = 0.5
    groupings = (3, 4, 6, 2)
    beats_per_segment = 72
    grouping_values = np.concatenate([
        np.full(beats_per_segment, grouping, dtype=np.int64)
        for grouping in groupings])
    beats = 1.0 + np.arange(len(grouping_values)) * beat_period
    positions = np.empty(len(beats), dtype=np.int64)
    offset = 0
    for grouping in groupings:
        positions[offset:offset + beats_per_segment] = (
            np.arange(beats_per_segment) % grouping) + 1
        offset += beats_per_segment
    downbeats = beats[positions == 1]
    duration_sec = int(math.ceil(beats[-1] + 2.0))
    frame_times = np.arange(0.0, duration_sec, 0.02, dtype=np.float64)
    frame_emit = np.maximum(1.0, np.ceil(
        (frame_times + 0.064) * sample_rate / BLOCK_SAMPLES))
    beat_emit = reference_emit(beats, sample_rate)

    with tempfile.TemporaryDirectory() as directory:
        audio = pathlib.Path(directory) / "m0b_changes.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(b"\0\0" * int(sample_rate * duration_sec))
        payload = replay_bar(
            binary, audio, np.zeros(len(frame_times)),
            oracle_channel(frame_times, downbeats), frame_emit, frame_times,
            beats, beat_emit)
    got_positions = np.asarray(payload["bar_replay_positions"], dtype=np.int64)
    got_meters = np.asarray(payload["bar_replay_meters"], dtype=np.int64)
    tails = {}
    offset = 0
    for grouping in groupings:
        tail = np.arange(offset + beats_per_segment - grouping,
                         offset + beats_per_segment)
        meter_ok = float(np.mean(got_meters[tail] == grouping))
        position_ok = float(np.mean(got_positions[tail] + 1 == positions[tail]))
        if meter_ok < 1.0 or position_ok < 0.75:
            raise InvariantError(
                f"synthetic change segment {grouping} failed: "
                f"meter={meter_ok}, position={position_ok}")
        tails[str(grouping)] = {"meter_accuracy": meter_ok,
                                "position_accuracy": position_ok}
        offset += beats_per_segment
    return {"passed": True, "sequence": list(groupings), "segment_tails": tails}


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
    predicted_downbeat = np.asarray(initial["activation_downbeat"], dtype=np.float64)
    predicted_all = np.asarray(initial["live_bar_beats_all"], dtype=np.float64)
    predicted_emit = np.asarray(initial["live_bar_emit_all"], dtype=np.float64)
    predicted_visible = np.asarray(initial["beats"], dtype=np.float64)
    predicted_visible_index = visible_indices(predicted_all, predicted_visible)
    ref_emit = reference_emit(ref_times, float(initial["sample_rate"]))

    definitions = {
        "A1": (ref_times, ref_emit, oracle_channel(frame_times, ref_downbeats), None),
        "A2": (ref_times, ref_emit, predicted_downbeat, None),
        "A3": (predicted_all, predicted_emit, oracle_channel(
            frame_times, project_downbeats_to_grid(predicted_all, ref_downbeats)),
               predicted_visible_index),
        "A4": (predicted_all, predicted_emit, predicted_downbeat,
               predicted_visible_index),
    }
    raw = {}
    for arm, (grid, emit, channel, select) in definitions.items():
        payload = replay_bar(binary, item["audio"], beat_activation, channel,
                             frame_emit, frame_times, grid, emit)
        positions = np.asarray(payload["bar_replay_positions"], dtype=np.float64)
        meters = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
        if len(positions) != len(grid) or len(meters) != len(grid):
            raise InvariantError(f"{item['name']} {arm}: incomplete bar replay")
        raw[arm] = ((grid[select], positions[select], meters[select])
                    if select is not None else (grid, positions, meters))
        if arm == "A4" and not (
            np.array_equal(positions, np.asarray(initial["live_bar_positions_all"]))
            and np.array_equal(meters, np.asarray(initial["live_bar_meters_all"]))
        ):
            raise InvariantError(f"{item['name']}: A4 causal parity failed")
    first = [beats[np.flatnonzero(meters > 0)[0]]
             for beats, _, meters in raw.values() if np.any(meters > 0)]
    common_start = float(max(first)) if first else 0.0
    arms = {arm: score_dynamic(*raw[arm], reference, common_start)
            for arm in ARMS}
    return {
        "name": item["name"], "corpus": item["corpus"],
        "work_id": item["work_id"], "primary_eligible": item["primary_eligible"],
        "groupings": item["groupings"], "meter_families": item["meter_families"],
        "annotation": digest(item["annotation"]), "common_start_sec": common_start,
        "arms": arms, "invariants": {"A4_causal_parity": True,
                                      "A1_A2_grid_identical": True,
                                      "A3_A4_grid_identical": True},
    }


def measure_outcome(item: dict, binary: pathlib.Path,
                    model: pathlib.Path) -> tuple[str, dict]:
    try:
        return "record", measure_one(item, binary, model)
    except InvariantError:
        raise
    except Exception as error:
        return "exclusion", {"name": item["name"], "corpus": item["corpus"],
                             "error_type": type(error).__name__,
                             "reason": _without_local_paths(str(error)),
                             "annotation": digest(item.get("annotation"))}


def _work_rows(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["primary_eligible"]:
            grouped[record["work_id"]].append(record)
    out = []
    for work_id, rows in grouped.items():
        block = {"work_id": work_id, "corpora": sorted({r["corpus"] for r in rows}),
                 "groupings": sorted({g for r in rows for g in r["groupings"]}),
                 "arms": {}}
        for arm in ARMS:
            block["arms"][arm] = {}
            for metric in ("phase_f1", "grouping_accuracy",
                           "grouping_balanced_accuracy", "position_accuracy",
                           "coverage", "false_confident_share",
                           "unnecessary_unknown_share"):
                block["arms"][arm][metric] = float(np.mean([
                    row["arms"][arm][metric] for row in rows]))
            total = sum(row["arms"][arm]["changes"]["total"] for row in rows)
            acquired = sum(row["arms"][arm]["changes"]["acquired"] for row in rows)
            within = sum(row["arms"][arm]["changes"]["within_two_bars"] for row in rows)
            block["arms"][arm]["change_acquisition"] = acquired / total if total else None
            block["arms"][arm]["change_within_two_bars"] = within / total if total else None
        out.append(block)
    return out


def summarise(records: list[dict]) -> dict:
    works = _work_rows(records)
    grouping_counts = {str(grouping): sum(grouping in work["groupings"] for work in works)
                       for grouping in sorted(SUPPORTED_GROUPINGS)}
    arms = {}
    for arm in ARMS:
        arms[arm] = {metric: _metric_ci([
            work["arms"][arm][metric] for work in works
            if work["arms"][arm].get(metric) is not None])
            for metric in ("phase_f1", "grouping_accuracy",
                           "grouping_balanced_accuracy", "position_accuracy",
                           "coverage", "false_confident_share",
                           "unnecessary_unknown_share", "change_acquisition",
                           "change_within_two_bars")}
    deltas = {}
    for metric in ("phase_f1", "grouping_balanced_accuracy"):
        deltas[metric] = _metric_ci([
            work["arms"]["A1"][metric] - work["arms"]["A4"][metric]
            for work in works])

    enough_groups = all(count >= MIN_WORKS_PER_GROUPING
                        for count in grouping_counts.values())
    enough_corpora = len({corpus for work in works for corpus in work["corpora"]}) >= 2
    a1_phase = arms["A1"]["phase_f1"]
    a1_group = arms["A1"]["grouping_balanced_accuracy"]
    a1_change = arms["A1"]["change_within_two_bars"]
    pass_result = (
        enough_groups and enough_corpora
        and a1_phase["mean"] is not None and a1_phase["mean"] >= PHASE_PASS
        and a1_phase["ci"][0] >= LOWER_CI_PASS
        and a1_group["mean"] is not None and a1_group["mean"] >= GROUPING_PASS
        and a1_group["ci"][0] >= LOWER_CI_PASS
        and a1_change["mean"] is not None
        and a1_change["mean"] >= CHANGE_WITHIN_TWO_BARS_PASS)
    hard_negative = (
        enough_groups and enough_corpora
        and ((a1_phase["ci"][1] is not None
              and a1_phase["ci"][1] < UPPER_CI_HARD_NEGATIVE)
             or (a1_group["ci"][1] is not None
                 and a1_group["ci"][1] < UPPER_CI_HARD_NEGATIVE)))
    decision = ("decoder_not_falsified" if pass_result else
                "decoder_bottleneck" if hard_negative else "inconclusive")
    by_corpus = {}
    for corpus in sorted({record["corpus"] for record in records}):
        corpus_works = _work_rows([
            record for record in records if record["corpus"] == corpus])
        by_corpus[corpus] = {
            "records": sum(record["corpus"] == corpus for record in records),
            "primary_works": len(corpus_works),
            "arms": {arm: {
                metric: _metric_ci([
                    work["arms"][arm][metric] for work in corpus_works
                    if work["arms"][arm].get(metric) is not None])
                for metric in ("phase_f1", "grouping_balanced_accuracy",
                               "position_accuracy", "change_within_two_bars")}
                for arm in ARMS},
        }
    return {
        "primary_records": sum(r["primary_eligible"] for r in records),
        "exploratory_records": sum(not r["primary_eligible"] for r in records),
        "independent_works": len(works), "works_by_grouping": grouping_counts,
        "arms": arms, "A1-A4": deltas, "by_corpus": by_corpus,
        "decision": {"verdict": decision,
                     "enough_grouping_coverage": enough_groups,
                     "enough_corpora": enough_corpora,
                     "thresholds": {
                         "phase_point": PHASE_PASS,
                         "grouping_point": GROUPING_PASS,
                         "lower_ci": LOWER_CI_PASS,
                         "hard_negative_upper_ci": UPPER_CI_HARD_NEGATIVE,
                         "change_within_two_bars": CHANGE_WITHIN_TWO_BARS_PASS,
                         "min_works_per_grouping": MIN_WORKS_PER_GROUPING}},
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-audio-verification", action="store_true",
                        help="diagnostic only; forces an inconclusive artifact")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    items, source_manifest = load_manifest(
        args.manifest, args.music_root,
        verify_audio=not args.skip_audio_verification)
    if args.limit:
        items = items[:args.limit]
    provenance = experiment_provenance(
        repository, files={"binary": args.binary, "model": args.model,
                           "manifest": args.manifest}, experiment="M0b",
        arms=list(ARMS), bootstrap_draws=2000, workers=args.workers)
    preflight = synthetic_preflight(args.binary)
    print(json.dumps({"event": "start", "recordings": len(items),
                      "workers": args.workers}), flush=True)
    started = time.perf_counter()
    ordered = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure_outcome, item, args.binary, args.model): index
                   for index, item in enumerate(items)}
        for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1):
            ordered[futures[future]] = future.result()
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"event": "progress", "done": completed,
                                  "total": len(futures), "elapsed_sec": round(
                                      time.perf_counter() - started, 1)}), flush=True)
    records = [payload for kind, payload in ordered if kind == "record"]
    exclusions = [payload for kind, payload in ordered if kind == "exclusion"]
    summary = summarise(records)
    if args.limit or args.skip_audio_verification:
        summary["decision"]["verdict"] = "inconclusive"
        summary["decision"]["diagnostic_only"] = True
    artifact = {
        "provenance": provenance, "synthetic_preflight": preflight,
        "source_profile": source_manifest.get("profile", {}),
        "source_technical_exclusions": source_manifest.get(
            "technical_exclusions", []),
        "selected": len(items), "scored": len(records),
        "technical_exclusions": exclusions, "records": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
