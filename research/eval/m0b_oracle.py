#!/usr/bin/env python3
"""Run the pre-registered M0b time-varying causal oracle ladder."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import tempfile
import time
import wave
from collections import defaultdict, deque
from datetime import datetime, timezone

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
SENSITIVITY_MARGIN = 0.05
SENSITIVITY_CONTROLS = ("profiled_oracle", "shifted_one_tactus")
CHECKPOINT_SCHEMA = "tiktak.m0b_checkpoint/v1"
PAUSED_EXIT_CODE = 75


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def _atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    """Write one durable JSON value without exposing a partial target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    with staged.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staged.replace(path)


def _require_outside_repository(path: pathlib.Path, repository: pathlib.Path,
                                label: str) -> None:
    try:
        path.resolve().relative_to(repository.resolve())
    except ValueError:
        return
    raise ValueError(
        f"{label} must be outside the repository to preserve tree_clean: {path}")


def checkpoint_identity(provenance: dict, items: list[dict], *,
                        workers: int, limit: int,
                        skip_audio_verification: bool) -> dict:
    selection = [{
        "corpus": item["corpus"], "name": item["name"],
        "audio_sha256": item.get("audio_sha256"),
        "annotation_sha256": item.get("annotation_sha256"),
    } for item in items]
    selection_digest = hashlib.sha256(json.dumps(
        selection, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "commit": provenance["commit"],
        "binary": provenance["binary"],
        "model": provenance["model"],
        "manifest": provenance["manifest"],
        "arms": list(ARMS),
        "sensitivity_controls": list(SENSITIVITY_CONTROLS),
        "sensitivity_margin": SENSITIVITY_MARGIN,
        "bootstrap_draws": 2000,
        "workers": workers,
        "limit": limit,
        "skip_audio_verification": skip_audio_verification,
        "selected": len(items),
        "selection_sha256": selection_digest,
    }


def _checkpoint_state(path: pathlib.Path, state: dict, *, status: str,
                      completed: int, total: int) -> None:
    updated = dict(state)
    updated.update({"status": status, "completed": completed,
                    "total": total, "updated_utc": _utc_now()})
    state.clear()
    state.update(updated)
    _atomic_write_json(path / "state.json", state)


def prepare_checkpoint(path: pathlib.Path, identity: dict, provenance: dict,
                       items: list[dict], *, resume: bool
                       ) -> tuple[list[object | None], dict, int]:
    expected_items = [[item["corpus"], item["name"]] for item in items]
    header_path = path / "header.json"
    outcomes_path = path / "outcomes"
    if resume:
        if not header_path.is_file() or not outcomes_path.is_dir():
            raise ValueError(f"cannot resume: incomplete checkpoint {path}")
        header = json.loads(header_path.read_text(encoding="utf-8"))
        if header.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("cannot resume: checkpoint schema changed")
        if header.get("identity") != identity:
            raise ValueError("cannot resume: checkpoint run identity changed")
        if header.get("items") != expected_items:
            raise ValueError("cannot resume: checkpoint item order changed")
        state_path = path / "state.json"
        state = (json.loads(state_path.read_text(encoding="utf-8"))
                 if state_path.is_file() else {})
        sessions = list(state.get("sessions", []))
        sessions.append({"utc": provenance["utc"],
                         "workers": identity["workers"], "resume": True})
        state = {"schema": CHECKPOINT_SCHEMA, "sessions": sessions}
    else:
        if path.exists():
            raise ValueError(
                f"checkpoint already exists; pass --resume or choose another: {path}")
        outcomes_path.mkdir(parents=True)
        _atomic_write_json(header_path, {
            "schema": CHECKPOINT_SCHEMA, "identity": identity,
            "provenance": provenance, "items": expected_items,
        })
        state = {"schema": CHECKPOINT_SCHEMA, "sessions": [{
            "utc": provenance["utc"], "workers": identity["workers"],
            "resume": False,
        }]}

    ordered: list[object | None] = [None] * len(items)
    for outcome_path in sorted(outcomes_path.glob("*.json")):
        try:
            index = int(outcome_path.stem)
        except ValueError as error:
            raise ValueError(
                f"cannot resume: invalid outcome filename {outcome_path.name}") from error
        if not 0 <= index < len(items) or ordered[index] is not None:
            raise ValueError(f"cannot resume: invalid outcome index {index}")
        saved = json.loads(outcome_path.read_text(encoding="utf-8"))
        if saved.get("schema") != CHECKPOINT_SCHEMA or saved.get("index") != index:
            raise ValueError(f"cannot resume: invalid outcome envelope {index}")
        if saved.get("item") != expected_items[index]:
            raise ValueError(f"cannot resume: outcome item changed at {index}")
        outcome = saved.get("outcome")
        if (not isinstance(outcome, list) or len(outcome) != 2
                or outcome[0] not in {"record", "exclusion"}
                or not isinstance(outcome[1], dict)):
            raise ValueError(f"cannot resume: malformed outcome {index}")
        ordered[index] = outcome
    completed = sum(outcome is not None for outcome in ordered)
    _checkpoint_state(path, state, status="ready", completed=completed,
                      total=len(items))
    return ordered, state, completed


def _save_checkpoint_outcome(path: pathlib.Path, index: int, item: dict,
                             outcome: tuple[str, dict]) -> None:
    target = path / "outcomes" / f"{index:06d}.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint outcome {index}")
    _atomic_write_json(target, {
        "schema": CHECKPOINT_SCHEMA, "index": index,
        "item": [item["corpus"], item["name"]],
        "outcome": list(outcome),
    })


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


def profiled_oracle_channels(
    frame_times: np.ndarray,
    predicted_channel: np.ndarray,
    reference_times: np.ndarray,
    reference_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the registered realistic oracle and its shifted positive control."""
    if len(frame_times) != len(predicted_channel) or len(frame_times) < 2:
        raise ValueError("profiled oracle requires matching non-trivial frames")
    if len(reference_times) != len(reference_positions) or len(reference_times) < 2:
        raise ValueError("profiled oracle requires matching reference tactus")
    if (not np.all(np.isfinite(frame_times))
            or not np.all(np.isfinite(predicted_channel))
            or not np.all(np.isfinite(reference_times))):
        raise ValueError("profiled oracle inputs must be finite")

    frame_step = float(np.median(np.diff(frame_times)))
    tactus_interval = float(np.median(np.diff(reference_times)))
    if frame_step <= 0.0 or tactus_interval <= 0.0:
        raise ValueError("profiled oracle clocks must increase")
    half_width_frames = max(1, int(math.ceil(tactus_interval / frame_step)))
    offsets = np.arange(-half_width_frames, half_width_frames + 1,
                        dtype=np.int64)
    source_center = int(np.argmax(predicted_channel))
    source_indices = source_center + offsets
    template = np.zeros(len(offsets), dtype=np.float64)
    source_valid = ((source_indices >= 0)
                    & (source_indices < len(predicted_channel)))
    template[source_valid] = predicted_channel[source_indices[source_valid]]

    def nearest_frame(time_sec: float) -> int:
        right = int(np.searchsorted(frame_times, time_sec, side="left"))
        candidates = [index for index in (right - 1, right)
                      if 0 <= index < len(frame_times)]
        return min(candidates, key=lambda index: (
            abs(frame_times[index] - time_sec), index))

    def stamp(targets: np.ndarray) -> np.ndarray:
        channel = np.zeros(len(frame_times), dtype=np.float64)
        for target in targets:
            target_indices = nearest_frame(float(target)) + offsets
            valid = ((target_indices >= 0) & (target_indices < len(channel)))
            indices = target_indices[valid]
            channel[indices] = np.maximum(channel[indices], template[valid])
        return channel

    downbeat_indices = np.flatnonzero(reference_positions == 1)
    downbeats = reference_times[downbeat_indices]
    shifted_indices = downbeat_indices[downbeat_indices + 1 < len(reference_times)] + 1
    shifted = reference_times[shifted_indices]
    return stamp(downbeats), stamp(shifted), {
        "source_peak_sec": float(frame_times[source_center]),
        "source_peak_value": float(predicted_channel[source_center]),
        "template_half_width_frames": half_width_frames,
        "template_half_width_sec": float(half_width_frames * frame_step),
        "median_tactus_interval_sec": tactus_interval,
        "profiled_targets": int(len(downbeats)),
        "shifted_targets": int(len(shifted)),
        "overlap_rule": "maximum",
    }


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

    profiled, shifted, sensitivity_profile = profiled_oracle_channels(
        frame_times, predicted_downbeat, ref_times, reference["positions"])
    sensitivity = {}
    for name, channel in zip(
            SENSITIVITY_CONTROLS, (profiled, shifted), strict=True):
        payload = replay_bar(binary, item["audio"], beat_activation, channel,
                             frame_emit, frame_times, ref_times, ref_emit)
        positions = np.asarray(payload["bar_replay_positions"], dtype=np.float64)
        meters = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
        if len(positions) != len(ref_times) or len(meters) != len(ref_times):
            raise InvariantError(
                f"{item['name']} {name}: incomplete sensitivity replay")
        sensitivity[name] = score_dynamic(
            ref_times, positions, meters, reference, common_start)
    return {
        "name": item["name"], "corpus": item["corpus"],
        "work_id": item["work_id"], "primary_eligible": item["primary_eligible"],
        "groupings": item["groupings"], "meter_families": item["meter_families"],
        "annotation": digest(item["annotation"]), "common_start_sec": common_start,
        # Same naming rule as `m0a_oracle`: only `A4_causal_parity` is checked,
        # by the raise above. `definitions` gives A1/A2 and A3/A4 the same grid
        # objects, so their identity is a property of this function rather than
        # a result of the run. M0b has not run yet, which is the cheapest moment
        # for an artifact's vocabulary to stop overstating itself.
        "arms": arms, "A1_sensitivity": sensitivity,
        "A1_sensitivity_profile": sensitivity_profile, "invariants": {
            "A4_causal_parity": True,
            "A1_A2_grid_identical_by_construction": True,
            "A3_A4_grid_identical_by_construction": True},
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
        block["A1_sensitivity"] = {
            control: {"phase_f1": float(np.mean([
                row["A1_sensitivity"][control]["phase_f1"] for row in rows]))}
            for control in SENSITIVITY_CONTROLS
        }
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

    profiled_delta = _metric_ci([
        work["A1_sensitivity"]["profiled_oracle"]["phase_f1"]
        - work["arms"]["A1"]["phase_f1"] for work in works])
    positive_drop = _metric_ci([
        work["A1_sensitivity"]["profiled_oracle"]["phase_f1"]
        - work["A1_sensitivity"]["shifted_one_tactus"]["phase_f1"]
        for work in works])
    sensitivity_passed = (
        profiled_delta["mean"] is not None
        and abs(profiled_delta["mean"]) <= SENSITIVITY_MARGIN
        and positive_drop["mean"] is not None
        and positive_drop["mean"] >= SENSITIVITY_MARGIN)

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
    decision = ("inconclusive" if not sensitivity_passed else
                "decoder_not_falsified" if pass_result else
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
            "A1_sensitivity": {
                control: {"phase_f1": _metric_ci([
                    work["A1_sensitivity"][control]["phase_f1"]
                    for work in corpus_works])}
                for control in SENSITIVITY_CONTROLS},
        }
    return {
        "primary_records": sum(r["primary_eligible"] for r in records),
        "exploratory_records": sum(not r["primary_eligible"] for r in records),
        "independent_works": len(works), "works_by_grouping": grouping_counts,
        "arms": arms, "A1-A4": deltas,
        "A1_sensitivity": {
            "profiled_oracle_minus_A1": profiled_delta,
            "profiled_oracle_minus_shifted_one_tactus": positive_drop,
            "passed": sensitivity_passed,
            "limitation": (
                "tests shape and amplitude, but not unaligned false competing "
                "evidence outside the copied template")},
        "by_corpus": by_corpus,
        "decision": {"verdict": decision,
                     "enough_grouping_coverage": enough_groups,
                     "enough_corpora": enough_corpora,
                     "sensitivity_passed": sensitivity_passed,
                     "thresholds": {
                         "phase_point": PHASE_PASS,
                         "grouping_point": GROUPING_PASS,
                         "lower_ci": LOWER_CI_PASS,
                         "hard_negative_upper_ci": UPPER_CI_HARD_NEGATIVE,
                         "change_within_two_bars": CHANGE_WITHIN_TWO_BARS_PASS,
                         "sensitivity_margin": SENSITIVITY_MARGIN,
                         "min_works_per_grouping": MIN_WORKS_PER_GROUPING}},
    }


def run_checkpointed(items: list[dict], binary: pathlib.Path,
                     model: pathlib.Path, *, workers: int,
                     checkpoint: pathlib.Path, state: dict,
                     ordered: list[object | None], pause_file: pathlib.Path
                     ) -> tuple[list[object | None], bool]:
    """Run a bounded queue, checkpointing each outcome before replacing it."""
    pending = deque(index for index, outcome in enumerate(ordered)
                    if outcome is None)
    completed = len(items) - len(pending)
    started = time.perf_counter()
    paused = pause_file.is_file()

    def save(index: int, outcome: tuple[str, dict]) -> None:
        nonlocal completed
        _save_checkpoint_outcome(checkpoint, index, items[index], outcome)
        ordered[index] = list(outcome)
        completed += 1
        _checkpoint_state(checkpoint, state, status="running",
                          completed=completed, total=len(items))
        if completed % 25 == 0 or completed == len(items):
            print(json.dumps({"event": "progress", "done": completed,
                              "total": len(items), "elapsed_sec": round(
                                  time.perf_counter() - started, 1)}), flush=True)

    _checkpoint_state(checkpoint, state, status="running",
                      completed=completed, total=len(items))
    active: dict[concurrent.futures.Future, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        def notice_pause() -> None:
            nonlocal paused
            if pause_file.is_file() and not paused:
                paused = True
                _checkpoint_state(checkpoint, state,
                                  status="pause_requested",
                                  completed=completed, total=len(items))
                print(json.dumps({"event": "pause_requested",
                                  "done": completed,
                                  "active": len(active)}), flush=True)

        def fill() -> None:
            while pending and len(active) < workers and not paused:
                index = pending.popleft()
                active[pool.submit(
                    measure_outcome, items[index], binary, model)] = index

        fill()
        try:
            while active:
                notice_pause()
                done, _ = concurrent.futures.wait(
                    active, timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in sorted(done, key=lambda value: active[value]):
                    index = active.pop(future)
                    save(index, future.result())
                notice_pause()
                fill()
        except KeyboardInterrupt:
            paused = True
            print(json.dumps({"event": "interrupt_draining",
                              "done": completed,
                              "active": len(active)}), flush=True)
            while active:
                done, _ = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in sorted(done, key=lambda value: active[value]):
                    index = active.pop(future)
                    save(index, future.result())
        except Exception:
            _checkpoint_state(checkpoint, state, status="failed",
                              completed=completed, total=len(items))
            raise

    if paused and completed < len(items):
        _checkpoint_state(checkpoint, state, status="paused",
                          completed=completed, total=len(items))
        print(json.dumps({"event": "paused", "done": completed,
                          "total": len(items),
                          "checkpoint": checkpoint.name}), flush=True)
        return ordered, True
    _checkpoint_state(checkpoint, state, status="complete",
                      completed=completed, total=len(items))
    return ordered, False


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--checkpoint", type=pathlib.Path,
        help="checkpoint directory; defaults to <output>.checkpoint")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume a compatible existing checkpoint")
    parser.add_argument(
        "--pause-file", type=pathlib.Path,
        help="when this file appears, drain active workers and exit 75")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-audio-verification", action="store_true",
                        help="diagnostic only; forces an inconclusive artifact")
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
    items, source_manifest = load_manifest(
        args.manifest, args.music_root,
        verify_audio=not args.skip_audio_verification)
    if args.limit:
        items = items[:args.limit]
    provenance = experiment_provenance(
        repository, files={"binary": args.binary, "model": args.model,
                           "manifest": args.manifest}, experiment="M0b",
        arms=list(ARMS), bootstrap_draws=2000, workers=args.workers)
    identity = checkpoint_identity(
        provenance, items, workers=args.workers, limit=args.limit,
        skip_audio_verification=args.skip_audio_verification)
    ordered, checkpoint_state, resumed = prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=args.resume)
    preflight = synthetic_preflight(args.binary)
    print(json.dumps({"event": "start", "recordings": len(items),
                      "workers": args.workers, "resumed": resumed}), flush=True)
    ordered, paused = run_checkpointed(
        items, args.binary, args.model, workers=args.workers,
        checkpoint=checkpoint, state=checkpoint_state, ordered=ordered,
        pause_file=pause_file)
    if paused:
        return PAUSED_EXIT_CODE
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
        "checkpoint": {
            "schema": CHECKPOINT_SCHEMA, "resumed_outcomes": resumed,
            "sessions": checkpoint_state.get("sessions", []),
        },
        "summary": summary,
    }
    _atomic_write_json(args.output, artifact)
    _checkpoint_state(checkpoint, checkpoint_state, status="artifact_written",
                      completed=len(items), total=len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
