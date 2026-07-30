#!/usr/bin/env python3
"""Corpus-wide benchmark for the causal live tracker.

The ground-truth bundle has a manifest because its four corpora keep audio in
different source folders.  This runner resolves that manifest, executes the
same ``dump_analysis --live`` path as the application, and reports both beat
quality and live tempo stability.

Octave switching is measured against a local reference tempo: the median of up
to ten annotated beat intervals around each one-second live observation.  A
state is half, normal, or double tempo when it is within eight percent of that
ratio.  The tracker's own lock/release hysteresis decides which observations
belong to an active tracking session.

Run from the research folder:

    .venv/Scripts/python -m eval.live_corpus_benchmark \
        --model ../models/beatnet_model_1.ttw --include-root-audio
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import mir_eval.beat
import mir_eval.util
import numpy as np

from eval.analysis import Analyser, DEFAULT_BINARY, Estimate
from eval.annotations import load_annotation
from eval.harness import evaluate

LOCK_CONFIDENCE = 0.25
RELEASE_CONFIDENCE = 0.02
OCTAVE_TOLERANCE = math.log2(1.08)
ROOT_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a"}


def _audio_path(ground_truth: pathlib.Path, row: dict[str, str]) -> pathlib.Path:
    relative = pathlib.Path(row["audio_relpath"])
    if row["dataset"] == "ballroom":
        return ground_truth / "sources" / "ballroom_audio" / relative
    if row["dataset"] == "gtzan":
        return ground_truth / "audio" / "gtzan-ready" / relative
    return ground_truth / relative


def load_corpus(
    manifest: pathlib.Path,
    music: pathlib.Path,
    include_root_audio: bool,
) -> list[dict[str, Any]]:
    ground_truth = manifest.parent
    items: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            has_audio = (
                row["dataset"] in {"ballroom", "gtzan", "smc"}
                or row["status"] == "audio-aligned"
            )
            if not has_audio:
                continue
            items.append(
                {
                    "corpus": row["dataset"],
                    "name": row["track_id"],
                    "audio": _audio_path(ground_truth, row),
                    "annotation": ground_truth / row["annotation_relpath"],
                    "annotated": True,
                }
            )
    if include_root_audio:
        items.extend(
            {
                "corpus": "root",
                "name": path.name,
                "audio": path,
                "annotation": None,
                "annotated": False,
            }
            for path in sorted(music.iterdir())
            if path.is_file() and path.suffix.lower() in ROOT_AUDIO_SUFFIXES
        )
    return items


def load_reference_beats(path: pathlib.Path) -> np.ndarray:
    """Read either the normalized bundle CSV or a regular beat annotation."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "time_seconds" in reader.fieldnames:
            return np.asarray(
                [float(row["time_seconds"]) for row in reader],
                dtype=np.float64,
            )
    return load_annotation(path).beats


def local_reference_bpm(beats: np.ndarray, time_sec: float) -> float:
    beats = np.asarray(beats, dtype=np.float64)
    index = int(np.searchsorted(beats, time_sec))
    start = max(0, index - 5)
    stop = min(len(beats), index + 6)
    intervals = np.diff(beats[start:stop])
    intervals = intervals[
        np.isfinite(intervals) & (intervals > 0.1) & (intervals < 3.0)
    ]
    if len(intervals) == 0:
        return 0.0
    return 60.0 / float(np.median(intervals))


def tempo_state(bpm: float, reference_bpm: float) -> str:
    if not (
        bpm > 0.0
        and reference_bpm > 0.0
        and math.isfinite(bpm)
        and math.isfinite(reference_bpm)
    ):
        return "zero"
    ratio = math.log2(bpm / reference_bpm)
    for octave, name in ((-1, "half"), (0, "same"), (1, "double")):
        if abs(ratio - octave) <= OCTAVE_TOLERANCE:
            return name
    return "other"


def octave_statistics(estimate: Estimate, beats: np.ndarray) -> dict[str, Any]:
    columns = (
        estimate.live_times,
        estimate.live_bpms,
        estimate.live_confidences,
        estimate.live_tempo_spreads_octaves,
    )
    count = min(map(len, columns))
    locked = False
    previous_state: str | None = None
    last_session_state: str | None = None
    new_session = False
    within_switches = 0
    reacquire_switches = 0
    sessions = 0
    active_samples = 0
    eligible_samples = 0
    states: Counter[str] = Counter()
    active_spreads: list[float] = []

    observations = zip(*(np.asarray(column)[:count] for column in columns))
    for time_sec, bpm, confidence, spread in observations:
        if not locked and confidence >= LOCK_CONFIDENCE:
            locked = True
            new_session = True
            sessions += 1
        elif locked and confidence < RELEASE_CONFIDENCE:
            locked = False
            previous_state = None
            new_session = False

        if time_sec < 5.0:
            continue
        eligible_samples += 1
        if not locked:
            continue

        active_samples += 1
        active_spreads.append(float(spread))
        state = tempo_state(
            float(bpm), local_reference_bpm(beats, float(time_sec))
        )
        states[state] += 1
        if state not in {"half", "same", "double"}:
            continue
        if previous_state is not None and state != previous_state:
            within_switches += 1
        elif (
            previous_state is None
            and new_session
            and last_session_state is not None
            and state != last_session_state
        ):
            reacquire_switches += 1
        previous_state = state
        last_session_state = state
        new_session = False

    final_time = (
        float(estimate.live_times[count - 1]) if count else float(beats[-1])
    )
    return {
        "switches": within_switches + reacquire_switches,
        "within_switches": within_switches,
        "reacquire_switches": reacquire_switches,
        "sessions": sessions,
        "active_samples": active_samples,
        "eligible_samples": eligible_samples,
        "states": dict(states),
        "final_ref_bpm": local_reference_bpm(beats, final_time),
        "final_state": tempo_state(
            float(estimate.live_bpm), local_reference_bpm(beats, final_time)
        ),
        "final_active": locked,
        "median_active_spread": (
            float(np.median(active_spreads)) if active_spreads else 0.0
        ),
    }


def _score_one(
    item: dict[str, Any],
    mode: str,
    binary: pathlib.Path,
    model: pathlib.Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        estimate = Analyser(binary).analyse_live_file(
            item["audio"], model=model if mode == "model" else None
        )
        result: dict[str, Any] = {
            "ok": True,
            "mode": mode,
            "corpus": item["corpus"],
            "name": item["name"],
            "annotated": item["annotated"],
            "duration": float(estimate.duration_sec),
            "wall": time.perf_counter() - started,
            "live_bpm": float(estimate.live_bpm),
            "live_confidence": float(estimate.live_confidence),
            "live_spread": float(estimate.live_tempo_spread_octaves),
            "beats": len(estimate.beats),
            "late": int(estimate.live_beats_late),
        }
        if not item["annotated"]:
            return result

        reference = load_reference_beats(item["annotation"])
        result.update(evaluate(reference, estimate.beats, trim=True))
        trimmed_reference = mir_eval.beat.trim_beats(
            np.asarray(reference), min_beat_time=5.0
        )
        trimmed_estimate = mir_eval.beat.trim_beats(
            np.asarray(estimate.beats), min_beat_time=5.0
        )
        matches = (
            mir_eval.util.match_events(
                trimmed_reference, trimmed_estimate, window=0.07
            )
            if len(trimmed_reference) and len(trimmed_estimate)
            else []
        )
        result["p70"] = (
            len(matches) / len(trimmed_estimate) if len(trimmed_estimate) else 0.0
        )
        result["r70"] = (
            len(matches) / len(trimmed_reference)
            if len(trimmed_reference)
            else float("nan")
        )
        result["coverage"] = (
            len(trimmed_estimate) / len(trimmed_reference)
            if len(trimmed_reference)
            else float("nan")
        )
        result.update(octave_statistics(estimate, reference))
        return result
    except Exception as error:  # keep a bad file from hiding the corpus result
        return {
            "ok": False,
            "mode": mode,
            "corpus": item["corpus"],
            "name": item["name"],
            "annotated": item["annotated"],
            "error": str(error),
            "wall": time.perf_counter() - started,
        }


def _finite_stat(values: list[float], function: Any) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(function(array)) if len(array) else None


def summarize(mode: str, results: list[dict[str, Any]], wall: float) -> dict:
    good = [result for result in results if result["ok"]]
    scored = [result for result in good if result["annotated"]]
    quality = {
        key: _finite_stat([result[key] for result in scored], np.mean)
        for key in ("f_measure", "cmlt", "amlt", "p70", "r70", "coverage")
    }
    state_counts: Counter[str] = Counter()
    for result in scored:
        state_counts.update(result["states"])
    total_states = sum(state_counts.values())
    final_states = Counter(result["final_state"] for result in scored)
    switching = [result for result in scored if result["switches"] > 0]
    scored_hours = sum(result["duration"] for result in scored) / 3600.0

    by_corpus = {}
    for corpus in ("ballroom", "gtzan", "harmonix", "smc"):
        part = [result for result in scored if result["corpus"] == corpus]
        switches = sum(result["switches"] for result in part)
        hours = sum(result["duration"] for result in part) / 3600.0
        finals = Counter(result["final_state"] for result in part)
        by_corpus[corpus] = {
            "n": len(part),
            "f_measure": _finite_stat(
                [result["f_measure"] for result in part], np.mean
            ),
            "cmlt": _finite_stat([result["cmlt"] for result in part], np.mean),
            "switch_tracks": sum(result["switches"] > 0 for result in part),
            "switches": switches,
            "switches_per_hour": switches / hours if hours else 0.0,
            "final_same_fraction": finals["same"] / len(part) if part else None,
            "final_half": finals["half"],
            "final_double": finals["double"],
            "final_other_or_zero": finals["other"] + finals["zero"],
        }

    wanted = {
        "0038_bringmetolife",
        "0439_lovethewayyoulie",
        "0446_midnightcity",
        "0471_paradise",
    }
    harmonix_four = [
        {
            key: result[key]
            for key in (
                "name",
                "live_bpm",
                "live_confidence",
                "live_spread",
                "final_ref_bpm",
                "final_state",
                "final_active",
                "switches",
                "within_switches",
                "reacquire_switches",
            )
        }
        for result in scored
        if result["name"] in wanted
    ]
    top = sorted(
        switching,
        key=lambda result: (result["switches"], result["within_switches"]),
        reverse=True,
    )[:12]
    duration = sum(result["duration"] for result in good)
    return {
        "mode": mode,
        "attempted": len(results),
        "success": len(good),
        "scored": len(scored),
        "failures": [
            {
                "corpus": result["corpus"],
                "name": result["name"],
                "error": result["error"],
            }
            for result in results
            if not result["ok"]
        ],
        "quality": quality,
        "silent_tracks": sum(result["beats"] == 0 for result in scored),
        "median_final_confidence": _finite_stat(
            [result["live_confidence"] for result in scored], np.median
        ),
        "median_final_spread_octaves": _finite_stat(
            [result["live_spread"] for result in scored], np.median
        ),
        "active_fraction": (
            sum(result["active_samples"] for result in scored)
            / max(1, sum(result["eligible_samples"] for result in scored))
        ),
        "octave": {
            "tracks_with_switch": len(switching),
            "track_fraction": len(switching) / len(scored) if scored else 0.0,
            "total_switches": sum(result["switches"] for result in scored),
            "within_lock_switches": sum(
                result["within_switches"] for result in scored
            ),
            "reacquire_switches": sum(
                result["reacquire_switches"] for result in scored
            ),
            "switches_per_audio_hour": (
                sum(result["switches"] for result in scored) / scored_hours
            ),
            "active_state_shares": {
                state: state_counts[state] / total_states if total_states else 0.0
                for state in ("same", "half", "double", "other", "zero")
            },
            "final_states": dict(final_states),
        },
        "by_corpus": by_corpus,
        "top_switching": [
            {
                "corpus": result["corpus"],
                "name": result["name"],
                "switches": result["switches"],
                "within": result["within_switches"],
                "reacquire": result["reacquire_switches"],
                "final_bpm": result["live_bpm"],
                "final_state": result["final_state"],
            }
            for result in top
        ],
        "harmonix_four": sorted(
            harmonix_four, key=lambda result: result["name"]
        ),
        "audio_hours": duration / 3600.0,
        "wall_seconds": wall,
        "rtf": wall / max(1.0, duration),
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=repository / "music" / "ground-truth" / "manifest.csv",
    )
    parser.add_argument(
        "--music", type=pathlib.Path, default=repository / "music"
    )
    parser.add_argument(
        "--binary", type=pathlib.Path, default=DEFAULT_BINARY
    )
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument(
        "--mode", choices=("baseline", "model", "both"), default="both"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-root-audio", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode in {"model", "both"} and args.model is None:
        parser.error("--model is required for model mode")

    items = load_corpus(args.manifest, args.music, args.include_root_audio)
    missing_audio = [str(item["audio"]) for item in items if not item["audio"].is_file()]
    missing_annotations = [
        str(item["annotation"])
        for item in items
        if item["annotated"] and not item["annotation"].is_file()
    ]
    if missing_audio or missing_annotations:
        print(
            json.dumps(
                {
                    "missing_audio": missing_audio,
                    "missing_annotations": missing_annotations,
                }
            )
        )
        return 2

    modes = ("baseline", "model") if args.mode == "both" else (args.mode,)
    print(
        json.dumps(
            {
                "event": "start",
                "canonical": len(items),
                "annotated": sum(item["annotated"] for item in items),
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    summaries = []
    for mode in modes:
        started = time.perf_counter()
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = [
                pool.submit(_score_one, item, mode, args.binary, args.model)
                for item in items
            ]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                results.append(future.result())
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "mode": mode,
                                "done": completed,
                                "total": len(futures),
                                "elapsed_sec": round(
                                    time.perf_counter() - started, 1
                                ),
                            }
                        ),
                        flush=True,
                    )
        summaries.append(
            summarize(mode, results, time.perf_counter() - started)
        )

    report = {
        "protocol": {
            "causal": True,
            "callback_samples": 512,
            "poll_seconds": 1.0,
            "warmup_seconds": 5.0,
            "lock_confidence": LOCK_CONFIDENCE,
            "release_confidence": RELEASE_CONFIDENCE,
            "octave_tolerance_percent": 8.0,
            "reference_bpm": (
                "local median of up to 10 neighbouring annotated intervals"
            ),
            "switch_definition": (
                "half/normal/double state change while hysteresis-active; "
                "reacquisition changes reported separately"
            ),
        },
        "summaries": summaries,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        "FINAL_JSON="
        + json.dumps(report, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
