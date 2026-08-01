"""Align downloaded Harmonix audio to the official reference mel spectrograms.

This is a batch-safe port of the reconstruction method in the official
``Audio Alignment.ipynb`` notebook.  The original Harmonix MP3 files are not
distributed here, so the official 80-band mel spectrograms are used as the
reference timeline.  A coarse subsequence DTW locates the matching recording;
a fine DTW then supplies one source frame per official reference frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf


SR = 22_050
N_MELS = 80
N_FFT = 2_048
HOP_LENGTH = 1_024
COARSE_FACTOR = 8
NONDIGONAL_MULTIPLIER = 2.0
FINE_MARGIN_SECONDS = 15.0


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_csv_index(path: Path, key: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row[key]: row for row in rows}


def pool_time_axis(features: np.ndarray, factor: int) -> np.ndarray:
    remainder = features.shape[1] % factor
    if remainder:
        features = np.pad(features, ((0, 0), (0, factor - remainder)), mode="edge")
    return features.reshape(features.shape[0], -1, factor).mean(axis=2)


def dtw_path(reference: np.ndarray, candidate: np.ndarray, subsequence: bool) -> np.ndarray:
    _, path = librosa.sequence.dtw(
        X=reference,
        Y=candidate,
        metric="euclidean",
        subseq=subsequence,
        backtrack=True,
        weights_mul=np.asarray([1.0, NONDIGONAL_MULTIPLIER, NONDIGONAL_MULTIPLIER]),
    )
    return path[::-1]


def map_reference_frames(path: np.ndarray, reference_frames: int) -> np.ndarray:
    mapping: dict[int, int] = {}
    for reference_frame, candidate_frame in path:
        mapping[int(reference_frame)] = int(candidate_frame)
    if len(mapping) == reference_frames:
        return np.asarray([mapping[index] for index in range(reference_frames)], dtype=np.int64)

    known_reference = np.asarray(sorted(mapping), dtype=np.int64)
    if not len(known_reference):
        raise ValueError("DTW produced an empty reference mapping")
    known_candidate = np.asarray([mapping[index] for index in known_reference], dtype=np.float64)
    return np.rint(
        np.interp(np.arange(reference_frames), known_reference, known_candidate)
    ).astype(np.int64)


def align_features(reference_db: np.ndarray, candidate_db: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    reference_coarse = pool_time_axis(reference_db, COARSE_FACTOR)
    candidate_coarse = pool_time_axis(candidate_db, COARSE_FACTOR)
    coarse_subsequence = reference_coarse.shape[1] <= candidate_coarse.shape[1]
    coarse_path = dtw_path(reference_coarse, candidate_coarse, coarse_subsequence)
    coarse_mapping = map_reference_frames(coarse_path, reference_coarse.shape[1])

    margin = int(round(FINE_MARGIN_SECONDS * SR / HOP_LENGTH))
    if candidate_db.shape[1] > reference_db.shape[1]:
        coarse_start = int(coarse_mapping[0] * COARSE_FACTOR)
        coarse_end = int(min(candidate_db.shape[1] - 1, (coarse_mapping[-1] + 1) * COARSE_FACTOR - 1))
        crop_start = max(0, coarse_start - margin)
        crop_end = min(candidate_db.shape[1], coarse_end + margin + 1)
    else:
        crop_start = 0
        crop_end = candidate_db.shape[1]

    candidate_crop = candidate_db[:, crop_start:crop_end]
    fine_subsequence = reference_db.shape[1] <= candidate_crop.shape[1]
    fine_path = dtw_path(reference_db, candidate_crop, fine_subsequence)
    fine_path[:, 1] += crop_start
    selected = map_reference_frames(fine_path, reference_db.shape[1])
    metrics = {
        "coarse_subsequence": coarse_subsequence,
        "fine_subsequence": fine_subsequence,
        "crop_start_frame": crop_start,
        "crop_end_frame": crop_end,
        "path_frames": int(len(fine_path)),
    }
    return selected, metrics


def quality_metrics(
    reference_power: np.ndarray,
    reference_db: np.ndarray,
    candidate_db: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    candidate_selected = candidate_db[:, selected]
    reference_centered = reference_db - np.mean(reference_db, axis=0, keepdims=True)
    candidate_centered = candidate_selected - np.mean(candidate_selected, axis=0, keepdims=True)
    denominator = np.linalg.norm(reference_centered, axis=0) * np.linalg.norm(
        candidate_centered, axis=0
    )
    correlations = np.sum(reference_centered * candidate_centered, axis=0) / np.maximum(
        denominator, 1e-12
    )
    active = (np.ptp(reference_db, axis=0) > 1.0) & (np.sum(reference_power, axis=0) > 0)
    active_correlations = correlations[active]
    if not len(active_correlations):
        active_correlations = correlations

    differences = reference_db - candidate_selected
    frame_rmse = np.sqrt(np.mean(differences * differences, axis=0))
    step_differences = np.diff(selected)
    return {
        "correlation_mean": float(np.mean(active_correlations)),
        "correlation_median": float(np.median(active_correlations)),
        "correlation_p10": float(np.quantile(active_correlations, 0.10)),
        "correlation_fraction_ge_0_8": float(np.mean(active_correlations >= 0.8)),
        "correlation_fraction_ge_0_9": float(np.mean(active_correlations >= 0.9)),
        "rmse_mean_db": float(np.mean(frame_rmse)),
        "rmse_median_db": float(np.median(frame_rmse)),
        "tempo_ratio": float(np.mean(step_differences)) if len(step_differences) else 1.0,
    }


def reconstruct_signal(audio: np.ndarray, selected: np.ndarray, target_samples: int) -> np.ndarray:
    output = np.zeros(target_samples, dtype=np.float32)
    block_count = min(len(selected), math.ceil(target_samples / HOP_LENGTH))
    for reference_frame in range(block_count):
        destination_start = reference_frame * HOP_LENGTH
        destination_end = min(target_samples, destination_start + HOP_LENGTH)
        source_start = int(selected[reference_frame]) * HOP_LENGTH
        source_end = min(len(audio), source_start + destination_end - destination_start)
        copied = max(0, source_end - source_start)
        if copied:
            output[destination_start : destination_start + copied] = audio[source_start:source_end]

    appended_start = len(selected) * HOP_LENGTH
    if appended_start < target_samples:
        source_start = int(selected[-1] + 1) * HOP_LENGTH
        copied = min(target_samples - appended_start, max(0, len(audio) - source_start))
        if copied:
            output[appended_start : appended_start + copied] = audio[
                source_start : source_start + copied
            ]
    return output


def alignment_passes(metrics: dict[str, float], source_span_ratio: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["correlation_mean"] < 0.90:
        reasons.append("correlation_mean_below_0.90")
    if metrics["correlation_median"] < 0.94:
        reasons.append("correlation_median_below_0.94")
    if metrics["correlation_fraction_ge_0_8"] < 0.80:
        reasons.append("fewer_than_80pct_frames_at_corr_0.8")
    if not 0.90 <= metrics["tempo_ratio"] <= 1.10:
        reasons.append("tempo_ratio_outside_0.90_1.10")
    if not 0.90 <= source_span_ratio <= 1.10:
        reasons.append("source_span_ratio_outside_0.90_1.10")
    return not reasons, reasons


def align_one(task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    file_id = task["file_id"]
    report_path = Path(task["report_dir"]) / f"{file_id}.json"
    try:
        reference_power = np.load(task["reference_path"]).astype(np.float32, copy=False)
        if reference_power.shape[0] != N_MELS:
            raise ValueError(f"expected {N_MELS} mel bands, got {reference_power.shape}")
        reference_db = librosa.power_to_db(reference_power)

        audio, _ = librosa.load(task["raw_path"], sr=SR, mono=True)
        reverse = bool(task["reverse"])
        if reverse:
            audio = audio[::-1].copy()
        candidate_power = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        candidate_db = librosa.power_to_db(candidate_power)
        selected, path_metrics = align_features(reference_db, candidate_db)
        metrics = quality_metrics(reference_power, reference_db, candidate_db, selected)

        source_start_seconds = float(selected[0] * HOP_LENGTH / SR)
        source_end_seconds = float((selected[-1] * HOP_LENGTH + HOP_LENGTH) / SR)
        source_span_seconds = max(0.0, source_end_seconds - source_start_seconds)
        target_duration = float(task["target_duration"])
        source_span_ratio = source_span_seconds / target_duration
        accepted, rejection_reasons = alignment_passes(metrics, source_span_ratio)

        result: dict[str, Any] = {
            "file_id": file_id,
            "status": "accepted" if accepted else "rejected",
            "raw_path": task["raw_path"],
            "output_path": task["output_path"] if accepted and not task["dry_run"] else "",
            "reference_path": task["reference_path"],
            "official_url": task["official_url"],
            "official_url_note": task["official_url_note"],
            "official_historical_alignment_score": task["official_historical_alignment_score"],
            "reverse_applied": reverse,
            "raw_duration_seconds": float(len(audio) / SR),
            "target_duration_seconds": target_duration,
            "reference_frames": int(reference_db.shape[1]),
            "candidate_frames": int(candidate_db.shape[1]),
            "source_start_seconds": source_start_seconds,
            "source_end_seconds": source_end_seconds,
            "source_span_seconds": source_span_seconds,
            "source_span_ratio": source_span_ratio,
            "rejection_reasons": ";".join(rejection_reasons),
            **path_metrics,
            **metrics,
        }

        if accepted and not task["dry_run"]:
            target_samples = int(round(target_duration * SR))
            output = reconstruct_signal(audio, selected, target_samples)
            output_path = Path(task["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".tmp.wav")
            sf.write(temporary, output, SR, subtype="PCM_16", format="WAV")
            os.replace(temporary, output_path)
            info = sf.info(output_path)
            result.update(
                {
                    "output_frames": int(info.frames),
                    "output_duration_seconds": float(info.duration),
                    "output_samplerate": int(info.samplerate),
                    "output_channels": int(info.channels),
                    "output_subtype": info.subtype,
                }
            )

        result["elapsed_seconds"] = float(time.monotonic() - started)
        atomic_json(report_path, result)
        return result
    except Exception as error:
        result = {
            "file_id": file_id,
            "status": "error",
            "raw_path": task["raw_path"],
            "output_path": "",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.monotonic() - started),
        }
        atomic_json(report_path, result)
        return result


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "traceback" and not isinstance(value, (dict, list))
    }


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_rows, metadata = load_csv_index(args.metadata, "File")
    _, urls = load_csv_index(args.urls, "File")
    _, historical_scores = load_csv_index(args.historical_scores, "File")
    requested_ids = set(args.ids or [])
    tasks: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for row in metadata_rows:
        file_id = row["File"]
        if requested_ids and file_id not in requested_ids:
            continue
        raw_path = args.raw_dir / f"{file_id}.mp3"
        reference_path = args.melspec_dir / f"{file_id}-mel.npy"
        url_value = urls.get(file_id, {}).get("URL", "")
        official_url, _, note_tail = url_value.partition(" (")
        note = f"({note_tail}" if note_tail else ""
        reverse = "backwards" in note.lower()
        if not raw_path.exists():
            unavailable.append(
                {
                    "file_id": file_id,
                    "status": "missing_raw",
                    "raw_path": str(raw_path),
                    "official_url": official_url,
                    "official_url_note": note,
                }
            )
            continue
        if not reference_path.exists():
            unavailable.append(
                {
                    "file_id": file_id,
                    "status": "missing_reference",
                    "raw_path": str(raw_path),
                    "official_url": official_url,
                    "official_url_note": note,
                }
            )
            continue

        report_path = args.report_dir / f"{file_id}.json"
        output_path = args.ready_dir / f"{file_id}.wav"
        if not args.force and report_path.exists():
            with report_path.open("r", encoding="utf-8") as handle:
                prior = json.load(handle)
            prior_status = prior.get("status")
            if prior_status == "rejected":
                continue
            if prior_status == "accepted" and (args.dry_run or output_path.exists()):
                continue

        tasks.append(
            {
                "file_id": file_id,
                "raw_path": str(raw_path),
                "reference_path": str(reference_path),
                "output_path": str(output_path),
                "report_dir": str(args.report_dir),
                "target_duration": float(metadata[file_id]["Duration"]),
                "official_url": official_url,
                "official_url_note": note,
                "official_historical_alignment_score": historical_scores.get(file_id, {}).get(
                    "score", ""
                ),
                "reverse": reverse,
                "dry_run": args.dry_run,
            }
        )
    return tasks, unavailable


def collect_reports(report_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*.json")):
        if path.name in {"summary.json", "validation.json"}:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                results.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue
    return results


def write_progress(report_dir: Path, unavailable: list[dict[str, Any]]) -> dict[str, Any]:
    results = collect_reports(report_dir)
    combined = sorted(results + unavailable, key=lambda row: row["file_id"])
    fields = sorted({key for row in combined for key in flatten_result(row)})
    if fields:
        atomic_csv(report_dir / "alignment_report.csv", map(flatten_result, combined), fields)
    counts: dict[str, int] = {}
    for row in combined:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"counts": counts, "rows": len(combined), "updated_unix": time.time()}
    atomic_json(report_dir / "summary.json", summary)
    return summary


def command_align(args: argparse.Namespace) -> int:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.ready_dir.mkdir(parents=True, exist_ok=True)
    tasks, unavailable = build_tasks(args)
    print(f"queued={len(tasks)} unavailable={len(unavailable)} workers={args.workers}", flush=True)
    write_progress(args.report_dir, unavailable)

    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            result = align_one(task)
            summary = write_progress(args.report_dir, unavailable)
            print(
                f"[{index}/{len(tasks)}] {result['file_id']} {result['status']} "
                f"elapsed={result['elapsed_seconds']:.1f}s counts={summary['counts']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(align_one, task): task for task in tasks}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                summary = write_progress(args.report_dir, unavailable)
                print(
                    f"[{completed}/{len(tasks)}] {result['file_id']} {result['status']} "
                    f"elapsed={result['elapsed_seconds']:.1f}s counts={summary['counts']}",
                    flush=True,
                )
    summary = write_progress(args.report_dir, unavailable)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not summary["counts"].get("error") else 1


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--ready-dir", type=Path, required=True)
    parser.add_argument("--melspec-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--urls", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    align = subparsers.add_parser("align")
    add_common_paths(align)
    align.add_argument("--workers", type=int, default=1)
    align.add_argument("--ids", nargs="*")
    align.add_argument("--dry-run", action="store_true")
    align.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "align":
        return command_align(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
