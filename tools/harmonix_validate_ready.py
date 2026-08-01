"""Validate and publish the quality-gated Harmonix aligned-audio manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SR = 22_050


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_reports(report_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in report_dir.glob("*.json"):
        if path.name in {"summary.json", "validation.json"}:
            continue
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        reports[report["file_id"]] = report
    combined_report = report_dir / "alignment_report.csv"
    if combined_report.exists():
        for report in read_csv(combined_report):
            reports.setdefault(report["file_id"], report)
    return reports


def last_annotation_time(path: Path) -> float:
    last = 0.0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            last = max(last, float(row["time_seconds"]))
    return last


def reference_trailing_silence(path: Path) -> float:
    power = np.load(path, mmap_mode="r")
    energy = np.sum(power, axis=0)
    if not len(energy) or float(np.max(energy)) <= 0:
        return max(0, len(energy) - 1) * 1_024 / SR
    active = np.flatnonzero(energy > float(np.max(energy)) * 1e-6)
    if not len(active):
        return max(0, len(energy) - 1) * 1_024 / SR
    return (len(energy) - 1 - int(active[-1])) * 1_024 / SR


def audio_signal_stats(path: Path) -> dict[str, float]:
    sample_count = 0
    sum_squares = 0.0
    peak = 0.0
    clipped = 0
    trailing_zero_samples = 0
    with sf.SoundFile(path) as audio_file:
        for block in audio_file.blocks(blocksize=1_048_576, dtype="float32", always_2d=False):
            block = np.asarray(block)
            if not np.all(np.isfinite(block)):
                raise ValueError(f"non-finite samples in {path.name}")
            absolute = np.abs(block)
            sample_count += len(block)
            sum_squares += float(np.dot(block.astype(np.float64), block.astype(np.float64)))
            peak = max(peak, float(np.max(absolute, initial=0.0)))
            clipped += int(np.count_nonzero(absolute >= (32767 / 32768)))
            nonzero = np.flatnonzero(block)
            if len(nonzero):
                trailing_zero_samples = len(block) - int(nonzero[-1]) - 1
            else:
                trailing_zero_samples += len(block)
    return {
        "rms": math.sqrt(sum_squares / sample_count) if sample_count else 0.0,
        "peak": peak,
        "clipped_fraction": clipped / sample_count if sample_count else 0.0,
        "trailing_zero_seconds": trailing_zero_samples / SR,
    }


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_rows = read_csv(args.metadata)
    metadata = {row["File"]: row for row in metadata_rows}
    reports = load_reports(args.report_dir)
    expected_ids = set(metadata)
    report_ids = set(reports)
    blockers: list[str] = []
    if report_ids != expected_ids:
        blockers.append(
            f"report coverage differs: missing={len(expected_ids - report_ids)} extra={len(report_ids - expected_ids)}"
        )

    accepted_ids = {file_id for file_id, row in reports.items() if row["status"] == "accepted"}
    ready_files = {path.stem: path for path in args.ready_dir.glob("*.wav")}
    if set(ready_files) != accepted_ids:
        blockers.append(
            f"ready WAV set differs: missing={len(accepted_ids - set(ready_files))} "
            f"extra={len(set(ready_files) - accepted_ids)}"
        )

    validation_rows: list[dict[str, Any]] = []
    for file_id in sorted(accepted_ids & set(ready_files)):
        report = reports[file_id]
        path = ready_files[file_id]
        info = sf.info(path)
        expected_frames = int(round(float(metadata[file_id]["Duration"]) * SR))
        annotation_last = last_annotation_time(args.normalized_dir / f"{file_id}.csv")
        reference_zero_tail = reference_trailing_silence(
            args.melspec_dir / f"{file_id}-mel.npy"
        )
        signal = audio_signal_stats(path)
        issues: list[str] = []
        if info.samplerate != SR:
            issues.append("samplerate")
        if info.channels != 1:
            issues.append("channels")
        if info.subtype != "PCM_16":
            issues.append("subtype")
        if info.frames != expected_frames:
            issues.append("duration")
        if annotation_last > info.duration + 0.050:
            issues.append("annotation_beyond_audio")
        if signal["rms"] < 0.001:
            issues.append("near_silent")
        if signal["trailing_zero_seconds"] > max(5.0, reference_zero_tail + 0.25):
            issues.append("long_zero_tail")
        if issues:
            blockers.append(f"{file_id}: {','.join(issues)}")
        validation_rows.append(
            {
                "file_id": file_id,
                "status": "valid" if not issues else "invalid",
                "issues": ";".join(issues),
                "duration_seconds": info.duration,
                "expected_duration_seconds": float(metadata[file_id]["Duration"]),
                "annotation_last_seconds": annotation_last,
                "samplerate": info.samplerate,
                "channels": info.channels,
                "subtype": info.subtype,
                "rms": signal["rms"],
                "peak": signal["peak"],
                "clipped_fraction": signal["clipped_fraction"],
                "trailing_zero_seconds": signal["trailing_zero_seconds"],
                "reference_zero_tail_seconds": reference_zero_tail,
                "correlation_mean": report["correlation_mean"],
                "correlation_median": report["correlation_median"],
                "correlation_fraction_ge_0_8": report["correlation_fraction_ge_0_8"],
                "tempo_ratio": report["tempo_ratio"],
                "source_span_ratio": report["source_span_ratio"],
            }
        )

    counts: dict[str, int] = {}
    for report in reports.values():
        counts[report["status"]] = counts.get(report["status"], 0) + 1
    summary = {
        "manifest_rows": len(metadata_rows),
        "report_rows": len(reports),
        "counts": counts,
        "ready_wav_count": len(ready_files),
        "validated_wav_count": len(validation_rows),
        "validation_blockers": blockers,
        "ready_for_testing": not blockers and len(reports) == len(metadata_rows),
        "audio_format": {"samplerate": SR, "channels": 1, "subtype": "PCM_16"},
        "alignment_method": "coarse subsequence DTW plus fine DTW and frame reconstruction, ported from official Audio Alignment.ipynb",
        "quality_gate": {
            "correlation_mean_min": 0.90,
            "correlation_median_min": 0.94,
            "fraction_frames_correlation_ge_0_8_min": 0.80,
            "tempo_ratio_range": [0.90, 1.10],
            "source_span_ratio_range": [0.90, 1.10],
        },
        "generated_unix": time.time(),
    }
    return summary, validation_rows


def update_manifest(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not summary["ready_for_testing"]:
        raise RuntimeError("refusing to update manifest while validation blockers exist")
    reports = load_reports(args.report_dir)
    rows = read_csv(args.manifest)
    backup = args.report_dir / "manifest.before_alignment.csv"
    if not backup.exists():
        shutil.copy2(args.manifest, backup)
    for row in rows:
        if row["dataset"] != "harmonix":
            continue
        report = reports[row["track_id"]]
        if report["status"] == "accepted":
            row["audio_relpath"] = f"harmonix-ready/{row['track_id']}.wav"
            row["status"] = "ok"
            row["notes"] = (
                "quality-gated DTW alignment to official Harmonix mel reference; "
                f"corr_mean={report['correlation_mean']:.4f}; "
                f"corr_median={report['correlation_median']:.4f}"
            )
        elif report["status"] == "rejected":
            row["audio_relpath"] = ""
            row["status"] = "alignment-rejected"
            row["notes"] = report.get("rejection_reasons", "alignment quality gate failed")
        elif report["status"] == "missing_raw":
            row["audio_relpath"] = ""
            row["status"] = "missing-audio"
            row["notes"] = "official YouTube URL unavailable or blocked; no substitute used"
        else:
            row["audio_relpath"] = ""
            row["status"] = "alignment-error"
            row["notes"] = report.get("error", report["status"])
    atomic_csv(args.manifest, rows, list(rows[0]))

    with args.validation_report.open("r", encoding="utf-8") as handle:
        validation_report = json.load(handle)
    harmonix = validation_report["harmonix"]
    harmonix.update(
        {
            "audio_available": "quality-gated DTW-aligned YouTube audio",
            "audio_downloaded_raw": summary["counts"].get("accepted", 0)
            + summary["counts"].get("rejected", 0),
            "audio_matched": summary["counts"].get("accepted", 0),
            "audio_alignment_rejected": summary["counts"].get("rejected", 0),
            "audio_missing_official_url": summary["counts"].get("missing_raw", 0),
            "audio_alignment_validation": "validation/harmonix_alignment/validation.json",
            "audio_format": summary["audio_format"],
        }
    )
    atomic_json(args.validation_report, validation_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--melspec-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--update-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, validation_rows = validate(args)
    fields = list(validation_rows[0]) if validation_rows else ["file_id", "status", "issues"]
    atomic_csv(args.report_dir / "ready_validation.csv", validation_rows, fields)
    atomic_json(args.report_dir / "validation.json", summary)
    if args.update_manifest:
        update_manifest(args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready_for_testing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
