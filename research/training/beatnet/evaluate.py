"""Product-path development evaluation for one S1 TTBN checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import time
from collections import defaultdict

import mir_eval.util
import numpy as np

from eval.analysis import Estimate
from eval.harness import evaluate, evaluate_downbeats
from eval.live_corpus_benchmark import score_estimate
from eval.m0b_oracle import load_annotation, load_manifest
from eval.m0e_non_oracle import BASELINE, _measure_arm
from eval.octave_veto_replay import run
from eval.provenance import digest, experiment_provenance

from .cache import _atomic_json, _outside_repository
from .data import file_sha256, fixed_split


SCHEMA = "tiktak.s1_evaluation/v1"
PRODUCT_BINARY_SHA256 = (
    "49c47437423f0d79c2f30dde3bcba506f1075099b9f3a7c780efcffe2eed647d")


def validate_product_binary(path: pathlib.Path) -> None:
    if file_sha256(path) != PRODUCT_BINARY_SHA256:
        raise ValueError(
            "S1 requires the M0e product binary with full internal bar traces")


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _prf(reference: np.ndarray, estimate: np.ndarray) -> dict:
    reference = reference[reference >= 5.0]
    estimate = estimate[estimate >= 5.0]
    matches = (mir_eval.util.match_events(reference, estimate, window=0.07)
               if len(reference) and len(estimate) else [])
    precision = len(matches) / len(estimate) if len(estimate) else 0.0
    recall = len(matches) / len(reference) if len(reference) else None
    f1 = (2 * precision * recall / (precision + recall)
          if recall is not None and precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def measure_one(item: dict, binary: pathlib.Path,
                model: pathlib.Path) -> dict:
    reference = load_annotation(item["annotation"])
    initial = run(binary, item["audio"], model, extra=["--live-bars"])
    arm = _measure_arm(item, BASELINE, binary, reference, initial)
    beats = np.asarray(initial["beats"], dtype=np.float64)
    positions = np.asarray(initial["live_bar_positions"], dtype=np.int64)
    if len(positions) != len(beats):
        raise RuntimeError(f"{item['name']}: beat/bar output lengths differ")
    ref_beats = reference["times"][reference["supported"]]
    ref_downbeats = reference["times"][(reference["positions"] == 1)
                                       & reference["supported"]]
    estimated_downbeats = beats[positions == 0]
    canonical = score_estimate(
        {**item, "annotated": True}, Estimate.from_json(initial), mode="S1")
    if canonical.get("ok") is not True:
        raise RuntimeError(
            f"{item['name']}: canonical scorer failed: "
            f"{canonical.get('error', 'unknown error')}")
    beat = _prf(ref_beats, beats)
    downbeat = _prf(ref_downbeats, estimated_downbeats)
    # Keep the harness values explicitly: `_prf` additionally exposes P/R.
    beat["f1"] = _finite(evaluate(ref_beats, beats)["f_measure"])
    downbeat["f1"] = _finite(evaluate_downbeats(
        ref_downbeats, estimated_downbeats)["downbeat_f_measure"])
    return {
        "corpus": item["corpus"], "name": item["name"],
        "work_id": item["work_id"],
        "audio": {"name": item["audio"].name,
                  "bytes": item["audio_bytes"],
                  "sha256": item["audio_sha256"]},
        "annotation": {"name": item["annotation"].name,
                       "bytes": item["annotation_bytes"],
                       "sha256": item["annotation_sha256"]}, "beat": beat,
        "downbeat": downbeat,
        "usable_strict": bool(canonical.get("usable_strict", False)),
        "phase": arm["score"], "static": arm["static"],
        "diagnostics": arm["diagnostics"],
    }


def _work_means(records: list[dict]) -> dict[str, dict[str, float]]:
    values = defaultdict(lambda: defaultdict(list))
    for row in records:
        metrics = {
            "beat_f1": row["beat"]["f1"],
            "beat_precision": row["beat"]["precision"],
            "beat_recall": row["beat"]["recall"],
            "downbeat_f1": row["downbeat"]["f1"],
            "downbeat_precision": row["downbeat"]["precision"],
            "downbeat_recall": row["downbeat"]["recall"],
            "usable_strict": float(row["usable_strict"]),
            "phase_f1": row["phase"]["phase_f1"],
            "position_accuracy": row["phase"]["position_accuracy"],
            "grouping_balanced_accuracy": row["phase"][
                "grouping_balanced_accuracy"],
            "coverage": row["phase"]["coverage"],
            "false_confident_share": row["phase"]["false_confident_share"],
            "unnecessary_unknown_share": row["phase"][
                "unnecessary_unknown_share"],
            "stable_exact_position": row["static"]["accuracy"],
            "false_switches_per_5min": (
                row["static"]["false_switches"] * 300.0
                / row["static"]["eligible_duration_sec"]
                if row["static"]["eligible_duration_sec"] else 0.0),
            "long_wrong_episodes_per_5min": (
                row["static"]["long_wrong_episodes_1bar"] * 300.0
                / row["static"]["eligible_duration_sec"]
                if row["static"]["eligible_duration_sec"] else 0.0),
            "wrong_episodes_per_5min": (
                row["static"]["wrong_episodes"] * 300.0
                / row["static"]["eligible_duration_sec"]
                if row["static"]["eligible_duration_sec"] else 0.0),
            "resolver_state_changes_per_5min": (
                row["diagnostics"]["resolver_path_state_changes"] * 300.0
                / row["static"]["eligible_duration_sec"]
                if row["static"]["eligible_duration_sec"] else 0.0),
            "held_state_changes_per_5min": (
                row["diagnostics"]["held_state_changes"] * 300.0
                / row["static"]["eligible_duration_sec"]
                if row["static"]["eligible_duration_sec"] else 0.0),
            "acquisition_latency_sec": (
                float(np.mean(row["phase"]["changes"]["latency_sec"]))
                if row["phase"]["changes"]["latency_sec"] else None),
        }
        work = f"{row['corpus']}::{row['work_id']}"
        for name, value in metrics.items():
            if value is not None and math.isfinite(float(value)):
                values[work][name].append(float(value))
    return {work: {name: float(np.mean(parts)) for name, parts in metrics.items()}
            for work, metrics in values.items()}


def evaluate_model(*, split: dict, manifest_path: pathlib.Path,
                   music_root: pathlib.Path, m0e_path: pathlib.Path,
                   binary: pathlib.Path, model: pathlib.Path,
                   workers: int) -> dict:
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    validate_product_binary(binary)
    items, _ = load_manifest(manifest_path, music_root, verify_audio=False)
    item_map = {(row["corpus"], row["name"]): row for row in items}
    source = json.loads(m0e_path.read_text(encoding="utf-8"))
    source_map = {(row["corpus"], row["name"]): row
                  for row in source["records"]}
    dev = []
    for row in split["records"]:
        if row["split"] != "dev":
            continue
        identity = (row["corpus"], row["name"])
        item = item_map[identity]
        if file_sha256(item["audio"]) != row["audio_sha256"]:
            raise ValueError(f"development audio digest changed: {identity}")
        item["source_m0b"] = source_map[identity]
        dev.append(item)
    started = time.perf_counter()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(measure_one, row, binary, model): index
                   for index, row in enumerate(dev)}
        ordered = [None] * len(dev)
        for future in concurrent.futures.as_completed(futures):
            ordered[futures[future]] = future.result()
        records = ordered
    works = _work_means(records)
    metric_names = sorted(next(iter(works.values())))
    means = {name: float(np.mean([row[name] for row in works.values()
                                 if name in row])) for name in metric_names}
    return {
        "schema": SCHEMA, "research_only": True,
        "model": digest(model), "binary": digest(binary),
        "manifest": digest(manifest_path), "m0e": digest(m0e_path),
        "dev_records": len(records), "dev_works": len(works),
        "work_corpora": {
            f"{row['corpus']}::{row['work_id']}": row["corpus"]
            for row in records},
        "wall_sec": time.perf_counter() - started,
        "work_metrics": works, "means": means, "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--m0e", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    try:
        _outside_repository(args.output, repository)
        split = fixed_split(args.manifest, args.m0e)
        result = evaluate_model(
            split=split, manifest_path=args.manifest,
            music_root=args.music_root, m0e_path=args.m0e,
            binary=args.binary, model=args.model, workers=args.workers)
        result["provenance"] = experiment_provenance(
            repository, files={"manifest": args.manifest, "m0e": args.m0e,
                               "binary": args.binary, "model": args.model},
            experiment="S1 frozen baseline", workers=args.workers)
        _atomic_json(args.output, result)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": "complete", "output": str(args.output),
                      "dev_works": result["dev_works"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
