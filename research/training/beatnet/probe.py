"""Non-binding 5% S1 throughput and GPU-memory probe."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from eval.provenance import experiment_provenance

from .cache import _atomic_json, _outside_repository
from .data import (
    load_cache_manifest, load_cached_recording, verify_cache_records)
from .model import BeatNetTrainable, configure_a3
from .run import _validate_config
from .trainer import ARMS, set_deterministic, train_epoch


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    try:
        _outside_repository(args.output, repository)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        _validate_config(config)
        cache = load_cache_manifest(args.cache)
        if cache.get("diagnostic_only") or cache.get("selected") != 980:
            raise ValueError("S1 throughput probe requires the complete cache")
        provenance = experiment_provenance(
            repository, files={"source": args.source, "cache": args.cache,
                               "config": args.config},
            experiment="S1 throughput probe", fraction=0.05)
        root = args.cache.parent
        verify_cache_records(cache, root)
        train = [row for row in cache["records"] if row["split"] == "train"]
        total_frames = sum(row["frames"] for row in train)
        target = int(total_frames * 0.05)
        selected = []
        selected_frames = 0
        for row in train:
            selected.append(row)
            selected_frames += row["frames"]
            if selected_frames >= target:
                break
        fraction = selected_frames / total_frames
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        loader = lambda row: load_cached_recording(row, root, verify=False)
        arms = {}
        for arm in ARMS:
            set_deterministic(17)
            model = BeatNetTrainable.from_checkpoint(args.source)
            configure_a3(model)
            model.to(device)
            optimizer = torch.optim.Adam(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad], lr=config["learning_rate"])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = train_epoch(
                model, optimizer, selected, arm=arm, seed=17,
                batch_size=config["batch_size"], device=device,
                gradient_clip=config["gradient_clip"], loader=loader)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall = time.perf_counter() - started
            arms[arm] = {
                "wall_sec": wall, "frames": result.frames,
                "supervised_frames": result.supervised_frames,
                "frames_per_sec": result.frames / wall,
                "peak_gpu_bytes": (torch.cuda.max_memory_allocated(device)
                                   if device.type == "cuda" else None),
            }
        mean_epoch = sum(row["wall_sec"] for row in arms.values()) / 2 / fraction
        payload = {
            "schema": "tiktak.s1_throughput/v1", "research_only": True,
            "diagnostic_only": True, "provenance": provenance,
            "device": str(device), "cache_bytes": cache["totals"]["bytes"],
            "sample_records": len(selected), "sample_fraction": fraction,
            "arms": arms, "projected_full_epoch_sec": mean_epoch,
            "projected_six_run_50_epoch_training_hours": mean_epoch * 300 / 3600,
            "projection_excludes_product_validation": True,
        }
        _atomic_json(args.output, payload)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": "complete", "output": str(args.output),
                      "projected_training_hours": payload[
                          "projected_six_run_50_epoch_training_hours"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
