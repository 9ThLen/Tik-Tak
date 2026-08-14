"""Run one preregistered S1 A3 arm/seed with epoch-boundary resume."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from eval.provenance import experiment_provenance

from .cache import _atomic_json, _outside_repository
from .data import (
    file_sha256, fixed_split, load_cache_manifest, load_cached_recording,
    verify_cache_records)
from .evaluate import PRODUCT_BINARY_SHA256, evaluate_model
from .export import export_ttbn, save_state_dict
from .model import BeatNetTrainable, SOURCE_SHA256, configure_a3
from .trainer import (
    ARMS, checkpoint_identity, load_checkpoint_payload, save_checkpoint,
    set_deterministic, train_epoch, validation_loss)


SCHEMA = "tiktak.s1_training/v1"


def _validate_config(config: dict) -> None:
    fixed = {
        "seeds": [17, 29, 43], "batch_size": 8, "learning_rate": 5e-4,
        "class_weights": [50.0, 400.0, 5.0], "gradient_clip": 5.0,
        "max_epochs": 50, "validate_every": 5, "patience_points": 4,
        "block_frames": 400, "warmup_frames": 100,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(
                f"S1 config {key} changed: expected {expected!r}, "
                f"got {config.get(key)!r}")


def _eligible_key(evaluation: dict, baseline: dict) -> tuple | None:
    means = evaluation["means"]
    base = baseline["means"]
    if means["beat_f1"] < base["beat_f1"] - 0.01:
        return None
    return (means["phase_f1"], means["downbeat_f1"], means["beat_f1"])


def c1_training_rows(subset: dict, train_rows: list[dict],
                     fraction: float | None, *, arm: str,
                     cache_sha256: str) -> tuple[list[dict], dict]:
    """Restrict the training rows to a registered C1 fraction.

    Membership comes from the subset artifact; the order comes from the cache
    manifest, and the two are not the same thing -- see `c1_subsets`. The
    identity digest is recomputed from the emitted rows rather than copied, so a
    filter that returned the right works in the wrong order is caught here
    instead of silently training on a different schedule.
    """
    from . import c1_subsets

    c1_subsets.require_registered_corpus(subset)
    if fraction is None:
        raise ValueError("C1 requires an explicit --fraction")
    # C1 registers exactly six runs. Nothing here stopped a subset run from
    # training A3_reset, or from training 1.00 -- which is not a C1 run at all
    # but the S1 anchor, and repeating it would quietly replace the thing the
    # six-run design depends on being reused.
    if arm != c1_subsets.C1_ARM:
        raise ValueError(f"C1 registers only {c1_subsets.C1_ARM}, not {arm}")
    if fraction not in c1_subsets.C1_TRAINED_FRACTIONS:
        raise ValueError(
            f"C1 trains only {sorted(c1_subsets.C1_TRAINED_FRACTIONS)}; "
            f"{fraction:.2f} is the S1 anchor and is reused, not rerun")
    key = f"{fraction:.2f}"
    block = subset.get("fractions", {}).get(key)
    if block is None:
        raise ValueError(f"fraction {key} is not in the subset artifact")
    order = c1_subsets.work_order(train_rows)
    selected = c1_subsets.subset_rows(
        train_rows, c1_subsets.members(order, fraction))
    if (len(selected) != block["records"]
            or c1_subsets.identity_digest(selected) != block["identity_sha256"]):
        raise ValueError(f"fraction {key} does not match the subset artifact")
    if fraction == 1.00:
        # The anchor claim in one assertion: at 100% the filter has to be the
        # identity, or C1's reuse of the S1 runs is not reuse.
        c1_subsets.assert_anchor_schedule({"records": train_rows})
    if subset.get("cache_sha256") not in (None, cache_sha256):
        raise ValueError("C1 subset was built from a different cache")
    return selected, {
        "fraction": fraction, "records": len(selected),
        "cache_sha256": cache_sha256,
        "subset_provenance": subset.get("provenance"),
        "works": block["works"], "frames": block["frames"],
        "identity_sha256": block["identity_sha256"],
        "salt": subset.get("salt"),
    }


def run_training(*, arm: str, seed: int, config: dict,
                 config_path: pathlib.Path,
                 source: pathlib.Path, cache_manifest_path: pathlib.Path,
                 output_root: pathlib.Path, repository: pathlib.Path,
                 device: torch.device, resume: bool,
                 pause_file: pathlib.Path | None,
                 baseline_path: pathlib.Path | None = None,
                 binary: pathlib.Path | None = None,
                 source_manifest: pathlib.Path | None = None,
                 m0e: pathlib.Path | None = None,
                 music_root: pathlib.Path | None = None,
                 eval_workers: int = 4,
                 subset: dict | None = None,
                 fraction: float | None = None) -> dict | None:
    if arm not in ARMS or seed not in config["seeds"]:
        raise ValueError("S1 arm or seed is outside the registration")
    cache = load_cache_manifest(cache_manifest_path)
    if cache.get("diagnostic_only") or cache.get("selected") != 980:
        raise ValueError("binding S1 requires the complete 980-record cache")
    records = cache["records"]
    train_rows = [row for row in records if row["split"] == "train"]
    dev_rows = [row for row in records if row["split"] == "dev"]
    if len(train_rows) != 765 or len(dev_rows) != 215:
        raise ValueError("S1 cache split population changed")

    # C1 restricts the training rows and nothing else. Without a subset this
    # function must stay byte-identical to the one that produced S1, because
    # C1's 100% arm is that run reused as an anchor rather than repeated.
    subset_identity = None
    if subset is not None:
        train_rows, subset_identity = c1_training_rows(
            subset, train_rows, fraction, arm=arm,
            cache_sha256=file_sha256(cache_manifest_path))
    product_inputs = (baseline_path, binary, source_manifest, m0e, music_root)
    if any(value is None for value in product_inputs):
        raise ValueError("binding S1 requires baseline and all product-eval inputs")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (baseline.get("schema") != "tiktak.s1_evaluation/v1"
            or baseline.get("dev_works") != 84
            or baseline.get("model", {}).get("sha256")
            != "812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1"
            or baseline.get("binary", {}).get("sha256")
            != PRODUCT_BINARY_SHA256
            or baseline.get("provenance", {}).get("tree_clean") is not True):
        raise ValueError("invalid frozen A0 development baseline")
    fixed = fixed_split(source_manifest, m0e)
    fixed_identities = [(row["corpus"], row["name"], row["split"])
                        for row in fixed["records"]]
    cache_identities = [(row["corpus"], row["name"], row["split"])
                        for row in records]
    if cache_identities != fixed_identities:
        raise ValueError("S1 cache does not contain the fixed split in order")
    for field, path in (("manifest", source_manifest), ("m0e", m0e),
                        ("binary", binary)):
        if baseline.get(field, {}).get("sha256") != file_sha256(path):
            raise ValueError(f"frozen A0 baseline {field} changed")

    provenance = experiment_provenance(
        repository, files={"source": source, "cache": cache_manifest_path,
                           "config": config_path,
                           "baseline": baseline_path, "binary": binary,
                           "manifest": source_manifest, "m0e": m0e},
        experiment="S1", arm=arm, seed=seed)
    identity = checkpoint_identity(
        config, source_sha256=file_sha256(source),
        split_sha256=file_sha256(cache_manifest_path),
        cache_sha256=file_sha256(cache_manifest_path),
        commit=provenance["commit"])
    identity["arm"] = arm
    identity["seed"] = seed
    identity["baseline_sha256"] = file_sha256(baseline_path)
    # Only when a subset is in play, so an S1 run's identity is unchanged and
    # its checkpoints stay resumable and its anchor argument stays true.
    if subset_identity is not None:
        identity["c1"] = subset_identity

    set_deterministic(seed)
    model = BeatNetTrainable.from_checkpoint(source)
    configure_a3(model)
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=config["learning_rate"])
    checkpoint = output_root / "checkpoint.pt"
    start_epoch = 0
    metadata = {"history": [], "best": None, "stale_points": 0}
    if resume:
        if (output_root / "result.json").exists():
            raise ValueError("S1 run is already complete")
        if not checkpoint.is_file():
            raise ValueError("--resume requested without S1 checkpoint")
        payload = load_checkpoint_payload(
            checkpoint, model, optimizer, identity=identity)
        start_epoch = int(payload["epoch"]) + 1
        metadata = payload["metadata"]
    elif checkpoint.exists() or (output_root / "result.json").exists():
        raise ValueError("refusing to overwrite an existing S1 run")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_manifest_path.parent
    verify_cache_records(cache, cache_root)
    loader = lambda row: load_cached_recording(row, cache_root, verify=False)
    started = time.perf_counter()
    stopped_early = metadata.get("stale_points", 0) >= config["patience_points"]

    for epoch in ([] if stopped_early else
                  range(start_epoch, config["max_epochs"])):
        if pause_file is not None and pause_file.exists():
            return None
        train = train_epoch(
            model, optimizer, train_rows, arm=arm, seed=seed + epoch,
            batch_size=config["batch_size"], device=device,
            gradient_clip=config["gradient_clip"], loader=loader)
        row = {"epoch": epoch + 1, "train_loss": train.loss,
               "train_frames": train.frames,
               "train_supervised_frames": train.supervised_frames}
        validate = ((epoch + 1) % config["validate_every"] == 0
                    or epoch + 1 == config["max_epochs"])
        if validate:
            neural = validation_loss(
                model, dev_rows, arm=arm, seed=seed,
                batch_size=config["batch_size"], device=device, loader=loader)
            candidate_root = output_root / "candidates" / f"epoch-{epoch + 1:03d}"
            candidate_pt = candidate_root / "model.pt"
            candidate_ttbn = candidate_root / "model.ttw"
            save_state_dict(candidate_pt, model)
            export_ttbn(candidate_pt, candidate_ttbn, repository=repository)
            evaluation = evaluate_model(
                split={"records": records}, manifest_path=source_manifest,
                music_root=music_root, m0e_path=m0e, binary=binary,
                model=candidate_ttbn, workers=eval_workers)
            evaluation.update({"arm": arm, "seed": seed, "epoch": epoch + 1,
                               "neural_dev_loss": neural.loss})
            evaluation_path = candidate_root / "evaluation.json"
            _atomic_json(evaluation_path, evaluation)
            key = _eligible_key(evaluation, baseline)
            row.update({"dev_loss": neural.loss,
                        "evaluation_sha256": file_sha256(evaluation_path),
                        "eligible": key is not None})
            old = metadata.get("best")
            if key is not None and (old is None or tuple(key) > tuple(old["key"])):
                metadata["best"] = {
                    "epoch": epoch + 1, "key": list(key),
                    "model": str(candidate_ttbn.relative_to(output_root)),
                    "evaluation": str(evaluation_path.relative_to(output_root)),
                }
                metadata["stale_points"] = 0
            else:
                metadata["stale_points"] += 1
            if metadata["stale_points"] >= config["patience_points"]:
                stopped_early = True
        metadata["history"].append(row)
        save_checkpoint(
            checkpoint, model, optimizer, epoch=epoch,
            identity=identity, metadata=metadata)
        print(json.dumps({"event": "epoch", "arm": arm, "seed": seed,
                          **row}), flush=True)
        if stopped_early:
            break
    result = {
        "schema": SCHEMA, "research_only": True, "provenance": provenance,
        "identity": identity, "arm": arm, "seed": seed,
        "device": str(device), "history": metadata["history"],
        "best": metadata.get("best"), "stopped_early": stopped_early,
        "wall_sec": time.perf_counter() - started,
        "complete": True, "eligible_checkpoint": bool(metadata.get("best")),
    }
    _atomic_json(output_root / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--m0e", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--pause-file", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subset", type=pathlib.Path,
                        help="C1 subset artifact; omit for a full S1 run")
    parser.add_argument("--fraction", type=float,
                        help="C1 registered fraction, required with --subset")
    args = parser.parse_args(argv)
    if (args.subset is None) != (args.fraction is None):
        parser.error("--subset and --fraction are used together or not at all")
    try:
        _outside_repository(args.output_root, repository)
        if args.pause_file is not None:
            _outside_repository(args.pause_file, repository)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        _validate_config(config)
        if file_sha256(args.source) != SOURCE_SHA256:
            raise ValueError("S1 source checkpoint digest changed")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        subset = (None if args.subset is None else
                  json.loads(args.subset.read_text(encoding="utf-8")))
        result = run_training(
            arm=args.arm, seed=args.seed, config=config, source=args.source,
            subset=subset, fraction=args.fraction,
            config_path=args.config,
            cache_manifest_path=args.cache, output_root=args.output_root,
            repository=repository, device=device, resume=args.resume,
            pause_file=args.pause_file, baseline_path=args.baseline,
            binary=args.binary, source_manifest=args.manifest, m0e=args.m0e,
            music_root=args.music_root, eval_workers=args.eval_workers)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if result is None:
        return 75
    print(json.dumps({"event": "complete", "arm": args.arm,
                      "seed": args.seed, "best": result["best"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
