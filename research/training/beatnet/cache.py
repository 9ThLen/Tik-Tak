"""Resumable, content-addressed S1 feature-cache preparation."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
import time

import numpy as np

from eval.beatnet_onnx import (
    FEATURES, log_filtered_spectrogram, resample_to_model_rate)
from eval.provenance import digest, experiment_provenance

from .data import (
    file_sha256, fixed_split, frame_labels, load_canonical_annotation)


SCHEMA = "tiktak.s1_cache/v1"


def _outside_repository(path: pathlib.Path, repository: pathlib.Path) -> None:
    try:
        path.resolve().relative_to(repository.resolve())
    except ValueError:
        return
    raise ValueError(f"S1 cache must be outside the repository: {path}")


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary_path = pathlib.Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_npz(path: pathlib.Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary_path = pathlib.Path(temporary)
    try:
        with temporary_path.open("wb") as output:
            np.savez(output, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_audio(path: pathlib.Path) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(path, always_2d=True, dtype="float64")
    return np.mean(audio, axis=1), int(rate)


def build_one(row: dict, *, manifest_root: pathlib.Path,
              music_root: pathlib.Path, output_root: pathlib.Path) -> dict:
    audio = (music_root / row["audio_relpath"]).resolve()
    annotation = (manifest_root / row["annotation_relpath"]).resolve()
    if file_sha256(audio) != row["audio_sha256"]:
        raise ValueError(f"audio digest changed: {row['corpus']}/{row['name']}")
    if file_sha256(annotation) != row["annotation_sha256"]:
        raise ValueError(
            f"annotation digest changed: {row['corpus']}/{row['name']}")
    samples, rate = _read_audio(audio)
    features = log_filtered_spectrogram(
        resample_to_model_rate(samples, rate)).astype(np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURES:
        raise RuntimeError(f"frontend shape changed for {row['name']}")
    labels, mask = frame_labels(
        len(features), load_canonical_annotation(annotation))
    if not np.any(mask):
        raise RuntimeError(f"no supervised frames for {row['name']}")
    relative = pathlib.Path(row["corpus"]) / f"{row['name']}.npz"
    target = output_root / "records" / relative
    _atomic_npz(target, features=features, labels=labels, mask=mask)
    return {
        **row, "cache_relpath": (pathlib.Path("records") / relative).as_posix(),
        "cache_sha256": file_sha256(target), "frames": len(features),
        "supervised_frames": int(np.sum(mask)), "bytes": target.stat().st_size,
    }


def _identity(split: dict, provenance: dict) -> dict:
    return {
        "schema": SCHEMA, "split_sources": split["sources"],
        "commit": provenance["commit"], "frontend": provenance["frontend"],
    }


def prepare_cache(split: dict, *, manifest_path: pathlib.Path,
                  music_root: pathlib.Path, output_root: pathlib.Path,
                  provenance: dict, pause_file: pathlib.Path | None = None,
                  limit: int = 0) -> tuple[dict | None, bool]:
    """Build or resume cache. Existing rows are accepted only by identity/hash."""
    identity = _identity(split, provenance)
    state_path = output_root / "state.json"
    state = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("S1 cache checkpoint identity mismatch")
        if state.get("status") == "complete":
            manifest = output_root / "manifest.json"
            if (not manifest.is_file()
                    or file_sha256(manifest) != state.get("manifest_sha256")):
                raise ValueError("completed S1 cache manifest changed")
            return json.loads(manifest.read_text(encoding="utf-8")), False
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        state = {"schema": "tiktak.s1_cache_state/v1", "identity": identity,
                 "completed": {}, "status": "running"}
        _atomic_json(state_path, state)
    rows = split["records"][:limit or None]
    manifest_root = manifest_path.parent
    for index, row in enumerate(rows):
        key = f"{row['corpus']}/{row['name']}"
        cached = state["completed"].get(key)
        if cached is not None:
            target = output_root / cached["cache_relpath"]
            if not target.is_file() or file_sha256(target) != cached["cache_sha256"]:
                raise ValueError(f"S1 cache checkpoint file changed: {key}")
            continue
        if pause_file is not None and pause_file.exists():
            state["status"] = "paused"
            _atomic_json(state_path, state)
            return None, True
        built = build_one(
            row, manifest_root=manifest_root, music_root=music_root,
            output_root=output_root)
        state["completed"][key] = built
        state["updated_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_json(state_path, state)
        print(json.dumps({"event": "cached", "done": index + 1,
                          "total": len(rows), "record": key}), flush=True)
    records = [state["completed"][f"{row['corpus']}/{row['name']}"]
               for row in rows]
    payload = {
        "schema": SCHEMA, "research_only": True, "identity": identity,
        "provenance": provenance, "diagnostic_only": bool(limit),
        "selected": len(rows), "records": records,
        "totals": {
            "frames": sum(row["frames"] for row in records),
            "supervised_frames": sum(
                row["supervised_frames"] for row in records),
            "bytes": sum(row["bytes"] for row in records),
        },
    }
    manifest = output_root / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("refusing to overwrite a different S1 cache manifest")
    else:
        _atomic_json(manifest, payload)
    state["status"] = "complete"
    state["manifest_sha256"] = file_sha256(manifest)
    _atomic_json(state_path, state)
    return payload, False


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--m0e", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--pause-file", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    try:
        _outside_repository(args.output_root, repository)
        if args.pause_file is not None:
            _outside_repository(args.pause_file, repository)
        split = fixed_split(args.manifest, args.m0e)
        provenance = experiment_provenance(
            repository, files={
                "manifest": args.manifest, "m0e": args.m0e,
                "frontend": pathlib.Path(__file__).resolve().parents[2]
                / "eval" / "beatnet_onnx.py",
            }, experiment="S1 feature cache", limit=args.limit)
        payload, paused = prepare_cache(
            split, manifest_path=args.manifest, music_root=args.music_root,
            output_root=args.output_root, provenance=provenance,
            pause_file=args.pause_file, limit=args.limit)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if paused:
        return 75
    print(json.dumps({"event": "complete", "records": payload["selected"],
                      "manifest": str(args.output_root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
