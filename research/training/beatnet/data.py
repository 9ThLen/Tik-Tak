"""Deterministic S1 split, labels, cache contract and contiguous scheduler."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np


FPS = 50
FEATURES = 272
BLOCK_FRAMES = 400
WARMUP_FRAMES = 100
SPLIT_SALT = "tiktak-s1-v1"
M0B_MANIFEST_SHA256 = (
    "484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd")
M0E_ARTIFACT_SHA256 = (
    "b866228e9c115c2c48c43acde1eac5e745bc99f8c8a70482d7d66d2d5502d278")
FIXED_WORK_COUNTS = {
    "bpsd": (31, 7), "candombe": (35, 7), "kraisler": (20, 4),
    "rubato": (14, 3), "rwc2": (314, 63),
}


def file_sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _split_key(corpus: str, work_id: str) -> tuple[str, str]:
    raw = f"{SPLIT_SALT}\0{corpus}\0{work_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), work_id


def fixed_split(manifest_path: pathlib.Path,
                m0e_path: pathlib.Path) -> dict:
    if file_sha256(manifest_path) != M0B_MANIFEST_SHA256:
        raise ValueError("S1 source manifest digest changed")
    if file_sha256(m0e_path) != M0E_ARTIFACT_SHA256:
        raise ValueError("S1 source M0e artifact digest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m0e = json.loads(m0e_path.read_text(encoding="utf-8"))
    successful = {(row["corpus"], row["name"]): row
                  for row in m0e["records"]}
    if len(successful) != 980 or m0e.get("technical_exclusions"):
        raise ValueError("S1 fixed successful population changed")
    records = []
    for row in manifest["records"]:
        identity = (row["corpus"], row["name"])
        source = successful.get(identity)
        if source is None:
            continue
        if row["work_id"] != source["work_id"]:
            raise ValueError(f"work identity changed: {identity}")
        records.append({
            "corpus": row["corpus"], "name": row["name"],
            "work_id": row["work_id"],
            "audio_relpath": row["audio_relpath"],
            "audio_sha256": row["audio_sha256"],
            "annotation_relpath": row["annotation_relpath"],
            "annotation_sha256": row["annotation_sha256"],
            "common_start_sec": source["common_start_sec"],
        })
    if len(records) != 980:
        raise ValueError(f"expected 980 S1 records, got {len(records)}")

    by_corpus: dict[str, set[str]] = {}
    for row in records:
        by_corpus.setdefault(row["corpus"], set()).add(row["work_id"])
    split_by_work = {}
    counts = {}
    for corpus, (expected, expected_dev) in FIXED_WORK_COUNTS.items():
        works = sorted(by_corpus.get(corpus, ()),
                       key=lambda work: _split_key(corpus, work))
        dev_count = math.ceil(0.20 * len(works))
        if len(works) != expected or dev_count != expected_dev:
            raise ValueError(f"S1 work profile changed for {corpus}")
        dev = set(works[:dev_count])
        for work in works:
            split_by_work[(corpus, work)] = "dev" if work in dev else "train"
        counts[corpus] = {"all": len(works), "dev": len(dev),
                          "train": len(works) - len(dev)}
    for row in records:
        row["split"] = split_by_work[(row["corpus"], row["work_id"])]
    records.sort(key=lambda row: (row["split"], row["corpus"],
                                  row["work_id"], row["name"]))
    return {
        "schema": "tiktak.s1_split/v1", "salt": SPLIT_SALT,
        "sources": {"manifest": M0B_MANIFEST_SHA256,
                    "m0e": M0E_ARTIFACT_SHA256},
        "research_only": True, "counts": counts, "records": records,
    }


def load_canonical_annotation(path: pathlib.Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name}: empty annotation")
    return {
        "times": np.asarray([float(row["time_seconds"]) for row in rows]),
        "positions": np.asarray([int(row["position"]) for row in rows]),
        "segments": np.asarray([int(row["segment_id"]) for row in rows]),
        "supported": np.asarray([
            str(row["supported"]).strip().lower() in {"1", "true"}
            for row in rows
        ], dtype=bool),
    }


def frame_labels(frames: int, annotation: dict[str, np.ndarray],
                 *, fps: int = FPS) -> tuple[np.ndarray, np.ndarray]:
    """Return hard beat/downbeat/null labels plus canonical-span loss mask."""
    labels = np.full(frames, 2, dtype=np.int64)
    mask = np.zeros(frames, dtype=bool)
    times = annotation["times"]
    positions = annotation["positions"]
    segments = annotation["segments"]
    supported = annotation.get("supported", np.ones(len(times), dtype=bool))
    if not (len(times) == len(positions) == len(segments) == len(supported)):
        raise ValueError("annotation columns differ in length")
    for segment in np.unique(segments):
        selected = np.flatnonzero(segments == segment)
        if not len(selected) or not np.all(supported[selected]):
            continue
        first = max(0, int(math.ceil(times[selected[0]] * fps)))
        last = min(frames, int(math.floor(times[selected[-1]] * fps)) + 1)
        if first < last:
            mask[first:last] = True
    snapped = np.floor(times * fps + 0.5 - 1e-12).astype(np.int64)
    for frame, position, usable in zip(
            snapped, positions, supported, strict=True):
        if usable and 0 <= frame < frames:
            value = 1 if position == 1 else 0
            if labels[frame] == 1 and value == 0:
                continue
            labels[frame] = value
    return labels, mask


@dataclass(frozen=True)
class Recording:
    identity: str
    work_id: str
    features: np.ndarray
    labels: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class ScheduledBlock:
    slot: int
    recording: str
    work_id: str
    index: int
    reset: bool
    end: bool
    features: np.ndarray
    labels: np.ndarray
    mask: np.ndarray


def _block(recording: Recording, index: int,
           block_frames: int, warmup_frames: int) -> tuple[np.ndarray, ...]:
    start = index * block_frames
    stop = min(start + block_frames, len(recording.features))
    features = np.zeros((block_frames, FEATURES), dtype=np.float32)
    labels = np.full(block_frames, 2, dtype=np.int64)
    mask = np.zeros(block_frames, dtype=bool)
    count = stop - start
    features[:count] = recording.features[start:stop]
    labels[:count] = recording.labels[start:stop]
    mask[:count] = recording.mask[start:stop]
    mask[:min(warmup_frames, block_frames)] = False
    return features, labels, mask


def contiguous_batches(
    recordings: list, *, batch_size: int, seed: int,
    block_frames: int = BLOCK_FRAMES, warmup_frames: int = WARMUP_FRAMES,
    loader: Callable[[object], Recording] | None = None,
) -> Iterator[list[ScheduledBlock]]:
    """Keep one recording in each slot until it ends; never cross slot state."""
    if batch_size < 1 or warmup_frames >= block_frames:
        raise ValueError("invalid batch or warm-up configuration")
    order = np.random.default_rng(seed).permutation(len(recordings)).tolist()
    slots: list[tuple[Recording, int] | None] = [None] * batch_size
    cursor = 0
    while cursor < len(order) or any(slot is not None for slot in slots):
        batch = []
        for slot_id in range(batch_size):
            reset = False
            if slots[slot_id] is None and cursor < len(order):
                source = recordings[order[cursor]]
                recording = loader(source) if loader is not None else source
                if not isinstance(recording, Recording):
                    raise TypeError("S1 scheduler loader did not return Recording")
                slots[slot_id] = (recording, 0)
                cursor += 1
                reset = True
            current = slots[slot_id]
            if current is None:
                continue
            recording, index = current
            count = max(1, math.ceil(len(recording.features) / block_frames))
            end = index + 1 == count
            features, labels, mask = _block(
                recording, index, block_frames, warmup_frames)
            batch.append(ScheduledBlock(
                slot_id, recording.identity, recording.work_id, index,
                reset, end, features, labels, mask))
            slots[slot_id] = None if end else (recording, index + 1)
        if batch:
            yield batch


def load_cached_recording(entry: dict, root: pathlib.Path,
                          *, verify: bool = True) -> Recording:
    path = root / entry["cache_relpath"]
    if verify and file_sha256(path) != entry["cache_sha256"]:
        raise ValueError(f"cache digest changed: {entry['name']}")
    with np.load(path, allow_pickle=False) as payload:
        return Recording(
            identity=f"{entry['corpus']}/{entry['name']}",
            work_id=entry["work_id"],
            features=np.asarray(payload["features"], dtype=np.float32),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            mask=np.asarray(payload["mask"], dtype=bool),
        )


def load_cache_manifest(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tiktak.s1_cache/v1":
        raise ValueError("not an S1 cache manifest")
    if payload.get("research_only") is not True:
        raise ValueError("S1 cache lost research_only marker")
    return payload


def verify_cache_records(payload: dict, root: pathlib.Path) -> None:
    """Pay the full cache hash cost once before an S1 run, not every epoch."""
    for entry in payload["records"]:
        path = root / entry["cache_relpath"]
        if not path.is_file() or file_sha256(path) != entry["cache_sha256"]:
            raise ValueError(f"cache digest changed: {entry['name']}")
