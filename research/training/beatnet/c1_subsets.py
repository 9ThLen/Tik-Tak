#!/usr/bin/env python3
"""Nested corpus-stratified training subsets for the preregistered C1 curve.

Two orderings do two different jobs here, and confusing them silently breaks the
experiment.

A hash order over works decides **membership**: which works are in the 25% and
50% fractions. Nothing else. The rows handed to the trainer afterwards are
rebuilt in **cache-manifest order**, because `contiguous_batches` draws
`np.random.default_rng(seed).permutation(len(recordings))` -- a permutation of
*positional* indices -- and then reads `recordings[order[cursor]]`. The batch
schedule is a pure function of the order of the list it is given and the seed.

Hand it hash-ordered rows and the same seed puts different recordings in
different slots at different steps. That would change the SGD trajectory at
every fraction *including 100%*, and a 100% arm that does not reproduce S1 is
not an anchor -- which is the whole reason C1 is six runs rather than nine.

`schedule_signature` is the check, and it compares the emitted schedule rather
than the row order, because the schedule is what actually has to match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from .cache import _atomic_json, _outside_repository
from .data import (BLOCK_FRAMES, FEATURES, Recording, WARMUP_FRAMES,
                   contiguous_batches, file_sha256, load_cache_manifest)

SCHEMA = "tiktak.c1_subsets/v1"
SALT = "tiktak-c1-v1"
FRACTIONS = (0.25, 0.50, 1.00)
TRAIN_RECORDS = 765
TRAIN_WORKS = 330
# Verified against the S1 cache; a different corpus mix is a different corpus.
CORPUS_WORKS = {"bpsd": 24, "candombe": 28, "kraisler": 16,
                "rubato": 11, "rwc2": 251}

# Registered prefix lengths, tabulated rather than rounded. `round` is
# ties-to-even in Python and round-half-up elsewhere, and 0.5 * 11 and
# 0.5 * 251 both land on a tie -- so the table is the specification and the
# arithmetic is only checked against it.
PREFIX = {
    0.25: {"bpsd": 6, "candombe": 7, "kraisler": 4, "rubato": 3, "rwc2": 63},
    0.50: {"bpsd": 12, "candombe": 14, "kraisler": 8, "rubato": 6, "rwc2": 126},
    1.00: dict(CORPUS_WORKS),
}
# Predicted by the registered rule; a mismatch is a defect, not a variant. Two
# implementations written independently from the specification landed on the
# same two figures, which is the property pinning the rule to the byte was for.
FRAME_FRACTIONS = {0.25: 0.2931, 0.50: 0.5568, 1.00: 1.0}
REAL_TOTAL_FRAMES = 10566912


def _order_key(corpus: str, work_id: str) -> tuple[str, bytes]:
    """Membership order, fixed to the byte.

    UTF-8, the manifest's own strings with no normalisation, a single NUL
    between fields and after the salt, digest ascending, ties by `work_id` as a
    byte string. Deliberately a different salt from `data._split_key`: sharing
    it would correlate which works land in a fraction with how train and dev
    were divided, so a fraction could inherit the split's structure instead of
    sampling independently of it.
    """
    raw = (SALT.encode("utf-8") + b"\x00" + corpus.encode("utf-8")
           + b"\x00" + work_id.encode("utf-8"))
    return hashlib.sha256(raw).hexdigest(), work_id.encode("utf-8")


def training_rows(cache: dict) -> list[dict]:
    """The train split in cache-manifest order, exactly as `run.py` builds it."""
    rows = [row for row in cache["records"] if row["split"] == "train"]
    if len(rows) != TRAIN_RECORDS:
        raise ValueError(f"expected {TRAIN_RECORDS} training records")
    return rows


def work_order(rows: list[dict]) -> dict[str, list[str]]:
    """Per corpus, the works in membership order."""
    seen: dict[str, dict[str, None]] = {}
    for row in rows:
        seen.setdefault(row["corpus"], {})[row["work_id"]] = None
    out = {}
    for corpus, works in seen.items():
        out[corpus] = sorted(works, key=lambda work: _order_key(corpus, work))
    if {corpus: len(works) for corpus, works in out.items()} != CORPUS_WORKS:
        raise ValueError("training corpus work counts changed")
    return out


def members(order: dict[str, list[str]], fraction: float) -> set[str]:
    """A registered prefix per corpus, so the fractions nest by construction."""
    lengths = PREFIX[fraction]
    chosen: set[str] = set()
    for corpus, works in order.items():
        take = lengths[corpus]
        if take > len(works):
            raise ValueError(f"{corpus}: prefix {take} exceeds {len(works)}")
        chosen.update(works[:take])
    return chosen


def subset_rows(rows: list[dict], chosen: set[str]) -> list[dict]:
    """Membership decided elsewhere; order is the manifest's. See the docstring."""
    return [row for row in rows if row["work_id"] in chosen]


def _light(row: dict) -> Recording:
    """A recording with the right length and no contents.

    The schedule depends on how many blocks a recording makes and on nothing
    inside them, so read-only broadcast views cost no memory and produce the
    same `(slot, identity, work, index, reset, end)` sequence as the real
    arrays would.
    """
    frames = int(row["frames"])
    return Recording(
        identity=f"{row['corpus']}/{row['name']}", work_id=row["work_id"],
        features=np.broadcast_to(
            np.zeros((1, FEATURES), dtype=np.float32), (frames, FEATURES)),
        labels=np.broadcast_to(np.full(1, 2, dtype=np.int64), (frames,)),
        mask=np.broadcast_to(np.zeros(1, dtype=bool), (frames,)))


def schedule_signature(rows: list[dict], *, seed: int, batch_size: int = 8
                       ) -> list[tuple]:
    """What the trainer would actually consume, values excluded."""
    signature = []
    for batch in contiguous_batches(
            rows, batch_size=batch_size, seed=seed,
            block_frames=BLOCK_FRAMES, warmup_frames=WARMUP_FRAMES,
            loader=_light):
        for block in batch:
            signature.append((block.slot, block.recording, block.work_id,
                              block.index, block.reset, block.end))
    return signature


def identity_digest(rows: list[dict]) -> str:
    payload = [[row["corpus"], row["name"], row["work_id"]] for row in rows]
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def build(cache: dict) -> dict:
    rows = training_rows(cache)
    order = work_order(rows)
    total_frames = sum(int(row["frames"]) for row in rows)
    fractions = {}
    previous: set[str] | None = None
    for fraction in FRACTIONS:
        chosen = members(order, fraction)
        if previous is not None and not previous <= chosen:
            raise ValueError(f"fraction {fraction} is not nested")
        previous = chosen
        selected = subset_rows(rows, chosen)
        by_corpus = {}
        for corpus in sorted(order):
            part = [row for row in selected if row["corpus"] == corpus]
            by_corpus[corpus] = {
                "works": len({row["work_id"] for row in part}),
                "records": len(part),
                "frames": sum(int(row["frames"]) for row in part),
            }
        frames = sum(block["frames"] for block in by_corpus.values())
        fractions[f"{fraction:.2f}"] = {
            "nominal_work_fraction": fraction,
            "works": len(chosen), "records": len(selected),
            "frames": frames,
            # An output, never an input: works differ in length, so the frame
            # axis is whatever the corpus makes it and any figure has to say
            # which axis it is drawn on.
            "frame_fraction": frames / total_frames,
            "by_corpus": by_corpus,
            "identity_sha256": identity_digest(selected),
        }
    full = fractions[f"{1.00:.2f}"]
    if (full["records"] != TRAIN_RECORDS or full["works"] != TRAIN_WORKS
            or full["identity_sha256"] != identity_digest(rows)):
        raise ValueError("the 100% fraction is not the original training set")
    # The registration derives 29.31% and 55.68% from the byte-exact rule, so a
    # generator landing elsewhere has a defect rather than a defensible reading.
    #
    # A first version skipped the comparison when `total_frames` differed from
    # the registered corpus, so that a fixture could still build. That disarmed
    # the guard in exactly the case it exists for: perturb a recording's length
    # and the total moves, the condition goes false, and nothing is checked.
    # It reports instead, and `require_registered_corpus` is what refuses.
    deviations = {}
    for fraction, expected in FRAME_FRACTIONS.items():
        got = fractions[f"{fraction:.2f}"]["frame_fraction"]
        if abs(got - expected) > 5e-5:
            deviations[f"{fraction:.2f}"] = got
    return {"schema": SCHEMA, "research_only": True, "salt": SALT,
            "train_records": TRAIN_RECORDS, "train_works": TRAIN_WORKS,
            "total_frames": total_frames, "fractions": fractions,
            "registered_corpus": (total_frames == REAL_TOTAL_FRAMES
                                  and not deviations),
            "frame_fraction_deviations": deviations}


def require_registered_corpus(payload: dict) -> None:
    """The binding path refuses anything but the corpus C1 registered."""
    if payload.get("total_frames") != REAL_TOTAL_FRAMES:
        raise ValueError(
            f"training frames {payload.get('total_frames')} is not the "
            f"registered {REAL_TOTAL_FRAMES}")
    if payload.get("frame_fraction_deviations"):
        raise ValueError(
            "frame shares differ from the registered rule: "
            f"{payload['frame_fraction_deviations']}")
    if not payload.get("registered_corpus"):
        raise ValueError("subset payload is not from the registered corpus")


def assert_anchor_schedule(cache: dict, *, seeds: tuple[int, ...] = (17, 18, 19)
                           ) -> dict:
    """The registered preflight: 100% must reproduce S1's schedule, not just its rows.

    `run.py` advances the scheduler seed per epoch (`seed + epoch`), so a single
    seed would only prove the first epoch matches.
    """
    rows = training_rows(cache)
    order = work_order(rows)
    selected = subset_rows(rows, members(order, 1.00))
    if [id(row) for row in selected] != [id(row) for row in rows]:
        raise ValueError("100% selection is not the original rows in order")
    checked = {}
    for seed in seeds:
        expected = schedule_signature(rows, seed=seed)
        actual = schedule_signature(selected, seed=seed)
        if actual != expected:
            raise ValueError(f"100% schedule differs from S1 at seed {seed}")
        checked[str(seed)] = len(expected)
    return {"passed": True, "seeds": list(seeds), "blocks_compared": checked}


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        _outside_repository(args.output, repository)
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        cache = load_cache_manifest(args.cache)
        if cache.get("diagnostic_only") or cache.get("selected") != 980:
            raise ValueError("C1 requires the complete 980-record cache")
        payload = build(cache)
        require_registered_corpus(payload)
        payload["preflight"] = assert_anchor_schedule(cache)
        payload["cache_sha256"] = file_sha256(args.cache)
        _atomic_json(args.output, payload)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for name, block in payload["fractions"].items():
        print(json.dumps({"fraction": name, "works": block["works"],
                          "records": block["records"],
                          "frame_fraction": round(block["frame_fraction"], 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
