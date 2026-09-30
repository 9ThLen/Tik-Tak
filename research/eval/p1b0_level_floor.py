"""The level floor on the P1-B0 room captures.

Registered in `PREREGISTERED_p1b0_level_floor.md`. Every aligned take goes
through `live_corpus_benchmark._score_one`, as P1-B0 scored them, once per arm
and seed, and the arms are paired seed by seed and excerpt by excerpt.

    cd research
    .venv/Scripts/python -m eval.p1b0_level_floor \\
        --takes ../music/replay-p1b0/aligned --session ../music/replay-p1b0/session.json \\
        --manifest ../music/ground-truth/manifest.csv --music ../music \\
        --binary <dump_analysis> --model <beatnet.ttw> --output <result.json>
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import pathlib
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.live_corpus_benchmark import _score_one, load_corpus  # noqa: E402
from eval.provenance import experiment_provenance, provenance  # noqa: E402

BASE_FLAGS = ("--live-sample-hz", "50")
ARMS = {"off": (), "floor": ("--live-level-floor", "-20")}
SEEDS = {"default": (), "1": ("--live-rng-seed", "1"), "2": ("--live-rng-seed", "2")}
QUIET = ("quiet__near__phone", "quiet__far__phone")
NORMAL = ("normal__near__phone", "normal__far__phone")
LOOPBACK = ("normal__loopback", "quiet__loopback")
MAX_WRONG_OCTAVE_SEC = 4.0
HALF_WIDTH = 0.05
DRAWS = 10_000
BOOTSTRAP_SEED = 20260930


def parse_take(path: pathlib.Path) -> tuple[str, str]:
    """`blues.00025__quiet__far__phone.wav` -> (`blues.00025`, `quiet__far__phone`)."""
    track, cell = path.stem.split("__", 1)
    return track, cell


def excerpt_deltas(records: Iterable[dict], cells: Iterable[str],
                   metric: str = "f_measure") -> dict[str, float]:
    """Floor minus off per excerpt, paired by seed, averaged over seeds, then cells.

    A take counts only when every seed of both arms scored, so a failed run
    shrinks the sample instead of unbalancing the pairing.
    """
    cells = set(cells)
    index = {(r["track"], r["cell"], r["arm"], r["seed"]): r
             for r in records if r.get("ok") and r.get(metric) is not None}
    per_take: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for (track, cell, arm, seed), row in index.items():
        if arm != "floor" or cell not in cells:
            continue
        off = index.get((track, cell, "off", seed))
        if off is not None:
            per_take[(track, cell)].append(float(row[metric]) - float(off[metric]))
    per_excerpt: dict[str, list[float]] = collections.defaultdict(list)
    for (track, _), deltas in per_take.items():
        if len(deltas) == len(SEEDS):
            per_excerpt[track].append(float(np.mean(deltas)))
    return {track: float(np.mean(values)) for track, values in per_excerpt.items()}


def interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = data[rng.integers(0, len(data), (DRAWS, len(data)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def captures_needed(values: list[float]) -> dict | None:
    """A2's formula, `n = (1.96 SD / h)^2`, with the SD's chi-square interval."""
    from scipy.stats import chi2

    if len(values) < 2:
        return None
    k = len(values) - 1
    sd = float(np.std(values, ddof=1))
    low, high = k * sd ** 2 / chi2.ppf(0.975, k), k * sd ** 2 / chi2.ppf(0.025, k)
    need = lambda variance: float(np.ceil(1.96 ** 2 * variance / HALF_WIDTH ** 2))
    return {"half_width": HALF_WIDTH, "estimate": need(sd ** 2), "ci95": [need(low), need(high)]}


def effect(records: list[dict], cells: Iterable[str], metric: str = "f_measure") -> dict:
    deltas = excerpt_deltas(records, cells, metric)
    values = list(deltas.values())
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "ci95": interval(values), "captures_needed": captures_needed(values)}


def per_arm(records: list[dict], cell: str) -> dict:
    """Seed-averaged level, the across-seed SD of the cell mean, and activity."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        rows = [r for r in records if r.get("ok") and r["cell"] == cell and r["arm"] == arm]
        by_seed = collections.defaultdict(list)
        for r in rows:
            by_seed[r["seed"]].append(r)
        means = [float(np.mean([r["f_measure"] for r in s])) for s in by_seed.values()]
        out[arm] = {
            "f": float(np.mean(means)) if means else None,
            "f_seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "usable": float(np.mean([bool(r["usable"]) for r in rows])) if rows else None,
            "episode_free": float(np.mean([(r.get("worst_wrong_octave_sec") or 0.0)
                                           <= MAX_WRONG_OCTAVE_SEC for r in rows]))
            if rows else None,
            "active": (sum(r["active_samples"] for r in rows)
                       / max(1, sum(r["eligible_samples"] for r in rows))) if rows else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takes", type=pathlib.Path, required=True)
    parser.add_argument("--session", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N takes only; a smoke test, never a result")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    items = {item["name"]: item for item in
             load_corpus(args.manifest, args.music, False, frozenset({"gtzan"}))}
    takes = sorted(args.takes.glob("*.wav"))
    if args.limit > 0:
        takes = takes[:args.limit]
    files: dict[str, pathlib.Path] = {"binary": args.binary, "model": args.model,
                                      "session": args.session, "manifest": args.manifest}
    files.update({f"take:{path.name}": path for path in takes})
    stamp = provenance if args.limit > 0 else experiment_provenance
    record = stamp(repository, files, experiment="p1b0_level_floor",
                   limit=args.limit, smoke=args.limit > 0)

    jobs = []
    for path in takes:
        track, cell = parse_take(path)
        for arm, arm_flags in ARMS.items():
            for seed, seed_flags in SEEDS.items():
                jobs.append((path, track, cell, arm, seed, BASE_FLAGS + arm_flags + seed_flags))

    def run(job):
        path, track, cell, arm, seed, flags = job
        scored = _score_one(dict(items[track], audio=path), "model", args.binary, args.model,
                            extra=flags)
        return dict(scored, track=track, cell=cell, arm=arm, seed=seed, take=path.name)

    records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(run, jobs):
            records.append(done)
    failures = [f"{r['take']} {r['arm']} {r['seed']}: {r.get('error')}"
                for r in records if not r.get("ok")]

    cells = sorted({r["cell"] for r in records})
    result = {
        "schema": "tiktak.p1b0_level_floor/v1",
        "provenance": record,
        "arms": {arm: list(BASE_FLAGS + flags) for arm, flags in ARMS.items()},
        "seeds": {seed: list(flags) for seed, flags in SEEDS.items()},
        "failures": failures,
        "primary_quiet": effect(records, QUIET),
        "harm_check_normal": effect(records, NORMAL),
        "cells": {cell: {"delta_f": effect(records, (cell,)),
                         "delta_usable": effect(records, (cell,), "usable"),
                         "arms": per_arm(records, cell)} for cell in cells},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("failures", "primary_quiet",
                                                   "harm_check_normal")},
                     indent=1, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
