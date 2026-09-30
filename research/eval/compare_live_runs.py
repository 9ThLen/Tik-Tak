"""Paired comparison of live_corpus_benchmark per-track outputs.

Usage: python -m eval.compare_live_runs BASELINE ARM [ARM ...]
       python -m eval.compare_live_runs --seeds RUN [RUN ...]

The arguments are the per-track files `live_corpus_benchmark --per-track`
writes. Macro over corpora with n >= 30, as the benchmark reports; the paired
bootstrap resamples recordings within each corpus and re-takes the macro
average.

**Read a paired interval against the seed noise, not on its own.** The live
tracker is chaotic, and two runs of identical code with different particle
seeds moved RWC episode-free by 0.026 with a paired interval clear of zero
(`quickfix_diagnostics`, Q1). The bootstrap here treats one draw as the truth,
so `--seeds` is how the floor is measured, and a difference is only a finding
once it clears that.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

MIN_CORPUS = 30
MAX_WRONG_OCTAVE_SEC = 4.0
RESAMPLES = 10_000
SEED = 20260930


def load(path):
    records = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in records:
        if not r.get("ok", False) or not r.get("annotated", False):
            continue
        out[(r["corpus"], r["name"])] = {
            "usable": float(bool(r["usable"])),
            "strict": float(bool(r["usable_strict"])),
            "episode_free": float((r.get("worst_wrong_octave_sec") or 0.0) <= MAX_WRONG_OCTAVE_SEC),
            "f": float(r["f_measure"]),
            "correct": float(r.get("correct_share_of_eligible") or 0.0),
        }
    return out


def corpora_of(keys):
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(k)
    return {c: ks for c, ks in by.items() if len(ks) >= MIN_CORPUS}


def macro(run, keys_by_corpus, metric):
    return float(np.mean([np.mean([run[k][metric] for k in ks]) for ks in keys_by_corpus.values()]))


def paired(base, arm, metric):
    keys = sorted(set(base) & set(arm))
    by = corpora_of(keys)
    delta = macro(arm, by, metric) - macro(base, by, metric)
    rng = np.random.default_rng(SEED)
    diffs = {c: np.array([arm[k][metric] - base[k][metric] for k in ks]) for c, ks in by.items()}
    boots = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        boots[i] = np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs.values()])
    better = sum(arm[k][metric] > base[k][metric] for ks in by.values() for k in ks)
    worse = sum(arm[k][metric] < base[k][metric] for ks in by.values() for k in ks)
    return delta, np.percentile(boots, [2.5, 97.5]), better, worse, macro(base, by, metric), macro(arm, by, metric)


def main(argv):
    if argv[0] == "--seeds":
        runs = [load(p) for p in argv[1:]]
        keys = sorted(set.intersection(*(set(r) for r in runs)))
        by = corpora_of(keys)
        for metric in ("usable", "episode_free", "strict", "f"):
            values = [macro(r, by, metric) for r in runs]
            print(f"{metric:13s} per seed " + " ".join(f"{v:.4f}" for v in values)
                  + f"   mean {np.mean(values):.4f}  SD {np.std(values, ddof=1):.4f}"
                  + f"  range {max(values) - min(values):.4f}")
        for metric in ("usable", "episode_free"):
            flips = sum(len({r[k][metric] for r in runs}) > 1 for ks in by.values() for k in ks)
            total = sum(len(ks) for ks in by.values())
            print(f"{metric:13s} verdict differs across seeds on {flips}/{total} "
                  f"recordings ({100.0 * flips / total:.1f}%)")
        return 0
    base = load(argv[0])
    for path in argv[1:]:
        arm = load(path)
        print(path.split("/")[-1].split("\\")[-1])
        for metric in ("usable", "episode_free", "strict", "f", "correct"):
            d, ci, better, worse, b, a = paired(base, arm, metric)
            print(f"  {metric:13s} {b:.4f} -> {a:.4f}  delta {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                  f"  better/worse {better}/{worse}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
