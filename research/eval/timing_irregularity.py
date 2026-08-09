#!/usr/bin/env python3
"""What kind of timing the causal decoder loses, and on which recordings.

Two questions, both answered from artifacts `oracle_activation.py` has already
written, so this costs no audio processing at all.

**Which irregularity.** The oracle-fed filter's recall correlates with the
coefficient of variation of the annotated beat intervals. That statistic is two
things at once: a slow tempo *drift* and a beat-to-beat residual around it. The
residual can contain expressive but structured timing, so this script calls it
``jitter`` only as a compact statistical label, not as a claim that it is random
or causally unpredictable.

**Locally steady baseline.** Replacing every annotated interval by its centred
local median produces one illustrative smooth pulse. It is anchored only at the
first beat, can accumulate phase error, and uses future intervals, so it is
neither causal nor a bound on what a causal tracker can recover. The comparison
only says how the filter fares against this particular smoothing strategy.

**Which recordings the anchor saves, and which it costs.** `oracle_activation`
scores every recording with the anchor on and off, so the per-track difference
splits the corpus into three groups. If some quantity available to the tracker
*at run time* separates them, that quantity is a candidate for making the
anchor's strength conditional instead of fixed — which is the one direction the
on/off measurement supports. Annotated tempo statistics are not available at run
time and are reported here only to characterise the groups, never as a proposed
signal.

Run after `oracle_activation.py`, on its output:

    .venv/Scripts/python -m eval.timing_irregularity \\
        --oracle results/oracle_activation_gtzan.json results/oracle_activation.json
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
sys.path.insert(0, str(RESEARCH))

from eval.live_corpus_benchmark import load_corpus  # noqa: E402
from eval.live_corpus_benchmark import load_reference_beats  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402
from eval.statistics import spearman  # noqa: E402

WINDOW_SEC = 0.070
WARMUP_SEC = 5.0
# Beats in the running median that separates drift from jitter. Nine is about
# two bars of 4/4: long enough that a single expressive beat cannot move it,
# short enough to follow a ritardando.
TREND_BEATS = 9


def split(beats: np.ndarray) -> tuple[float, float] | None:
    """Drift and jitter, each as a fraction of the mean interval."""
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.1) & (intervals < 3.0)]
    if len(intervals) < 2 * TREND_BEATS:
        return None
    half = TREND_BEATS // 2
    padded = np.pad(intervals, (half, half), mode="edge")
    trend = np.array([np.median(padded[i:i + TREND_BEATS])
                      for i in range(len(intervals))])
    mean = float(np.mean(intervals))
    return float(np.std(trend) / mean), float(np.std(intervals - trend) / mean)


def steady_pulse(beats: np.ndarray) -> np.ndarray | None:
    """One non-causal baseline, walked from beat zero at median intervals."""
    intervals = np.diff(beats)
    if len(intervals) < 2 * TREND_BEATS:
        return None
    half = TREND_BEATS // 2
    padded = np.pad(intervals, (half, half), mode="edge")
    trend = np.array([np.median(padded[i:i + TREND_BEATS])
                      for i in range(len(intervals))])
    return beats[0] + np.concatenate([[0.0], np.cumsum(trend)])


def recall(reference: np.ndarray, found: np.ndarray) -> float:
    used = np.zeros(len(found), dtype=bool)
    hits = 0
    for beat in reference:
        near = np.flatnonzero((np.abs(found - beat) <= WINDOW_SEC) & ~used)
        if len(near):
            used[near[np.argmin(np.abs(found[near] - beat))]] = True
            hits += 1
    return hits / len(reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", nargs="+", type=pathlib.Path, required=True,
                        help="oracle_activation.py outputs to read")
    parser.add_argument("--output", type=pathlib.Path)
    # How much better or worse the anchor has to make a recording before it is
    # called saved or cost rather than unaffected. Two beats' worth of a
    # thirty-second excerpt, roughly.
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    items = {(item["corpus"], item["name"]): item
             for item in load_corpus(
                 REPOSITORY / "music" / "ground-truth" / "manifest.csv",
                 REPOSITORY / "music", False)}

    source_files = {f"oracle_{index}": path
                    for index, path in enumerate(args.oracle)}
    report: dict = {
        "provenance": provenance(REPOSITORY, source_files,
                                 trend_beats=TREND_BEATS, margin=args.margin),
        "window_sec": WINDOW_SEC, "by_corpus": {},
    }

    for path in args.oracle:
        loaded = json.load(io.open(path, encoding="utf-8"))
        for corpus, block in loaded["by_corpus"].items():
            rows = []
            for track in block.get("tracks", ()):
                item = items.get((corpus, track["name"]))
                if item is None:
                    continue
                beats = load_reference_beats(item["annotation"])
                beats = beats[np.isfinite(beats)]
                parts = split(beats)
                pulse = steady_pulse(beats)
                scored = beats[beats >= WARMUP_SEC]
                if parts is None or pulse is None or len(scored) < 8:
                    continue
                rows.append({
                    "name": track["name"],
                    "drift": parts[0],
                    "jitter": parts[1],
                    "steady": recall(scored, pulse),
                    "oracle": track["bump"],
                    "oracle_no_anchor": track.get("bump_no_anchor"),
                    "real": track["real"],
                })
            if len(rows) < 20:
                print(f"{corpus}: too few usable rows")
                continue

            drift = np.array([r["drift"] for r in rows])
            jitter = np.array([r["jitter"] for r in rows])
            steady = np.array([r["steady"] for r in rows])
            oracle = np.array([r["oracle"] for r in rows])
            drift_rho = spearman(drift, oracle, min_samples=20)
            jitter_rho = spearman(jitter, oracle, min_samples=20)
            drift_rho_text = f"{drift_rho:+.2f}" if drift_rho is not None else "n/a"
            jitter_rho_text = (
                f"{jitter_rho:+.2f}" if jitter_rho is not None else "n/a"
            )

            summary = {
                "n": len(rows),
                "median_drift": float(np.median(drift)),
                "median_jitter": float(np.median(jitter)),
                "rho_drift_vs_oracle": drift_rho,
                "rho_jitter_vs_oracle": jitter_rho,
                "steady_pulse_recall": float(steady.mean()),
                "oracle_recall": float(oracle.mean()),
            }

            print(f"\n{corpus}: {len(rows)} recordings")
            print(f"   drift  median {np.median(drift):.4f}   "
                  f"rho vs oracle recall {drift_rho_text}")
            print(f"   jitter median {np.median(jitter):.4f}   "
                  f"rho vs oracle recall {jitter_rho_text}")
            print(f"   a locally steady pulse recalls {steady.mean():.1%}, "
                  f"our filter {oracle.mean():.1%}")

            # The anchor's per-recording effect, and whether anything separates
            # the recordings it saves from the ones it costs.
            paired = [r for r in rows if r["oracle_no_anchor"] is not None]
            if len(paired) >= 20:
                delta = np.array([r["oracle"] - r["oracle_no_anchor"]
                                  for r in paired])
                groups = {
                    "anchor saves": delta > args.margin,
                    "no effect": np.abs(delta) <= args.margin,
                    "anchor costs": delta < -args.margin,
                }
                summary["anchor"] = {
                    "margin": args.margin,
                    "groups": {name: int(mask.sum())
                               for name, mask in groups.items()},
                }
                print(f"   the anchor, per recording (margin {args.margin:.2f}):")
                for name, mask in groups.items():
                    if not mask.any():
                        continue
                    sub = [paired[i] for i in np.flatnonzero(mask)]
                    stats = {
                        key: float(np.median([r[key] for r in sub]))
                        for key in ("drift", "jitter", "real", "steady")
                    }
                    summary["anchor"][name] = {"n": len(sub), **stats}
                    print(f"      {name:<13} {len(sub):>4}   "
                          f"drift {stats['drift']:.4f}  "
                          f"jitter {stats['jitter']:.4f}  "
                          f"real-activation recall {stats['real']:.1%}")

            report["by_corpus"][corpus] = summary

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
