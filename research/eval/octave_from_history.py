#!/usr/bin/env python3
"""Take the octave from a long window and the rate from a short one.

Everything measured so far says the live path's remaining failure is the
anchor's choice of metrical level, that the filter follows the anchor faithfully
(94%), and that no rule reading the anchor's own confidence improves it —
gating was worse, and holding the level through an unconvinced anchor is worse
monotonically.

What has not been tried is splitting the anchor's two jobs. activation_tempo.hpp
already says why they are different: "an octave is a global fact about a
recording" while the rate is local, and its window was set to six seconds
because *end to end* a fresh rate beats an accurate one. That measurement moved
one window and so moved both jobs at once. A long window is a better look at the
octave and a worse look at the rate; if the two are taken from different windows
neither has to be traded for the other.

The ceiling is measurable without writing any of it: dump the anchor at both
windows and combine them — the octave from the long series, the rate from the
short one. This is the same shape of finding as the resolver's metre and phase
wanting different cue weights, which is why it is worth checking rather than
assuming.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
sys.path.insert(0, str(RESEARCH))

from eval.analysis import DEFAULT_BINARY  # noqa: E402
from eval.annotations import load_annotation  # noqa: E402

DATA = RESEARCH / "data"
MODEL = REPOSITORY / "models" / "beatnet_model_1.ttw"
TOLERANCE = math.log2(1.08)


def local_bpm(beats: np.ndarray, at: float) -> float:
    i = int(np.searchsorted(beats, at))
    window = np.diff(beats[max(0, i - 5): min(len(beats), i + 6)])
    window = window[(window > 0.1) & (window < 3.0)]
    return 60.0 / float(np.median(window)) if len(window) else 0.0


def anchor_series(audio: pathlib.Path, window: float):
    args = [str(DEFAULT_BINARY), str(audio), "--live",
            "--live-model", str(MODEL)]
    if window > 0:
        args += ["--live-anchor-window", repr(window),
                 "--live-anchor-min-window", repr(min(window, 6.0))]
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    raw = json.loads(done.stdout)
    times = np.asarray(raw.get("live_times", []), dtype=np.float64)
    bpm = np.asarray(raw.get("live_anchor_bpm", []), dtype=np.float64)
    n = min(len(times), len(bpm))
    return (times[:n], bpm[:n]) if n >= 10 else None


def one(task):
    audio, long_window = task
    short = anchor_series(audio, 0.0)
    long = anchor_series(audio, long_window)
    if short is None or long is None:
        return None
    reference = load_annotation(audio.with_suffix(".beats")).beats
    if len(reference) < 8:
        return None
    n = min(len(short[0]), len(long[0]))
    times = short[0][:n]
    truth = np.array([local_bpm(reference, float(t)) for t in times])
    return times, short[1][:n], long[1][:n], truth


def right(bpm: np.ndarray, truth: np.ndarray, usable: np.ndarray) -> float:
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs(np.log2(bpm[usable] / truth[usable])) <= TOLERANCE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=["ballroom", "gtzan"])
    parser.add_argument("--long", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for corpus in args.corpora:
        folder = DATA / corpus / corpus
        files = sorted(p for p in folder.rglob("*.wav")
                       if p.with_suffix(".beats").is_file())
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            got = [g for g in pool.map(one, [(f, args.long) for f in files]) if g]
        print(f"\n{corpus}: {len(got)} recordings, long window {args.long:g} s")

        rows = {"short only (ships)": [], f"long only": [],
                "octave from long, rate from short": []}
        for times, short, long, truth in got:
            usable = (times >= 5.0) & (short > 0.0) & (long > 0.0) & (truth > 0.0)
            # The hybrid: keep the short window's rate, then move it to the
            # metrical level the long window is in. Nothing about how fast the
            # music is comes from the long window; only which pulse is counted.
            octaves = np.zeros_like(short)
            good = (short > 0.0) & (long > 0.0)
            octaves[good] = np.round(np.log2(long[good] / short[good]))
            hybrid = short * (2.0 ** octaves)
            rows["short only (ships)"].append(right(short, truth, usable))
            rows["long only"].append(right(long, truth, usable))
            rows["octave from long, rate from short"].append(
                right(hybrid, truth, usable))

        for label, values in rows.items():
            v = np.array(values)
            v = v[np.isfinite(v)]
            print(f"   {label:36} {v.mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
