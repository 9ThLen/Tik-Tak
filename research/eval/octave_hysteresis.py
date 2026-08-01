#!/usr/bin/env python3
"""Would holding the octave through an unconvinced anchor help, and by how much?

The attribution says the filter follows the anchor 94% of the time and that
68-85% of the seconds at the wrong metrical level are the anchor's choice. So
the anchor is the thing to fix, and it reports a quantity that looks like it
knows: on ballroom its octave margin is 0.758 where it is right and 0.037 where
it is wrong.

Gating on that margin has already been tried and rejected — live.hpp records
F 0.752 ungated against 0.738 at 0.15 — but gating means *refusing to anchor*,
which drops the filter back to a fixed prior that is worse than either octave.
Holding the octave already established is a different move and was never
measured: it keeps the anchor's tempo and only declines its change of level.

This simulates it on the dumped series, before any of it is written in C++.
The objective is the anchor's own accuracy about the metrical level, because
that is what the filter then follows.
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


def series(audio: pathlib.Path):
    done = subprocess.run([str(DEFAULT_BINARY), str(audio), "--live",
                           "--live-model", str(MODEL)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return None
    raw = json.loads(done.stdout)
    keys = ("live_times", "live_anchor_bpm", "live_anchor_margin",
            "live_anchor_confidence")
    columns = [np.asarray(raw.get(k, []), dtype=np.float64) for k in keys]
    n = min(len(c) for c in columns)
    if n < 10:
        return None
    reference = load_annotation(audio.with_suffix(".beats")).beats
    if len(reference) < 8:
        return None
    times, bpm, margin, confidence = (c[:n] for c in columns)
    truth = np.array([local_bpm(reference, float(t)) for t in times])
    return times, bpm, margin, confidence, truth


def hold(bpm: np.ndarray, margin: np.ndarray, floor: float) -> np.ndarray:
    """The anchor, with a change of metrical level refused below `floor`.

    The tempo itself is always taken — a singer drifting is followed exactly as
    before. Only the *octave* is held: an answer that is half or double what is
    currently held, and unconvinced about it, is rescaled back to the level in
    hand rather than being ignored or obeyed.
    """
    out = bpm.copy()
    held = 0.0
    for i, value in enumerate(bpm):
        if not (value > 0.0):
            continue
        if held <= 0.0:
            held = value
            out[i] = value
            continue
        ratio = math.log2(value / held)
        octaves = round(ratio)
        if octaves != 0 and margin[i] < floor:
            out[i] = value / (2.0 ** octaves)   # keep the level, take the rate
        else:
            out[i] = value
        held = out[i]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=["ballroom", "gtzan"])
    parser.add_argument("--floors", type=float, nargs="*",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for corpus in args.corpora:
        folder = DATA / corpus / corpus
        files = sorted(p for p in folder.rglob("*.wav")
                       if p.with_suffix(".beats").is_file())
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            got = [g for g in pool.map(series, files) if g]
        print(f"\n{corpus}: {len(got)} recordings")
        print(f"{'hold below margin':>19}{'anchor at the right level':>28}"
              f"{'changes of level kept':>24}")
        for floor in args.floors:
            right = []
            changes = 0
            total_changes = 0
            for times, bpm, margin, _confidence, truth in got:
                fixed = hold(bpm, margin, floor)
                usable = (times >= 5.0) & (bpm > 0.0) & (truth > 0.0)
                if not usable.any():
                    continue
                ratio = np.abs(np.log2(fixed[usable] / truth[usable]))
                right.append(np.mean(ratio <= TOLERANCE))
                raw_steps = np.abs(np.round(np.log2(
                    bpm[1:][bpm[1:] > 0] / np.maximum(bpm[:-1][bpm[1:] > 0], 1e-9))))
                kept_steps = np.abs(np.round(np.log2(
                    fixed[1:][fixed[1:] > 0] / np.maximum(fixed[:-1][fixed[1:] > 0], 1e-9))))
                total_changes += int(np.sum(raw_steps >= 1))
                changes += int(np.sum(kept_steps >= 1))
            share = changes / total_changes if total_changes else float("nan")
            mark = "  <- ships" if floor == 0.0 else ""
            print(f"{floor:>19.1f}{np.mean(right):>28.1%}{share:>24.1%}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
