#!/usr/bin/env python3
"""Offline beat tracker benchmark on synthetic material.

    cd research && .venv/bin/python -m eval.benchmark

Synthetic, not a public dataset: the annotated corpora either need a request
form or ship audio with no stated licence (see docs/ml-models.md). This gives a
regression signal we control today. It is not a substitute for real material —
synthetic clips have exact onsets and no room, and will flatter any tracker.
"""

import sys

import numpy as np

from tiktak.synth import make_clip
from tiktak.odf import compute_odf, OdfConfig
from tiktak.tempo import estimate_tempo
from tiktak.tracker import track_beats
from eval.harness import evaluate, evaluate_many

def run(clip, **kw):
    o = compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))
    return track_beats(o.full, o.times, o.fps, **kw), o

cases = []
for bpm in (72, 90, 100, 120, 140, 168, 190):
    cases.append((f"steady {bpm}", make_clip(bpm=bpm, duration_sec=25, seed=bpm)))
cases.append(("drift 120->140", make_clip(bpm=120, duration_sec=25, tempo_drift=20, seed=1)))
cases.append(("swing 120",      make_clip(bpm=120, duration_sec=25, swing=0.3, seed=2)))
cases.append(("noisy 120 (6dB)",make_clip(bpm=120, duration_sec=25, noise_db=6, seed=3)))
cases.append(("lead-in 120",    make_clip(bpm=120, duration_sec=25, silence_lead=3.0, seed=4)))
cases.append(("3/4 at 150",     make_clip(bpm=150, beats_per_bar=3, duration_sec=25, seed=5)))
cases.append(("sparse 100",     make_clip(bpm=100, duration_sec=25, sparse=True, seed=6)))

pairs = []
print(f"{'case':18} {'true':>6} {'est':>7} {'conf':>5} | {'F':>5} {'CMLt':>5} {'AMLt':>5} | note")
print("-"*76)
for name, clip in cases:
    res, o = run(clip)
    m = evaluate(clip.beats, res.beats)
    pairs.append((clip.beats, res.beats))
    ratio = res.bpm/clip.bpm
    note = ""
    if abs(ratio-1) > 0.04:
        for r, lbl in ((2,"double"), (0.5,"half"), (3,"triple"), (1/3,"third"), (1.5,"3:2"), (2/3,"2:3")):
            if abs(ratio-r) < 0.06: note = f"{lbl} tempo"; break
        else: note = f"x{ratio:.2f}"
    print(f"{name:18} {clip.bpm:6.0f} {res.bpm:7.1f} {res.tempo_confidence:5.2f} | "
          f"{m['f_measure']:5.2f} {m['cmlt']:5.2f} {m['amlt']:5.2f} | {note}")

agg = evaluate_many(pairs)
print("-"*76)
print(f"{'MEAN':18} {'':6} {'':7} {'':5} | {agg['mean']['f_measure']:5.2f} "
      f"{agg['mean']['cmlt']:5.2f} {agg['mean']['amlt']:5.2f} | n={agg['n_scored']}/{agg['n_clips']}")


if __name__ == "__main__":
    sys.exit(0)
