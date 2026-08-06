"""Lightweight helpers for comparing a live BPM with an annotated local BPM."""

from __future__ import annotations

import math

import numpy as np

OCTAVE_TOLERANCE = math.log2(1.08)


def local_reference_bpm(beats: np.ndarray, time_sec: float) -> float:
    beats = np.asarray(beats, dtype=np.float64)
    index = int(np.searchsorted(beats, time_sec))
    start = max(0, index - 5)
    stop = min(len(beats), index + 6)
    intervals = np.diff(beats[start:stop])
    intervals = intervals[
        np.isfinite(intervals) & (intervals > 0.1) & (intervals < 3.0)
    ]
    if len(intervals) == 0:
        return 0.0
    return 60.0 / float(np.median(intervals))


def tempo_state(bpm: float, reference_bpm: float) -> str:
    if not (
        bpm > 0.0
        and reference_bpm > 0.0
        and math.isfinite(bpm)
        and math.isfinite(reference_bpm)
    ):
        return "zero"
    ratio = math.log2(bpm / reference_bpm)
    for octave, name in ((-1, "half"), (0, "same"), (1, "double")):
        if abs(ratio - octave) <= OCTAVE_TOLERANCE:
            return name
    return "other"
