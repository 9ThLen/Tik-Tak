"""CMLt and AMLt: beat-tracking accuracy that distinguishes its failure modes.

F-measure treats a grid at half the reference tempo as roughly half wrong, and
a grid on the offbeat as entirely wrong, which throws away exactly the
distinction that decides what to fix. These metrics keep it:

* **CMLt** — Correct Metrical Level. The fraction of beats that land on a
  reference beat *and* whose spacing matches, so a half-tempo grid scores near
  zero however neatly it sits on the music.
* **AMLt** — Allowed Metrical Level. The same, but the reference is also tried
  at double and half tempo and on the offbeat, and the best is taken. A tracker
  that heard the music correctly at a different metrical level scores high here
  and low on CMLt, which is the signature of an octave error rather than of a
  lost grid.

So AMLt near 1 with CMLt near 0 means "right music, wrong level" — a tempo
hypothesis problem. Both low means the grid is genuinely lost.

Following Davies, Degara and Plumbley's formulation: a beat counts when it and
its predecessor both fall inside a tolerance window of consecutive reference
beats, and the interval between them matches the reference interval to the same
tolerance. The window is a proportion of the local reference interval rather
than a fixed number of milliseconds, so the metric does not quietly get
stricter as the music gets faster.
"""

from __future__ import annotations

import numpy as np

__all__ = ["continuity", "cmlt", "amlt", "TOLERANCE"]

# The standard 17.5% of the inter-annotation interval.
TOLERANCE = 0.175


def _trim(beats: np.ndarray, start: float = 5.0) -> np.ndarray:
    """Drops the first few seconds, where a tracker is still acquiring.

    Standard in this family of metrics, and honest rather than generous: what
    is being measured is how well a grid holds, not how fast it starts, and
    every method compared here pays the same.
    """
    return beats[beats >= start]


def continuity(estimate, reference, tolerance: float = TOLERANCE) -> tuple[float, float]:
    """Returns (continuous fraction, total fraction) at the reference's level."""
    estimate = _trim(np.asarray(estimate, dtype=float))
    reference = _trim(np.asarray(reference, dtype=float))
    if len(estimate) < 2 or len(reference) < 2:
        return 0.0, 0.0

    intervals = np.diff(reference)
    correct = np.zeros(len(estimate), dtype=bool)

    for i in range(1, len(estimate)):
        # The reference beat this one claims to be, and the one before it.
        j = int(np.argmin(np.abs(reference - estimate[i])))
        if j == 0:
            continue
        window = intervals[j - 1] * tolerance
        if abs(estimate[i] - reference[j]) > window:
            continue
        if abs(estimate[i - 1] - reference[j - 1]) > window:
            continue
        # The spacing has to match too, or a grid that happens to pass near
        # every other reference beat would score as if it were tracking.
        if abs((estimate[i] - estimate[i - 1]) - intervals[j - 1]) > window:
            continue
        correct[i] = True

    # The first beat has no predecessor and so cannot be judged by a rule that
    # asks about consecutive pairs. Scoring over the beats that can be judged
    # keeps a perfect grid at exactly 1.0 instead of at 1 - 1/n, which would
    # otherwise make short excerpts look worse than long ones for no reason.
    judged = correct[1:]
    if len(judged) == 0:
        return 0.0, 0.0
    total = float(judged.mean())
    longest, run = 0, 0
    for ok in judged:
        run = run + 1 if ok else 0
        longest = max(longest, run)
    return longest / len(judged), total


def _variations(reference: np.ndarray) -> list[np.ndarray]:
    """The reference at the metrical levels a listener could also have chosen.

    Double time, half time on each of its two phases, and the offbeat. These
    are the readings of the same music that are musically defensible, which is
    what separates AMLt from CMLt.
    """
    reference = np.asarray(reference, dtype=float)
    out = [reference]
    if len(reference) >= 2:
        interpolated = np.empty(len(reference) * 2 - 1)
        interpolated[0::2] = reference
        interpolated[1::2] = (reference[:-1] + reference[1:]) / 2.0
        out.append(interpolated)                     # double time
        out.append(reference[0::2])                  # half time, on the beat
        out.append(reference[1::2])                  # half time, offbeat
        out.append(interpolated[1::2])               # the offbeat itself
    return [v for v in out if len(v) >= 2]


def cmlt(estimate, reference, tolerance: float = TOLERANCE) -> float:
    return continuity(estimate, reference, tolerance)[1]


def amlt(estimate, reference, tolerance: float = TOLERANCE) -> float:
    scores = [continuity(estimate, v, tolerance)[1] for v in _variations(reference)]
    return max(scores) if scores else 0.0
