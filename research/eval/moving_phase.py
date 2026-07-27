"""A bar-line decoder whose phase is allowed to move, for evaluation only.

The shipping resolver commits to one (metre, phase) pair for a whole track.
Measured against a Beat This! reference on forty-three releases, that is the
binding constraint rather than the cue quality — but only once the salience is
good enough for the constraint to be what is in the way:

    salience               one global phase   this decoder   batch it came from
    the built-in cues          F 0.415           F 0.415       8 groups
                               F 0.658           F 0.658      35 albums
    a learned activation       F 0.772           F 0.950       8 groups
                               F 0.775           F 0.947      35 albums

The second batch was collected after the first was measured, and the shape
replicated: the same gain on an activation, and on the cues a gain of exactly
nothing, twice. That is why this exists, and it carries an ordering with it —
a decoder is worth nothing on the cues the core computes today, so activations
and decoder are one piece of work rather than two.

Nothing here runs in the product. It lives in research so the port has a number
attached before anyone writes it in C++, where the resolver's exact-arithmetic
machinery makes this a much more delicate change.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decode", "bar_positions", "salience_from_cues", "SINGLE_PHASE"]

# A switch cost this large can never be paid back by the emission terms, so the
# decoder collapses to a single global phase and reproduces the resolver.
SINGLE_PHASE = float("inf")


def _emissions(salience: np.ndarray, meter: int) -> np.ndarray:
    """Per-beat score of each phase offset, one column per offset.

    A beat that a phase calls a downbeat earns its salience; every other beat
    in the bar pays an equal share of it back. Summed over a track with the
    phase held fixed, that is the same mean(in) - mean(out) contrast the
    resolver maximises, only written per beat so it can be decoded over time
    instead of scored once.
    """
    n = len(salience)
    offsets = np.arange(meter)
    position = (np.arange(n)[:, None] - offsets[None, :]) % meter
    return np.where(position == 0, salience[:, None], -salience[:, None] / (meter - 1))


def bar_positions(salience, meter: int, switch_cost: float = SINGLE_PHASE):
    """Best phase offset per beat, by Viterbi over the offsets.

    Staying on a phase is free and changing costs `switch_cost`, in the same
    units as the salience. Returns one offset per beat.
    """
    salience = np.asarray(salience, dtype=float)
    if meter < 2:
        raise ValueError(f"a bar needs at least two beats, got {meter}")
    n = len(salience)
    if n == 0:
        return np.zeros(0, dtype=int)

    emit = _emissions(salience, meter)
    score = emit[0].copy()
    back = np.zeros((n, meter), dtype=np.int64)

    for i in range(1, n):
        # Every phase can be reached either by staying, or by switching from
        # whichever other phase is currently best. Only the best and second
        # best are ever needed: the best is the winner for every phase except
        # itself, which falls back to the second.
        #
        # No switching until a full bar has gone by. Choosing where to start is
        # free, and without this a phase that holds for a single beat pays the
        # cost once instead of twice — on and off — which is cheap enough that
        # the first beat of a track gets labelled its own bar and emitted as a
        # spurious downbeat. Nothing before the first bar line is a phase
        # *change* yet; it is still the opening choice.
        if i < meter:
            score = score + emit[i]
            back[i] = np.arange(meter)
            continue

        order = np.argsort(score)[::-1]
        best, second = int(order[0]), int(order[1])
        came_from = np.where(np.arange(meter) == best, second, best)
        switched = score[came_from] - switch_cost

        stay = score >= switched
        score = np.where(stay, score, switched) + emit[i]
        back[i] = np.where(stay, np.arange(meter), came_from)

    path = np.empty(n, dtype=np.int64)
    state = int(np.argmax(score))
    for i in range(n - 1, -1, -1):
        path[i] = state
        state = int(back[i, state])
    return path


def decode(salience, meter: int, switch_cost: float = SINGLE_PHASE):
    """Indices of the beats that begin a bar."""
    salience = np.asarray(salience, dtype=float)
    path = bar_positions(salience, meter, switch_cost)
    if len(path) == 0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero((np.arange(len(salience)) - path) % meter == 0)


def salience_from_cues(estimate, harmony_scale: float = 12.0,
                       harmony_floor: float = 0.05,
                       low_weight: float = 1.0,
                       harmony_weight: float = 1.0) -> np.ndarray:
    """cueSalience() from the core, recomputed so the weights can be swept.

    Kept in step with core/src/analysis/downbeat.cpp by hand. The constants are
    the core's own defaults; passing different weights is the point of having
    it here, since sweeping them in C++ would mean a rebuild per candidate.
    """
    low = np.asarray(estimate["cue_low"], dtype=float)
    harmony = np.maximum(np.asarray(estimate["cue_harmony"], dtype=float) - harmony_floor, 0.0)
    spread = low.std()
    low = (low - low.mean()) / spread if spread > 0 else np.zeros_like(low)
    return low_weight * low + harmony_weight * harmony_scale * harmony
