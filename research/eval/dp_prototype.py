"""Does letting the bar phase move actually recover bar lines?

The current resolver commits to one (metre, phase) for a whole track. Measured
on real songs that is the binding constraint, not the cue quality. This tests
the replacement before anyone writes it in C++: a Viterbi over bar position
that may change phase at a cost, decoded from the same per-beat salience the
resolver already sees.

lambda = infinity reproduces the current resolver exactly, which is the
baseline every row is measured against.
"""
import pickle, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
from groups import OF

NEG_INF = -1e18


def viterbi(sal, M, switch_cost):
    """Best sequence of bar-phase offsets. Returns downbeat beat indices."""
    n = len(sal)
    # Emission: a downbeat earns its salience, every other beat in the bar
    # pays a share of it, so the score is the same contrast the resolver
    # maximises — just written per beat so it can be decoded over time.
    emit = np.empty((n, M))
    for s in range(M):
        pos = (np.arange(n) - s) % M
        emit[:, s] = np.where(pos == 0, sal, -sal / (M - 1))

    score = emit[0].copy()
    back = np.zeros((n, M), dtype=np.int8)
    for i in range(1, n):
        stay = score
        best_other = np.empty(M)
        prev = np.empty(M, dtype=np.int8)
        order = np.argsort(score)[::-1]
        top, second = order[0], order[1] if M > 1 else order[0]
        for s in range(M):
            alt = top if top != s else second
            best_other[s] = score[alt] - switch_cost
            prev[s] = alt
        keep = stay >= best_other
        score = np.where(keep, stay, best_other) + emit[i]
        back[i] = np.where(keep, np.arange(M), prev)

    s = int(np.argmax(score))
    path = np.empty(len(sal), dtype=np.int8)
    for i in range(len(sal) - 1, -1, -1):
        path[i] = s
        s = int(back[i, s])
    return np.flatnonzero((np.arange(len(sal)) - path) % M == 0), float(np.max(score))


def salience_from_cues(e, harmony_scale=12.0, harmony_floor=0.05,
                       low_w=1.0, harmony_w=1.0):
    """The core's cueSalience(), recomputed here so weights can be swept."""
    low = np.asarray(e["cue_low"], dtype=float)
    har = np.maximum(np.asarray(e["cue_harmony"], dtype=float) - harmony_floor, 0.0)
    sd = low.std()
    low = (low - low.mean()) / sd if sd > 0 else np.zeros_like(low)
    return low_w * low + harmony_w * harmony_scale * har


def recall(pred_times, ref_times, tol=0.07):
    if len(pred_times) == 0 or len(ref_times) == 0:
        return 0.0
    return float(np.mean([np.min(np.abs(pred_times - t)) <= tol for t in ref_times]))
