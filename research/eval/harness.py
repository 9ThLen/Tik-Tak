"""mir_eval-based evaluation harness for beat and downbeat tracking.

Conventions (deliberately fixed here so every experiment is comparable):

* **Trimming.** MIREX convention: beats in the first 5 seconds are ignored,
  because most trackers need a warm-up period and the reference annotations
  of real datasets are unreliable there. We apply
  ``mir_eval.beat.trim_beats`` (min_beat_time=5.0) to *both* reference and
  estimate by default. For short synthetic clips (or the file/offline mode,
  which has the whole track up front and no warm-up excuse) pass
  ``trim=False`` — then everything from t=0 counts.
* **F-measure tolerance.** ±70 ms, the MIREX standard, for both beats and
  downbeats. Same tolerance for downbeats so the two F-measures are directly
  comparable.
* **Degenerate inputs.** Empty estimates, a single estimated beat, etc. are
  scored as 0.0 on every metric (mir_eval's own behaviour, warnings
  silenced) rather than raising: a tracker that outputs nothing has failed,
  and the harness must keep aggregating over the rest of the dataset. An
  empty *reference* (e.g. everything trimmed away on a <5 s clip with
  trim=True) yields NaN — the clip is unscorable, not failed — and NaNs are
  excluded from aggregation.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence, Tuple

import numpy as np
import mir_eval.beat

__all__ = [
    "METRIC_NAMES",
    "evaluate",
    "evaluate_downbeats",
    "evaluate_many",
    "format_report",
]

METRIC_NAMES = (
    "f_measure",
    "cmlc",
    "cmlt",
    "amlc",
    "amlt",
    "p_score",
    "information_gain",
)

F_MEASURE_TOLERANCE = 0.07  # seconds, MIREX standard
TRIM_MIN_BEAT_TIME = 5.0    # seconds, MIREX convention


def _sanitize(beats: np.ndarray | Sequence[float]) -> np.ndarray:
    """1-D, float64, sorted, unique, non-negative — what mir_eval expects."""
    b = np.asarray(beats, dtype=np.float64).ravel()
    b = b[np.isfinite(b)]
    b = b[b >= 0.0]
    return np.unique(b)


def evaluate(
    reference_beats: np.ndarray,
    estimated_beats: np.ndarray,
    trim: bool = True,
) -> dict:
    """Score one clip. Returns a dict with the metrics in METRIC_NAMES.

    ``trim=True`` (default) drops beats before 5 s from both sequences
    (MIREX convention, see module docstring). All metrics are in [0, 1];
    ``information_gain`` is normalised by mir_eval 0.8 to [0, 1] as well.
    """
    ref = _sanitize(reference_beats)
    est = _sanitize(estimated_beats)
    if trim:
        ref = mir_eval.beat.trim_beats(ref, min_beat_time=TRIM_MIN_BEAT_TIME)
        est = mir_eval.beat.trim_beats(est, min_beat_time=TRIM_MIN_BEAT_TIME)

    if len(ref) < 2:
        # Unscorable clip: no usable reference after trimming.
        return {name: float("nan") for name in METRIC_NAMES}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = mir_eval.beat.f_measure(ref, est, f_measure_threshold=F_MEASURE_TOLERANCE)
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(ref, est)
        p = mir_eval.beat.p_score(ref, est)
        ig = mir_eval.beat.information_gain(ref, est)

    return {
        "f_measure": float(f),
        "cmlc": float(cmlc),
        "cmlt": float(cmlt),
        "amlc": float(amlc),
        "amlt": float(amlt),
        "p_score": float(p),
        "information_gain": float(ig),
    }


def evaluate_downbeats(
    reference_downbeats: np.ndarray,
    estimated_downbeats: np.ndarray,
    trim: bool = True,
) -> dict:
    """Downbeat F-measure with the same ±70 ms tolerance as beats.

    Downbeats are just a sparser beat sequence, so the beat F-measure
    machinery applies unchanged; continuity metrics are less meaningful at
    bar level and are intentionally not reported here.
    """
    ref = _sanitize(reference_downbeats)
    est = _sanitize(estimated_downbeats)
    if trim:
        ref = mir_eval.beat.trim_beats(ref, min_beat_time=TRIM_MIN_BEAT_TIME)
        est = mir_eval.beat.trim_beats(est, min_beat_time=TRIM_MIN_BEAT_TIME)
    if len(ref) == 0:
        return {"downbeat_f_measure": float("nan")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = mir_eval.beat.f_measure(ref, est, f_measure_threshold=F_MEASURE_TOLERANCE)
    return {"downbeat_f_measure": float(f)}


def evaluate_many(
    pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    trim: bool = True,
) -> dict:
    """Evaluate (reference, estimate) pairs and aggregate.

    Returns::

        {
          "n_clips": int,           # pairs evaluated
          "n_scored": int,          # pairs with a usable reference (non-NaN)
          "per_clip": [dict, ...],  # evaluate() output per pair, in order
          "mean":   {metric: float, ...},   # NaN-aware
          "median": {metric: float, ...},
          "std":    {metric: float, ...},
        }
    """
    per_clip = [evaluate(ref, est, trim=trim) for ref, est in pairs]
    agg_mean, agg_median, agg_std = {}, {}, {}
    n_scored = 0
    if per_clip:
        n_scored = int(np.sum([not np.isnan(r["f_measure"]) for r in per_clip]))
    for name in METRIC_NAMES:
        vals = np.array([r[name] for r in per_clip], dtype=np.float64)
        if len(vals) == 0 or np.all(np.isnan(vals)):
            agg_mean[name] = agg_median[name] = agg_std[name] = float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agg_mean[name] = float(np.nanmean(vals))
                agg_median[name] = float(np.nanmedian(vals))
                agg_std[name] = float(np.nanstd(vals))
    return {
        "n_clips": len(per_clip),
        "n_scored": n_scored,
        "per_clip": per_clip,
        "mean": agg_mean,
        "median": agg_median,
        "std": agg_std,
    }


def format_report(results: dict) -> str:
    """Readable text table for evaluate() or evaluate_many() output."""
    lines = []
    if "per_clip" in results:  # aggregated results
        lines.append(
            f"Beat evaluation over {results['n_clips']} clip(s)"
            f" ({results['n_scored']} scored)"
        )
        header = f"{'metric':<18}{'mean':>10}{'median':>10}{'std':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for name in METRIC_NAMES:
            lines.append(
                f"{name:<18}"
                f"{results['mean'][name]:>10.3f}"
                f"{results['median'][name]:>10.3f}"
                f"{results['std'][name]:>10.3f}"
            )
    else:  # single clip
        header = f"{'metric':<18}{'value':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, value in results.items():
            lines.append(f"{name:<18}{value:>10.3f}")
    return "\n".join(lines)
