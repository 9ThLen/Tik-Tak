"""Scoring bar lines, and choosing the margin threshold from data.

Beat tracking has a settled metric and downbeat tracking mostly borrows it: the
±70 ms F-measure, applied to the sparser sequence. That is the right number for
comparing this analysis against a model later, and :func:`score` reports it.

It is not, on its own, the number the app needs. Two reasons:

* **F-measure cannot see a metre error.** Call a 4/4 track 2/4 and every real
  bar line is still hit — recall is perfect, precision is a half, and the
  F-measure lands near 0.67, which reads like a decent result for an answer
  that is simply wrong. So the metre is scored separately, and a clip only
  counts as right if the metre *and* the phase are both right.

* **The app does not have to answer.** Below a margin threshold the harness
  withholds the accent and counts from the first beat instead, because an accent
  on the wrong beat is worse to play along to than no accent at all. That makes
  this a decision with three outcomes and not two, and the interesting question
  is not "how accurate is it" but "how much can it cover at an acceptable wrong
  rate". :func:`sweep` produces that curve and :func:`choose_margin` reads a
  threshold off it.

The asymmetry is the whole point and is worth stating once: withholding costs
the user a feature, and a wrong accent costs them the take. The threshold is
therefore chosen by capping the wrong rate and maximising coverage under that
cap — never by maximising accuracy, which a threshold of 1.0 would win by
answering almost never.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import mir_eval.beat

from eval.harness import F_MEASURE_TOLERANCE, _sanitize

__all__ = [
    "Verdict",
    "ClipScore",
    "SweepRow",
    "score",
    "sweep",
    "choose_margin",
    "format_scores",
    "format_sweep",
]

# What fraction of the reference bar lines the estimate has to land on before
# the phase counts as found. Well away from a half: with the metre correct,
# the bar lines are either almost all right or almost all wrong, so anything
# in the middle means the beat grid itself has slipped and the clip deserves to
# be looked at rather than scored on a coin toss.
PHASE_RECALL = 0.5


class Verdict:
    """What happened on one clip, from the point of view of a player.

    ``NO_ANSWER`` and ``WITHHELD`` are both "no accent shown" and differ only in
    why: the first is the analysis finding no bar-level pattern at all, the
    second is it finding one it is not confident enough to use. Keeping them
    apart is what tells a low coverage number caused by a weak threshold from
    one caused by material the cues cannot read.
    """

    CORRECT = "correct"
    WRONG_METER = "wrong metre"
    WRONG_PHASE = "wrong phase"
    WITHHELD = "withheld"
    NO_ANSWER = "no answer"

    SHOWN = (CORRECT, WRONG_METER, WRONG_PHASE)
    WRONG = (WRONG_METER, WRONG_PHASE)


@dataclass
class ClipScore:
    name: str
    beat_f: float
    downbeat_f: float
    reference_meter: int
    estimated_meter: int
    margin: float
    strength: float
    # None when the reference has no bar lines annotated, so the question was
    # never asked. Distinct from False, which means it was asked and missed.
    meter_correct: bool | None
    phase_correct: bool | None
    scorable: bool          # whether the reference carried bar lines at all
    meter_is_stable: bool = True

    def verdict(self, min_margin: float) -> str:
        if self.estimated_meter <= 0:
            return Verdict.NO_ANSWER
        if self.margin < min_margin:
            return Verdict.WITHHELD
        if not self.meter_correct:
            return Verdict.WRONG_METER
        if not self.phase_correct:
            return Verdict.WRONG_PHASE
        return Verdict.CORRECT


def _f_measure(reference: np.ndarray, estimate: np.ndarray) -> float:
    if len(reference) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            mir_eval.beat.f_measure(reference, estimate,
                                    f_measure_threshold=F_MEASURE_TOLERANCE)
        )


def _recall(reference: np.ndarray, estimate: np.ndarray,
            tolerance: float = F_MEASURE_TOLERANCE) -> float:
    """Fraction of reference times an estimate lands on.

    Deliberately not mir_eval's F-measure: this is asked only once the metre is
    known to be right, and there the question is purely "are the bar lines in
    the right place", which is recall. Precision would double-count the metre
    error that has already been ruled out.
    """
    if len(reference) == 0:
        return float("nan")
    if len(estimate) == 0:
        return 0.0
    nearest = np.abs(reference[:, None] - estimate[None, :]).min(axis=1)
    return float(np.mean(nearest <= tolerance))


def score(reference, estimate, name: str | None = None) -> ClipScore:
    """Scores one clip.

    ``reference`` is an :class:`eval.annotations.Reference`; ``estimate`` is an
    :class:`eval.analysis.Estimate` — anything with the same attributes will do,
    which is what lets a model be scored here later without touching this file.
    """
    ref_beats = _sanitize(reference.beats)
    ref_downbeats = _sanitize(reference.downbeats)
    est_beats = _sanitize(estimate.beats)
    est_downbeats = _sanitize(estimate.downbeats)

    scorable = len(ref_downbeats) > 0 and reference.beats_per_bar > 0

    meter_correct: bool | None = None
    phase_correct: bool | None = None
    if scorable and estimate.beats_per_bar > 0:
        meter_correct = estimate.beats_per_bar == reference.beats_per_bar
        # Phase is only a separate question once the metre is right. With the
        # wrong metre the estimated bar lines are a different sequence
        # altogether and "is the phase right" has no answer worth recording.
        phase_correct = (
            _recall(ref_downbeats, est_downbeats) >= PHASE_RECALL
            if meter_correct
            else None
        )

    return ClipScore(
        name=name if name is not None else getattr(reference, "name", ""),
        beat_f=_f_measure(ref_beats, est_beats),
        downbeat_f=_f_measure(ref_downbeats, est_downbeats) if scorable else float("nan"),
        reference_meter=int(reference.beats_per_bar),
        estimated_meter=int(estimate.beats_per_bar),
        margin=float(estimate.downbeat_margin),
        strength=float(estimate.downbeat_strength),
        meter_correct=meter_correct,
        phase_correct=phase_correct,
        scorable=scorable,
        meter_is_stable=getattr(reference, "meter_is_stable", True),
    )


@dataclass
class SweepRow:
    min_margin: float
    n: int              # clips with bar lines in the reference
    shown: int          # clips the app would have accented
    correct: int
    wrong_meter: int
    wrong_phase: int
    withheld: int
    no_answer: int

    @property
    def coverage(self) -> float:
        """Fraction of clips that get an accent at all."""
        return self.shown / self.n if self.n else float("nan")

    @property
    def precision(self) -> float:
        """Of the accents shown, the fraction that are right."""
        return self.correct / self.shown if self.shown else float("nan")

    @property
    def wrong_rate(self) -> float:
        """Fraction of *all* clips given a wrong accent.

        The one to hold down. Precision alone can be pushed to 1.0 by answering
        almost never, which is not a better product.
        """
        return (self.wrong_meter + self.wrong_phase) / self.n if self.n else float("nan")


def sweep(scores: Sequence[ClipScore],
          thresholds: Iterable[float] | None = None) -> list[SweepRow]:
    """Outcome counts across a range of margin thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.001, 0.05), 3)

    scorable = [s for s in scores if s.scorable]
    rows = []
    for threshold in thresholds:
        verdicts = [s.verdict(float(threshold)) for s in scorable]
        rows.append(
            SweepRow(
                min_margin=float(threshold),
                n=len(scorable),
                shown=sum(v in Verdict.SHOWN for v in verdicts),
                correct=verdicts.count(Verdict.CORRECT),
                wrong_meter=verdicts.count(Verdict.WRONG_METER),
                wrong_phase=verdicts.count(Verdict.WRONG_PHASE),
                withheld=verdicts.count(Verdict.WITHHELD),
                no_answer=verdicts.count(Verdict.NO_ANSWER),
            )
        )
    return rows


def choose_margin(rows: Sequence[SweepRow], max_wrong_rate: float = 0.05) -> SweepRow | None:
    """The most generous threshold whose wrong rate stays inside the budget.

    "Most generous" means the lowest threshold, i.e. the widest coverage — the
    tie is broken towards showing the accent, because among thresholds that are
    all acceptably safe the useful one is the one that answers most often.

    Returns None when no threshold meets the budget, which is a real answer: it
    says the cues cannot support an automatic accent on this material at that
    level of safety, and the honest response is to leave the feature off rather
    than to raise the budget until it passes.

    Choose on a validation split and report on a held-out one. A threshold
    picked and reported on the same clips is a description of those clips.
    """
    # ``shown`` has to be non-zero: a threshold above every clip's margin has a
    # wrong rate of zero and would win on the budget alone, while describing a
    # feature that never fires. That is not a calibration, it is the feature
    # being off, and the caller asked which threshold to switch it on at.
    acceptable = [r for r in rows if r.n and r.shown and r.wrong_rate <= max_wrong_rate]
    if not acceptable:
        return None
    return min(acceptable, key=lambda r: (r.min_margin, -r.coverage))


def format_scores(scores: Sequence[ClipScore], min_margin: float) -> str:
    lines = [
        f"{'clip':<28}{'beat F':>8}{'db F':>7}{'metre':>12}{'margin':>8}  verdict",
        "-" * 78,
    ]
    for s in scores:
        metre = f"{s.estimated_meter or '-'}/{s.reference_meter or '?'}"
        note = "" if s.meter_is_stable else "  (metre changes)"
        lines.append(
            f"{s.name[:27]:<28}{s.beat_f:>8.2f}{s.downbeat_f:>7.2f}"
            f"{metre:>12}{s.margin:>8.2f}  {s.verdict(min_margin)}{note}"
        )
    return "\n".join(lines)


def format_sweep(rows: Sequence[SweepRow]) -> str:
    lines = [
        f"{'margin':>7}{'coverage':>10}{'precision':>11}{'wrong rate':>12}"
        f"{'correct':>9}{'metre':>7}{'phase':>7}{'held':>6}{'none':>6}",
        "-" * 75,
    ]
    for r in rows:
        lines.append(
            f"{r.min_margin:>7.2f}{r.coverage:>10.2f}{r.precision:>11.2f}"
            f"{r.wrong_rate:>12.2f}{r.correct:>9d}{r.wrong_meter:>7d}"
            f"{r.wrong_phase:>7d}{r.withheld:>6d}{r.no_answer:>6d}"
        )
    return "\n".join(lines)
