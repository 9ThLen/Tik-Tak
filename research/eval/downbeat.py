"""Scoring bar lines, and choosing the margin thresholds from data.

Beat tracking has a settled metric and downbeat tracking mostly borrows it: the
±70 ms F-measure, applied to the sparser sequence. That is the right number for
comparing this analysis against a model later, and :func:`score` reports it.

It is not, on its own, the number the app needs. Three reasons:

* **F-measure cannot see a metre error.** Call a 4/4 track 2/4 and every real
  bar line is still hit — recall is perfect, precision is a half, and the
  F-measure lands near 0.67, which reads like a decent result for an answer
  that is simply wrong. So the metre is scored separately, and a clip only
  counts as right if the metre *and* the phase are both right.

* **The app does not have to answer.** Below its thresholds the analysis
  withholds the accent and the click stays even, because an accent on the wrong
  beat is worse to play along to than no accent at all. That makes this a
  decision with three outcomes and not two, and the interesting question is not
  "how accurate is it" but "how much can it cover at an acceptable wrong rate".

* **There are two separate doubts, so there are two thresholds.** The phase
  margin says which beat starts the bar is settled; the metre margin says no
  other bar length fits nearly as well. Sweeping only the first is what let a
  4/4 track read as three with a phase margin of 0.69 — inside that wrong
  metre, the phase really was unambiguous. :func:`sweep` crosses both.

The asymmetry is the whole point and is worth stating once: withholding costs
the user a feature, and a wrong accent costs them the take. Thresholds are
therefore chosen by capping the error and maximising coverage under that cap —
never by maximising accuracy, which answering almost never would win.

Two error rates are reported because they fail differently.  ``wrong_rate`` is
wrong answers over *all* clips: it is what the user experiences across their
library, but it can be driven to zero by never answering.  ``conditional_error``
is wrong answers over the clips actually accented: it is what the user
experiences *when the feature fires*, and a threshold with 10% coverage and 40%
conditional error is a bad feature even though its wrong rate is only 4%.
:func:`choose_margins` bounds both.
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
    "choose_margins",
    "thresholds_from",
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
    apart is what tells a low coverage number caused by strict thresholds from
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
    phase_margin: float
    meter_margin: float
    strength: float
    # None when the reference has no bar lines annotated, so the question was
    # never asked. Distinct from False, which means it was asked and missed.
    meter_correct: bool | None
    phase_correct: bool | None
    scorable: bool          # whether the reference carried bar lines at all
    meter_is_stable: bool = True
    shipped_confident: bool = False   # what the core's own defaults concluded

    def verdict(self, min_phase_margin: float, min_meter_margin: float) -> str:
        if self.estimated_meter <= 0:
            return Verdict.NO_ANSWER
        if self.phase_margin < min_phase_margin or self.meter_margin < min_meter_margin:
            return Verdict.WITHHELD
        if not self.meter_correct:
            return Verdict.WRONG_METER
        if not self.phase_correct:
            return Verdict.WRONG_PHASE
        return Verdict.CORRECT

    def shipped_verdict(self) -> str:
        """What the C++ core decided under the thresholds it actually ships."""
        if self.estimated_meter <= 0:
            return Verdict.NO_ANSWER
        if not self.shipped_confident:
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
        phase_margin=float(estimate.downbeat_phase_margin),
        meter_margin=float(estimate.downbeat_meter_margin),
        strength=float(estimate.downbeat_strength),
        meter_correct=meter_correct,
        phase_correct=phase_correct,
        scorable=scorable,
        meter_is_stable=getattr(reference, "meter_is_stable", True),
        shipped_confident=bool(getattr(estimate, "downbeat_confident", False)),
    )


@dataclass
class SweepRow:
    min_phase_margin: float
    min_meter_margin: float
    n: int              # clips eligible for calibration
    shown: int          # clips the app would have accented
    correct: int
    wrong_meter: int
    wrong_phase: int
    withheld: int
    no_answer: int

    @property
    def wrong(self) -> int:
        return self.wrong_meter + self.wrong_phase

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

        What the user meets across a library. Can be pushed to zero by never
        answering, which is why it is never the only bound.
        """
        return self.wrong / self.n if self.n else float("nan")

    @property
    def conditional_error(self) -> float:
        """Fraction of the accents actually shown that are wrong.

        What the user meets when the feature fires. Blind to coverage, so it is
        never the only bound either — the two are bounded together.
        """
        return self.wrong / self.shown if self.shown else float("nan")


def thresholds_from(values: Sequence[float]) -> list[float]:
    """Candidate thresholds drawn from the margins actually observed.

    A fixed 0…1 grid was wrong in both directions: it wasted most of its rows on
    a range the data never occupies, and it stopped at 1.0 while real margins
    reach past 3, so every clip above 1.0 was lumped together and the top of the
    curve was invisible. Observed margins put the candidates exactly where the
    decisions change.

    Every distinct decision boundary is included. A 100–150 clip calibration
    produces at most a few tens of thousands of threshold pairs, which is small
    enough to evaluate exactly and avoids a quantile grid skipping the optimum.

    Zero is always included, so "accent everything the analysis offers" stays
    on the curve as the baseline the thresholds have to beat. Because a margin
    equal to the threshold is still shown, the boundary is the next
    representable float above each observed value.
    """
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(finite) == 0:
        return [0.0]
    return sorted({
        0.0,
        *(float(np.nextafter(value, np.inf)) for value in np.unique(finite)),
    })


def eligible(scores: Sequence[ClipScore]) -> list[ClipScore]:
    """The clips a threshold may be calibrated on.

    Excludes clips with no annotated bar lines, and clips whose metre changes
    partway: for those no single answer is right for the whole take, so counting
    them either way would move the threshold on the strength of a question the
    analysis was never asked. They are reported separately instead.
    """
    return [s for s in scores if s.scorable and s.meter_is_stable]


def sweep(scores: Sequence[ClipScore],
          phase_thresholds: Iterable[float] | None = None,
          meter_thresholds: Iterable[float] | None = None) -> list[SweepRow]:
    """Outcome counts across the cross product of both thresholds."""
    usable = eligible(scores)
    if phase_thresholds is None:
        phase_thresholds = thresholds_from([s.phase_margin for s in usable])
    if meter_thresholds is None:
        meter_thresholds = thresholds_from([s.meter_margin for s in usable])

    rows = []
    for phase in phase_thresholds:
        for meter in meter_thresholds:
            verdicts = [s.verdict(float(phase), float(meter)) for s in usable]
            rows.append(
                SweepRow(
                    min_phase_margin=float(phase),
                    min_meter_margin=float(meter),
                    n=len(usable),
                    shown=sum(v in Verdict.SHOWN for v in verdicts),
                    correct=verdicts.count(Verdict.CORRECT),
                    wrong_meter=verdicts.count(Verdict.WRONG_METER),
                    wrong_phase=verdicts.count(Verdict.WRONG_PHASE),
                    withheld=verdicts.count(Verdict.WITHHELD),
                    no_answer=verdicts.count(Verdict.NO_ANSWER),
                )
            )
    return rows


def choose_margins(rows: Sequence[SweepRow],
                   max_wrong_rate: float = 0.05,
                   max_conditional_error: float = 0.10) -> SweepRow | None:
    """The threshold pair with the widest coverage inside both error budgets.

    Coverage is what is maximised, not accuracy: among pairs that are all
    acceptably safe, the useful one is the one that answers most often. Ties go
    to the lower thresholds, so a pair is not made stricter than the evidence
    requires.

    Both budgets are enforced because either alone has a degenerate optimum —
    ``wrong_rate`` is minimised by never answering, and ``conditional_error`` is
    minimised by answering only on the easiest clip in the set.

    Returns None when nothing meets both, which is a real answer: it says the
    cues cannot support an automatic accent on this material at that level of
    safety, and the honest response is to leave the feature off rather than to
    raise the budget until it passes.

    Choose on a validation split and report on a held-out one. A threshold
    picked and reported on the same clips is a description of those clips.
    """
    acceptable = [
        r for r in rows
        # ``shown`` must be non-zero: a pair above every clip's margins has a
        # wrong rate of zero and would win on the budget alone, while describing
        # a feature that never fires. That is not a calibration, it is the
        # feature being off, and the caller asked what to switch it on at.
        if r.n and r.shown
        and r.wrong_rate <= max_wrong_rate
        and r.conditional_error <= max_conditional_error
    ]
    if not acceptable:
        return None
    return max(
        acceptable,
        key=lambda r: (r.coverage, -r.min_phase_margin, -r.min_meter_margin),
    )


def frontier(rows: Sequence[SweepRow]) -> list[SweepRow]:
    """The rows worth looking at: no other row is both safer and wider.

    A full cross product is mostly noise — many threshold pairs produce
    identical outcomes. This keeps the trade-off and drops the rest.
    """
    best: dict[int, SweepRow] = {}
    for row in rows:
        current = best.get(row.shown)
        if current is None or row.wrong < current.wrong:
            best[row.shown] = row
    return sorted(best.values(), key=lambda r: (-r.shown, r.wrong))


def format_scores(scores: Sequence[ClipScore]) -> str:
    lines = [
        f"{'clip':<24}{'beat F':>8}{'db F':>7}{'metre':>9}"
        f"{'phase':>8}{'metre m':>9}  verdict",
        "-" * 82,
    ]
    for s in scores:
        metre = f"{s.estimated_meter or '-'}/{s.reference_meter or '?'}"
        note = "" if s.meter_is_stable else "  (metre changes)"
        lines.append(
            f"{s.name[:23]:<24}{s.beat_f:>8.2f}{s.downbeat_f:>7.2f}{metre:>9}"
            f"{s.phase_margin:>8.2f}{s.meter_margin:>9.2f}  "
            f"{s.shipped_verdict()}{note}"
        )
    return "\n".join(lines)


def format_sweep(rows: Sequence[SweepRow]) -> str:
    lines = [
        f"{'phase':>7}{'metre':>7}{'coverage':>10}{'wrong rate':>12}"
        f"{'cond err':>10}{'correct':>9}{'metre':>7}{'phase':>7}{'held':>6}{'none':>6}",
        "-" * 81,
    ]
    for r in rows:
        lines.append(
            f"{r.min_phase_margin:>7.2f}{r.min_meter_margin:>7.2f}{r.coverage:>10.2f}"
            f"{r.wrong_rate:>12.2f}{r.conditional_error:>10.2f}{r.correct:>9d}"
            f"{r.wrong_meter:>7d}{r.wrong_phase:>7d}{r.withheld:>6d}{r.no_answer:>6d}"
        )
    return "\n".join(lines)
