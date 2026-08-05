#!/usr/bin/env python3
"""Does the low/high phase relationship break *before* the level does?

The hypothesis, fixed before any number was looked at: a causal change in the
settled phase relationship between the low and high ODF bands precedes the
onset of a live wrong-metrical-level episode by one to four seconds, well
enough to hold the octave through it at a tolerable cost in correct locked
time.

**Change in the relationship, not the relationship itself.** Kick and snare
sitting stably in antiphase is ordinary rhythm, not a fault. What would be
evidence is their habitual offset coming apart. So the feature is measured
against what the offset has recently been, and everything here is causal: a
gate that needed the future could not run in the tracker.

This is a research spike and deliberately not in the core. It answers one
question — is there a warning signal at all — and the answer decides whether
anything gets built, not what.

What it does *not* do is choose the octave. Phase concentration at a period P
is equally strong at P/2, because events one period apart land at the same
angle when the candidate is half as long; it cancels only at 2P, where they
alternate between 0 and pi. So concentration rejects half-time and is blind to
double-time, and double is where the residual error actually is — 16.6% against
2.3% on Harmonix. Nothing in this file can decide a metrical level. It can only
notice that the evidence has become unreliable, which is a different and much
smaller claim.

--------------------------------------------------------------------- method

For each band c, one complex accumulator driven by the onset function and read
against the tracker's own beat clock:

    z_c(t) = sum over tau <= t of  d^(t-tau) * o_c(tau)^3 * exp(i*phi(tau))

`d` is a half-life of 2.5 s, the exponent is 3, and the Rayleigh threshold for
believing a phase is 3 — all three taken unchanged from `tracking::SyncConfig`
rather than fitted here, because a spike that tunes its own constants on the
corpus it is tested on has measured nothing.

Two implementation points that the formula as usually written gets wrong:

*One clock, not two.* `phi` is shared by both bands. A phase difference between
two accumulators is only meaningful if both were measured against the same
reference; giving each band its own clock at the same period would produce two
identical clocks anyway, and at different periods would produce a difference
that means nothing.

*The clock is integrated, not evaluated.* Writing the angle as `2*pi*tau/P(t)`
is only correct while P is constant. The tracker changes P continuously, and
that expression jumps the phase every time it does — manufacturing exactly the
discontinuity this file is looking for, at exactly the moments it is looking.
So the phase is advanced by `2*pi*dt/P(t)` per frame and stays continuous
through a tempo change.

Then:

    rho_c   = |z_c| / (m_c + eps)            concentration, 0..1
    N_c     = m_c^2 / m_sq_c                  effective onsets behind it
    dphi    = arg(z_low) - arg(z_high)
    G       = min(rho_low, rho_high) * (1 - cos(dphi - dphi_settled))

`dphi_settled` is a causal circular mean of `dphi` over the recent past,
updated only while the tracker is locked and both bands clear the Rayleigh
threshold — a weak or smeared vector has an essentially random angle and would
otherwise drag the reference around.

------------------------------------------------------------------ the test

Showing that G is large *during* a failure would prove nothing: by then the
anchor has already moved to another octave, and the phase relationship would be
disturbed by the very thing it is supposed to predict. So three windows are
scored separately:

    predictor  [onset-4, onset-1]   before anything has gone wrong
    detector   [onset,   onset+2]   at and just after the transition
    negative   matched 3 s windows drawn from locked, correct stretches

If only the detector window separates, the signal is a consequence and not a
warning, and the idea is finished.

And the comparison that decides whether the phase part earns its complexity:
three simpler signals are scored through the identical windows —

    coherence drop      -min(rho_low, rho_high)
    band balance        |log(E_low/E_high) - its own settled value|
    anchor margin       -live_anchor_margin, the tracker's own octave margin

If the phase feature is not better than a plain fall in coherence, the
elaborate part is not carrying anything.

**Measured, 2026-08-05, and the hypothesis is negative.** RWC-Pop (100
recordings, 178 episodes) chose the threshold; Harmonix (581 recordings, 1,063
episodes) received it as a number. Harmonix is a **threshold-transfer corpus
and not a held-out one** — it has been spent before, on the ensemble seam
experiment, and calling it held out would claim more than it can carry.

Area under the ROC, positives against the matched negatives, with how many
episodes each signal could see at all:

                    RWC-Pop                    Harmonix
                predict  detect  seen      predict  detect  seen
    phase         0.564   0.532  160/178     0.635   0.609   957/1063
    coherence     0.571   0.526  178/178     0.660   0.631  1063/1063
    balance       0.442   0.400  156/178     0.445   0.398   926/1063
    margin        0.866   0.878  178/178     0.895   0.932  1063/1063

The threshold fixed where 20% of clean negative windows trigger on RWC-Pop,
then carried to Harmonix unchanged. `warned` counts every episode over four
seconds, so an episode a signal had no answer for counts as missed:

                     RWC-Pop                 Harmonix (transfer)
                warned  windows  frames   warned  windows  frames
    phase        15.2%    20.0%    6.8%    16.3%    14.5%    4.9%
    coherence    22.5%    20.0%    2.0%    37.5%    20.4%    2.1%
    balance      12.9%    20.0%    0.1%    12.7%    22.8%    0.2%
    margin       84.3%    20.0%   18.6%    85.9%    18.7%   16.9%

**The phase feature does not predict.** 16% of episodes warned is what a coin
does. It is not better than a plain fall in coherence — which now leads it on
*both* corpora — and it is the only one of the four besides `balance` that
cannot even see every episode, missing 10% of them because no settled phase
ever formed. The elaborate part is not carrying anything and should not be
built.

**A trap worth recording, because it nearly produced a result.** Set the
threshold at a quantile of *frames* and then report the warned rate from
*window maxima*, and `balance` — an AUC of 0.44, worse than chance — comes back
warning 87% of episodes for 20% of frames, looking level with `margin`. The two
statistics are different populations: a window takes its maximum over 281
frames, so a threshold that 20% of frames clear is one that nearly every window
clears. The threshold and the rate it is quoted against have to come from the
same statistic; the table above fixes both on negative windows.

**And there is a mechanism, not just a low number.** The per-band phase is
mostly not believable in the first place. Measured on three recordings, the
share of frames where *both* bands clear the Rayleigh threshold of 3.0 that
`SyncConfig` already uses to decide whether a phase means anything: 2.9%, 5.5%
and 45.8%, with median per-band Rayleigh statistics of 0.72, 1.33 and 10.16.
The gate does keep firing to the end of each recording — so the reference is
sparse rather than frozen, and this is a verdict on the idea and not on the
implementation — but a relationship between two angles that are individually
this unreliable has little to come apart. Splitting the ODF in two costs most
of the evidence that made the full-band phase worth trusting.

**What did come out of it was not the hypothesis.** `live_anchor_margin` — how
far ahead the activation-tempo estimator's winning octave is of its runner-up,
a number the live path already computes on every frame and uses for nothing
else — warns of **85.9% of every Harmonix episode over four seconds, one to
four seconds before it starts**, at a threshold chosen on RWC-Pop and carried
across unchanged, with **16.9% of correct locked frames above that threshold**.
It sees all 1,063 of them, so that rate has no hidden denominator. It is
anticipatory rather than merely concurrent: 0.895 on the window that ends a
second *before* the onset against 0.932 on the transition itself.

That 16.9% is not "16.9% of good time lost" and must not be quoted as one. It
is the share of correct locked frames at which the signal is above the
threshold. What a gate costs depends on what the gate *does* — freezing the
octave while leaving tempo and phase free costs almost nothing on a frame where
the octave was not going to change anyway — and only replaying the tracker
under the policy can turn this column into a cost.

That is a finding about the tracker rather than about this file, and it belongs
to the same family as the downbeat channel `tracking/live.hpp` discards: the
evidence needed to see the failure coming is already being computed and thrown
away. It is also not a licence to gate on it — a tracker that abstains does not
spend time at the wrong level, so `no_wrong_level_episode_fraction` can be
improved by simply saying less, and any use of this has to be scored on correct
locked time kept as well as episodes avoided.

**Two defects in the first pass of this evaluator, recorded because both
changed a headline.** Every signal was scored on the *phase* feature's
availability, so `margin` was read only where a phase had settled and its rate
had a denominator of 996 of 1,063 episodes that the report did not mention;
with its own mask it sees all of them. And `_period_at` clipped its index to
zero before testing it, handing every frame ahead of the tracker's first poll a
tempo from the future — small in effect, since the windows all sit after
warm-up, but fatal to the claim that the thing could run causally at all.
`tests/test_phase_instability.py` pins both.

Kept rather than deleted, like `eval/oscillator.py` and `eval/am_hierarchy.py`
before it: the idea is well founded, the implementation is small, and a
documented negative is what stops it being proposed again with nothing to
answer.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

from eval.analysis import Analyser, Estimate
from eval.live_corpus_benchmark import (
    LOCK_CONFIDENCE,
    MAX_WRONG_OCTAVE_SEC,
    RELEASE_CONFIDENCE,
    WARMUP_SEC,
    local_reference_bpm,
    tempo_state,
)
from tiktak.odf import OdfConfig, compute_odf

__all__ = ["episodes", "phase_features", "windows", "score"]

# Taken from tracking::SyncConfig, not fitted here. See the module docstring.
HALF_LIFE_SEC = 2.5
ONSET_EXPONENT = 3.0
MIN_RAYLEIGH = 3.0

# `LiveConfig::onset_peak_tau_sec`. The tracker normalises its onset against a
# decaying peak follower rather than against the file, and so must this:
# dividing by the whole recording's maximum would let every frame know how loud
# the music is going to get, which is not a mistake a causal gate can make.
ONSET_PEAK_TAU_SEC = 3.0

# How far back the "settled" relationship looks. Longer than the accumulator's
# own memory on purpose: the reference has to survive the disturbance it is
# being compared against, so it cannot be built from the same 2.5 seconds.
SETTLED_HALF_LIFE_SEC = 8.0

# The warning window, and the window that tells a predictor from a detector.
PREDICT_WINDOW = (-4.0, -1.0)
DETECT_WINDOW = (0.0, 2.0)
# Negatives are the same width as positives so the maximum inside them is taken
# over the same number of frames; a wider window would score higher for nothing.
NEGATIVE_WIDTH = PREDICT_WINDOW[1] - PREDICT_WINDOW[0]
# Correct stretches touching an episode are not clean negatives: the approach
# to a failure is exactly what the positives are made of.
NEGATIVE_GUARD_SEC = 8.0

# Every nth correct locked frame is kept for the per-frame cost. At 93.75 fps
# this is about three samples a second, which is far more than the shape needs
# and far less than storing all of them.
FRAME_COST_STRIDE = 32

SAMPLE_RATE = 48000.0


@dataclass
class Episode:
    """One locked stretch at the wrong metrical level."""

    onset_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.onset_sec


@dataclass
class Timeline:
    """Everything measured per ODF frame, plus the tracker's own state."""

    times: np.ndarray
    g: np.ndarray
    rho_min: np.ndarray
    balance: np.ndarray
    anchor_margin: np.ndarray
    # Per signal, the frames where that signal has an answer at all — not
    # "no warning" but "no answer", and scoring those as zero would credit a
    # feature with every quiet opening it slept through.
    #
    # One mask per feature and not one shared mask, which is the bug this
    # replaced. Sharing the phase feature's availability made every other
    # signal inherit it, so `margin` was scored only where a *phase* had
    # settled — discarding whole recordings in which the phase never became
    # believable but the tracker's own octave margin was perfectly readable.
    # That silently changed the denominator of the one number the spike
    # eventually reported.
    available: dict[str, np.ndarray] = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)


def episodes(estimate: Estimate, beats: np.ndarray
             ) -> tuple[list[Episode], np.ndarray, np.ndarray]:
    """Wrong-level episodes, and the locked / correct masks behind them.

    The lock hysteresis, the octave tolerance and the warm-up are the
    benchmark's own, imported rather than restated, so an episode here is the
    same event `no_wrong_level_episode_fraction` counts. Any drift between the
    two would make this spike measure something the product does not.
    """
    count = min(len(estimate.live_times), len(estimate.live_bpms),
                len(estimate.live_confidences))
    times = np.asarray(estimate.live_times[:count], dtype=np.float64)
    bpms = np.asarray(estimate.live_bpms[:count], dtype=np.float64)
    confidences = np.asarray(estimate.live_confidences[:count], dtype=np.float64)

    locked_mask = np.zeros(count, dtype=bool)
    correct_mask = np.zeros(count, dtype=bool)
    found: list[Episode] = []

    locked = False
    wrong_since: float | None = None
    for index in range(count):
        time_sec = float(times[index])
        confidence = float(confidences[index])
        if not locked and confidence >= LOCK_CONFIDENCE:
            locked = True
        elif locked and confidence < RELEASE_CONFIDENCE:
            locked = False
            # A release ends the episode wherever it had got to. It does not
            # make the recording right, but it does end this stretch.
            if wrong_since is not None:
                found.append(Episode(wrong_since, time_sec))
                wrong_since = None
        if not locked or time_sec < WARMUP_SEC:
            continue

        locked_mask[index] = True
        state = tempo_state(float(bpms[index]),
                            local_reference_bpm(beats, time_sec))
        if state == "same":
            correct_mask[index] = True
            if wrong_since is not None:
                found.append(Episode(wrong_since, time_sec))
                wrong_since = None
        elif wrong_since is None:
            wrong_since = time_sec

    if wrong_since is not None and count:
        found.append(Episode(wrong_since, float(times[count - 1])))

    return ([e for e in found if e.duration_sec > MAX_WRONG_OCTAVE_SEC],
            times, correct_mask)


def _period_at(times: np.ndarray, live_times: np.ndarray,
               live_bpms: np.ndarray) -> np.ndarray:
    """The tracker's beat period at each ODF frame, causally.

    `searchsorted` with side="right" minus one is the most recent poll at or
    before the frame — never a later one, which would let the clock know a
    tempo the tracker had not yet reported.
    """
    # NaN, not the first poll, before the tracker has reported anything.
    # Clipping the index to zero first and testing it afterwards -- which this
    # did -- silently hands every opening frame a tempo from the future, and a
    # gate that cannot run causally is not a gate.
    index = np.searchsorted(live_times, times, side="right") - 1
    known = index >= 0
    bpm = np.where(known, live_bpms[np.clip(index, 0, max(len(live_bpms) - 1, 0))],
                   np.nan)
    # Hold the last usable tempo across gaps rather than letting the clock
    # stop: a stopped clock puts every onset at the same angle and would read
    # as perfect coherence.
    usable = np.isfinite(bpm) & (bpm > 1.0)
    if not usable.any():
        return np.full(len(times), np.nan)
    # Forward fill only. Frames before the *first* usable poll stay NaN rather
    # than borrowing it backwards, which is the same leak in another dress.
    source = np.maximum.accumulate(np.where(usable, np.arange(len(bpm)), -1))
    filled = np.where(source >= 0, bpm[np.clip(source, 0, len(bpm) - 1)], np.nan)
    return 60.0 / filled


def phase_features(odf, estimate: Estimate) -> Timeline:
    """The per-frame accumulators and the four candidate signals."""
    times = np.asarray(odf.times, dtype=np.float64)
    low = np.asarray(odf.low, dtype=np.float64)
    high = np.asarray(odf.high, dtype=np.float64)
    period = _period_at(times, np.asarray(estimate.live_times),
                        np.asarray(estimate.live_bpms))

    n = len(times)
    dt = 1.0 / float(odf.fps)
    decay = 0.5 ** (dt / HALF_LIFE_SEC)
    settled_decay = 0.5 ** (dt / SETTLED_HALF_LIFE_SEC)

    peak_decay = math.exp(-dt / ONSET_PEAK_TAU_SEC)
    peak_low = peak_high = 0.0

    z_low = z_high = 0.0 + 0.0j
    m_low = m_high = 0.0
    sq_low = sq_high = 0.0
    settled = 0.0 + 0.0j
    phase = 0.0

    g = np.zeros(n)
    rho_min = np.zeros(n)
    defined = np.zeros(n, dtype=bool)
    # The clock only exists once the tracker has reported a tempo. Before that
    # the accumulators are not merely empty, they are meaningless, and the
    # bands' coherence must not be read off them either.
    clock = np.isfinite(period) & (period > 0.0)
    for i in range(n):
        if not clock[i]:
            continue
        phase += 2.0 * math.pi * dt / period[i]
        rotor = complex(math.cos(phase), math.sin(phase))

        # The core's peak follower, per band: rises instantly to a new loudest
        # onset and forgets one over a few seconds.
        peak_low = max(low[i], peak_low * peak_decay)
        peak_high = max(high[i], peak_high * peak_decay)
        w_low = min(1.0, low[i] / (peak_low + 1e-6)) ** ONSET_EXPONENT
        w_high = min(1.0, high[i] / (peak_high + 1e-6)) ** ONSET_EXPONENT

        z_low = z_low * decay + w_low * rotor
        z_high = z_high * decay + w_high * rotor
        m_low = m_low * decay + w_low
        m_high = m_high * decay + w_high
        # Decayed at the square of the rate, matching PhaseSync::evidence --
        # decaying it at the plain rate understates the participation ratio by
        # very nearly a factor of two.
        sq_low = sq_low * decay * decay + w_low ** 2
        sq_high = sq_high * decay * decay + w_high ** 2

        r_low = abs(z_low) / (m_low + 1e-12)
        r_high = abs(z_high) / (m_high + 1e-12)
        n_low = (m_low * m_low) / (sq_low + 1e-12)
        n_high = (m_high * m_high) / (sq_high + 1e-12)

        dphi = math.atan2(z_low.imag, z_low.real) - math.atan2(z_high.imag,
                                                               z_high.real)
        rho_min[i] = min(r_low, r_high)

        believable = (r_low * r_low * n_low >= MIN_RAYLEIGH
                      and r_high * r_high * n_high >= MIN_RAYLEIGH)
        if settled != 0:
            defined[i] = True
            reference = math.atan2(settled.imag, settled.real)
            g[i] = rho_min[i] * (1.0 - math.cos(dphi - reference))
        # Updated *after* the read, so the frame is scored against what the
        # relationship was before it, never including itself.
        if believable:
            settled = settled * settled_decay + (1.0 - settled_decay) * complex(
                math.cos(dphi), math.sin(dphi)) * rho_min[i]

    balance = _balance_signal(low, high, settled_decay)
    live_times = np.asarray(estimate.live_times)
    margin = _resample_step(times, live_times,
                            np.asarray(estimate.live_anchor_margin))
    # Each signal's own availability, and nothing borrowed from another.
    #
    # `margin` is available from the tracker's first poll: it is the tracker's
    # own number and owes nothing to the band split. `coherence` needs the
    # clock and some accumulated mass. `balance` is a ratio of two ODF bands
    # and needs neither, only enough frames for its own running mean. `phase`
    # is the strictest because it alone needs a settled reference to differ
    # from.
    polled = (times >= float(live_times[0])) if live_times.size else np.zeros(
        n, dtype=bool)
    warm = times >= float(times[0]) + SETTLED_HALF_LIFE_SEC
    return Timeline(
        times=times, g=g, rho_min=rho_min, balance=balance,
        anchor_margin=margin, available={
            "phase": defined,
            "coherence": clock & (rho_min > 0.0),
            "balance": warm,
            "margin": polled,
        })


def _balance_signal(low: np.ndarray, high: np.ndarray,
                    decay: float) -> np.ndarray:
    """How far the low/high energy ratio has moved from its own settled value.

    The cheapest thing that could explain a phase disturbance: the bands simply
    changed loudness relative to each other, an arrangement change rather than
    a rhythmic one. If this predicts as well as the phase feature, the phase
    feature is not the reason.
    """
    ratio = np.log((low + 1e-9) / (high + 1e-9))
    settled = np.zeros_like(ratio)
    running = ratio[0] if len(ratio) else 0.0
    for i, value in enumerate(ratio):
        settled[i] = abs(value - running)
        running = running * decay + (1.0 - decay) * value
    return settled


def _resample_step(times: np.ndarray, source_times: np.ndarray,
                   values: np.ndarray) -> np.ndarray:
    """A poll-rate series held at ODF frame rate, causally."""
    if len(source_times) == 0 or len(values) == 0:
        return np.zeros(len(times))
    count = min(len(source_times), len(values))
    index = np.searchsorted(source_times[:count], times, side="right") - 1
    return np.where(index >= 0, values[:count][np.clip(index, 0, count - 1)], 0.0)


# Every candidate signal, oriented so that larger always means "more reason to
# distrust the level". The three beside `phase` are the controls: if a plain
# fall in coherence predicts as well, the phase construction is not what is
# doing the work and should not be built.
FEATURES = ("phase", "coherence", "balance", "margin")


def _feature(timeline: Timeline, name: str) -> np.ndarray:
    if name == "phase":
        return timeline.g
    if name == "coherence":
        return -timeline.rho_min
    if name == "balance":
        return timeline.balance
    if name == "margin":
        return -timeline.anchor_margin
    raise KeyError(name)


def windows(timeline: Timeline, found: list[Episode],
            correct: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """The maximum of each signal inside every scored window.

    Three kinds, and the middle one is the control that matters most: a signal
    that only lights up once the level has already moved is a detector, and a
    detector is worth nothing to a tracker that has to decide *before*.
    """
    out = {name: {"predict": [], "detect": [], "negative": [], "frames": [],
                  "episodes_total": len(found), "episodes_scored": 0}
           for name in FEATURES}
    times = timeline.times

    def peak(values: np.ndarray, available: np.ndarray,
             start: float, stop: float) -> float | None:
        mask = (times >= start) & (times <= stop) & available
        return float(values[mask].max()) if mask.any() else None

    guarded = np.zeros(len(times), dtype=bool)
    for episode in found:
        guarded |= ((times >= episode.onset_sec - NEGATIVE_GUARD_SEC)
                    & (times <= episode.end_sec + NEGATIVE_GUARD_SEC))

    # Negatives are tiled rather than drawn at random, so a long correct
    # stretch contributes in proportion to its length — which is also how the
    # cost of a false alarm is actually paid.
    for name in FEATURES:
        values = _feature(timeline, name)
        available = timeline.available[name]

        for episode in found:
            scored = False
            for kind, (lo, hi) in (("predict", PREDICT_WINDOW),
                                   ("detect", DETECT_WINDOW)):
                value = peak(values, available, episode.onset_sec + lo,
                             episode.onset_sec + hi)
                if value is not None:
                    out[name][kind].append(value)
                    scored = scored or kind == "predict"
            out[name]["episodes_scored"] += int(scored)

        # Negatives are tiled rather than drawn at random, so a long correct
        # stretch contributes in proportion to its length. Per feature, since
        # the eligible stretches depend on that feature's own availability.
        eligible = correct & available & ~guarded
        for start, stop in _runs(times, eligible):
            for edge in np.arange(start, stop - NEGATIVE_WIDTH, NEGATIVE_WIDTH):
                value = peak(values, available, edge, edge + NEGATIVE_WIDTH)
                if value is not None:
                    out[name]["negative"].append(value)

        # And the honest cost denominator beside the windowed one: every
        # correct locked frame where this signal has an answer, guard and
        # windowing removed. A window counts as gated when any one of its 281
        # frames crosses, which overstates how much time a gate would actually
        # suppress; this does not. Subsampled because the shape is what is
        # wanted, not every sample of it.
        frames = values[correct & available]
        if frames.size:
            out[name]["frames"].extend(
                float(v) for v in frames[::FRAME_COST_STRIDE])
    return out


def _runs(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True stretches of `mask`, as (start, stop) in seconds."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask) - 1)
    return [(float(times[a]), float(times[b])) for a, b in zip(starts, stops)]


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Rank-based area under the ROC, ties counted as half.

    Written out rather than imported: it is six lines, and the alternative is a
    dependency whose tie handling would have to be checked anyway.
    """
    if not positive or not negative:
        return None
    values = np.concatenate([np.asarray(positive), np.asarray(negative)])
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # Average the ranks of tied values, or a signal that is constant would
    # score 1.0 against itself.
    _, inverse, counts = np.unique(values, return_inverse=True,
                                   return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    n_pos = len(positive)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * len(negative)))


def measure(item: dict, model: pathlib.Path | None,
            binary: pathlib.Path | None) -> dict:
    """One recording: run the tracker, build the signals, cut the windows."""
    import soundfile
    from scipy.signal import resample_poly

    analyser = Analyser(binary) if binary else Analyser()
    estimate = analyser.analyse_live_file(item["audio"], model=model)
    beats = item["beats"]

    audio, rate = soundfile.read(str(item["audio"]), dtype="float32",
                                 always_2d=True)
    samples = audio.mean(axis=1)
    if int(rate) != int(SAMPLE_RATE):
        samples = resample_poly(samples, int(SAMPLE_RATE), int(rate))

    odf = compute_odf(samples, OdfConfig(sample_rate=SAMPLE_RATE))
    timeline = phase_features(odf, estimate)
    found, poll_times, correct_polls = episodes(estimate, beats)
    correct = _resample_step(timeline.times, poll_times,
                             correct_polls.astype(np.float64)) > 0.5
    return {
        "name": item["name"], "corpus": item["corpus"],
        "episodes": len(found),
        "windows": windows(timeline, found, correct),
    }


def _pool(records: list[dict]) -> dict[str, dict]:
    """Every recording's windows, concatenated, per feature."""
    out = {name: {"predict": [], "detect": [], "negative": [], "frames": [],
                  "episodes_total": 0, "episodes_scored": 0}
           for name in FEATURES}
    for record in records:
        for name in FEATURES:
            block = record["windows"][name]
            for kind in ("predict", "detect", "negative", "frames"):
                out[name][kind].extend(block[kind])
            for key in ("episodes_total", "episodes_scored"):
                out[name][key] += block[key]
    return out


def score(pooled: dict[str, dict]) -> list[dict]:
    """AUC for each signal, predictor and detector kept apart.

    `episodes_scored` against `episodes_total` is reported per feature and not
    once for the run: each signal now has its own availability, so each sees a
    different number of episodes, and an AUC over a subset means nothing
    without saying how large the subset was.
    """
    rows = []
    for name in FEATURES:
        block = pooled[name]
        rows.append({
            "feature": name,
            "episodes_total": block["episodes_total"],
            "episodes_scored": block["episodes_scored"],
            "coverage": (block["episodes_scored"] / block["episodes_total"]
                         if block["episodes_total"] else None),
            "n_predict": len(block["predict"]),
            "n_detect": len(block["detect"]),
            "n_negative": len(block["negative"]),
            "auc_predict": auc(block["predict"], block["negative"]),
            "auc_detect": auc(block["detect"], block["negative"]),
        })
    return rows


def operating_curve(positive: list[float], negative: list[float],
                    frames: list[float], episodes_total: int,
                    points: int = 21) -> list[dict]:
    """What a threshold would warn about, and what it would cost.

    Three columns rather than one, because the obvious single number is the
    one that flatters:

    `warned_of_scored` is the share of the episodes this signal could see at
    all. `warned_of_all` divides by every episode over four seconds, counting
    the ones the signal had no answer for as missed, which is what a tracker
    would actually experience.

    `negative_windows_triggered` is the share of clean three-second windows
    containing at least one crossing. It is a ranking quantity and *not* a
    share of time: a window counts as triggered on one frame in 281, guarded
    stretches near failures are excluded from it, and a window is not a unit
    anybody experiences.

    `correct_frames_triggered` is the share of correct locked frames above the
    threshold, with no guard and no windowing. It is the closer of the two to
    the cost a gate would really impose — and still not the cost itself, which
    only a replay of the tracker under the policy can give.
    """
    if not positive or not negative:
        return []
    positives = np.asarray(positive)
    negatives = np.asarray(negative)
    frame_values = np.asarray(frames) if frames else None
    rows = []
    for threshold in np.quantile(negatives, np.linspace(0.0, 1.0, points)):
        rows.append({
            "threshold": float(threshold),
            "warned_of_scored": float(np.mean(positives >= threshold)),
            "warned_of_all": float(
                np.sum(positives >= threshold) / episodes_total)
            if episodes_total else None,
            "negative_windows_triggered": float(np.mean(negatives >= threshold)),
            "correct_frames_triggered": float(
                np.mean(frame_values >= threshold))
            if frame_values is not None and frame_values.size else None,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    import concurrent.futures

    from eval.live_corpus_benchmark import load_corpus, load_reference_beats

    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth"
                        / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path,
                        default=repository / "music")
    parser.add_argument("--corpora", nargs="+", default=["harmonix"])
    parser.add_argument("--model", type=pathlib.Path,
                        default=repository / "models" / "beatnet_model_1.ttw")
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int,
                        help="stop after this many recordings, for a smoke run")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False, set(args.corpora))
    for item in items:
        item["beats"] = load_reference_beats(item["annotation"])
    if args.limit:
        items = items[:args.limit]
    print(f"   {len(items)} recordings, {args.workers} workers")

    records = []
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(measure, item, args.model, args.binary): item
                   for item in items}
        for done in concurrent.futures.as_completed(futures):
            try:
                records.append(done.result())
            except Exception as error:  # noqa: BLE001 - one bad file is not a run
                print(f"   skipped {futures[done]['name']}: {error}")

    pooled = _pool(records)
    rows = score(pooled)
    episodes_total = sum(record["episodes"] for record in records)
    print(f"\n   {len(records)} scored, {episodes_total} episodes over 4 s")
    print(f"\n   {'signal':<12}{'AUC predict':>13}{'AUC detect':>12}"
          f"{'episodes seen':>15}{'n neg':>8}")
    for row in rows:
        predict = "—" if row["auc_predict"] is None else f"{row['auc_predict']:.3f}"
        detect = "—" if row["auc_detect"] is None else f"{row['auc_detect']:.3f}"
        seen = (f"{row['episodes_scored']}/{row['episodes_total']}")
        print(f"   {row['feature']:<12}{predict:>13}{detect:>12}"
              f"{seen:>15}{row['n_negative']:>8}")

    curves = {name: operating_curve(pooled[name]["predict"],
                                    pooled[name]["negative"],
                                    pooled[name]["frames"],
                                    pooled[name]["episodes_total"])
              for name in FEATURES}
    for name in ("phase", "margin"):
        print(f"\n   {name}: warned (of all episodes) against cost")
        for row in curves[name][::4]:
            frames = ("—" if row["correct_frames_triggered"] is None
                      else f"{row['correct_frames_triggered'] * 100:5.1f}%")
            print(f"   thr {row['threshold']:9.4f}   warned "
                  f"{row['warned_of_all'] * 100:5.1f}%   frames {frames}"
                  f"   windows {row['negative_windows_triggered'] * 100:5.1f}%")

    if args.output:
        args.output.write_text(json.dumps({
            "corpora": args.corpora, "scored": len(records),
            "episodes": episodes_total, "score": rows, "curves": curves,
            # The window maxima themselves, not only the curve fitted through
            # them. A threshold chosen on one corpus has to be applied to
            # another *as a number*, and a curve sampled at this corpus's own
            # quantiles cannot answer that without interpolating — which is
            # exactly the step a transfer test should not have to take.
            "windows": {name: {kind: pooled[name][kind]
                               for kind in ("predict", "detect", "negative",
                                            "frames")}
                        for name in FEATURES},
        }, indent=2), encoding="utf-8")
        print(f"\n   wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
