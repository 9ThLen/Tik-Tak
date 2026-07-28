#!/usr/bin/env python3
"""A bank of nonlinear oscillators as a tempo estimator.

Why this exists. The shipped estimator scores each candidate period by the
onset function's autocorrelation and weighs it with a log-normal prior over
tempo. Autocorrelation is symmetric about the octave — a train of beats
correlates with itself just as well at twice the period — so the prior is the
only thing choosing between t and t/2, and a prior can only decide *where* the
crossover sits, never which side a given recording belongs on. That is measured,
not asserted: over 698 annotated Ballroom recordings, moving the centre from 120
to 140 took half-tempo errors from 186 to 119 and gave 43 of them back as
double-tempo errors. See ``TempoConfig::prior_centre_bpm``.

The claim this module tests is Large's: a bank of *nonlinear* oscillators does
not inherit that symmetry. A driven Hopf oscillator saturates — its response is
not proportional to the drive — so an oscillator at the beat rate, which is
pushed on every beat, and one at half the beat rate, which is pushed on every
other beat, do not end up with the same amplitude even though the linear
correlation at both lags is the same. Whether that asymmetry is large enough to
decide the octave on real music is the empirical question, and it is the only
reason this file is here.

The canonical model, in the normal form Large writes it in:

    dz/dt = z (alpha + i*omega + beta |z|^2) + k x(t)

`alpha` below zero makes the oscillator damped, so it is silent unless driven;
`beta` below zero saturates it, which is the nonlinearity the whole argument
rests on; `x(t)` is the onset function. Amplitude |z| after the transient is the
resonance, and the bank's peak is the tempo.

**Measured, and the answer is interesting but negative.** Counting how often the
peak lands on the annotated tempo, on its half, or on its double:

                                       1x            octave-wrong
    Ballroom (698)
      shipped: autocorrelation x prior  439 (62.9%)   174 (24.9%)
      bank alone                        339 (48.6%)   240 (34.4%)
      bank x prior / f^1.5              433 (62.0%)   136 (19.5%)
    GTZAN (998, held out)
      shipped: autocorrelation x prior  700 (70.1%)   194 (19.4%)
      bank x prior / f^1.5              617 (61.8%)   216 (21.6%)
      bank x prior / f^2.0              648 (64.9%)   173 (17.3%)

Three things in that table, in order of how much they matter.

The nonlinearity is real. A bare bank almost never picks the half — 8 of 698
against the shipped estimator's 119 — so the octave symmetry genuinely is
broken. It is broken the wrong way: the bank prefers the subdivision, 232 of 698
at double the annotated tempo, because a faster oscillator is driven more often
per second and accumulates more amplitude for that reason alone.

Dividing the amplitude by f^p corrects that bias, and on Ballroom the corrected
bank beats what ships on octave errors, 136 against 174, at the same accuracy.
This is not the prior in disguise: pushing the shipped prior's centre down to
120 or 100 — the same direction f^-p pushes — makes Ballroom worse, 199 and 225.
So the bank carries information the autocorrelation does not.

And it still loses. The exponent that wins on Ballroom does not transfer: on
GTZAN, which it was not chosen on, f^1.5 is behind the shipped estimator on both
columns, and the f^2.0 that ties on octave errors gives up 52 recordings of
outright correctness to do it. A method that needs a different exponent per
corpus has not earned a default.

Kept as a documented negative rather than deleted, so the idea is not proposed
again with nothing to answer it.

Research code. It runs offline over a whole recording, in numpy, and was written
to be measured against the shipped estimator rather than shipped beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "OscillatorConfig",
    "OscillatorBank",
    "Resonance",
    "estimate_tempo",
]


@dataclass(frozen=True)
class OscillatorConfig:
    """The bank's shape.

    The tempo range matches ``TempoConfig`` deliberately: a comparison between
    two estimators that searched different ranges would be measuring the ranges.
    """

    min_bpm: float = 40.0
    max_bpm: float = 220.0
    count: int = 240
    # Damping. Negative so an oscillator with nothing to resonate to falls
    # silent instead of ringing forever and reporting itself as a tempo.
    alpha: float = -0.4
    # Saturation, and the reason this is not just a filter bank. Zero here would
    # make the whole model linear, and a linear bank has exactly the octave
    # symmetry the autocorrelation already has — there would be no experiment.
    beta: float = -1.0
    # Input gain. Large enough that a normalised onset function drives the bank
    # into saturation, which is where the nonlinearity does its work.
    drive: float = 1.0
    # Fraction of the recording discarded before amplitudes are read, so the
    # answer is the steady state and not the transient every oscillator shows
    # while it is filling up.
    settle: float = 0.25

    def validate(self) -> None:
        if not 0.0 < self.min_bpm < self.max_bpm:
            raise ValueError("bad tempo range")
        if self.count < 2:
            raise ValueError("a bank needs at least two oscillators")
        if self.alpha >= 0.0:
            raise ValueError("alpha must be negative, or the bank self-oscillates")
        if self.beta >= 0.0:
            raise ValueError("beta must be negative, or the amplitude diverges")
        if not 0.0 <= self.settle < 1.0:
            raise ValueError("settle must be in [0, 1)")


@dataclass
class Resonance:
    """What the bank ended up doing, per oscillator."""

    bpm: np.ndarray       # the bank's frequencies, in BPM
    amplitude: np.ndarray  # steady-state mean |z|, same length

    @property
    def peak_bpm(self) -> float:
        """The strongest oscillator's tempo, or zero when none resonated.

        The zero check is not defensive padding. argmax over an all-zero bank
        returns its first entry, so silence would be reported as the slowest
        tempo in the range — a confident answer with nothing behind it, which is
        the one output worse than admitting there is none.
        """
        if len(self.amplitude) == 0:
            return 0.0
        peak = float(np.max(self.amplitude))
        if not np.isfinite(peak) or peak <= 0.0:
            return 0.0
        return float(self.bpm[int(np.argmax(self.amplitude))])

    def normalised(self) -> np.ndarray:
        """Amplitudes scaled to a peak of one, or zeros if nothing resonated."""
        if len(self.amplitude) == 0:
            return self.amplitude
        peak = float(np.max(self.amplitude))
        if not np.isfinite(peak) or peak <= 0.0:
            return np.zeros_like(self.amplitude)
        return self.amplitude / peak


class OscillatorBank:
    """Integrates the bank over an onset function."""

    def __init__(self, config: OscillatorConfig = OscillatorConfig()):
        config.validate()
        self.config = config
        # Log-spaced, because tempo is heard in ratios: an even spacing in BPM
        # would crowd the fast end and starve the slow one.
        self.bpm = np.geomspace(config.min_bpm, config.max_bpm, config.count)
        self.omega = 2.0 * np.pi * self.bpm / 60.0

    def run(self, odf: np.ndarray, fps: float) -> Resonance:
        """Drive the bank with `odf` sampled at `fps` and read its amplitudes."""
        values = np.asarray(odf, dtype=np.float64).ravel()
        if values.size < 2 or not np.isfinite(fps) or fps <= 0.0:
            return Resonance(self.bpm, np.zeros(len(self.bpm)))

        # Normalised so `drive` means the same thing whatever the recording's
        # level. A constant onset function carries no rhythm and is refused
        # rather than divided by zero.
        spread = float(np.std(values))
        if not np.isfinite(spread) or spread <= 0.0:
            return Resonance(self.bpm, np.zeros(len(self.bpm)))
        x = (values - float(np.mean(values))) / spread

        dt = 1.0 / float(fps)
        alpha, beta, gain = self.config.alpha, self.config.beta, self.config.drive
        iw = 1j * self.omega

        # The fastest oscillator sets the step: omega*dt near or above one is
        # where a fixed-step integrator stops representing a rotation at all.
        # Substepping costs linearly and is cheaper than being wrong.
        substeps = max(1, int(np.ceil(float(np.max(self.omega)) * dt / 0.2)))
        h = dt / substeps

        z = np.zeros(len(self.bpm), dtype=np.complex128)
        start = int(len(x) * self.config.settle)
        total = np.zeros(len(self.bpm))
        counted = 0

        def derivative(state: np.ndarray, forcing: float) -> np.ndarray:
            return state * (alpha + iw + beta * np.abs(state) ** 2) + gain * forcing

        for n in range(len(x) - 1):
            # Linear interpolation across the substeps, so the drive is not a
            # staircase at the rate the oscillators are being asked to resolve.
            for s in range(substeps):
                u = (s + 0.5) / substeps
                forcing = x[n] * (1.0 - u) + x[n + 1] * u
                k1 = derivative(z, forcing)
                k2 = derivative(z + 0.5 * h * k1, forcing)
                k3 = derivative(z + 0.5 * h * k2, forcing)
                k4 = derivative(z + h * k3, forcing)
                z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if n >= start:
                total += np.abs(z)
                counted += 1

        if counted == 0:
            return Resonance(self.bpm, np.zeros(len(self.bpm)))
        amplitude = total / counted
        # A diverged oscillator is a bug in the integration, not a tempo. Report
        # nothing rather than a number that would win every argmax it entered.
        amplitude[~np.isfinite(amplitude)] = 0.0
        return Resonance(self.bpm, amplitude)


def estimate_tempo(odf: np.ndarray, fps: float,
                   config: OscillatorConfig = OscillatorConfig()) -> float:
    """The bank's peak, in BPM. Zero when nothing resonated."""
    return OscillatorBank(config).run(odf, fps).peak_bpm
