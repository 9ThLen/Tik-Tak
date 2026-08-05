"""Tempo estimation from an onset detection function.

Autocorrelation of the ODF, weighted by a log-normal prior over tempo. The
prior is not cosmetic: autocorrelation peaks just as happily at half and double
the true period, and without a prior the estimate flips octaves between
neighbouring windows on perfectly ordinary music. It is the single most common
failure mode of beat trackers.

The public result carries the whole posterior, not just its peak, because the
beat tracker downstream needs to know when two tempi are nearly tied — that is
the difference between "confident" and "guessing", and the UI is supposed to
show it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TempoConfig:
    min_bpm: float = 40.0
    max_bpm: float = 220.0
    # 140, not the 120 this carried before, and the difference was measured
    # rather than chosen. Taken alone a log-normal prior centred at c prefers
    # t/2 over t once t > c*sqrt(2) — the two are equidistant in log2 when
    # log2(t/c) = 1/2 — so the pull starts at 170 BPM for a centre of 120 and at
    # 198 for 140. That single fact was the tracker's largest failure mode: over
    # 698 ballroom recordings it landed on exactly half the annotated tempo on
    # 186 of them, and the 120 prior independently prefers the half on 184 of
    # the same 698. Cause, not correlation.
    #
    # Tuned on ballroom, validated on GTZAN, which it was not tuned on:
    #
    #             ballroom (698)              GTZAN (999, held out)
    #     centre  F      CMLt   octave        F      CMLt   octave
    #     120     0.746  0.553  18.8%         0.769  0.628  14.8%
    #     140     0.763  0.579  17.0%         0.782  0.649  14.6%
    #     150     0.772  0.583  16.2%         0.779  0.619  17.5%
    #
    # 140 is the only point that improves both. 150 buys more on ballroom and
    # gives it back on GTZAN, which is what fitting one corpus looks like.
    #
    # The move is close to zero sum: two thirds of the half errors it removes
    # come back as double errors on slower material, and the octave failure
    # survives at 17%. A global prior can only choose where the half/double
    # crossover sits, never tell fast music from slow.
    #
    # The core is the other half of this number. It carried 140 for a month
    # while this file still said 120, because the re-centring commit changed
    # core/src/analysis/tempo.hpp and never came back here — and the parity
    # gate that exists to catch exactly that had never run in CI. The two
    # centres must move together or the reference stops being one.
    prior_centre_bpm: float = 140.0
    # Standard deviation of the prior in octaves. Wide enough not to fight real
    # music, narrow enough to break the octave tie.
    #
    # Left at 0.7. Narrowing to 0.6 was the best point on ballroom and lost on
    # GTZAN, so it is corpus fitting. Widening is worse everywhere: at 1.5 CMLt
    # falls to 0.390 and at 3.0 to 0.142 — the prior carries real weight, it was
    # merely aimed wrong.
    prior_width_octaves: float = 0.7
    grid_size: int = 512

    # A candidate period can be scored by a comb: its own autocorrelation plus
    # that of its multiples, on the theory that a real beat period is supported
    # at every metrical level above it while a spurious peak is not.
    #
    # Off by default (1 = score each period by its own lag alone), because
    # measurement disagreed with that theory. Over 140 synthetic clips spanning
    # 60-196 BPM the comb was worse on every metric:
    #
    #     harmonics=1   F 0.900   CMLt 0.702   AMLt 0.991   non-metrical 0/140
    #     harmonics=3   F 0.889   CMLt 0.681   AMLt 0.977   non-metrical 2/140
    #     harmonics=4   F 0.880   CMLt 0.660   AMLt 0.970   non-metrical 4/140
    #
    # and it introduced the very error it was supposed to remove: candidates at
    # two-thirds of the true period collect full support from every third
    # multiple. Restricting the comb to powers of two does not rescue it
    # (F 0.874); more levels is monotonically worse.
    #
    # Kept as a parameter rather than deleted: comb scoring is standard in the
    # literature (Klapuri, Davies) and this evidence is entirely synthetic, from
    # one generator. Re-measure on real annotated audio before concluding it is
    # useless in general. See docs/PLAN.md section 10.
    comb_harmonics: int = 1
    # Weight of harmonic k is k**-comb_weight_decay. Higher metrical levels
    # carry real but weaker evidence, so they should count for less.
    comb_weight_decay: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.min_bpm < self.max_bpm:
            raise ValueError("bad bpm range")
        if self.prior_centre_bpm <= 0.0:
            raise ValueError("prior_centre_bpm must be positive")
        if self.prior_width_octaves <= 0.0:
            raise ValueError("prior_width_octaves must be positive")
        if self.grid_size < 8:
            raise ValueError("grid_size too small to resolve anything")
        if self.comb_harmonics < 1:
            raise ValueError("comb_harmonics must be at least 1")
        if self.comb_weight_decay < 0.0:
            raise ValueError("comb_weight_decay must be non-negative")


@dataclass
class TempoEstimate:
    bpm: float
    confidence: float       # 0..1; peak sharpness against the rest of the grid
    bpm_grid: np.ndarray    # log-spaced candidate tempi
    posterior: np.ndarray   # prior-weighted autocorrelation, peak-normalised

    def top_candidates(self, count: int = 3, min_separation_octaves: float = 0.2):
        """Distinct peaks, strongest first, as (bpm, strength) pairs.

        Used to spot octave ambiguity: if the runner-up sits at half or double
        the winner with a similar score, the estimate is a coin toss and the
        caller should say so rather than commit.
        """
        order = np.argsort(self.posterior)[::-1]
        chosen: list[tuple[float, float]] = []
        for i in order:
            bpm = float(self.bpm_grid[i])
            if any(
                abs(np.log2(bpm / other)) < min_separation_octaves for other, _ in chosen
            ):
                continue
            chosen.append((bpm, float(self.posterior[i])))
            if len(chosen) == count:
                break
        return chosen


def autocorrelation(values: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation for non-negative lags, via FFT."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.zeros(0)

    # Mean removal matters: the ODF is non-negative, so its DC component would
    # otherwise swamp every real periodicity.
    centred = values - values.mean()

    size = 1 << int(np.ceil(np.log2(max(2 * n - 1, 2))))
    spectrum = np.fft.rfft(centred, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]

    # Unbiased: lag k is the average of only n-k products, so without this the
    # tail sags and slow tempi are penalised for no musical reason.
    counts = np.arange(n, 0, -1, dtype=np.float64)
    return acf / counts


def comb_score(
    acf: np.ndarray,
    bpm_grid: np.ndarray,
    fps: float,
    config: TempoConfig,
) -> np.ndarray:
    """Score each candidate tempo by a comb over its metrical multiples.

    The idea is that summing over multiples requires a candidate to be
    supported at every metrical level above it, which a spurious peak at
    two-thirds or three-halves of the beat is not.

    In practice it did not survive measurement and is disabled by default; see
    TempoConfig.comb_harmonics for the numbers. With `comb_harmonics == 1` this
    reduces to interpolating the autocorrelation at each candidate's own lag.
    """
    lags = 60.0 * fps / bpm_grid
    index = np.arange(len(acf))

    total = np.zeros_like(bpm_grid)
    weight_total = np.zeros_like(bpm_grid)

    for k in range(1, config.comb_harmonics + 1):
        harmonic_lags = lags * k
        usable = (harmonic_lags >= 1.0) & (harmonic_lags < len(acf) - 1)
        if not usable.any():
            break

        weight = float(k) ** -config.comb_weight_decay
        strength = np.interp(harmonic_lags[usable], index, acf)
        total[usable] += weight * np.maximum(strength, 0.0)
        weight_total[usable] += weight

    # Normalise by the weight actually used: slow tempi have fewer multiples
    # inside the analysed span, and would otherwise be penalised for the length
    # of the recording rather than for anything musical.
    return np.divide(total, weight_total, out=np.zeros_like(total), where=weight_total > 0.0)


def tempo_prior(bpm_grid: np.ndarray, config: TempoConfig) -> np.ndarray:
    octaves = np.log2(bpm_grid / config.prior_centre_bpm)
    return np.exp(-0.5 * (octaves / config.prior_width_octaves) ** 2)


def estimate_tempo(
    odf_values: np.ndarray,
    fps: float,
    config: TempoConfig | None = None,
) -> TempoEstimate:
    """Single global tempo estimate over the whole ODF."""
    config = config or TempoConfig()
    config.validate()
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    odf_values = np.asarray(odf_values, dtype=np.float64)
    bpm_grid = np.geomspace(config.min_bpm, config.max_bpm, config.grid_size)

    acf = autocorrelation(odf_values)
    posterior = np.zeros_like(bpm_grid)

    if len(acf) > 2 and np.any(odf_values > 0.0):
        posterior = comb_score(acf, bpm_grid, fps, config)
        posterior *= tempo_prior(bpm_grid, config)

    peak = posterior.max(initial=0.0)
    if peak <= 0.0:
        return TempoEstimate(
            bpm=float(config.prior_centre_bpm),
            confidence=0.0,
            bpm_grid=bpm_grid,
            posterior=posterior,
        )

    posterior = posterior / peak
    best = int(np.argmax(posterior))

    # Confidence is how periodic the signal actually is at the chosen tempo:
    # the autocorrelation there as a fraction of the autocorrelation at lag
    # zero, which is the signal's variance. 1.0 means the ODF repeats exactly
    # at that period, 0.0 means it does not repeat at all.
    #
    # This replaced a "peak sharpness against the rest of the grid" measure that
    # was measuring the wrong thing: white noise produces a sharp, meaningless
    # peak and scored 0.76 by it — *higher* than a clean beat at 0.84 in some
    # cases — which is exactly backwards for a UI whose job is to distinguish
    # "sure" from "guessing". The same cases score 0.02 and 0.71 here.
    confidence = 0.0
    if acf[0] > 0.0:
        lag = 60.0 * fps / float(bpm_grid[best])
        strength = float(np.interp(lag, np.arange(len(acf)), acf))
        confidence = float(np.clip(strength / acf[0], 0.0, 1.0))

    return TempoEstimate(
        bpm=float(bpm_grid[best]),
        confidence=confidence,
        bpm_grid=bpm_grid,
        posterior=posterior,
    )


def estimate_tempo_windowed(
    odf_values: np.ndarray,
    fps: float,
    window_sec: float = 8.0,
    hop_sec: float = 1.0,
    config: TempoConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tempo over time: (centre_times, bpm, confidence).

    The online tracker sees only a rolling window, so this is what its tempo
    input actually looks like — including the lag before it notices a change.
    """
    config = config or TempoConfig()
    odf_values = np.asarray(odf_values, dtype=np.float64)

    window = max(2, int(round(window_sec * fps)))
    hop = max(1, int(round(hop_sec * fps)))
    if len(odf_values) < window:
        estimate = estimate_tempo(odf_values, fps, config)
        centre = len(odf_values) / (2.0 * fps)
        return (
            np.array([centre]),
            np.array([estimate.bpm]),
            np.array([estimate.confidence]),
        )

    starts = np.arange(0, len(odf_values) - window + 1, hop)
    times = (starts + window / 2.0) / fps
    bpms = np.zeros(len(starts))
    confidences = np.zeros(len(starts))

    for i, start in enumerate(starts):
        estimate = estimate_tempo(odf_values[start : start + window], fps, config)
        bpms[i] = estimate.bpm
        confidences[i] = estimate.confidence

    return times, bpms, confidences
