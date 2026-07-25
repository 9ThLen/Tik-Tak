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
    prior_centre_bpm: float = 120.0
    # Standard deviation of the prior in octaves. Wide enough not to fight real
    # music, narrow enough to break the octave tie.
    prior_width_octaves: float = 0.7
    grid_size: int = 512

    # A candidate period is scored by a comb: its own autocorrelation plus that
    # of its multiples. A real beat period is supported at every metrical level
    # above it, while a spurious peak at, say, two-thirds of the true period is
    # supported at none — those are the errors this removes outright. It does
    # *not* resolve the octave question, which is genuine ambiguity, not noise.
    comb_harmonics: int = 4
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

    Scoring a period by its own autocorrelation alone picks up peaks that are
    not metrical at all — with a kick/snare pattern the strongest peak often
    sits at two-thirds or three-halves of the beat, where unlike events happen
    to line up. Summing over multiples requires a candidate to be supported at
    every level above it, which those peaks are not.
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

    # Confidence as peak sharpness: how far the winner stands above the typical
    # candidate. A flat posterior means "no periodicity found", not "120 BPM".
    median = float(np.median(posterior[posterior > 0.0]))
    confidence = float(np.clip(1.0 - median, 0.0, 1.0))

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
