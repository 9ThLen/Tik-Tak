#!/usr/bin/env python3
"""Bar-level salience from an amplitude-modulation hierarchy.

The idea is borrowed from speech, where the amplitude envelope carries a
three-level modulation hierarchy — stress around 2 Hz, syllables around 5 Hz,
phonemes around 20 Hz — and band-passing the envelope at those rates parses
speech into its units without knowing any words. The musical analogue is
bar / beat / tatum, and the level this file wants is the slowest of the three:
if a recording modulates its amplitude once per bar, a band-pass at the bar rate
should be largest at the downbeat.

What it produces is one number per beat, which is exactly what
``core/src/analysis/downbeat.hpp`` takes through the ``--salience`` seam. So this
is a third backend beside the built-in cues and the Beat This! activation, run
through the same resolver and scored by the same benchmark, and the three
columns are directly comparable. Anything less than that would not settle
whether the idea works.

The bar band is not a fixed frequency. It is derived from the beat grid the
caller already has: bars are some small integer number of beats, so the bar rate
lies between the beat rate over eight and the beat rate over two. Choosing which
integer is the resolver's job and this file does not pre-empt it.

**Measured, and it does not work.** Over 698 annotated Ballroom recordings,
scored through the same resolver and the same benchmark as the other two
backends:

    backend            downbeat F   median   accented   wrong
    cues                    0.487    0.427    304/697     108
    Beat This!              0.702    0.929    388/697       2
    am_hierarchy (onset)    0.362    0.143      0/697       0
    am_hierarchy (envelope) 0.280    0.000      1/697       0

Worse than the built-in cues on both readings of the envelope, and it almost
never clears its own confidence gate — the zeros in the last column are
abstention, not accuracy. Beat F is identical in every row because the grid is;
only the salience differs, which is what makes the comparison a controlled one.

Kept rather than deleted, for the same reason the comb scoring in
``analysis/tempo.hpp`` was kept: the idea is well founded elsewhere, the
implementation is small, and a documented negative result is what stops it being
proposed again with nothing to answer it. See AmConfig.source for the part of
this worth reading twice — the mechanism does work on synthetic audio with the
modulation built in, and that is exactly why the corpus result matters.

Research code, and deliberately not in the core: the point was to find out
whether the number is worth anything before any of it ships. It is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiktak.odf import OdfConfig, compute_odf

__all__ = [
    "AmConfig",
    "bar_band_envelope",
    "bar_salience",
]


@dataclass(frozen=True)
class AmConfig:
    """Where the bar band sits, relative to the beat rate.

    Ratios rather than hertz: a bar is a count of beats, so the band that could
    hold it is fixed relative to the beat rate and not to the clock. The bounds
    cover everything from a two-beat bar to an eight-beat one, which is wider
    than the metres the resolver will consider and is meant to be — narrowing it
    to the answer would be assuming the answer.
    """

    lowest_bar_in_beats: float = 8.0
    highest_bar_in_beats: float = 2.0
    # Which signal the band-pass runs on. "envelope" is what the speech model
    # does — the amplitude itself, whose slow modulation is the hierarchy —
    # and "onset" runs on the onset function, which is already a rectified
    # first difference and so attenuates the very modulation being looked for.
    #
    # On synthetic audio with a bar line built in as a louder click, "envelope"
    # is clearly the better of the two: it separates downbeats from other beats
    # by 0.64 against 0.43. On 698 annotated Ballroom recordings it is the worse
    # of the two, downbeat F 0.280 against 0.362. Both are far below the built-
    # in cues at 0.487.
    #
    # That reversal is the finding, and it is worth more than either number:
    # the mechanism works when the modulation is there by construction, and real
    # dance music does not carry the bar at the amplitude's bar rate. "onset"
    # is the default only because it is the less bad of two options that lose.
    source: str = "onset"
    # Envelope compression before the band-pass. A bar line is often marked by a
    # loud kick, and without this the band is dominated by whichever bar happens
    # to be loudest rather than by the periodicity.
    compress: bool = True
    # The three bands compute_odf already separates. Weighted equally: the
    # question is whether bar-rate modulation exists at all, and picking weights
    # per band on the same corpus the result is read from would be fitting.
    use_bands: tuple[str, ...] = ("low", "full", "high")

    def validate(self) -> None:
        if not 0.0 < self.highest_bar_in_beats < self.lowest_bar_in_beats:
            raise ValueError("bad bar band")
        if not self.use_bands:
            raise ValueError("at least one band is needed")
        if self.source not in ("envelope", "onset"):
            raise ValueError("source must be 'envelope' or 'onset'")


def _band_pass(signal: np.ndarray, fps: float,
               low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase band-pass, done in the frequency domain.

    An FFT mask rather than a designed IIR filter, for one reason that matters
    here: the bar rate is well under 1 Hz against a frame rate near 100, and an
    IIR at that ratio is numerically delicate and would need care that would
    itself become a thing to verify. Zero phase is not a nicety either — a phase
    shift at the bar rate is a shift of a fraction of a bar, which is precisely
    the quantity being measured.
    """
    n = len(signal)
    if n < 4:
        return np.zeros(n)
    spectrum = np.fft.rfft(signal - float(np.mean(signal)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return np.zeros(n)
    spectrum = np.where(mask, spectrum, 0.0)
    return np.fft.irfft(spectrum, n=n)


def bar_band_envelope(audio: np.ndarray, sample_rate: float,
                      beat_period_sec: float,
                      config: AmConfig = AmConfig()) -> tuple[np.ndarray, np.ndarray]:
    """The bar-rate modulation of the envelope, and its frame times.

    Returns the summed band-pass response across the onset function's spectral
    bands, rectified: only the positive half is evidence for "louder than the
    local average here", and the negative half is evidence for the opposite,
    which is not the same thing as evidence for a bar line elsewhere.
    """
    config.validate()
    if not np.isfinite(beat_period_sec) or beat_period_sec <= 0.0:
        return np.zeros(0), np.zeros(0)

    odf = compute_odf(np.asarray(audio, dtype=np.float64).ravel(),
                      OdfConfig(sample_rate=float(sample_rate),
                                mel_max_hz=min(16000.0, sample_rate * 0.5)))
    if len(odf) < 4:
        return np.zeros(0), np.zeros(0)

    beat_hz = 1.0 / beat_period_sec
    low_hz = beat_hz / config.lowest_bar_in_beats
    high_hz = beat_hz / config.highest_bar_in_beats

    if config.source == "envelope":
        bands = _band_envelopes(np.asarray(audio, dtype=np.float64).ravel(),
                                float(sample_rate), odf.times, config)
    else:
        bands = [np.asarray(getattr(odf, name), dtype=np.float64)
                 for name in config.use_bands]

    combined = np.zeros(len(odf.times))
    for band in bands:
        if config.compress:
            band = np.log1p(np.maximum(band, 0.0))
        combined += _band_pass(band, odf.fps, low_hz, high_hz)

    return np.maximum(combined, 0.0), odf.times


def _band_envelopes(audio: np.ndarray, sample_rate: float,
                    times: np.ndarray, config: AmConfig) -> list[np.ndarray]:
    """Amplitude envelopes in the same three bands, sampled at `times`.

    Not the onset function: this is the magnitude itself, smoothed, which is the
    quantity the speech hierarchy modulates. The band edges are the ones
    OdfConfig already splits at, so "low" means the same thing in both sources.
    """
    settings = OdfConfig(sample_rate=sample_rate,
                         mel_max_hz=min(16000.0, sample_rate * 0.5))
    frame, hop = settings.frame_size, settings.hop_size
    window = np.hanning(frame)
    edges = (settings.low_band_hz, settings.high_band_hz)

    count = len(times)
    out = [np.zeros(count) for _ in config.use_bands]
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    masks = {
        "low": freqs < edges[0],
        "high": freqs >= edges[1],
        "full": np.ones(len(freqs), dtype=bool),
    }
    for i in range(count):
        start = i * hop
        chunk = audio[start:start + frame]
        if len(chunk) < frame:
            break
        spectrum = np.abs(np.fft.rfft(chunk * window))
        for j, name in enumerate(config.use_bands):
            mask = masks.get(name, masks["full"])
            out[j][i] = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0
    return out


def bar_salience(audio: np.ndarray, sample_rate: float,
                 beat_times: np.ndarray,
                 config: AmConfig = AmConfig()) -> np.ndarray:
    """One value per beat, scaled to [0, 1].

    Scaled per recording because the resolver's calibration is a threshold on
    these numbers, and an unbounded unit would make one threshold mean different
    things on a loud recording and a quiet one.
    """
    beats = np.asarray(beat_times, dtype=np.float64).ravel()
    if beats.size == 0:
        return np.zeros(0)
    if beats.size < 2:
        return np.zeros(len(beats))

    period = float(np.median(np.diff(beats)))
    envelope, times = bar_band_envelope(audio, sample_rate, period, config)
    if len(envelope) == 0:
        return np.zeros(len(beats))

    # Sampled the same way the model backend is sampled — the peak inside a
    # window centred on the beat — so the two columns differ by the activation
    # and not by how it was read.
    from eval.backends import sample_at_beats
    values = sample_at_beats(envelope, times, beats)

    span = float(np.max(values)) - float(np.min(values))
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros(len(beats))
    return (values - float(np.min(values))) / span
