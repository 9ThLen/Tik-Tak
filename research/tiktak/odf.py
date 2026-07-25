"""Onset detection function — reference implementation.

This mirrors `core/src/dsp/odf.cpp` step for step. It exists so the algorithm
can be developed and measured here, where a change costs seconds instead of an
Xcode build, and so the C++ can be checked against something known-good rather
than against intuition.

Any change here is a change to the specification. Keep the two in step, and use
`tools/parity/` to prove it rather than assuming it.

One deliberate difference: this works in float64 while the core works in
float32. The reference should be the more accurate of the two, so parity is
checked with a tolerance rather than bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Guards the whitening division in true digital silence.
_EPSILON = 1e-12

# The global peak that sets the relative floor decays more slowly than the
# per-band peaks: it stands for "how loud is this material", which should
# survive a rest, while per-band peaks track individual instruments.
_GLOBAL_TAU_MULTIPLIER = 4.0


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.exp(np.asarray(mel, dtype=np.float64) / 1127.0) - 1.0)


def hann_periodic(size: int) -> np.ndarray:
    """DFT-even Hann, matching `core/src/dsp/window.cpp`."""
    if size <= 1:
        return np.ones(size, dtype=np.float64)
    n = np.arange(size, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / size))


def mel_filterbank(
    fft_size: int,
    sample_rate: float,
    bands: int,
    min_hz: float,
    max_hz: float,
) -> np.ndarray:
    """Peak-normalised triangular filterbank, shape (bands, fft_size // 2 + 1).

    Peak- rather than area-normalised: the flux takes a log and then differences
    in time, so what matters is that a band is comparable with its own past, not
    that bands are comparable in absolute energy. Area normalisation would let
    wide high bands dominate purely by being wide.
    """
    nyquist = sample_rate * 0.5
    max_hz = min(max_hz, nyquist)
    if not (0.0 <= min_hz < max_hz):
        raise ValueError(f"bad filterbank range: {min_hz}..{max_hz} Hz")

    n_bins = fft_size // 2 + 1
    bin_to_hz = sample_rate / fft_size
    bin_hz = np.arange(n_bins, dtype=np.float64) * bin_to_hz

    # bands + 2 edges: filter b spans [edge[b], edge[b+2]] peaking at edge[b+1],
    # so neighbours overlap by half.
    edges = mel_to_hz(np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), bands + 2))

    weights = np.zeros((bands, n_bins), dtype=np.float64)
    for b in range(bands):
        left, centre, right = edges[b], edges[b + 1], edges[b + 2]

        if centre > left:
            rising = (bin_hz > left) & (bin_hz < centre)
            weights[b, rising] = (bin_hz[rising] - left) / (centre - left)
        if right > centre:
            falling = (bin_hz >= centre) & (bin_hz < right)
            weights[b, falling] = (right - bin_hz[falling]) / (right - centre)

        if not weights[b].any():
            # The triangle fell between two FFT bins — happens at the low end
            # with a short window. Snap it to the nearest bin so no band is
            # permanently silent.
            weights[b, min(n_bins - 1, int(round(centre / bin_to_hz)))] = 1.0

    return weights


@dataclass(frozen=True)
class OdfConfig:
    sample_rate: float = 48000.0
    frame_size: int = 2048
    hop_size: int = 512
    mel_bands: int = 81
    mel_min_hz: float = 27.5      # A0
    mel_max_hz: float = 16000.0   # clamped to Nyquist
    low_band_hz: float = 200.0    # kick and bass below here
    high_band_hz: float = 4000.0  # hi-hat and cymbals above here
    whitening: bool = True
    whitening_tau: float = 1.0
    whitening_floor_rel: float = 1e-3
    whitening_strength: float = 0.5

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop_size

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_size < 2 or self.frame_size & (self.frame_size - 1):
            raise ValueError("frame_size must be a power of two >= 2")
        if not 1 <= self.hop_size <= self.frame_size:
            raise ValueError("hop_size must be within the window")
        if self.mel_bands < 1:
            raise ValueError("mel_bands must be positive")
        if not 0.0 <= self.mel_min_hz < self.mel_max_hz:
            raise ValueError("bad mel range")
        if not 0.0 < self.low_band_hz < self.high_band_hz:
            raise ValueError("bad band split")
        if self.whitening:
            if self.whitening_tau <= 0.0:
                raise ValueError("whitening_tau must be positive")
            if not 0.0 < self.whitening_floor_rel < 1.0:
                raise ValueError("whitening_floor_rel must be in (0, 1)")
            if not 0.0 <= self.whitening_strength <= 1.0:
                raise ValueError("whitening_strength must be in [0, 1]")


@dataclass
class OdfResult:
    """Onset strengths, one row per hop.

    All three bands are the *mean* rise per mel band, not the sum: the high band
    spans several times more mel bands than the low one, so sums would make
    `high` structurally larger regardless of the audio.
    """

    times: np.ndarray  # window centres, seconds
    full: np.ndarray
    low: np.ndarray
    high: np.ndarray
    fps: float

    def __len__(self) -> int:
        return len(self.times)


def _stft_magnitude(samples: np.ndarray, config: OdfConfig) -> tuple[np.ndarray, np.ndarray]:
    """Framed magnitude spectra and the window-centre time of each frame."""
    frame, hop = config.frame_size, config.hop_size
    if len(samples) < frame:
        empty = np.zeros((0, frame // 2 + 1), dtype=np.float64)
        return empty, np.zeros(0, dtype=np.float64)

    n_frames = (len(samples) - frame) // hop + 1
    starts = np.arange(n_frames) * hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )

    windowed = frames * hann_periodic(frame)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1))

    # The frame describes its window's centre, so that is what it is stamped
    # with — a beat tracker compares these against predicted beat times, and
    # half a window of systematic bias would matter.
    times = (starts + frame * 0.5) / config.sample_rate
    return magnitude, times


def compute_odf(samples: np.ndarray, config: OdfConfig | None = None) -> OdfResult:
    """Half-wave rectified spectral flux over a whitened log mel spectrogram."""
    config = config or OdfConfig()
    config.validate()

    samples = np.ascontiguousarray(np.asarray(samples, dtype=np.float64).ravel())
    magnitude, times = _stft_magnitude(samples, config)

    filters = mel_filterbank(
        config.frame_size,
        config.sample_rate,
        config.mel_bands,
        config.mel_min_hz,
        config.mel_max_hz,
    )
    centres = mel_to_hz(
        np.linspace(
            hz_to_mel(config.mel_min_hz),
            hz_to_mel(min(config.mel_max_hz, config.sample_rate * 0.5)),
            config.mel_bands + 2,
        )
    )[1:-1]

    bands = config.mel_bands
    low_split = int(np.clip(np.searchsorted(centres, config.low_band_hz), 1, bands))
    high_split = int(np.clip(np.searchsorted(centres, config.high_band_hz), 0, bands - 1))

    mel = magnitude @ filters.T  # (frames, bands)

    if config.whitening:
        mel = _whiten(mel, config)

    log_mel = np.log1p(mel)

    n = len(times)
    full = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)

    if n >= 2:
        # Half-wave rectified: only rising energy is an onset. Falling energy is
        # a note ending, which is not a rhythmic event we want. Frame 0 has no
        # predecessor and stays zero rather than inventing an onset at t=0.
        rise = np.diff(log_mel, axis=0)
        np.maximum(rise, 0.0, out=rise)

        full[1:] = rise.mean(axis=1)
        low[1:] = rise[:, :low_split].mean(axis=1)
        high[1:] = rise[:, high_split:].mean(axis=1)

    return OdfResult(times=times, full=full, low=low, high=high, fps=config.fps)


def _whiten(mel: np.ndarray, config: OdfConfig) -> np.ndarray:
    """Adaptive whitening (Stowell & Plumbley 2007) with a relative floor.

    Two departures from the textbook version, both load-bearing:

    * The floor is a fraction of the loudest band seen recently, not an absolute
      constant. With an absolute floor a band picking up nothing but spectral
      leakage gets divided by its own tiny peak and normalises straight to full
      scale, so after a silent passage every empty band reports a full-strength
      onset.

    * `whitening_strength` is an exponent. At 1.0 — the textbook value — every
      band on a rising edge lands on exactly full scale, which makes level
      invariance perfect and erases the balance between bands. The low/high
      bands exist to carry exactly that balance, so the two properties trade off
      against each other. See docs/PLAN.md; the operating point is a tuning
      question, not a correctness one.
    """
    frames, bands = mel.shape
    decay = np.exp(-1.0 / (config.whitening_tau * config.fps))
    global_decay = np.exp(-1.0 / (config.whitening_tau * _GLOBAL_TAU_MULTIPLIER * config.fps))

    out = np.empty_like(mel)
    band_peak = np.zeros(bands, dtype=np.float64)
    global_peak = 0.0

    # Sequential by nature: each frame's normaliser depends on the previous one.
    for i in range(frames):
        row = mel[i]
        global_peak = max(row.max(initial=0.0), global_peak * global_decay)
        floor = global_peak * config.whitening_floor_rel

        decayed = band_peak * decay
        divisor = np.maximum(np.maximum(row, decayed), max(floor, _EPSILON))
        band_peak = np.maximum(row, decayed)

        out[i] = row / divisor**config.whitening_strength

    return out
