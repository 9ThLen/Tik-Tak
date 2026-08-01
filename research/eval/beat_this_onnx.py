"""Beat This! through ONNX Runtime, as a salience source for our resolver.

The model emits a per-frame downbeat activation and has no opinion about metre;
our C++ resolver turns that into a bar length and a phase. That division is the
whole point of the seam in ``core/src/analysis/downbeat.hpp``, and it is why
this file is short: everything downstream of the activation already exists.

**Every constant here was read out of a reference implementation, not guessed.**
The preprocessing is transcribed from `mosynthkey/beat_this_cpp`
(Source/MelSpectrogram.{h,cpp}, Source/InferenceProcessor.{h,cpp}), which is the
MIT-licensed C++ port that ships the converted ONNX. Getting any of it subtly
wrong produces an activation that still looks plausible and is quietly worse
than the model really is, so the awkward details are spelled out:

* Hann window is **periodic** — ``0.5*(1-cos(2*pi*i/N))``, dividing by N and not
  N-1. scipy's default is symmetric and would be a slow leak of accuracy.
* The mel filterbank uses the **Slaney** mel scale but **no** Slaney area
  normalisation: plain triangles, ``max(0, min(rising, falling))``.
* The spectrum is an **amplitude** (power=1), divided by ``sqrt(win_length)``.
* Compression is ``log1p(1000 * energy)``, not dB.
* Chunks are 1500 frames with a 6-frame border, and overlaps keep the **first**
  chunk's prediction, which is why the aggregation walks the chunks backwards.

Verified end to end against a click track of known tempo and metre — see
tests/test_beat_this_onnx.py. That check is what stands between this file and
a plausible-looking transcription error.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

__all__ = ["BeatThisOnnx", "MODEL_PATH", "FPS"]

SAMPLE_RATE = 22050
N_FFT = 1024
WIN_LENGTH = 1024
HOP = 441                      # 22050 / 441 = exactly 50 frames per second
FPS = SAMPLE_RATE / HOP
N_MELS = 128
F_MIN = 30.0
F_MAX = 11000.0
LOG_MULTIPLIER = 1000.0
AMIN = 1e-10
CHUNK_FRAMES = 1500
BORDER_FRAMES = 6

MODEL_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "models" / "beat_this.onnx"
)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = hz / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    high = hz >= min_log_hz
    mels = np.where(high, min_log_mel + np.log(np.maximum(hz, 1e-9) / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3.0
    freqs = f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    high = mel >= min_log_mel
    return np.where(high, min_log_hz * np.exp(logstep * (mel - min_log_mel)), freqs)


def mel_filterbank() -> np.ndarray:
    """(n_fft//2+1, n_mels) triangles on the Slaney scale, unnormalised."""
    mel_points = np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2)
    hz_points = _mel_to_hz(mel_points)
    freqs = np.arange(N_FFT // 2 + 1) * SAMPLE_RATE / N_FFT

    left = hz_points[:-2][None, :]
    centre = hz_points[1:-1][None, :]
    right = hz_points[2:][None, :]
    f = freqs[:, None]

    rising = np.divide(f - left, centre - left,
                       out=np.zeros_like(f + left), where=(centre - left) != 0)
    falling = np.divide(right - f, right - centre,
                        out=np.zeros_like(f + right), where=(right - centre) != 0)
    return np.maximum(0.0, np.minimum(rising, falling))


def _hann_periodic(size: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(size) / size))


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """(frames, n_mels), matching the reference port frame for frame."""
    audio = np.asarray(audio, dtype=np.float64).ravel()
    pad = N_FFT // 2
    if audio.size <= pad:
        return np.zeros((0, N_MELS), dtype=np.float32)

    # Reflection that does not repeat the edge sample, which is numpy's
    # "reflect" and what the reference does by hand.
    padded = np.concatenate([audio[pad:0:-1], audio, audio[-2:-pad - 2:-1]])

    frames = (padded.size - N_FFT) // HOP + 1
    if frames <= 0:
        return np.zeros((0, N_MELS), dtype=np.float32)

    window = _hann_periodic(WIN_LENGTH)
    indices = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    blocks = padded[indices] * window[None, :]

    spectrum = np.abs(np.fft.rfft(blocks, n=N_FFT, axis=1)) / np.sqrt(WIN_LENGTH)
    energy = spectrum @ mel_filterbank()
    return np.log1p(LOG_MULTIPLIER * np.maximum(energy, AMIN)).astype(np.float32)


def resample_to_model_rate(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    if abs(sample_rate - SAMPLE_RATE) < 1e-6:
        return np.asarray(audio, dtype=np.float64).ravel()
    from math import gcd
    from scipy.signal import resample_poly

    rate = int(round(sample_rate))
    divisor = gcd(rate, SAMPLE_RATE)
    return resample_poly(np.asarray(audio, dtype=np.float64).ravel(),
                         SAMPLE_RATE // divisor, rate // divisor)


@dataclass
class Activations:
    """Per-frame logits at FPS frames per second."""

    beat: np.ndarray
    downbeat: np.ndarray

    @property
    def frame_times(self) -> np.ndarray:
        return np.arange(len(self.beat)) / FPS

    def downbeat_probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.downbeat))

    def beat_probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.beat))


class BeatThisOnnx:
    """Loads the converted model once and runs whole pieces through it."""

    def __init__(self, model_path: pathlib.Path | str = MODEL_PATH):
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"{self.model_path} is missing. It is deliberately not in git — "
                f"see models/README.md."
            )
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"])

    def activations(self, audio: np.ndarray, sample_rate: float) -> Activations:
        spectrogram = log_mel_spectrogram(
            resample_to_model_rate(audio, sample_rate))
        total = len(spectrogram)
        if total == 0:
            return Activations(np.zeros(0), np.zeros(0))

        # Starts run from -border so the first real frame is never a chunk edge.
        # The last start is pulled back to cover the tail exactly once, which is
        # what makes keep-first aggregation total.
        step = CHUNK_FRAMES - 2 * BORDER_FRAMES
        starts = list(range(-BORDER_FRAMES, total - BORDER_FRAMES, step))
        if total > step:
            starts[-1] = total - (CHUNK_FRAMES - BORDER_FRAMES)

        beat = np.full(total, -1000.0, dtype=np.float64)
        downbeat = np.full(total, -1000.0, dtype=np.float64)

        chunks = []
        for start in starts:
            begin = max(start, 0)
            end = min(start + CHUNK_FRAMES, total)
            piece = spectrogram[begin:end]
            left = max(0, -start)
            right = max(0, min(BORDER_FRAMES, start + CHUNK_FRAMES - total))
            if left or right:
                piece = np.pad(piece, ((left, right), (0, 0)))
            chunks.append(piece)

        predictions = [
            self.session.run(["beat", "downbeat"],
                             {"input_spectrogram": chunk[None, ...].astype(np.float32)})
            for chunk in chunks
        ]

        # Backwards, so an earlier chunk overwrites a later one where they
        # overlap: the reference calls this "keep_first".
        for start, (beat_chunk, downbeat_chunk) in reversed(
                list(zip(starts, predictions))):
            b = np.asarray(beat_chunk).ravel()
            d = np.asarray(downbeat_chunk).ravel()
            if len(b) < 2 * BORDER_FRAMES:
                lo, hi = 0, len(b)
            else:
                lo, hi = BORDER_FRAMES, len(b) - BORDER_FRAMES
            for j in range(lo, hi):
                target = start + j
                if 0 <= target < total:
                    beat[target] = b[j]
                    downbeat[target] = d[j]

        return Activations(beat, downbeat)


def pick_peaks(logits: np.ndarray) -> np.ndarray:
    """Frame indices of the model's peaks, exactly as the reference port does.

    Max-pool with a width-7 window, keep frames that equal their pooled value
    and are positive (a logit above zero is a probability above a half), then
    drop peaks one frame apart. Transcribed rather than invented: a different
    peak picker would change every number downstream and make a comparison
    against published results meaningless.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return np.zeros(0, dtype=int)

    padded = np.pad(logits, 3, mode="constant", constant_values=-np.inf)
    pooled = np.max(np.lib.stride_tricks.sliding_window_view(padded, 7), axis=1)
    frames = np.flatnonzero((logits == pooled) & (logits > 0))
    if frames.size == 0:
        return frames
    keep = [frames[0]]
    for f in frames[1:]:
        if f - keep[-1] > 1:
            keep.append(f)
    return np.asarray(keep, dtype=int)


def beats_and_downbeats(activations: "Activations") -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times in seconds, downbeats snapped onto beats.

    Snapping matters: the two heads are independent, so a downbeat peak can land
    a frame off its beat, and an unsnapped reference would then score its own
    tracker's exact answer as a near miss.
    """
    beats = pick_peaks(activations.beat) / FPS
    downbeats = pick_peaks(activations.downbeat) / FPS
    if beats.size and downbeats.size:
        nearest = np.abs(downbeats[:, None] - beats[None, :]).argmin(axis=1)
        downbeats = np.unique(beats[nearest])
    return beats, downbeats
