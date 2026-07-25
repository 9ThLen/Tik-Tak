"""Synthetic test clips with exact, by-construction ground truth.

Public beat-tracking datasets need downloads and licences; this module gives us
material we fully control. The contract is: **beat times are generated first**
(analytically, from the tempo curve), and audio is then rendered at exactly
those times. Ground truth is therefore exact no matter what knobs are turned.

Main entry point: :func:`make_clip`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Clip", "beat_times", "make_clip", "make_sparse_clip"]


@dataclass
class Clip:
    audio: np.ndarray        # float32, mono, in [-1, 1]
    sample_rate: int
    beats: np.ndarray        # beat times, seconds, float64
    downbeats: np.ndarray    # subset of beats, seconds
    bpm: float               # nominal (starting) tempo
    beats_per_bar: int
    meta: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / self.sample_rate


# ---------------------------------------------------------------------------
# Ground-truth beat grid
# ---------------------------------------------------------------------------

def beat_times(
    bpm: float,
    duration_sec: float,
    tempo_drift: float = 0.0,
    offset_sec: float = 0.0,
) -> np.ndarray:
    """Beat times (seconds) for a tempo that ramps linearly over the clip.

    Instantaneous tempo at time t (measured from ``offset_sec``) is::

        bpm(t) = bpm + tempo_drift * t / duration_sec

    i.e. ``tempo_drift`` is the total BPM change from start to end of the
    clip (0 = constant tempo). Beat n falls where the integrated beat phase

        phi(t) = (bpm * t + tempo_drift * t**2 / (2 * duration_sec)) / 60

    equals n; for drift != 0 that quadratic is solved in closed form, so the
    grid is exact, not an Euler approximation.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if bpm + tempo_drift <= 0:
        raise ValueError("end tempo (bpm + tempo_drift) must stay positive")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    span = duration_sec - offset_sec
    if span <= 0:
        return np.empty(0, dtype=np.float64)

    # Total number of beats that fit: phi(span).
    total_phase = span * (bpm + tempo_drift * span / (2.0 * duration_sec)) / 60.0
    n = np.arange(int(np.floor(total_phase)) + 1, dtype=np.float64)

    if tempo_drift == 0.0:
        t = n * 60.0 / bpm
    else:
        # a * t^2 + b * t - n = 0, take the physical (positive, increasing) root.
        a = tempo_drift / (120.0 * duration_sec)
        b = bpm / 60.0
        t = (-b + np.sqrt(b * b + 4.0 * a * n)) / (2.0 * a)

    t = t + offset_sec
    return t[t < duration_sec - 1e-9]


# ---------------------------------------------------------------------------
# Percussion voices — short decaying bursts, not identical impulses
# ---------------------------------------------------------------------------

def _kick(sr: int, rng: np.random.Generator) -> np.ndarray:
    """Low kick: exponentially decaying sine with a downward pitch sweep."""
    dur = 0.12 * rng.uniform(0.9, 1.1)
    t = np.arange(int(dur * sr)) / sr
    f0 = rng.uniform(85.0, 110.0)
    f1 = rng.uniform(45.0, 55.0)
    freq = f1 + (f0 - f1) * np.exp(-t / 0.02)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t / 0.045)
    return (np.sin(phase) * env).astype(np.float64)


def _snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    """Mid snare: 200 Hz body tone plus band-limited noise."""
    dur = 0.10 * rng.uniform(0.9, 1.1)
    t = np.arange(int(dur * sr)) / sr
    body = 0.5 * np.sin(2.0 * np.pi * rng.uniform(180.0, 220.0) * t) * np.exp(-t / 0.03)
    noise = rng.standard_normal(len(t))
    # crude band-pass around the mids: difference of moving averages
    noise = noise - np.convolve(noise, np.ones(48) / 48.0, mode="same")
    noise = np.convolve(noise, np.ones(6) / 6.0, mode="same")
    return body + noise * np.exp(-t / 0.05)


def _hat(sr: int, rng: np.random.Generator, accent: float = 1.0) -> np.ndarray:
    """High hi-hat: high-passed noise with a fast decay."""
    dur = 0.04 * rng.uniform(0.8, 1.2)
    n = max(int(dur * sr), 8)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # first difference ~ high-pass
    t = np.arange(n) / sr
    return noise * np.exp(-t / 0.012) * accent


def _add_at(audio: np.ndarray, burst: np.ndarray, t_sec: float, sr: int, gain: float) -> None:
    i = int(round(t_sec * sr))
    if i >= len(audio) or i < 0:
        return
    j = min(i + len(burst), len(audio))
    audio[i:j] += gain * burst[: j - i]


# ---------------------------------------------------------------------------
# Clip generators
# ---------------------------------------------------------------------------

def make_clip(
    bpm: float = 120.0,
    beats_per_bar: int = 4,
    duration_sec: float = 20.0,
    sample_rate: int = 48000,
    seed: int = 0,
    tempo_drift: float = 0.0,
    swing: float = 0.0,
    noise_db: float | None = None,
    silence_lead: float = 0.0,
    subdivisions: int = 2,
    sparse: bool = False,
) -> Clip:
    """Render a synthetic rhythmic mix with exact ground-truth beats.

    Parameters
    ----------
    bpm : starting tempo.
    beats_per_bar : meter (downbeat every ``beats_per_bar`` beats).
    duration_sec : total clip length, including ``silence_lead``.
    sample_rate : output sample rate.
    seed : RNG seed; controls per-hit velocity/timbre variation and noise.
    tempo_drift : total BPM change linearly over the clip (e.g. +20 ramps
        120 -> 140). Ground-truth beats follow the ramp exactly.
    swing : fraction in [0, 1); the *subdivision* hits between beats are
        delayed by ``swing`` of a half-beat. Annotated beats do not move —
        swing shifts only the off-beat hi-hats, which is the musically
        meaningful (and adversarial) case.
    noise_db : if set, add white noise at this signal-to-noise ratio in dB
        (larger = cleaner; e.g. 10.0 is quite noisy).
    silence_lead : seconds of silence before the music starts; ground truth
        is shifted accordingly.
    subdivisions : hi-hat hits per beat (1 = beats only, 2 = eighths, ...).
    sparse : if True, render sustained tones changing on beats instead of
        percussion (stand-in for a bowed or sung line, no sharp onsets).
    """
    if not 0.0 <= swing < 1.0:
        raise ValueError("swing must be in [0, 1)")
    if silence_lead >= duration_sec:
        raise ValueError("silence_lead must be shorter than duration_sec")

    rng = np.random.default_rng(seed)
    sr = sample_rate
    n_samples = int(round(duration_sec * sr))
    audio = np.zeros(n_samples, dtype=np.float64)

    beats = beat_times(bpm, duration_sec, tempo_drift=tempo_drift, offset_sec=silence_lead)
    downbeats = beats[::beats_per_bar]

    if sparse:
        _render_sparse(audio, beats, sr, rng)
    else:
        _render_percussion(audio, beats, beats_per_bar, sr, rng, swing, subdivisions)

    # Normalize to a fixed peak, then add noise at the requested SNR.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio *= 0.9 / peak
    if noise_db is not None:
        sig_power = np.mean(audio**2)
        noise_power = sig_power / (10.0 ** (noise_db / 10.0))
        audio += rng.standard_normal(n_samples) * np.sqrt(noise_power)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio *= 0.99 / peak

    return Clip(
        audio=audio.astype(np.float32),
        sample_rate=sr,
        beats=beats.astype(np.float64),
        downbeats=downbeats.astype(np.float64),
        bpm=float(bpm),
        beats_per_bar=int(beats_per_bar),
        meta={
            "seed": seed,
            "tempo_drift": tempo_drift,
            "swing": swing,
            "noise_db": noise_db,
            "silence_lead": silence_lead,
            "subdivisions": subdivisions,
            "sparse": sparse,
        },
    )


def make_sparse_clip(**kwargs) -> Clip:
    """Convenience wrapper: ``make_clip(sparse=True, ...)``."""
    kwargs["sparse"] = True
    return make_clip(**kwargs)


def _render_percussion(
    audio: np.ndarray,
    beats: np.ndarray,
    beats_per_bar: int,
    sr: int,
    rng: np.random.Generator,
    swing: float,
    subdivisions: int,
) -> None:
    for k, t in enumerate(beats):
        pos = k % beats_per_bar
        vel = rng.uniform(0.8, 1.0)
        if pos == 0:
            _add_at(audio, _kick(sr, rng), t, sr, 1.0 * vel)
        elif pos % 2 == 1:  # backbeats (2 and 4 in 4/4)
            _add_at(audio, _snare(sr, rng), t, sr, 0.8 * vel)
        # hi-hat on every beat, accented on the beat itself
        _add_at(audio, _hat(sr, rng), t, sr, 0.5 * vel)

        # subdivision hi-hats between this beat and the next, softer,
        # optionally swung late; these are NOT annotated beats.
        if k + 1 < len(beats) and subdivisions > 1:
            ibi = beats[k + 1] - t
            for s in range(1, subdivisions):
                frac = s / subdivisions
                if subdivisions == 2 or s % 2 == 1:
                    frac += swing * (1.0 / subdivisions)
                _add_at(audio, _hat(sr, rng), t + frac * ibi, sr,
                        0.25 * rng.uniform(0.7, 1.0))


_PENTATONIC = np.array([220.0, 247.5, 277.2, 330.0, 370.0, 440.0])


def _render_sparse(
    audio: np.ndarray,
    beats: np.ndarray,
    sr: int,
    rng: np.random.Generator,
) -> None:
    """Sustained tones that change pitch on each beat — no percussion.

    Each note has a soft 15 ms attack and sustains (with slow decay) until
    just past the next beat, so the only rhythmic information is the pitch
    change and its gentle amplitude bump at the annotated beat times.
    """
    if len(beats) == 0:
        return
    end_times = np.append(beats[1:], min(beats[-1] + 1.0, len(audio) / sr))
    prev_idx = -1
    for t0, t1 in zip(beats, end_times):
        idx = int(rng.integers(0, len(_PENTATONIC)))
        if idx == prev_idx:  # force an audible change on every beat
            idx = (idx + 1) % len(_PENTATONIC)
        prev_idx = idx
        f = _PENTATONIC[idx] * rng.uniform(0.995, 1.005)
        dur = (t1 - t0) + 0.03
        n = int(dur * sr)
        tt = np.arange(n) / sr
        tone = np.sin(2.0 * np.pi * f * tt) + 0.3 * np.sin(2.0 * np.pi * 2 * f * tt)
        attack = np.minimum(tt / 0.015, 1.0)
        release = np.minimum((dur - tt) / 0.03, 1.0)
        env = attack * release * np.exp(-tt / 2.0)
        _add_at(audio, tone * env, t0, sr, 0.6 * rng.uniform(0.85, 1.0))
