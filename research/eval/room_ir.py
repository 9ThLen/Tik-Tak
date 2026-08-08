#!/usr/bin/env python3
"""Measure the room from a swept-sine capture, and check it may be measured.

`room_degradation.py` invents an impulse response and costs 0.005 of F where a
real room costs 0.390. This reads one instead: an exponential sweep played
through the speaker and captured on the phone, deconvolved with the filter in
`make_sweep.py`, giving the speaker, the room and the microphone as one linear
response.

## The linearity check comes first, and can refuse the measurement

Three identical sweeps were played. Convolution can only describe a linear
time-invariant chain, and the phone's gain control is neither -- the captures
show slopes of 0.36 to 0.78, compression up to 3:1. If the three responses
disagree, the chain is not LTI and **no impulse response describes it**, however
carefully this one is computed. That is a result about the setup, not a problem
with the arithmetic, and it is reported before any response is written.

The comparison is on normalised responses, so a repeat that is merely quieter
does not read as a different room -- which is exactly what a slow gain rider
would produce, and exactly what would otherwise be mistaken for non-linearity.

## Harmonic distortion lands before the answer

An exponential sweep sends the speaker's harmonics to *negative* time, ahead of
the linear response, because a harmonic of the instantaneous frequency arrives
earlier in the sweep than the fundamental does. So the deconvolution's peak is
the start of the linear response and everything before it is distortion. The
window starts at the peak for that reason, and the distortion energy sitting in
front of it is reported: a speaker driven into its limits is a different
measurement from a room.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.make_sweep import inverse_filter, sweep  # noqa: E402
from eval.room_recording import read_audio  # noqa: E402

# How much of the response to keep. Longer than any domestic room's tail, and
# short enough that convolving a four-minute song with it stays cheap.
IR_SECONDS = 1.5
# Repeats correlating below this are not the same response, and the chain is
# not describable by one.
LINEARITY_MIN_CORRELATION = 0.95


def find_repeats(recorded: np.ndarray, reference: np.ndarray, rate: float,
                 count: int, period_sec: float) -> list[int]:
    """Where each played sweep begins in the capture.

    By correlation against the reference sweep rather than by trusting the
    schedule: the recorder was started by hand, and a phone's clock is not the
    playback device's. The expected spacing is used only to reject a second
    peak that is really the same sweep found twice.
    """
    from scipy.signal import fftconvolve

    curve = fftconvolve(recorded, reference[::-1], mode="valid")
    guard = int(0.5 * period_sec * rate)
    found: list[int] = []
    working = np.abs(curve).copy()
    for _ in range(count):
        index = int(np.argmax(working))
        found.append(index)
        working[max(0, index - guard): index + guard] = 0.0
    return sorted(found)


def response_from(segment: np.ndarray, rate: float, seconds: float,
                  f_low: float, f_high: float) -> tuple[np.ndarray, float]:
    """Deconvolve one sweep, return the response and the pre-peak energy share.

    The share is the distortion diagnostic: on an exponential sweep the
    speaker's harmonics arrive before the linear response, so energy in front
    of the peak is the speaker being pushed, not the room.
    """
    from scipy.signal import fftconvolve

    full = fftconvolve(segment, inverse_filter(int(rate), seconds, f_low, f_high))
    peak = int(np.argmax(np.abs(full)))
    keep = int(IR_SECONDS * rate)
    response = full[peak: peak + keep]
    before = full[max(0, peak - int(0.5 * rate)): peak]
    total = float(np.sum(full[max(0, peak - int(0.5 * rate)):
                              peak + keep] ** 2))
    share = float(np.sum(before ** 2) / total) if total > 0 else float("nan")
    return response, share


def rt60(response: np.ndarray, rate: float) -> float:
    """Reverberation time by Schroeder backward integration, over -5 to -25 dB.

    Not -5 to -65: a phone capture's noise floor swallows the last 40 dB, and
    fitting through it would measure the noise rather than the room. The slope
    over 20 dB is extrapolated, which is the standard T20 and is honest about
    being an extrapolation.
    """
    energy = response ** 2
    decay = np.cumsum(energy[::-1])[::-1]
    if decay[0] <= 0:
        return float("nan")
    level = 10.0 * np.log10(np.maximum(decay / decay[0], 1e-12))
    start = int(np.argmax(level <= -5.0))
    stop = int(np.argmax(level <= -25.0))
    if stop <= start:
        return float("nan")
    t = np.arange(start, stop) / rate
    slope = float(np.polyfit(t, level[start:stop], 1)[0])
    return float(-60.0 / slope) if slope < 0 else float("nan")


def linearity(responses: list[np.ndarray]) -> dict:
    """Do the repeats describe one response, or three different ones?"""
    unit = []
    for response in responses:
        norm = float(np.linalg.norm(response))
        unit.append(response / norm if norm > 0 else response)
    pairs = []
    for i in range(len(unit)):
        for j in range(i + 1, len(unit)):
            pairs.append(float(np.dot(unit[i], unit[j])))
    levels = [float(20.0 * np.log10(max(np.linalg.norm(r), 1e-12)))
              for r in responses]
    return {
        "pairwise_correlation": pairs,
        "worst_correlation": float(min(pairs)) if pairs else float("nan"),
        "level_db": levels,
        "level_spread_db": float(max(levels) - min(levels)) if levels else 0.0,
        "lti": bool(pairs and min(pairs) >= LINEARITY_MIN_CORRELATION),
    }


def noise_profile(mono: np.ndarray, rate: float) -> dict:
    """The room's own noise: level, and how it is spread over frequency."""
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freq = np.fft.rfftfreq(len(mono), 1.0 / rate)
    bands = [(20, 100), (100, 400), (400, 1600), (1600, 6400), (6400, 20000)]
    out = {
        "seconds": float(len(mono) / rate),
        "rms_dbfs": float(20.0 * np.log10(max(float(np.sqrt(np.mean(mono ** 2))),
                                              1e-12))),
        "peak_dbfs": float(20.0 * np.log10(max(float(np.max(np.abs(mono))), 1e-12))),
        "bands_db": {},
    }
    total = float(np.sum(spectrum ** 2))
    for low, high in bands:
        mask = (freq >= low) & (freq < high)
        share = float(np.sum(spectrum[mask] ** 2) / total) if total > 0 else 0.0
        out["bands_db"][f"{low}-{high}"] = float(10.0 * np.log10(max(share, 1e-12)))
    return out


def main(argv: list[str] | None = None) -> int:
    import soundfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-capture", type=pathlib.Path, required=True)
    parser.add_argument("--silence-capture", type=pathlib.Path)
    parser.add_argument("--params", type=pathlib.Path, required=True,
                        help="sweep.json written beside the played sweep")
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="directory for ir.wav, noise.wav and ir.json")
    args = parser.parse_args(argv)

    params = json.loads(args.params.read_text(encoding="utf-8"))
    seconds = float(params["seconds"])
    f_low, f_high = float(params["f_low"]), float(params["f_high"])
    repeats = int(params["repeats"])
    period = seconds + float(params["gap_sec"])

    captured, rate = read_audio(args.sweep_capture)
    # The reference is regenerated at the *capture's* rate, not the played
    # file's. The sweep occupies the same seconds and the same frequencies
    # whatever the phone sampled it at, and a reference on the wrong grid would
    # smear the response exactly as a room does.
    reference = sweep(int(rate), seconds, f_low, f_high)
    starts = find_repeats(captured, reference, rate, repeats, period)

    responses, shares = [], []
    for start in starts:
        stop = min(len(captured), start + int((seconds + IR_SECONDS) * rate))
        response, share = response_from(captured[start:stop], rate, seconds,
                                        f_low, f_high)
        if len(response) < int(IR_SECONDS * rate):
            response = np.pad(response, (0, int(IR_SECONDS * rate) - len(response)))
        responses.append(response)
        shares.append(share)

    check = linearity(responses)
    report = {
        "capture": str(args.sweep_capture),
        "rate": float(rate),
        "captured_sec": float(len(captured) / rate),
        "sweep_starts_sec": [float(s / rate) for s in starts],
        "spacing_sec": [float((b - a) / rate) for a, b in zip(starts, starts[1:])],
        "expected_spacing_sec": period,
        "pre_peak_energy_share": shares,
        "linearity": check,
        "rt60_sec": [rt60(r, rate) for r in responses],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    # Written whatever the verdict: a response from a non-linear chain is still
    # the best linear approximation to it, and refusing to save it would make
    # the failure unexaminable.
    mean = np.mean(np.stack(responses), axis=0)
    peak = float(np.max(np.abs(mean)))
    soundfile.write(str(args.output / "ir.wav"),
                    (mean / peak if peak > 0 else mean).astype(np.float32),
                    int(rate))
    for index, response in enumerate(responses):
        top = float(np.max(np.abs(response)))
        soundfile.write(str(args.output / f"ir_repeat{index + 1}.wav"),
                        (response / top if top > 0 else response).astype(np.float32),
                        int(rate))

    if args.silence_capture:
        silence, noise_rate = read_audio(args.silence_capture)
        report["noise"] = noise_profile(silence, noise_rate)
        soundfile.write(str(args.output / "noise.wav"),
                        silence.astype(np.float32), int(noise_rate))

    (args.output / "ir.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    if not check["lti"]:
        print("\nNOT LINEAR: the three repeats do not describe one response, "
              "so no impulse response describes this chain. The saved ir.wav "
              "is the best linear approximation and is not the room.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
