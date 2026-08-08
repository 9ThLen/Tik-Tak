#!/usr/bin/env python3
"""Generate the sweep that measures a room, and the filter that reads it back.

The simulated room in `room_degradation.py` costs 0.005 of F from reverberation
where a real one costs 0.390, so its impulse response is not slightly wrong but
wrong in character: exponentially decaying noise starting at sample zero has no
early reflections and no comb structure, which is most of what a room is. The
fix is to stop inventing the response and measure it.

An exponential sine sweep played through the speaker and captured on the phone
measures **speaker, room and microphone at once** — every linear part of the
chain that is currently being guessed. Deconvolving the capture with the
inverse filter below returns their combined impulse response.

## Why exponential and not white noise or a click

A click puts almost no energy into the room and measures mostly the noise
floor. An exponential sweep spends the same energy per octave, so the low end —
where a speaker is weakest and a room is most resonant — is measured as well as
the top. It also separates the harmonic distortion of the speaker into
*negative* time, before the impulse response, where it can be seen and cut away
instead of being averaged into the answer.

## What it cannot measure

Everything non-linear, and the phone has a large one. The captures show gain
slopes of 0.36 to 0.78, which is compression up to 3:1, and no convolution
reproduces it. Worse, it will distort **this** measurement too, because a sweep
holds a level that music does not — so play the sweep at roughly the loudness
the music was played at, and treat a response that changes between repeats as
evidence of the gain control rather than of the room.

Three repeats are written for exactly that reason. They are not there to be
averaged for the sake of it: if they disagree, the chain is not linear and the
disagreement is the finding.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

RATE = 48000
F_LOW = 20.0
F_HIGH = 20000.0
SECONDS = 10.0
REPEATS = 3
# Long enough for any domestic room's tail to die away completely before the
# next sweep starts; a tail that laps into the next repeat would be measured as
# part of it.
GAP_SEC = 3.0
LEAD_SEC = 1.0
# -6 dBFS, the level `docs/ROOM_PROTOCOL.md` asks for. Clipping is a separate
# degradation with its own character and would be indistinguishable here from a
# room.
AMPLITUDE = 0.5
FADE_SEC = 0.05


def sweep(rate: int = RATE, seconds: float = SECONDS, f_low: float = F_LOW,
          f_high: float = F_HIGH) -> np.ndarray:
    """Farina's exponential sine sweep, faded at both ends."""
    count = int(round(seconds * rate))
    t = np.arange(count) / rate
    ratio = np.log(f_high / f_low)
    phase = 2.0 * np.pi * f_low * seconds / ratio * (np.exp(t / seconds * ratio) - 1.0)
    out = np.sin(phase)
    # A sweep that starts and ends on a discontinuity puts a click into the
    # room, and a click is broadband: it would appear in the response as a
    # second, earlier arrival that no room produced.
    fade = int(FADE_SEC * rate)
    if fade > 0:
        window = np.hanning(2 * fade)
        out[:fade] *= window[:fade]
        out[-fade:] *= window[fade:]
    return out


def inverse_filter(rate: int = RATE, seconds: float = SECONDS,
                   f_low: float = F_LOW, f_high: float = F_HIGH) -> np.ndarray:
    """The filter whose convolution with the sweep is an impulse.

    Time-reversed, and amplitude-corrected for the sweep spending longer at low
    frequencies than at high ones: without the correction the result is not an
    impulse but a low-passed one, and the response would read as a room with no
    treble.

    The envelope goes on **after** the reversal, not before. Reversed, sample
    zero is the sweep's top octave and the last sample is its bottom one, so
    `exp(-t/T * ln(f2/f1))` attenuates the low end by the 60 dB it was
    over-represented by. Applied before the reversal the same expression
    attenuates the *high* end instead, and the self-test below returns a 1.9 dB
    blob rather than a spike -- which against a real recording would have been
    indistinguishable from a reverberant room.
    """
    count = int(round(seconds * rate))
    t = np.arange(count) / rate
    ratio = np.log(f_high / f_low)
    return sweep(rate, seconds, f_low, f_high)[::-1] * np.exp(-t / seconds * ratio)


def deconvolve(recorded: np.ndarray, rate: int = RATE,
               seconds: float = SECONDS, f_low: float = F_LOW,
               f_high: float = F_HIGH) -> np.ndarray:
    """Impulse response of whatever the sweep passed through.

    Kept here rather than in the analysis script so the filter and the signal
    it inverts cannot drift apart: a sweep generated with one set of parameters
    and read back with another gives a plausible-looking response that is
    wrong, and nothing downstream would notice.
    """
    from scipy.signal import fftconvolve

    return fftconvolve(recorded, inverse_filter(rate, seconds, f_low, f_high))


def self_test(rate: int = RATE, seconds: float = SECONDS) -> dict:
    """Deconvolve the sweep with itself: the answer has to be one spike.

    This is the control on the arithmetic above, and it runs before anybody
    plays anything. If the convention were reversed or the amplitude correction
    applied the wrong way, the result would be a smeared blob rather than a
    spike -- and against a real recording that blob would be indistinguishable
    from a reverberant room, which is precisely the thing being measured.
    """
    response = deconvolve(sweep(rate, seconds), rate, seconds)
    peak = int(np.argmax(np.abs(response)))
    height = float(np.abs(response[peak]))
    # Everything more than a millisecond from the spike is what a real room's
    # tail would have to stand out from.
    guard = int(0.001 * rate)
    mask = np.ones(len(response), dtype=bool)
    mask[max(0, peak - guard): peak + guard] = False
    sidelobe = float(np.max(np.abs(response[mask])))
    return {
        "peak_index": peak,
        "expected_index": int(round(seconds * rate)) - 1,
        "peak_to_sidelobe_db": float(20.0 * np.log10(height / max(sidelobe, 1e-30))),
    }


def main(argv: list[str] | None = None) -> int:
    import soundfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="where to write sweep.wav and sweep.json")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)

    check = self_test()
    print(f"self-test: spike at {check['peak_index']} "
          f"(expected {check['expected_index']}), "
          f"{check['peak_to_sidelobe_db']:.1f} dB above everything else")
    if check["peak_to_sidelobe_db"] < 40.0:
        print("the inverse filter does not invert the sweep; not writing a file")
        return 1

    one = sweep()
    gap = np.zeros(int(GAP_SEC * RATE))
    signal = [np.zeros(int(LEAD_SEC * RATE))]
    for _ in range(args.repeats):
        signal += [one, gap]
    track = AMPLITUDE * np.concatenate(signal)

    args.output.mkdir(parents=True, exist_ok=True)
    wav = args.output / "sweep.wav"
    soundfile.write(str(wav), track.astype(np.float32), RATE, subtype="PCM_16")
    (args.output / "sweep.json").write_text(json.dumps({
        "rate": RATE, "seconds": SECONDS, "f_low": F_LOW, "f_high": F_HIGH,
        "repeats": args.repeats, "gap_sec": GAP_SEC, "lead_sec": LEAD_SEC,
        "amplitude": AMPLITUDE, "fade_sec": FADE_SEC,
        "self_test": check,
        "read_back_with": "research/eval/make_sweep.py:deconvolve",
    }, indent=2), encoding="utf-8")

    print(f"wrote {wav}  ({len(track) / RATE:.0f} s, {args.repeats} sweeps, "
          f"{RATE} Hz, peak {AMPLITUDE:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
