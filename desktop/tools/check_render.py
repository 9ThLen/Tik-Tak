#!/usr/bin/env python3
"""Checks that `tiktak render` put every beat where it said it would.

    python3 desktop/tools/check_render.py metronome.wav --bpm 137 --start 0.5

This is the part of the harness that runs without a sound card, and it is the
reason `render` exists at all: a CI runner has no speakers, but the samples a
speaker would have received can still be checked. Standard library only, so it
needs nothing installed.

The tolerance is one millisecond. The renderer is accurate to half a sample —
ten microseconds — so anything approaching a millisecond means something is
wrong, not that the check is tight.
"""

import argparse
import struct
import sys
import wave


def onsets(samples, threshold_ratio=0.002, skip=4000):
    """Sample index of each click. `skip` clears the click's own tail.

    The gate is deliberately near the floor rather than a fraction of the peak.
    A render is exactly silent between clicks, so the first sample that moves at
    all *is* the onset; a higher gate would find the click a sample or two late
    and, worse, later for a quiet subdivision than for a loud downbeat, turning
    a difference in loudness into what looks like a difference in timing.
    """
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return []
    threshold = peak * threshold_ratio

    found = []
    i = 0
    while i < len(samples):
        if abs(samples[i]) > threshold:
            found.append(i)
            i += skip
        else:
            i += 1
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--start", type=float, required=True,
                        help="stream time of the first beat")
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--tolerance-ms", type=float, default=1.0)
    args = parser.parse_args()

    with wave.open(args.wav) as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            sys.exit(f"{args.wav}: expected 16-bit mono")
        rate = f.getframerate()
        frames = f.getnframes()
        samples = struct.unpack(f"<{frames}h", f.readframes(frames))

    found = onsets(samples)
    if len(found) < 4:
        sys.exit(f"{args.wav}: found {len(found)} clicks, expected a metronome")

    step_sec = 60.0 / (args.bpm * args.sub)
    tolerance = args.tolerance_ms / 1000.0

    worst = 0.0
    failures = []
    for i, at in enumerate(found):
        expected = args.start + step_sec * i
        actual = at / rate
        error = actual - expected
        worst = max(worst, abs(error))
        if abs(error) > tolerance:
            failures.append(f"  beat {i}: {actual:.6f} s, expected {expected:.6f} s "
                            f"({error * 1000:+.3f} ms)")

    print(f"{len(found)} beats at {args.bpm} BPM, worst error {worst * 1000:+.3f} ms")

    if failures:
        print(f"\n{len(failures)} beat(s) outside {args.tolerance_ms} ms:")
        print("\n".join(failures[:20]))
        sys.exit(1)

    # Drift is the failure this is really watching for, so it is reported
    # separately: a constant offset is a detector artefact, a growing one is a
    # metronome that speeds up.
    first_error = found[0] / rate - args.start
    last_error = found[-1] / rate - (args.start + step_sec * (len(found) - 1))
    drift = last_error - first_error
    print(f"drift from first beat to last: {drift * 1000:+.3f} ms")
    if abs(drift) > tolerance:
        sys.exit("the metronome drifted")


if __name__ == "__main__":
    main()
