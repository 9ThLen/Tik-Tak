#!/usr/bin/env python3
"""Prove that the C++ core and the Python reference compute the same ODF.

The research plan is to develop the algorithm in Python, where an experiment
costs seconds, and port it to C++ for the app. That only works if the two are
actually the same function. This runs both over identical audio and reports the
difference.

    cmake -S tools/parity -B tools/parity/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build tools/parity/build
    research/.venv/bin/python tools/parity/check_parity.py

Exact agreement is not the bar: the core works in float32 and the reference in
float64, deliberately, because the reference should be the more accurate of the
two. What matters is that the difference stays at the level of float32 rounding
rather than growing into a difference in behaviour.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from tiktak.odf import OdfConfig, compute_odf  # noqa: E402
from tiktak.synth import make_clip  # noqa: E402
from tiktak.tempo import estimate_tempo  # noqa: E402
from tiktak.tracker import track_beats  # noqa: E402

# float32 spectral flux accumulated over 81 mel bands: a few times the float32
# epsilon per operation is expected. An order of magnitude above this means the
# two implementations disagree about something real.
TOLERANCE = 2e-4


def run_cpp(binary: pathlib.Path, audio: np.ndarray, sample_rate: int, block: int):
    with tempfile.NamedTemporaryFile(suffix=".f32") as handle:
        handle.write(np.asarray(audio, dtype=np.float32).tobytes())
        handle.flush()

        completed = subprocess.run(
            [str(binary), handle.name, str(sample_rate), str(block)],
            capture_output=True,
            text=True,
            check=True,
        )

    rows = [line.split(",") for line in completed.stdout.strip().splitlines()[1:]]
    table = np.array(rows, dtype=np.float64)
    return {
        "times": table[:, 0],
        "full": table[:, 1],
        "low": table[:, 2],
        "high": table[:, 3],
    }


def compare(name: str, clip, binary: pathlib.Path, block: int) -> bool:
    cpp = run_cpp(binary, clip.audio, clip.sample_rate, block)
    reference = compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))

    if len(cpp["times"]) != len(reference):
        print(f"  {name}: FRAME COUNT differs — C++ {len(cpp['times'])}, Python {len(reference)}")
        return False

    ok = True
    time_error = np.max(np.abs(cpp["times"] - reference.times)) if len(reference) else 0.0
    if time_error > 1e-9:
        print(f"  {name}: timestamps differ by up to {time_error:.3g} s")
        ok = False

    scale = max(reference.full.max(), 1e-12)
    parts = []
    for band in ("full", "low", "high"):
        error = np.max(np.abs(cpp[band] - getattr(reference, band))) / scale
        parts.append(f"{band} {error:.2e}")
        if error > TOLERANCE:
            ok = False

    print(f"  {name:22} frames {len(reference):5d}  max rel err: {', '.join(parts)}"
          f"  {'ok' if ok else 'FAIL'}")
    return ok


def run_cpp_beats(binary: pathlib.Path, audio: np.ndarray, sample_rate: int, block: int,
                  bpm_hint: float = 0.0):
    with tempfile.NamedTemporaryFile(suffix=".f32") as handle:
        handle.write(np.asarray(audio, dtype=np.float32).tobytes())
        handle.flush()

        completed = subprocess.run(
            [str(binary), handle.name, str(sample_rate), str(block), str(bpm_hint)],
            capture_output=True,
            text=True,
            check=True,
        )

    header, _, body = completed.stdout.partition("\n\n")
    fields = dict(line.split("=", 1) for line in header.strip().splitlines())
    beats = np.array([float(line) for line in body.strip().splitlines()] or [],
                     dtype=np.float64)
    return fields, beats


def compare_beats(name: str, clip, binary: pathlib.Path, block: int, bpm_hint: float) -> bool:
    fields, cpp_beats = run_cpp_beats(binary, clip.audio, clip.sample_rate, block, bpm_hint)

    odf = compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))
    estimate = estimate_tempo(odf.full, odf.fps)
    tracked = track_beats(odf.full, odf.times, odf.fps,
                          bpm=bpm_hint if bpm_hint > 0.0 else None)

    ok = True
    notes = []

    # The tempo grid has 512 log-spaced points, so neighbouring candidates are
    # 0.5% apart. Landing on the neighbouring point is float32 rounding; landing
    # anywhere else is a difference in behaviour.
    bpm_error = abs(float(fields["bpm"]) - tracked.bpm) / tracked.bpm
    notes.append(f"bpm {bpm_error:.2e}")
    if bpm_error > 0.006:
        ok = False

    estimated_error = abs(float(fields["estimated_bpm"]) - estimate.bpm) / estimate.bpm
    notes.append(f"est {estimated_error:.2e}")
    if estimated_error > 0.006:
        ok = False

    confidence_error = abs(float(fields["confidence"]) - tracked.tempo_confidence)
    notes.append(f"conf {confidence_error:.2e}")
    if confidence_error > 5e-3:
        ok = False

    # Beat times are quantised to ODF frames, so agreement is exact or it is a
    # different decision. One frame of slack allows for a tie the two
    # implementations broke differently; more than that is a real divergence.
    hop = 512 / clip.sample_rate
    if len(cpp_beats) != len(tracked.beats):
        notes.append(f"BEAT COUNT {len(cpp_beats)} vs {len(tracked.beats)}")
        ok = False
    else:
        beat_error = np.max(np.abs(cpp_beats - tracked.beats)) if len(cpp_beats) else 0.0
        notes.append(f"beats {beat_error / hop:.2f} frames")
        if beat_error > hop * 1.5:
            ok = False

    print(f"  {name:22} beats {len(tracked.beats):4d}  {', '.join(notes)}"
          f"  {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=ROOT / "tools" / "parity" / "build" / "dump_odf",
        help="path to the dump_odf executable",
    )
    parser.add_argument(
        "--beats-binary",
        type=pathlib.Path,
        default=ROOT / "tools" / "parity" / "build" / "dump_beats",
        help="path to the dump_beats executable",
    )
    args = parser.parse_args()

    missing = [p for p in (args.binary, args.beats_binary) if not p.exists()]
    if missing:
        print("not found: " + ", ".join(str(p) for p in missing))
        print("Build them first — see the module docstring.")
        return 2

    cases = [
        ("steady 120", make_clip(bpm=120, duration_sec=12, seed=1), 137),
        ("fast 180", make_clip(bpm=180, duration_sec=12, seed=2), 512),
        ("sparse tones", make_clip(bpm=100, duration_sec=12, sparse=True, seed=3), 64),
        ("noisy", make_clip(bpm=120, duration_sec=12, noise_db=6.0, seed=4), 1024),
        ("silence lead", make_clip(bpm=120, duration_sec=12, silence_lead=3.0, seed=5), 137),
    ]

    print(f"comparing {args.binary} against research/tiktak/odf.py")
    print(f"tolerance: {TOLERANCE:.0e} relative to the peak ODF value\n")
    odf_ok = all(compare(name, clip, args.binary, block) for name, clip, block in cases)

    # The tracker makes discrete choices, so it needs its own comparison: the
    # ODF can agree to seven digits while the beat sequence still differs.
    print(f"\ncomparing {args.beats_binary} against research/tiktak/tracker.py\n")
    beat_cases = [(name, clip, block, 0.0) for name, clip, block in cases]
    beat_cases.append(("manual 100", make_clip(bpm=100, duration_sec=12, seed=6), 137, 100.0))
    beat_cases.append(("drift 120->140",
                       make_clip(bpm=120, duration_sec=15, tempo_drift=20, seed=7), 137, 0.0))
    beats_ok = all(compare_beats(name, clip, args.beats_binary, block, hint)
                   for name, clip, block, hint in beat_cases)

    print()
    if odf_ok and beats_ok:
        print("PARITY OK — the core and the reference agree to float32 precision.")
        return 0
    print("PARITY FAILED — the implementations have diverged. Fix before trusting any metric.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
