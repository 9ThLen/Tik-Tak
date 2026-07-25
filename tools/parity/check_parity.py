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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=ROOT / "tools" / "parity" / "build" / "dump_odf",
        help="path to the dump_odf executable",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"dump_odf not found at {args.binary}\nBuild it first — see the module docstring.")
        return 2

    print(f"comparing {args.binary} against research/tiktak/odf.py")
    print(f"tolerance: {TOLERANCE:.0e} relative to the peak ODF value\n")

    cases = [
        ("steady 120", make_clip(bpm=120, duration_sec=12, seed=1), 137),
        ("fast 180", make_clip(bpm=180, duration_sec=12, seed=2), 512),
        ("sparse tones", make_clip(bpm=100, duration_sec=12, sparse=True, seed=3), 64),
        ("noisy", make_clip(bpm=120, duration_sec=12, noise_db=6.0, seed=4), 1024),
        ("silence lead", make_clip(bpm=120, duration_sec=12, silence_lead=3.0, seed=5), 137),
    ]

    passed = all(compare(name, clip, args.binary, block) for name, clip, block in cases)

    print()
    if passed:
        print("PARITY OK — the core and the reference agree to float32 precision.")
        return 0
    print("PARITY FAILED — the implementations have diverged. Fix before trusting any metric.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
