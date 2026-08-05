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

WEIGHTS = ROOT / "models" / "beatnet_model_1.ttw"

# The network's own tolerance, and it needs a different number from the ODF's.
# Two layers of LSTM carry state from the first frame to the last, so a float32
# rounding difference in frame 0 is still present, slightly rearranged, in frame
# 15000 — there is no averaging to hide behind. A few times 1e-6 on a
# probability that sums to one with two others is float32; 1e-3 is a different
# function.
BEATNET_TOLERANCE = 5e-5

# The core computes this spectrogram in float32 and the reference in float64,
# over a 1024-point transform and 128 triangles. A few times 1e-6 relative is
# that difference; 1e-3 is a different front end.
BEAT_THIS_TOLERANCE = 1e-4

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


def run_cpp_beatnet(binary: pathlib.Path, audio: np.ndarray, sample_rate: int,
                    block: int, features: bool):
    with tempfile.NamedTemporaryFile(suffix=".f32") as handle:
        handle.write(np.asarray(audio, dtype=np.float32).tobytes())
        handle.flush()

        command = [str(binary), handle.name, str(sample_rate), str(WEIGHTS), str(block)]
        if features:
            command.append("--features")
        completed = subprocess.run(command, capture_output=True, text=True, check=True)

    body = completed.stdout.strip().splitlines()
    if not features:
        body = body[1:]
    return np.array([line.split(",") for line in body], dtype=np.float64)


def compare_beatnet(name: str, clip, binary: pathlib.Path, block: int, network) -> bool:
    from eval.beatnet_onnx import log_filtered_spectrogram, resample_to_model_rate

    cpp_features = run_cpp_beatnet(binary, clip.audio, clip.sample_rate, block, True)
    reference_features = log_filtered_spectrogram(
        resample_to_model_rate(clip.audio, clip.sample_rate))

    # The streaming implementation is one frame shorter, and should be. The
    # reference zero-pads the end of the file and computes a final frame that is
    # mostly padding; a tracker running live has no such file to pad, and
    # inventing that frame would be inventing 32 ms of future.
    shortfall = len(reference_features) - len(cpp_features)
    ok = shortfall in (0, 1)
    if not ok:
        print(f"  {name}: FRAME COUNT differs by {shortfall} — C++ {len(cpp_features)}, "
              f"Python {len(reference_features)}")

    count = min(len(cpp_features), len(reference_features))
    notes = []

    times = cpp_features[:count, 0]
    expected_times = np.arange(count) * 441.0 / 22050.0
    time_error = np.max(np.abs(times - expected_times)) if count else 0.0
    notes.append(f"t {time_error:.1e}")
    if time_error > 1e-9:
        ok = False

    feature_error = np.max(np.abs(cpp_features[:count, 1:] - reference_features[:count]))
    notes.append(f"features {feature_error:.1e}")
    if feature_error > BEATNET_TOLERANCE:
        ok = False

    cpp_activation = run_cpp_beatnet(binary, clip.audio, clip.sample_rate, block, False)
    probabilities = network.activations(clip.audio, clip.sample_rate)
    count = min(len(cpp_activation), len(probabilities))

    beat_error = np.max(np.abs(cpp_activation[:count, 1]
                               - (probabilities[:count, 0] + probabilities[:count, 1])))
    downbeat_error = np.max(np.abs(cpp_activation[:count, 2] - probabilities[:count, 1]))
    notes.append(f"beat {beat_error:.1e}, downbeat {downbeat_error:.1e}")
    if max(beat_error, downbeat_error) > BEATNET_TOLERANCE:
        ok = False

    print(f"  {name:22} frames {count:5d}  {', '.join(notes)}  {'ok' if ok else 'FAIL'}")
    return ok


def check_beatnet(binary: pathlib.Path) -> bool | None:
    """None when the weights are absent — a missing artifact is not a failure."""
    if not binary.exists() or not WEIGHTS.exists():
        print(f"\nskipping BeatNet: need {binary.name} and {WEIGHTS.name}")
        print("  research/.venv/bin/python models/export_beatnet.py "
              "models/beatnet_model_1_weights.pt models/beatnet_model_1.ttw")
        return None

    from eval.beatnet_onnx import WEIGHTS_PATH, BeatNet

    if not WEIGHTS_PATH.is_file():
        print(f"\nskipping BeatNet: the reference needs {WEIGHTS_PATH.name}")
        return None

    # The reference runs the network through PyTorch, deliberately: an
    # independent implementation of the LSTM is the whole point of comparing
    # against it, and one written here would only be agreeing with itself.
    # Absent torch this check cannot run, and saying so is better than the
    # traceback it used to produce.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\nskipping BeatNet: the reference runs on torch, which is not installed")
        return None

    print(f"\ncomparing {binary} against research/eval/beatnet_onnx.py")
    print(f"tolerance: {BEATNET_TOLERANCE:.0e} absolute\n")
    network = BeatNet(WEIGHTS_PATH)

    # Different capture rates on purpose: 48 kHz and 44.1 kHz resample to the
    # model's 22050 by different ratios, and 22050 itself takes the path where
    # no resampling happens at all.
    cases = [
        ("48k, block 137", make_clip(bpm=120, duration_sec=6, seed=1), 48000, 137),
        ("48k, block 1024", make_clip(bpm=120, duration_sec=6, seed=1), 48000, 1024),
        ("44.1k sparse", make_clip(bpm=100, duration_sec=6, sparse=True, seed=3,
                                   sample_rate=44100), 44100, 512),
        ("22.05k, no resample", make_clip(bpm=140, duration_sec=6, seed=4,
                                          sample_rate=22050), 22050, 441),
        ("noisy 48k", make_clip(bpm=120, duration_sec=6, noise_db=6.0, seed=5), 48000, 137),
    ]
    return all(compare_beatnet(name, clip, binary, block, network)
               for name, clip, _rate, block in cases)


def run_cpp_beat_this(binary: pathlib.Path, audio: np.ndarray):
    with tempfile.NamedTemporaryFile(suffix=".f32") as handle:
        handle.write(np.asarray(audio, dtype=np.float32).tobytes())
        handle.flush()
        completed = subprocess.run([str(binary), handle.name],
                                   capture_output=True, text=True, check=True)
    body = completed.stdout.strip().splitlines()
    if not body:
        return np.zeros((0, 128))
    return np.array([line.split(",") for line in body], dtype=np.float64)


def check_beat_this_end_to_end(binary: pathlib.Path) -> bool | None:
    """The whole offline path: our front end, the runtime, our peak picker.

    Off by default and skipped here unless both the tool and the model exist.
    The tool needs -DTIKTAK_BUILD_ML=ON, which downloads ONNX Runtime, and the
    model is fetched separately — neither belongs in a CI job that could not
    run the check anyway.
    """
    from eval.beat_this_onnx import MODEL_PATH, SAMPLE_RATE

    if not binary.exists() or not MODEL_PATH.is_file():
        print(f"\nskipping Beat This! end to end: need {binary.name} "
              f"(-DTIKTAK_BUILD_ML=ON) and {MODEL_PATH.name}")
        return None

    from eval.beat_this_onnx import BeatThisOnnx, beats_and_downbeats

    print(f"\ncomparing {binary} against research/eval/beat_this_onnx.py")
    print("beats and downbeats must agree exactly — they are frame indices, "
          "not measurements\n")

    network = BeatThisOnnx()
    cases = [
        ("click 120", make_clip(bpm=120, duration_sec=40, seed=1, sample_rate=SAMPLE_RATE)),
        ("sparse 100", make_clip(bpm=100, duration_sec=40, sparse=True, seed=3,
                                 sample_rate=SAMPLE_RATE)),
        ("noisy", make_clip(bpm=120, duration_sec=40, noise_db=6.0, seed=4,
                            sample_rate=SAMPLE_RATE)),
        ("drift 120->140", make_clip(bpm=120, duration_sec=40, tempo_drift=20, seed=7,
                                     sample_rate=SAMPLE_RATE)),
        ("silence lead", make_clip(bpm=120, duration_sec=40, silence_lead=3.0, seed=5,
                                   sample_rate=SAMPLE_RATE)),
        ("fast 180", make_clip(bpm=180, duration_sec=40, seed=2, sample_rate=SAMPLE_RATE)),
    ]

    ok = True
    for name, clip in cases:
        audio = np.asarray(clip.audio, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".f32") as handle:
            handle.write(audio.tobytes())
            handle.flush()
            completed = subprocess.run(
                [str(binary), handle.name, str(MODEL_PATH), "--beats"],
                capture_output=True, text=True, check=True)

        head, _, tail = completed.stdout.partition("\ndownbeats\n")
        cpp_beats = np.array([float(x) for x in head.strip().splitlines()[1:]])
        cpp_downbeats = np.array([float(x) for x in tail.strip().splitlines()]
                                 or [], dtype=np.float64)

        reference_beats, reference_downbeats = beats_and_downbeats(
            network.activations(audio.astype(np.float64), SAMPLE_RATE))

        def agree(a, b):
            return len(a) == len(b) and (len(a) == 0 or float(np.max(np.abs(a - b))) < 1e-9)

        beats_ok = agree(cpp_beats, reference_beats)
        downbeats_ok = agree(cpp_downbeats, reference_downbeats)
        ok = ok and beats_ok and downbeats_ok
        print(f"  {name:16} beats {len(cpp_beats):4d} vs {len(reference_beats):4d} "
              f"{'ok' if beats_ok else 'FAIL'}   downbeats {len(cpp_downbeats):3d} vs "
              f"{len(reference_downbeats):3d} {'ok' if downbeats_ok else 'FAIL'}")
    return ok


def check_beat_this(binary: pathlib.Path) -> bool | None:
    """The mel front end Beat This! expects, which the exported graph omits.

    Only the spectrogram: the network itself is the ONNX, and comparing that
    against itself would prove nothing. What can go wrong is entirely here — a
    symmetric window, the wrong mel curve, a power spectrum where an amplitude
    was wanted — and all of it produces something that still looks like a
    spectrogram.
    """
    if not binary.exists():
        print(f"\nskipping Beat This! features: {binary.name} is not built")
        return None

    from eval.beat_this_onnx import SAMPLE_RATE, log_mel_spectrogram

    print(f"\ncomparing {binary} against research/eval/beat_this_onnx.py")
    print(f"tolerance: {BEAT_THIS_TOLERANCE:.0e} relative to the peak value\n")

    cases = [
        ("click 120", make_clip(bpm=120, duration_sec=8, seed=1, sample_rate=SAMPLE_RATE)),
        ("sparse 100", make_clip(bpm=100, duration_sec=8, sparse=True, seed=3,
                                 sample_rate=SAMPLE_RATE)),
        ("noisy 120", make_clip(bpm=120, duration_sec=8, noise_db=6.0, seed=4,
                                sample_rate=SAMPLE_RATE)),
        ("silence lead", make_clip(bpm=120, duration_sec=8, silence_lead=3.0, seed=5,
                                   sample_rate=SAMPLE_RATE)),
    ]

    ok = True
    for name, clip in cases:
        audio = np.asarray(clip.audio, dtype=np.float32)
        cpp = run_cpp_beat_this(binary, audio)
        reference = log_mel_spectrogram(audio.astype(np.float64))

        if len(cpp) != len(reference):
            print(f"  {name}: FRAME COUNT differs — C++ {len(cpp)}, Python {len(reference)}")
            ok = False
            continue

        scale = max(float(reference.max()), 1e-12)
        error = float(np.max(np.abs(cpp - reference))) / scale if len(cpp) else 0.0
        good = error <= BEAT_THIS_TOLERANCE
        ok = ok and good
        print(f"  {name:22} frames {len(reference):5d}  max rel err {error:.2e}"
              f"  {'ok' if good else 'FAIL'}")
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
    parser.add_argument(
        "--beatnet-binary",
        type=pathlib.Path,
        default=ROOT / "tools" / "parity" / "build" / "dump_beatnet",
        help="path to the dump_beatnet executable",
    )
    parser.add_argument(
        "--beat-this-binary",
        type=pathlib.Path,
        default=ROOT / "tools" / "parity" / "build" / "dump_beat_this_features",
        help="path to the dump_beat_this_features executable",
    )
    parser.add_argument(
        "--beat-this-model-binary",
        type=pathlib.Path,
        default=ROOT / "tools" / "parity" / "build" / "dump_beat_this",
        help="path to the dump_beat_this executable (needs -DTIKTAK_BUILD_ML=ON)",
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

    beatnet_ok = check_beatnet(args.beatnet_binary)
    beat_this_ok = check_beat_this(args.beat_this_binary)
    end_to_end_ok = check_beat_this_end_to_end(args.beat_this_model_binary)

    print()
    if (odf_ok and beats_ok and beatnet_ok is not False
            and beat_this_ok is not False and end_to_end_ok is not False):
        if beatnet_ok is None:
            print("PARITY OK — the core and the reference agree to float32 precision "
                  "(BeatNet not checked).")
        else:
            print("PARITY OK — the core and the reference agree to float32 precision.")
        return 0
    print("PARITY FAILED — the implementations have diverged. Fix before trusting any metric.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
