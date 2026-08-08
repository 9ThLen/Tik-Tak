#!/usr/bin/env python3
"""Score a recording made in an actual room against the file it was played from.

`room_degradation.py` sweeps reverberation and noise in simulation. This takes
the real thing: someone played a corpus recording through a speaker, captured it
with a microphone, and the capture is scored against the same annotations the
clean baseline used. One room is one sample, so the number that matters is the
**gap** against the clean run on the same recording — and, once there is a gap,
which simulated cell it corresponds to, which is what turns the rest of the
sweep from a guess into an extrapolation from a measured point.

## Alignment is the whole risk

The annotations belong to the original file and were not moved, so a capture
that is offset or stretched against it scores a time shift rather than a room.
Two things are therefore checked before any beat is scored, and either one
failing voids the recording:

**Offset.** Found by cross-correlating onset envelopes rather than waveforms.
A room inverts phase, colours the spectrum and adds a tail; the waveform
correlation of a reverberant capture against its source is weak and can peak in
the wrong place, while the envelope survives all three.

**Drift.** The offset is fitted twice — over the first third and over the last
third — and the two must agree. Playback and capture run off different clocks,
and a part-per-thousand difference is 0.2 s over a three-minute song, which is
three beats. A constant offset is fine and is corrected; a drifting one is not
correctable by a constant and the run is void.

## What it does not do

It does not resample the capture to remove drift. That would be repairing the
measurement rather than reporting it, and a setup whose clocks disagree is a
fact about the setup that the log should carry.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.analysis import Estimate  # noqa: E402
from eval.live_corpus_benchmark import _score_one, load_corpus  # noqa: E402

SAMPLE_HZ = 50.0
# The envelope the alignment runs on. 100 Hz is ten times finer than a beat at
# 200 BPM and coarse enough that a three-minute correlation is instant.
ENVELOPE_HZ = 100.0
# Above this, the two ends of the recording disagree about where it starts by
# more than a beat at any sane tempo, and no constant offset describes it.
MAX_DRIFT_SEC = 0.030


def envelope(mono: np.ndarray, rate: float) -> np.ndarray:
    """A crude onset envelope: rectified first difference of a smoothed level.

    Resampled onto an **exact** ENVELOPE_HZ grid rather than left on the hop
    grid, and both of those words are load-bearing. `rate / step` is only
    approximately ENVELOPE_HZ — at 22.05 kHz a 220-sample hop is 100.23 Hz, and
    reading the correlation index as if it were 100 Hz puts a 4.7 s offset out
    by 10 ms. Worse, the corpus is 22.05 kHz and a phone records at 44.1 or 48,
    so the two signals would land on grids of *different* pitch and the
    correlation would measure the mismatch. A common exact grid removes both.
    """
    step = max(1, int(rate / ENVELOPE_HZ))
    trimmed = mono[: len(mono) // step * step]
    if len(trimmed) == 0:
        return np.zeros(0, dtype=np.float64)
    level = np.sqrt(np.mean(trimmed.reshape(-1, step) ** 2, axis=1))
    hop_hz = rate / step
    out = np.diff(level, prepend=level[:1])
    np.maximum(out, 0.0, out=out)

    duration = len(out) / hop_hz
    grid = np.arange(0.0, duration, 1.0 / ENVELOPE_HZ)
    out = np.interp(grid, np.arange(len(out)) / hop_hz, out)

    peak = float(np.max(out)) if len(out) else 0.0
    return out / peak if peak > 0 else out


def best_offset(reference: np.ndarray, capture: np.ndarray) -> tuple[float, float]:
    """Seconds the capture lags the reference, and the correlation peak height."""
    if len(reference) == 0 or len(capture) == 0:
        return (0.0, 0.0)
    size = 1
    while size < len(reference) + len(capture):
        size *= 2
    spectrum = (np.fft.rfft(capture, size)
                * np.conj(np.fft.rfft(reference, size)))
    correlation = np.fft.irfft(spectrum, size)
    index = int(np.argmax(correlation[: len(capture)]))
    peak = float(correlation[index])

    # Sub-grid refinement by fitting a parabola to the peak and its neighbours.
    # Without it the answer is quantised to the envelope's 10 ms, which is a
    # seventh of the 70 ms beat tolerance and -- as the synthetic check showed
    # -- lands on the same side every time, so it is a bias rather than noise.
    shift = 0.0
    if 0 < index < len(correlation) - 1:
        left, right = correlation[index - 1], correlation[index + 1]
        denominator = left - 2.0 * peak + right
        if denominator != 0.0:
            shift = 0.5 * (left - right) / denominator
            shift = float(np.clip(shift, -0.5, 0.5))

    norm = float(np.linalg.norm(reference) * np.linalg.norm(capture))
    return ((index + shift) / ENVELOPE_HZ, peak / norm if norm > 0 else 0.0)


def align(original: np.ndarray, capture: np.ndarray, rate_a: float,
          rate_b: float) -> dict:
    a = envelope(original, rate_a)
    b = envelope(capture, rate_b)
    whole, quality = best_offset(a, b)

    # The same fit over each end. If the clocks agree these coincide; if they do
    # not, the difference is the drift across that span and no constant offset
    # describes the recording.
    # Both slices start at the same point in each signal, so a constant lag
    # shows up identically in the two fits and their difference is the drift
    # accumulated across the two thirds between them.
    third = len(a) // 3
    head, head_q = best_offset(a[:third], b[: third + int(2 * ENVELOPE_HZ)])
    tail, tail_q = best_offset(a[2 * third:], b[2 * third:])
    drift = abs(tail - head)

    return {"offset_sec": whole, "quality": quality,
            "head_offset_sec": head, "tail_offset_sec": tail,
            "head_quality": head_q, "tail_quality": tail_q,
            "drift_sec": drift, "drift_ok": drift <= MAX_DRIFT_SEC}


def measure_one(item: dict, capture_path: pathlib.Path, binary: pathlib.Path,
                model: pathlib.Path) -> dict:
    import soundfile

    original, rate_a = soundfile.read(str(item["audio"]), dtype="float32",
                                      always_2d=True)
    capture, rate_b = soundfile.read(str(capture_path), dtype="float32",
                                     always_2d=True)
    original = original.mean(axis=1)
    capture = capture.mean(axis=1)

    alignment = align(original, capture, float(rate_a), float(rate_b))
    offset = alignment["offset_sec"]

    # Trimmed so sample zero of the written file is sample zero of the original,
    # which is what makes the untouched annotations apply.
    start = int(round(offset * rate_b))
    trimmed = capture[max(0, start):]
    alignment["captured_sec"] = len(trimmed) / float(rate_b)

    out: dict = {"name": item["name"], "corpus": item["corpus"],
                 "capture": str(capture_path), "alignment": alignment}
    if not alignment["drift_ok"]:
        out["void"] = "playback and capture clocks disagree; see drift_sec"
        return out

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "aligned.wav"
        soundfile.write(str(path), trimmed, int(rate_b))
        arms = {"room": str(path), "clean": str(item["audio"])}
        for arm, audio in arms.items():
            done = subprocess.run(
                [str(binary), audio, "--live", "--live-model", str(model),
                 "--live-sample-hz", repr(SAMPLE_HZ)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=False)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip()[:200])
            scored = _score_one(item, "model", binary, model,
                                estimate=Estimate.from_json(json.loads(done.stdout)))
            out[arm] = {
                "usable": bool(scored.get("usable", False)),
                "reasons": list(scored.get("reasons", [])),
                "f_measure": scored.get("f_measure"),
                "p70": scored.get("p70"),
                "r70": scored.get("r70"),
                "acquired_at": scored.get("acquired_at"),
                "switches": scored.get("switches"),
                "correct_share_of_eligible": scored.get("correct_share_of_eligible"),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", default=["harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--captures", type=pathlib.Path, required=True,
        help="directory of room recordings, each named <track>.wav")
    parser.add_argument("--notes", type=str, default="",
                        help="speaker, microphone, room, distance, levels")
    args = parser.parse_args(argv)

    items = {i["name"]: i for i in
             load_corpus(args.manifest, args.music, False, frozenset(args.corpora))}
    captures = sorted(args.captures.glob("*.wav"))
    if not captures:
        print(f"no .wav files in {args.captures}", file=sys.stderr)
        return 1

    records, failures = [], []
    for capture in captures:
        item = items.get(capture.stem)
        if item is None:
            failures.append({"capture": capture.name,
                             "error": f"{capture.stem} is not in the corpus"})
            continue
        try:
            records.append(measure_one(item, capture, args.binary, args.model))
        except Exception as error:  # noqa: BLE001
            failures.append({"capture": capture.name, "error": str(error)[:300]})
        print(f"{len(records) + len(failures)}/{len(captures)}",
              file=sys.stderr, flush=True)

    scored = [r for r in records if "room" in r]
    summary = {}
    if scored:
        for arm in ("clean", "room"):
            summary[arm] = {
                "n": len(scored),
                "usable_rate": float(np.mean([r[arm]["usable"] for r in scored])),
                "mean_f": float(np.mean([r[arm]["f_measure"] for r in scored])),
                "mean_r70": float(np.mean([r[arm]["r70"] for r in scored])),
                "mean_p70": float(np.mean([r[arm]["p70"] for r in scored])),
            }
        summary["voided"] = len([r for r in records if "void" in r])

    payload = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=repository).stdout.strip(),
        "notes": args.notes, "captures": str(args.captures),
        "failures": failures, "summary": summary, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
