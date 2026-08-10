#!/usr/bin/env python3
"""Simulate the room from what was measured, and check it against the room.

Registered in `PREREGISTERED_room_simulation.md`. Corpus audio convolved with
the impulse response measured by `room_ir.py`, plus the silence recorded in the
same session, scaled to the SNR the captures actually had. Nothing is fitted:
the response, the noise and the level all come from the session.

The validation is the point, not the simulation. `room_degradation.py` invented
a room, was never checked against one, and cost 0.005 of F from reverberation
where a real room costs 0.390. Any replacement is worth exactly what it scores
against the captures from its own session, so that comparison is what this
prints and what decides whether the simulator may be used.

## Two things it deliberately does not model

**The gain control.** Measured at 0.36 to 0.78, compression up to 3:1, and a
convolution cannot express it. Emulating it would mean choosing an attack, a
release and a ratio -- three parameters fitted on the two recordings the answer
is read on.

**The bottom two octaves.** The three sweep repeats disagree by 14 dB below
60 Hz, so no single response describes them. The response is used as measured
rather than repaired: a fix there would be inventing again, which is the thing
being replaced.
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
from eval.provenance import experiment_provenance as provenance  # noqa: E402

SAMPLE_HZ = 50.0
# Within this of the real capture and the simulation passes on level.
LEVEL_TOLERANCE = 0.05


def resample(signal: np.ndarray, source: float, target: float) -> np.ndarray:
    """Rate conversion by Fourier resampling, on a length that keeps duration.

    The response was captured at 48 kHz and the corpus is 22.05 kHz. Convolving
    across the two grids without converting would stretch the room in time --
    a 0.33 s tail would become 0.72 s -- and the simulation would measure a
    hall that nobody stood in.
    """
    from scipy.signal import resample_poly
    from math import gcd

    if source == target:
        return signal
    a, b = int(round(target)), int(round(source))
    divisor = gcd(a, b)
    return resample_poly(signal, a // divisor, b // divisor)


def simulate(mono: np.ndarray, rate: float, response: np.ndarray,
             response_rate: float, noise: np.ndarray, noise_rate: float,
             snr_db: float, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import fftconvolve

    impulse = resample(response, response_rate, rate)
    peak = int(np.argmax(np.abs(impulse)))
    # The direct path is the response's own peak, and the convolution must not
    # delay the audio against annotations that were not moved. Trimming to the
    # peak is what keeps sample zero at sample zero.
    impulse = impulse[peak:]
    energy = float(np.sqrt(np.sum(impulse * impulse)))
    if energy > 0.0:
        impulse = impulse / energy
    out = fftconvolve(mono, impulse)[: len(mono)]

    room_noise = resample(noise, noise_rate, rate)
    if len(room_noise) < len(out):
        repeats = -(-len(out) // len(room_noise))
        room_noise = np.tile(room_noise, repeats)
    # A random start, so a track is not always laid over the same second of
    # room tone; the generator is seeded per recording so it is reproducible.
    start = int(rng.integers(0, max(1, len(room_noise) - len(out))))
    room_noise = room_noise[start: start + len(out)]

    signal_rms = float(np.sqrt(np.mean(out * out)))
    noise_rms = float(np.sqrt(np.mean(room_noise * room_noise)))
    if noise_rms > 0.0 and signal_rms > 0.0:
        gain = signal_rms / noise_rms * float(np.power(10.0, -snr_db / 20.0))
        out = out + gain * room_noise

    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.98) if peak > 1.0 else out


def main(argv: list[str] | None = None) -> int:
    import soundfile

    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", default=["harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--ir", type=pathlib.Path, required=True)
    parser.add_argument("--noise", type=pathlib.Path, required=True)
    parser.add_argument("--snr-db", type=float, required=True,
                        help="measured on the captures, not chosen")
    parser.add_argument("--validate-against", type=pathlib.Path, required=True,
                        help="room_recording JSON from the same session")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    run_provenance = provenance(
        repository,
        {"manifest": args.manifest, "binary": args.binary,
         "model": args.model, "ir": args.ir, "noise": args.noise,
         "validate_against": args.validate_against},
    )

    items = {i["name"]: i for i in
             load_corpus(args.manifest, args.music, False, frozenset(args.corpora))}
    reference = json.loads(args.validate_against.read_text(encoding="utf-8"))

    # What the real capture scored, and for a void one the interval its
    # candidate alignments span. A simulation inside that interval is not
    # confirmed by it, only not refuted, and the field name says so.
    truth: dict = {}
    for record in reference["records"]:
        if "room" in record:
            truth[record["name"]] = {"real_f": record["room"]["f_measure"],
                                     "clean_f": record["clean"]["f_measure"],
                                     "kind": "scored"}
        elif "sensitivity" in record:
            values = [arm["room"]["f_measure"]
                      for arm in record["sensitivity"].values()]
            truth[record["name"]] = {"interval": [min(values), max(values)],
                                     "kind": "void_interval"}

    response, response_rate = soundfile.read(str(args.ir), dtype="float64",
                                             always_2d=True)
    response = response.mean(axis=1)
    noise, noise_rate = soundfile.read(str(args.noise), dtype="float64",
                                       always_2d=True)
    noise = noise.mean(axis=1)

    records = []
    for index, (name, expected) in enumerate(sorted(truth.items())):
        item = items.get(name)
        if item is None:
            continue
        audio, rate = soundfile.read(str(item["audio"]), dtype="float64",
                                     always_2d=True)
        mono = audio.mean(axis=1)
        simulated = simulate(mono, float(rate), response, float(response_rate),
                             noise, float(noise_rate), args.snr_db,
                             np.random.default_rng(index))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "simulated.wav"
            soundfile.write(str(path), simulated.astype(np.float32), int(rate))
            done = subprocess.run(
                [str(args.binary), str(path), "--live", "--live-model",
                 str(args.model), "--live-sample-hz", repr(SAMPLE_HZ)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=False)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip()[:300])
            scored = _score_one(item, "model", args.binary, args.model,
                                estimate=Estimate.from_json(json.loads(done.stdout)))
        row = {"name": name, "simulated_f": scored.get("f_measure"),
               "simulated_usable": bool(scored.get("usable", False)),
               "simulated_r70": scored.get("r70"),
               "simulated_p70": scored.get("p70"), **expected}
        if expected["kind"] == "scored":
            row["error"] = row["simulated_f"] - expected["real_f"]
            row["level_ok"] = bool(abs(row["error"]) <= LEVEL_TOLERANCE)
        else:
            low, high = expected["interval"]
            row["not_refuted"] = bool(low - LEVEL_TOLERANCE <= row["simulated_f"]
                                      <= high + LEVEL_TOLERANCE)
        records.append(row)

    # Ordering: whatever really collapsed must be simulated below whatever
    # really survived. A simulation that damages everything equally repeats the
    # failure it replaces.
    def real_level(row: dict) -> float:
        return (row["real_f"] if row["kind"] == "scored"
                else float(np.mean(row["interval"])))

    ordering_ok = None
    if len(records) >= 2:
        by_real = sorted(records, key=real_level)
        by_sim = sorted(records, key=lambda r: r["simulated_f"])
        ordering_ok = [r["name"] for r in by_real] == [r["name"] for r in by_sim]

    level_ok = all(r.get("level_ok", True) for r in records)
    payload = {
        "provenance": run_provenance, "snr_db": args.snr_db,
        "ir": str(args.ir), "noise": str(args.noise),
        "level_ok": bool(level_ok), "ordering_ok": ordering_ok,
        "approved_for_augmentation": bool(level_ok and ordering_ok),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
