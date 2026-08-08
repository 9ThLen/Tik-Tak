#!/usr/bin/env python3
"""What does a room cost the live path?

Every live number in this repository was measured on clean decoded files fed
straight to the tracker. The product's first scenario is a released track
playing through a speaker into a phone microphone, and there is not one number
about that case.

This is **not** that measurement. A real room needs a speaker, a microphone and
a physical space, and it brings a microphone's frequency response, automatic
gain control, clipping and the tracker hearing its own click — none of which is
here. What is here is the acoustic part, isolated and parameterised:
reverberation and additive noise, swept, reproducibly, with no hardware.

Read it as "how fast does the live path fall apart as the acoustics get worse",
not as "the tracker scores X in a room". The real-room protocol is in
`docs/` and needs a person with a microphone.

## The two knobs

**Reverberation.** An exponentially decaying noise impulse response of the given
RT60. **Its first sample is the direct path at unit gain**, which matters more
than the tail: any other construction delays the audio against annotations that
were not moved, and the whole run would measure a time shift rather than a room.

**Noise.** Pink, because room noise is not white — ventilation, traffic and
crowd all fall off with frequency, and white noise would put its energy exactly
where the onset detector is most sensitive and overstate the damage. Scaled to a
target SNR against the signal's own RMS.

Both are seeded per recording, so the same recording gets the same room every
time and two conditions differ in the condition alone.
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

# RT60 in seconds, and SNR in dB against the signal RMS. `inf` SNR is no noise.
# A living room is roughly 0.4, a hall 0.8; 0.0 is the clean control and has to
# reproduce the published baseline exactly or the pipeline is adding something.
CONDITIONS = (
    (0.0, float("inf")),
    (0.4, float("inf")),
    (0.8, float("inf")),
    (0.0, 10.0),
    (0.4, 10.0),
    (0.8, 10.0),
)


def label(rt60: float, snr: float) -> str:
    return f"rt{rt60:g}_snr{'inf' if snr == float('inf') else f'{snr:g}'}"


def impulse_response(rt60: float, rate: float, rng: np.random.Generator
                     ) -> np.ndarray:
    """Exponentially decaying noise, direct path first and at unit gain."""
    if rt60 <= 0.0:
        return np.array([1.0], dtype=np.float64)
    length = max(2, int(1.5 * rt60 * rate))
    time = np.arange(length) / rate
    tail = rng.standard_normal(length) * np.power(10.0, -3.0 * time / rt60)
    # The direct path is sample zero and is not part of the decaying tail, so
    # the convolution introduces no delay at all against the annotations.
    tail[0] = 0.0
    energy = float(np.sqrt(np.sum(tail * tail)))
    if energy > 0.0:
        # A tail carrying about as much energy as the direct sound: wet enough
        # to matter, not so wet that the direct path is inaudible.
        tail *= 0.7 / energy
    tail[0] = 1.0
    return tail


def pink_noise(count: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by shaping white noise in the frequency domain."""
    spectrum = np.fft.rfft(rng.standard_normal(count))
    frequency = np.arange(len(spectrum))
    frequency[0] = 1
    shaped = spectrum / np.sqrt(frequency)
    out = np.fft.irfft(shaped, n=count)
    peak = float(np.max(np.abs(out)))
    return out / peak if peak > 0.0 else out


def degrade(mono: np.ndarray, rate: float, rt60: float, snr_db: float,
            seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = mono.astype(np.float64)
    if rt60 > 0.0:
        response = impulse_response(rt60, rate, rng)
        out = np.convolve(out, response)[: len(mono)]
    if np.isfinite(snr_db):
        signal_rms = float(np.sqrt(np.mean(out * out)))
        noise = pink_noise(len(out), rng)
        noise_rms = float(np.sqrt(np.mean(noise * noise)))
        if noise_rms > 0.0 and signal_rms > 0.0:
            gain = signal_rms / noise_rms * float(np.power(10.0, -snr_db / 20.0))
            out = out + gain * noise
    peak = float(np.max(np.abs(out)))
    # Scaled, never clipped: clipping is a separate degradation with its own
    # character, and mixing the two would make neither readable.
    return (out / peak * 0.98) if peak > 1.0 else out


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path,
                seed: int) -> dict:
    import soundfile

    audio, rate = soundfile.read(str(item["audio"]), dtype="float32",
                                 always_2d=True)
    mono = audio.mean(axis=1)
    rate = float(rate)

    out: dict = {"name": item["name"], "corpus": item["corpus"]}
    with tempfile.TemporaryDirectory() as directory:
        for rt60, snr in CONDITIONS:
            path = pathlib.Path(directory) / f"{label(rt60, snr)}.wav"
            soundfile.write(str(path),
                            degrade(mono, rate, rt60, snr, seed).astype(np.float32),
                            int(rate))
            done = subprocess.run(
                [str(binary), str(path), "--live", "--live-model", str(model),
                 "--live-sample-hz", repr(SAMPLE_HZ)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=False)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip()[:200])
            scored = _score_one(item, "model", binary, model,
                                estimate=Estimate.from_json(json.loads(done.stdout)))
            out[label(rt60, snr)] = {
                "usable": bool(scored.get("usable", False)),
                "reasons": list(scored.get("reasons", [])),
                "f_measure": scored.get("f_measure"),
                "p70": scored.get("p70"),
                "r70": scored.get("r70"),
                "acquired_at": scored.get("acquired_at"),
            }
    return out


def summarise(records: list[dict]) -> dict:
    out: dict = {"n": len(records)}
    for rt60, snr in CONDITIONS:
        key = label(rt60, snr)
        rows = [r[key] for r in records if key in r]
        if not rows:
            continue
        out[key] = {
            "usable_rate": float(np.mean([r["usable"] for r in rows])),
            "mean_f": float(np.mean([r["f_measure"] for r in rows
                                     if r["f_measure"] is not None])),
            "mean_r70": float(np.mean([r["r70"] for r in rows
                                       if r["r70"] is not None])),
            "mean_p70": float(np.mean([r["p70"] for r in rows
                                       if r["p70"] is not None])),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import concurrent.futures
    import os

    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False, frozenset(args.corpora))
    seen = sorted({item["corpus"] for item in items})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1
    if args.limit and len(items) > args.limit:
        items = items[:: -(-len(items) // args.limit)]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=repository).stdout.strip()
    clean = not subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               cwd=repository).stdout.strip()

    records: list[dict] = []
    failures: list[dict] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure_one, item, args.binary, args.model, seed): item
                   for seed, item in enumerate(items)}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            done += 1
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001
                failures.append({"name": item["name"], "error": str(error)[:300]})
            if done % 20 == 0 or done == len(items):
                print(f"{done}/{len(items)}", file=sys.stderr, flush=True)

    payload = {"commit": commit, "clean": clean, "corpora": args.corpora,
               "conditions": [label(a, b) for a, b in CONDITIONS],
               "requested": len(items), "failures": failures,
               "summary": summarise(records) if records else {},
               "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
