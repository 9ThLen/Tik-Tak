#!/usr/bin/env python3
"""Does Beat This!'s advantage survive not being allowed to see the future?

`Beat This!` is the one measured better observation available — through the core
it gains +0.102 beat F and +0.138 CMLt on GTZAN. It is also **not causal**: it
is a transformer over the whole spectrogram, so every beat it places was decided
knowing how the recording ends. The live path cannot use that, and no causal
replacement has been measured at all.

Before anyone trains one, this asks the cheap question: **how much of the
advantage is the model, and how much is the lookahead?**

## What is measured

For a lookahead `L`, a beat at time `t` may be decided from audio up to `t + L`
and no later. That is exactly "run the model on the prefix `[0, t + L]`", so the
sweep is a set of prefix runs and nothing more elaborate:

* activations are computed on prefixes at one-second granularity, **once**;
* for each `L`, the beats emitted in `(t - 1, t]` are taken from the prefix run
  of length `t + L`.

**The lookahead is a range, not a point, and the labels say so.** A beat at `t`
taken from the prefix `t + L` saw exactly `L` seconds past itself; a beat at
`t - 1` in the same window saw `L + 1`. So the sweep bounds the lookahead from
above by `L + STEP_SEC` and that is what the reported keys are named after.
Calling the `L = 0` arm "strictly causal" would be wrong by up to a second, and
a second is more than the whole margin being measured.

Every lookahead therefore reads the same set of model passes, which is what
makes a six-point sweep cost the same as one. `L = inf` is the whole file — the
offline number, and the top of the curve by construction.

## Two things this cannot say

**The absolute level.** `models/beat_this.onnx` is the `final0` checkpoint,
trained on the full corpus, and that corpus includes GTZAN, Ballroom and RWC.
Every absolute figure here is recall of training material. The *shape* of the
degradation is the result; the height of the curve is not quotable.

**How a model trained to be causal would behave.** Depriving a bidirectional
model of its right-hand context measures what that context was worth to *it*.
A model trained without it would learn different features. This is an upper
bound on what a truncation-based causal Beat This! gives, and says nothing
about what a purpose-built one could.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.beat_this_onnx import BeatThisOnnx, beats_and_downbeats  # noqa: E402
from eval.live_corpus_benchmark import (load_corpus,  # noqa: E402
                                        load_reference_beats)

# inf is the whole file. 0 is the tightest arm reachable at this step size, and
# it is *not* strictly causal — see _key: a beat may still be read from a prefix
# up to STEP_SEC past it.
LOOKAHEADS = (float("inf"), 4.0, 2.0, 1.0, 0.5, 0.0)

STEP_SEC = 1.0
WINDOW_SEC = 0.070
# The same five-second warm-up every live number in this repository uses, so
# the F here is comparable with the ones it is being weighed against.
WARMUP_SEC = 5.0


def _key(lookahead: float) -> str:
    """Named for the *upper* bound on how far past a beat the model could hear.

    A beat at `t` read from the prefix `t + L` saw L seconds of its future; the
    beat a step earlier in the same window saw `L + STEP_SEC`. The bound is what
    a reader needs, so the bound is what the key says.
    """
    if lookahead == float("inf"):
        return "offline"
    return f"at_most_{lookahead + STEP_SEC:g}s"


def prefix_beats(session: BeatThisOnnx, mono: np.ndarray, rate: float,
                 seconds: float) -> np.ndarray:
    """Beats the model places when it has heard exactly `seconds` of audio."""
    clip = mono[: max(1, int(seconds * rate))]
    if len(clip) < int(0.5 * rate):
        return np.zeros(0, dtype=np.float64)
    beats, _ = beats_and_downbeats(session.activations(clip, rate))
    return np.asarray(beats, dtype=np.float64)


def causal_beats(prefixes: dict[float, np.ndarray], duration: float,
                 lookahead: float) -> np.ndarray:
    """Stitch a beat list where nothing was decided more than `lookahead` early."""
    if lookahead == float("inf"):
        return prefixes[max(prefixes)]
    out: list[float] = []
    step = STEP_SEC
    t = step
    while t <= duration + step:
        available = t + lookahead
        # The nearest prefix that does not exceed what the model may hear.
        usable = [p for p in prefixes if p <= available + 1e-9]
        if usable:
            beats = prefixes[max(usable)]
            out.extend(beats[(beats > t - step) & (beats <= t)])
        t += step
    return np.asarray(sorted(out), dtype=np.float64)


def f_measure(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference[reference >= WARMUP_SEC]
    estimate = estimate[estimate >= WARMUP_SEC]
    if len(reference) == 0 or len(estimate) == 0:
        return 0.0
    used = np.zeros(len(estimate), dtype=bool)
    hits = 0
    for beat in reference:
        near = np.flatnonzero((np.abs(estimate - beat) <= WINDOW_SEC) & ~used)
        if len(near):
            used[near[np.argmin(np.abs(estimate[near] - beat))]] = True
            hits += 1
    precision = hits / len(estimate)
    recall = hits / len(reference)
    return (2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0)


def measure_one(item: dict, model: pathlib.Path) -> dict:
    import soundfile

    audio, rate = soundfile.read(str(item["audio"]), dtype="float32",
                                 always_2d=True)
    mono = audio.mean(axis=1)
    rate = float(rate)
    duration = len(mono) / rate
    reference = load_reference_beats(item["annotation"])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 8 or duration < 10.0:
        raise RuntimeError("too short to score")

    session = _session(model)
    # One pass per second of audio, shared by every lookahead.
    lengths = list(np.arange(STEP_SEC, duration + STEP_SEC, STEP_SEC))
    if lengths[-1] < duration:
        lengths.append(duration)
    prefixes = {float(length): prefix_beats(session, mono, rate, float(length))
                for length in lengths}

    out = {"name": item["name"], "corpus": item["corpus"],
           "duration": duration, "passes": len(prefixes)}
    for lookahead in LOOKAHEADS:
        out[_key(lookahead)] = f_measure(
            reference, causal_beats(prefixes, duration, lookahead))
    return out


_sessions: dict[str, BeatThisOnnx] = {}


def _session(model: pathlib.Path) -> BeatThisOnnx:
    """One session per thread name; building it costs about five seconds."""
    import threading
    key = threading.current_thread().name
    if key not in _sessions:
        _sessions[key] = BeatThisOnnx(model)
    return _sessions[key]


def summarise(records: list[dict]) -> dict:
    out: dict = {}
    for corpus in sorted({r["corpus"] for r in records}):
        rows = [r for r in records if r["corpus"] == corpus]
        block = {"n": len(rows)}
        for lookahead in LOOKAHEADS:
            key = _key(lookahead)
            block[key] = float(np.mean([r[key] for r in rows]))
        out[corpus] = block
    return out


def main(argv: list[str] | None = None) -> int:
    import concurrent.futures

    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False, frozenset(args.corpora))
    seen = sorted({item["corpus"] for item in items})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1
    if args.limit and len(items) > args.limit:
        # Stride rather than truncate: GTZAN is filed by genre, and the first
        # N recordings are the first N genres.
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
        futures = {pool.submit(measure_one, item, args.model): item
                   for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            done += 1
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001
                failures.append({"name": item["name"], "error": str(error)[:300]})
            if done % 10 == 0 or done == len(items):
                print(f"{done}/{len(items)}", file=sys.stderr, flush=True)

    payload = {"commit": commit, "clean": clean, "model": str(args.model),
               "corpora": args.corpora, "requested": len(items),
               "step_sec": STEP_SEC, "warmup_sec": WARMUP_SEC,
               "failures": failures,
               "summary": summarise(records) if records else {},
               "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
