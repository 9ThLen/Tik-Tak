#!/usr/bin/env python3
"""Is a bar period recoverable from BeatNet's outputs at all?

`eval/PREREGISTERED_downbeat_audit.md`. The previous experiment showed that an
accurate bar period is worth 19.1 points of episode-freeness and that one
autocorrelation over the downbeat channel could not supply one. It could not say
which of two things was wrong: the channel carries nothing usable, or
`ActivationTempo` is the wrong instrument. This separates them, offline, with no
change to the live core.

**Beat-synchronous, because that is what the head answers.** The downbeat output
is not a slow tempo detector — it says *which of these beats begins a bar*, a
question that only means anything at beat positions. Autocorrelating a 50 fps
probability track asks it something it was never trained for, and spends the
beat grid, which is the one thing already known accurately.

So the decoder reads the downbeat probability at each beat of a given grid and,
for every metre in {2, 3, 4, 6} and every bar phase within it, scores the
contrast between the beats that would be downbeats and the beats that would not:

    score(m, p) = mean(d[i] for i mod m == p) - mean(d[i] for i mod m != p)

The grid's score is the best over all (m, p); its margin is the gap to the best
score of any *other* metre.

**The octave question, asked directly.** Score the annotated grid, then score
that grid doubled — every beat plus every midpoint. On a true 4/4 the downbeats
land every four beats of the first and every eight of the second, and eight is
not a metre, so the doubled grid should score worse. Which grid the evidence
prefers is the whole quantity this line of work has been circling, and it has
never been measured directly.

Whole-recording rather than over the last few bars, deliberately: this is a
ceiling. A causal decoder seeing two to four bars cannot beat what an offline
one seeing all of them extracts, so a failure here is a failure for any reading
of this channel, which is what makes a negative worth stopping on.

**The controls decide whether any of it means anything.** `shuffled` permutes
the downbeat probabilities across beat positions, destroying their alignment to
the grid while keeping the marginal distribution — anything above it is reading
structure rather than level. `beat-as-downbeat` feeds the beat channel where the
downbeat channel belongs; the beat channel is high at *every* beat, so a decoder
scoring well on it is finding periodicity in the grid it was handed rather than
downbeat evidence. A result that does not clear both is not a result.

**Measured 2026-08-05, and the answer is split.** GTZAN 991 scored, Harmonix
579:

                        metre                 octave separation
    GTZAN
      beat-sync         60.8% [57.7, 63.9]    76.2% [73.4, 78.8]
      shuffled          23.5% [20.9, 26.3]    84.2% [81.7, 86.4]
      beat-as-downbeat  38.1% [35.1, 41.2]    23.6% [21.0, 26.4]
    Harmonix
      beat-sync         82.9% [79.6, 85.9]    79.6% [76.1, 82.8]
      shuffled          30.1% [26.3, 34.0]    84.1% [80.9, 87.0]
      beat-as-downbeat  60.1% [56.0, 64.1]     7.8% [ 5.7, 10.3]

**The metre is carried and the octave is not.** On metre, `beat-sync` clears
shuffled by 37 and 53 points and `beat-as-downbeat` by 23 on each — and that
second comparison is the load-bearing one, since the beat channel alone reaches
38% and 60% from the grid's own periodicity.

On the octave it scores 76.2% and 79.6% while **shuffled noise scores 84.2% and
84.1%**. It is behind its own null on both corpora, and the intervals do not
overlap on GTZAN. A2 asked for fifteen points ahead; the result is eight and
four and a half points behind, so by the terms fixed before the run this closes
the downbeat head for octave correction.

Read that null carefully, because without it the 76-80% would have been reported
as a success. It is high for a structural reason and not because the task is
easy: the doubled grid carries twice as many points, so the maximum over
(metre, phase) of a noise contrast is smaller there, and the comparison tilts
toward the shorter grid before any evidence is consulted. `beat-as-downbeat`
shows the same instrument from the other side — at 23.6% and 7.8% it prefers the
doubled grid outright, which is what finding periodicity in the handed grid
looks like.

The `autocorr` arm named in the pre-registration is not implemented here, so P1
is unmeasured and is recorded as a gap rather than dropped. It cannot move a
verdict that turns on the controls.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np

__all__ = ["activations", "decode_grid", "double_grid", "audit_one", "METRES"]

# The metres this product is for. Eight is deliberately absent: it is what a
# doubled 4/4 implies, and admitting it would let the wrong octave call itself a
# long bar — the same reasoning that shaped `barEndorsedOctave` before it was
# reverted.
METRES = (2, 3, 4, 6)

MODEL_RATE = 22050.0
ARMS = ("beat-sync", "shuffled", "beat-as-downbeat")


def activations(binary: pathlib.Path, audio: pathlib.Path,
                weights: pathlib.Path) -> np.ndarray:
    """`(frames, 3)` of time, beat, downbeat, from the core's own network.

    Through `dump_beatnet` rather than a Python reimplementation, for the reason
    the parity tools exist at all: the thing being measured has to be the thing
    that ships.
    """
    import soundfile
    from scipy.signal import resample_poly

    samples, rate = soundfile.read(str(audio), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if int(rate) != int(MODEL_RATE):
        mono = resample_poly(mono, int(MODEL_RATE), int(rate))
        rate = MODEL_RATE

    handle = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    try:
        handle.write(np.asarray(mono, dtype=np.float32).tobytes())
        handle.close()
        done = subprocess.run(
            [str(binary), handle.name, repr(float(rate)), str(weights)],
            capture_output=True, text=True)
    finally:
        pathlib.Path(handle.name).unlink(missing_ok=True)

    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip()[:200] or "dump_beatnet failed")
    body = done.stdout.strip().split("\n")[1:]
    if not body:
        raise RuntimeError("no activation rows")
    return np.loadtxt(io.StringIO("\n".join(body)), delimiter=",")


def sample_at(track: np.ndarray, channel: int, times: np.ndarray) -> np.ndarray:
    """The channel's value at each time, from the nearest frame.

    Nearest rather than interpolated: the activation is a probability per frame,
    not a continuous signal, and a beat that falls between two frames is better
    described by the frame it is nearest than by a blend of two.
    """
    frame_times = track[:, 0]
    index = np.clip(np.searchsorted(frame_times, times), 1, len(frame_times) - 1)
    left = np.abs(times - frame_times[index - 1])
    right = np.abs(frame_times[index] - times)
    pick = np.where(left <= right, index - 1, index)
    return track[pick, channel]


def double_grid(beats: np.ndarray) -> np.ndarray:
    """The same grid at twice the rate: every beat plus every gap's centre."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 2:
        return beats
    midpoints = 0.5 * (beats[:-1] + beats[1:])
    return np.sort(np.concatenate([beats, midpoints]))


def decode_grid(values: np.ndarray) -> dict:
    """The best metre and bar phase for one grid, and how far ahead it is.

    `margin` is the gap to the best score of a *different* metre, not to the
    second-best phase of the same one. A metre that wins only because one of its
    own phases came second has not been distinguished from anything.
    """
    values = np.asarray(values, dtype=np.float64)
    best = {"metre": 0, "phase": 0, "score": -np.inf}
    per_metre: dict[int, float] = {}
    for metre in METRES:
        if len(values) < metre * 2:
            continue
        index = np.arange(len(values)) % metre
        metre_best = -np.inf
        for phase in range(metre):
            on = values[index == phase]
            off = values[index != phase]
            if on.size == 0 or off.size == 0:
                continue
            score = float(on.mean() - off.mean())
            metre_best = max(metre_best, score)
            if score > best["score"]:
                best = {"metre": metre, "phase": phase, "score": score}
        if np.isfinite(metre_best):
            per_metre[metre] = metre_best

    others = [s for m, s in per_metre.items() if m != best["metre"]]
    best["margin"] = float(best["score"] - max(others)) if others else 0.0
    best["answered"] = bool(np.isfinite(best["score"]))
    return best


def audit_one(item: dict, binary: pathlib.Path, weights: pathlib.Path,
              seed: int = 0) -> dict:
    """Every arm's reading of one recording."""
    from eval.live_corpus_benchmark import (load_reference_beats,
                                            load_reference_downbeats)

    track = activations(binary, item["audio"], weights)
    beats = load_reference_beats(item["annotation"])
    downbeats = load_reference_downbeats(item["annotation"])
    if len(beats) < 8:
        raise RuntimeError("too few annotated beats")

    inside = (beats >= track[0, 0]) & (beats <= track[-1, 0])
    beats = beats[inside]
    doubled = double_grid(beats)

    rng = np.random.default_rng(seed)
    channels = {
        "beat-sync": sample_at(track, 2, beats),
        "beat-as-downbeat": sample_at(track, 1, beats),
    }
    channels["shuffled"] = rng.permutation(channels["beat-sync"])

    doubled_channels = {
        "beat-sync": sample_at(track, 2, doubled),
        "beat-as-downbeat": sample_at(track, 1, doubled),
    }
    doubled_channels["shuffled"] = rng.permutation(doubled_channels["beat-sync"])

    # The annotated metre, as beats per bar, from the annotation's own downbeats
    # rather than from a manifest column: the column is missing on some corpora
    # and disagrees with the times on others.
    true_metre = 0
    if len(downbeats) >= 3 and len(beats) >= 2:
        beat_period = float(np.median(np.diff(beats)))
        bar_period = float(np.median(np.diff(downbeats)))
        if beat_period > 0:
            true_metre = int(round(bar_period / beat_period))

    out = {"name": item["name"], "corpus": item["corpus"],
           "beats": int(len(beats)), "true_metre": true_metre,
           "true_bar_sec": float(np.median(np.diff(downbeats)))
           if len(downbeats) >= 3 else 0.0,
           "beat_sec": float(np.median(np.diff(beats))) if len(beats) >= 2 else 0.0}

    for arm in ARMS:
        true_read = decode_grid(channels[arm])
        doubled_read = decode_grid(doubled_channels[arm])
        out[arm] = {
            "metre": true_read["metre"],
            "phase": true_read["phase"],
            "score": true_read["score"],
            "margin": true_read["margin"],
            # The octave question: does the evidence prefer the annotated grid
            # over the same grid doubled? Compared on the winning score, which
            # is the quantity a decoder would actually act on.
            "prefers_true_octave": bool(true_read["score"] > doubled_read["score"]),
            "octave_gap": float(true_read["score"] - doubled_read["score"]),
            "doubled_metre": doubled_read["metre"],
        }
    return out


def summarise(records: list[dict]) -> dict:
    """Per arm, the quantities the pre-registration named."""
    scored = [r for r in records if r.get("true_metre") in METRES]
    out: dict = {"scored": len(scored), "records": len(records)}
    for arm in ARMS:
        metre_right = [r[arm]["metre"] == r["true_metre"] for r in scored]
        octave_right = [r[arm]["prefers_true_octave"] for r in scored]
        bar_right = []
        for r in scored:
            recovered = r[arm]["metre"] * r["beat_sec"]
            true_bar = r["true_bar_sec"]
            bar_right.append(bool(true_bar > 0 and
                                  abs(np.log2(recovered / true_bar)) <= np.log2(1.08)))
        out[arm] = {
            "metre_accuracy": float(np.mean(metre_right)) if scored else None,
            "octave_separation": float(np.mean(octave_right)) if scored else None,
            "bar_period_accuracy": float(np.mean(bar_right)) if scored else None,
            "median_margin": float(np.median([r[arm]["margin"] for r in scored]))
            if scored else None,
            "median_octave_gap": float(
                np.median([r[arm]["octave_gap"] for r in scored])) if scored else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import concurrent.futures

    from eval.live_corpus_benchmark import load_corpus

    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth"
                        / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path,
                        default=repository / "music")
    parser.add_argument("--corpora", nargs="+", default=["gtzan"])
    parser.add_argument("--binary", type=pathlib.Path,
                        default=repository / "tools" / "parity" / "build"
                        / "RelWithDebInfo" / "dump_beatnet.exe")
    parser.add_argument("--weights", type=pathlib.Path,
                        default=repository / "models" / "beatnet_model_1.ttw")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False, set(args.corpora))
    if args.limit:
        items = items[:args.limit]
    print(f"   {len(items)} recordings, {args.workers} workers")

    records = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(audit_one, item, args.binary, args.weights,
                               index): item
                   for index, item in enumerate(items)}
        for done in concurrent.futures.as_completed(futures):
            try:
                records.append(done.result())
            except Exception as error:  # noqa: BLE001 - one bad file is not a run
                print(f"   skipped {futures[done]['name']}: {error}")

    report = summarise(records)
    print(f"\n   {report['scored']} of {report['records']} had a metre to score")
    print(f"\n   {'arm':<18}{'metre':>9}{'octave':>9}{'bar period':>12}"
          f"{'margin':>9}{'oct gap':>9}")
    for arm in ARMS:
        block = report[arm]
        print(f"   {arm:<18}{block['metre_accuracy']:>8.1%}"
              f"{block['octave_separation']:>9.1%}"
              f"{block['bar_period_accuracy']:>12.1%}"
              f"{block['median_margin']:>9.4f}{block['median_octave_gap']:>9.4f}")

    if args.output:
        args.output.write_text(json.dumps(
            {"corpora": args.corpora, "summary": report, "records": records},
            indent=2), encoding="utf-8")
        print(f"\n   wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
