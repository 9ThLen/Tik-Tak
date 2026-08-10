#!/usr/bin/env python3
"""How much of Beat This!'s advantage survives a lookahead the live path affords?

Answers `eval/PREREGISTERED_causal_teacher.md`, the last gate before training a
front end. Two existing numbers cannot be composed: Beat This! through the
shipped decoder is +0.138 mean F on GTZAN but unbounded, and Beat This! bounded
to one second loses 3.1 points but through its own postprocessor at a
granularity coarser than the margin. This measures the bounded model through the
shipped decoder, at 0.1 s.

    cd research
    .venv/Scripts/python -m eval.causal_teacher \
        --manifest <ground-truth>/manifest.csv --music <music> \
        --binary <dump_analysis> --model <beatnet.ttw> \
        --teacher <beat_this.onnx> --output ../research/results/causal_teacher_gtzan.json

Every arm enters the same `LiveTracker` through `--live-activation`, BeatNet
included: its activation is dumped with `--dump-activation` and replayed by the
identical path, so no arm gets a delivery the others did not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]

from eval.beat_this_front_end import (SAMPLE_HZ, score,  # noqa: E402
                                      through_tracker)
from eval.beat_this_onnx import FPS, BeatThisOnnx  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402
from eval.live_corpus_benchmark import load_corpus  # noqa: E402

# One pass per this many seconds of audio, shared by every lookahead. The
# tighter this is the finer the bound and the more quadratically it costs:
# Beat This! is a transformer, and a 30 s prefix costs 18x a 5 s one.
STEP_SEC = 0.1

# Lookaheads in seconds. Reported as L + STEP_SEC, never as L, because a frame
# one step earlier in the same window saw a step more of its future.
LOOKAHEADS = (0.0, 0.1, 0.2, 0.4)


def arm_name(lookahead: float) -> str:
    return f"at_most_{lookahead + STEP_SEC:g}s"


def sample(items: list, limit: int) -> list:
    """A subset that spans the corpus instead of stopping partway through it.

    Copied in spirit from `oracle_activation.sample`: GTZAN is ordered by genre,
    so truncating to the first N drops whole genres. Rounding the stride up lets
    the stride do the limiting and always reach the end of the list.
    """
    if not limit or len(items) <= limit:
        return items
    return items[:: -(-len(items) // limit)][:limit]


def prefix_activations(session, mono: np.ndarray, rate: float,
                       duration: float) -> dict[float, np.ndarray]:
    """The beat probability the model reports having heard exactly T seconds."""
    lengths = list(np.arange(STEP_SEC, duration + STEP_SEC, STEP_SEC))
    if lengths[-1] < duration:
        lengths.append(duration)
    out: dict[float, np.ndarray] = {}
    for length in lengths:
        clip = mono[: max(1, int(length * rate))]
        if len(clip) < int(0.5 * rate):
            continue
        out[float(length)] = np.asarray(
            session.activations(clip, rate).beat_probability(),
            dtype=np.float64)
    return out


def bounded(prefixes: dict[float, np.ndarray], frames: int,
            lookahead: float) -> np.ndarray:
    """Assemble one activation in which frame t saw no audio past t + lookahead.

    The shortest prefix reaching `t + lookahead` is the one that may decide
    frame `t`; a longer one has heard more than the bound allows. Frames beyond
    every prefix keep the longest available, which only happens at the tail.
    """
    lengths = sorted(prefixes)
    out = np.zeros(frames, dtype=np.float64)
    for frame in range(frames):
        needed = frame / FPS + lookahead
        usable = [length for length in lengths if length >= needed - 1e-9]
        chosen = prefixes[usable[0] if usable else lengths[-1]]
        out[frame] = chosen[frame] if frame < len(chosen) else 0.0
    return out


def beatnet_activation(binary: pathlib.Path, audio: pathlib.Path,
                       model: pathlib.Path) -> np.ndarray:
    done = subprocess.run(
        [str(binary), str(audio), "--live", "--live-model", str(model),
         "--live-sample-hz", repr(SAMPLE_HZ), "--dump-activation"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{audio.name}: {done.stderr.strip()[:300]}")
    return np.asarray(json.loads(done.stdout)["activation_beat"],
                      dtype=np.float64)


def measure_one(item: dict, session: BeatThisOnnx, binary: pathlib.Path,
                model: pathlib.Path) -> dict:
    import soundfile

    samples, rate = soundfile.read(str(item["audio"]), dtype="float64",
                                   always_2d=True)
    mono = samples.mean(axis=1)
    rate = float(rate)
    duration = len(mono) / rate
    if duration < 10.0:
        raise RuntimeError("too short to score")

    out = {"name": item["name"], "corpus": item["corpus"],
           "duration": duration}

    activation = beatnet_activation(binary, item["audio"], model)
    out["beatnet"] = score(item, through_tracker(binary, item["audio"],
                                                 activation), binary, model)

    offline = np.asarray(
        session.activations(mono, rate).beat_probability(), dtype=np.float64)
    out["offline"] = score(item, through_tracker(binary, item["audio"], offline),
                           binary, model)

    prefixes = prefix_activations(session, mono, rate, duration)
    out["passes"] = len(prefixes)
    for lookahead in LOOKAHEADS:
        series = bounded(prefixes, len(offline), lookahead)
        out[arm_name(lookahead)] = score(
            item, through_tracker(binary, item["audio"], series), binary, model)
    return out


def summarise(records: list[dict]) -> dict:
    arms = ["beatnet"] + [arm_name(l) for l in LOOKAHEADS] + ["offline"]
    means = {arm: float(np.mean([r[arm]["f_measure"] for r in records
                                 if r[arm]["f_measure"] is not None]))
             for arm in arms}
    usable = {arm: float(np.mean([r[arm]["usable"] for r in records]))
              for arm in arms}
    span = means["offline"] - means["beatnet"]
    survival = {
        arm: ((means[arm] - means["beatnet"]) / span) if span > 0 else None
        for arm in arms}
    return {"n": len(records), "mean_f": means, "usable_rate": usable,
            "advantage_offline_over_beatnet": span,
            "survival_share": survival}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", default=["gtzan"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=int, default=40)
    args = parser.parse_args()

    items = sample(load_corpus(args.manifest, args.music, False,
                               frozenset(args.corpora)), args.clips)
    if not items:
        print("no items loaded", file=sys.stderr)
        return 1

    session = BeatThisOnnx(args.teacher)
    records, failures = [], []
    for index, item in enumerate(items, 1):
        try:
            records.append(measure_one(item, session, args.binary,
                                       args.model))
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            failures.append({"name": item["name"], "error": str(error)[:300]})
        print(f"{index}/{len(items)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    summary = summarise(records) if records else {}
    verdict = {}
    if summary:
        tightest = arm_name(LOOKAHEADS[0])
        share = summary["survival_share"][tightest]
        span = summary["advantage_offline_over_beatnet"]
        # Three states, not two. A share is a fraction of an advantage, so when
        # the unbounded teacher has no advantage to begin with there is nothing
        # for a bound to preserve and the question is unanswerable -- which is
        # not the same as answered no. A three-clip probe hit exactly that:
        # BeatNet scored 0.991 against the teacher's 0.963 on material with no
        # room to improve, and a two-state verdict would have called it a
        # failed gate.
        verdict = {
            "tightest_arm": tightest,
            "advantage_offline_over_beatnet": span,
            "survival_share": share,
            "outcome": ("not_measurable" if share is None
                        else "passed" if share >= 0.50 else "failed"),
            "at_least_half_survives": bool(share is not None and share >= 0.50),
        }

    # The shared wrapper rather than an inline `git status`: it raises on a
    # dirty or unreadable tree instead of recording a flag beside numbers that
    # are already written. An artifact that says `clean: false` is one a reader
    # has to notice; one that was never produced cannot be misread.
    sources = {"binary": args.binary, "model": args.model,
               "teacher": args.teacher, "manifest": args.manifest}
    run_provenance = provenance(REPOSITORY, sources)
    commit = run_provenance["commit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "provenance": run_provenance,
        "registered_in": "research/eval/PREREGISTERED_causal_teacher.md",
        "step_sec": STEP_SEC, "lookaheads": list(LOOKAHEADS),
        "sample_hz": SAMPLE_HZ, "clips_requested": len(items),
        "clips": [item["name"] for item in items],
        "failures": failures, "verdict": verdict, "summary": summary,
        "records": records,
    }, indent=2), encoding="utf-8")

    if summary:
        for arm, value in summary["mean_f"].items():
            share = summary["survival_share"][arm]
            tail = "" if share is None else f"  survival {share:+.3f}"
            print(f"{arm:14s} F {value:.4f}{tail}")
        print(f"advantage {verdict['advantage_offline_over_beatnet']:+.4f}  "
              f"outcome {verdict['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
