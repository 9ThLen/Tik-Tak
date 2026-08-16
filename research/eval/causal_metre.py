#!/usr/bin/env python3
"""Does the metre survive a causal trailing window?

`eval/PREREGISTERED_downbeat_audit.md`, "The causal arm, registered 2026-08-08".

The audit read the downbeat channel over a whole recording on the *annotated*
grid and found the metre carried decisively — 82.9% on Harmonix, 60.8% on GTZAN,
against a shuffled null of 30.1% and 23.5%. It called that a ceiling and said a
causal decoder would need its own registration. `44c8c56` built one: 32 beats of
the tracker's own grid, re-resolved on every beat, by `analysis::resolveMeter`
inside the shipping core.

**Measured through the core, not re-implemented.** The whole point of the seam
is that the thing measured is the thing that ships, and a Python trailing-window
decoder would answer a different question from the one the flag turns on.

**The arms differ in one file.** The model runs once per recording; the beat
channel, the frame-release schedule and the model's own timestamps are then
handed back exactly as `eval/octave_veto_replay` established they must be, and
each arm supplies a different *downbeat* file through `--live-downbeat`:

* `beat_sync` — the model's downbeat channel;
* `shuffled` — that channel permuted across frames, destroying its alignment to
  the grid while keeping its marginal distribution;
* `beat_as_downbeat` — the *beat* channel where the downbeat channel belongs.
  It is high at every beat, so an arm scoring well on it is finding periodicity
  in the grid it was handed rather than downbeat evidence.

Because only the downbeat file differs, and nothing reads the bar decision back
into the beat grid, all three arms must publish **identical beat lists**. That
is checked on every recording rather than assumed, and a mismatch is a leak in
the core rather than a result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.octave_veto_experiment import write_activation_cache  # noqa: E402
from eval.octave_veto_replay import run, run_activation  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402

ARMS = ("beat_sync", "shuffled", "beat_as_downbeat")

# The tolerance every beat-level agreement in this repository uses.
DOWNBEAT_TOLERANCE_SEC = 0.070

# `random_phase` is not an arm of the core: it is `beat_sync`'s own metre and
# settled grid with the bar line moved to a phase drawn from a seeded generator.
# With the metre almost always four a random phase is right one time in four, so
# this and not a shuffle is what the phase result has to clear -- the lesson the
# metre arm's missing constant baseline taught, applied before the run.
PHASE_NULL = "random_phase"

# The metres resolveMeter considers. Anything else in an annotation is outside
# the hypothesis set and cannot be got right, so it is excluded from scoring
# rather than counted as a failure of the window.
METRES = (2, 3, 4, 6)

# The same 8% the rest of this work uses for "the same metrical level".
OCTAVE_TOLERANCE = np.log2(1.08)

SAMPLE_HZ = 50.0


def annotated_metre(beats: np.ndarray, downbeats: np.ndarray) -> int:
    """Beats per bar from the annotation's own times.

    From the times and not from a manifest column, exactly as
    `downbeat_audit.audit_one` does it: the column is missing on some corpora
    and disagrees with the times on others.
    """
    if len(downbeats) < 3 or len(beats) < 2:
        return 0
    beat_period = float(np.median(np.diff(beats)))
    bar_period = float(np.median(np.diff(downbeats)))
    if not beat_period > 0.0:
        return 0
    return int(round(bar_period / beat_period))


def tracked_at_annotated_level(live_beats: np.ndarray,
                               reference: np.ndarray) -> bool:
    """Is the tracker's grid the annotated one, or an octave off it?

    Registered as confound 1: a causal metre read off a doubled grid is wrong
    for a reason that has nothing to do with causality, so the registered
    answer is restricted to recordings where this is true.
    """
    if len(live_beats) < 4 or len(reference) < 4:
        return False
    live_period = float(np.median(np.diff(live_beats)))
    true_period = float(np.median(np.diff(reference)))
    if not (live_period > 0.0 and true_period > 0.0):
        return False
    return bool(abs(np.log2(live_period / true_period)) <= OCTAVE_TOLERANCE)


def measure_one(item: dict, binary: pathlib.Path, weights: pathlib.Path,
                seed: int = 0) -> dict:
    """Every arm's reading of one recording."""
    from eval.live_corpus_benchmark import (load_reference_beats,
                                            load_reference_downbeats)

    reference = load_reference_beats(item["annotation"])
    downbeats = load_reference_downbeats(item["annotation"])
    true_metre = annotated_metre(reference, downbeats)

    initial = run(binary, item["audio"], weights, sample_hz=SAMPLE_HZ)
    beat_channel = np.asarray(initial["activation_beat"], dtype=np.float64)
    downbeat_channel = np.asarray(initial["activation_downbeat"], dtype=np.float64)
    baseline_beats = np.asarray(initial.get("beats", []), dtype=np.float64)

    # Seeded per recording rather than once for the run, so a recording's
    # control does not depend on how many recordings preceded it in the pool.
    rng = np.random.default_rng(seed)
    channels = {
        "beat_sync": downbeat_channel,
        "shuffled": rng.permutation(downbeat_channel),
        "beat_as_downbeat": beat_channel,
    }

    out = {
        "name": item["name"],
        "corpus": item["corpus"],
        "true_metre": true_metre,
        "reference_beats": int(len(reference)),
        "baseline_beats": int(len(baseline_beats)),
        "tracked_at_level": tracked_at_annotated_level(baseline_beats, reference),
    }

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        activation_path = root / "beat.txt"
        emit_path = root / "emit.txt"
        times_path = root / "times.txt"
        write_activation_cache(initial, activation_path, emit_path, times_path)

        for arm in ARMS:
            downbeat_path = root / f"{arm}.txt"
            np.savetxt(downbeat_path, channels[arm], fmt="%.17g")
            payload = run_activation(
                binary, item["audio"], activation_path,
                sample_hz=SAMPLE_HZ,
                extra=["--live-bars", "--live-downbeat", str(downbeat_path)],
                emit_path=emit_path, times_path=times_path)
            out[arm] = read_arm(payload, baseline_beats, true_metre,
                                reference_downbeats=downbeats)

    return out


def score_phase(beats: np.ndarray, positions: np.ndarray, meters: np.ndarray,
                reference_downbeats: np.ndarray,
                shift: int = 0, start_sec: float | None = None) -> dict:
    """F1 and F2 from the pre-registration's bar-phase addition.

    Only beats after the metre first answered are scored: before that there is no
    phase to be right or wrong about, and counting the unsettled prefix would
    mix "had not decided yet" into "decided wrong".

    `shift` rotates the bar line by that many positions, which is how the
    `random_phase` null is built — the same metre, the same settled grid, a
    different bar line.

    `start_sec` adds an experiment-wide suffix cut after the arm has first
    answered. M0a uses one value for all four arms; S0 uses its registered 2 s
    common cold-start cut.
    """
    empty = {"f1": None, "precision": None, "recall": None,
             "phase_correct_share": None, "scored_beats": 0, "downbeats": 0}
    if len(beats) == 0 or len(beats) != len(positions) or len(meters) != len(beats):
        return empty
    if len(reference_downbeats) < 2:
        return empty

    decided = meters > 0
    if not decided.any():
        return empty
    first = int(np.argmax(decided))
    beats = beats[first:]
    positions = positions[first:]
    meters = meters[first:]
    usable = (positions >= 0) & (meters > 0)
    if start_sec is not None:
        usable &= beats >= start_sec
    if not usable.any():
        return empty
    beats = beats[usable]
    positions = positions[usable]
    meters = meters[usable]

    shifted = np.mod(positions - shift, meters)

    # Only downbeats inside the span the tracker actually covered: a recording
    # whose first eight bars went by before the metre settled has not missed
    # them, it was not asked about them.
    span_low, span_high = beats[0], beats[-1]
    inside = reference_downbeats[(reference_downbeats >= span_low - DOWNBEAT_TOLERANCE_SEC)
                                 & (reference_downbeats <= span_high + DOWNBEAT_TOLERANCE_SEC)]
    if len(inside) == 0:
        return empty

    claimed = beats[shifted == 0]
    if len(claimed) == 0:
        return {**empty, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "phase_correct_share": 0.0,
                "scored_beats": int(len(beats)), "downbeats": int(len(inside))}

    def nearest(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        index = np.clip(np.searchsorted(targets, values), 1, len(targets) - 1)
        left = np.abs(values - targets[index - 1])
        right = np.abs(targets[index] - values)
        return np.minimum(left, right)

    hit = nearest(claimed, inside) <= DOWNBEAT_TOLERANCE_SEC
    precision = float(np.mean(hit))
    found = nearest(inside, claimed) <= DOWNBEAT_TOLERANCE_SEC
    recall = float(np.mean(found))
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)

    # F2: the share of scored beats whose position agrees with the annotated bar
    # phase. Derived from where the bar lines are rather than from a metre
    # label, so it needs no annotation column that may not exist.
    to_downbeat = nearest(beats, inside)
    should_be_zero = to_downbeat <= DOWNBEAT_TOLERANCE_SEC
    agree = (shifted == 0) == should_be_zero
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "phase_correct_share": float(np.mean(agree)),
        "scored_beats": int(len(beats)),
        "downbeats": int(len(inside)),
    }


def read_arm(payload: dict, baseline_beats: np.ndarray, true_metre: int,
             reference_downbeats: np.ndarray | None = None) -> dict:
    """One arm's answer, and the invariant that makes the arms comparable."""
    beats = np.asarray(payload.get("beats", []), dtype=np.float64)
    meters = np.asarray(payload.get("live_bar_meters", []), dtype=np.float64)
    positions = np.asarray(payload.get("live_bar_positions", []), dtype=np.float64)
    final = int(payload.get("live_beats_per_bar", 0))

    # The beat grid must not depend on the downbeat file. Checked, not assumed:
    # if it moves, the bar decision is leaking into the beat path and every
    # number here is about a tracker that does not ship.
    identical = bool(len(beats) == len(baseline_beats)
                     and np.array_equal(beats, baseline_beats))

    answered = meters[meters > 0]
    decided = meters > 0
    correct_points = (float(np.mean(meters[decided] == true_metre))
                      if decided.any() and true_metre else 0.0)

    # Beats before the held metre stops changing, counted from the end so that
    # a decoder that settles and stays is distinguished from one that happens
    # to end on the right answer.
    stable_after = len(meters)
    if len(answered) > 0:
        last = meters[-1]
        i = len(meters)
        while i > 0 and meters[i - 1] == last:
            i -= 1
        stable_after = i

    switches = int(np.sum(answered[1:] != answered[:-1])) if len(answered) > 1 else 0

    phase: dict = {}
    if reference_downbeats is not None:
        phase["actual"] = score_phase(beats, positions, meters, reference_downbeats)
        if final > 0:
            # The null: the same metre and the same settled grid with the bar
            # line placed uniformly at random. Computed as the mean over *all*
            # `final` rotations rather than by drawing one, which is the exact
            # expectation with no sampling noise and needs no seed.
            #
            # Rotation zero is included, and that is the point. An earlier
            # version excluded it "so the null cannot be the arm itself", which
            # made it not a random phase but a guaranteed-different one: it
            # scored exactly 0.0 whenever the arm was right, so it measured the
            # arm's correctness instead of baselining it. A uniform phase is
            # right one time in `final`, and that is the number to clear.
            rotations = [
                score_phase(beats, positions, meters, reference_downbeats,
                            shift=shift)
                for shift in range(final)
            ]
            valid = [r for r in rotations if r["f1"] is not None]
            phase[PHASE_NULL] = (
                {**valid[0],
                 "f1": float(np.mean([r["f1"] for r in valid])),
                 "precision": float(np.mean([r["precision"] for r in valid])),
                 "recall": float(np.mean([r["recall"] for r in valid])),
                 "phase_correct_share": float(
                     np.mean([r["phase_correct_share"] for r in valid]))}
                if valid else {"f1": None})

    return {
        "final_metre": final,
        "beats_identical": identical,
        "answered_beats": int(len(answered)),
        "total_beats": int(len(meters)),
        "correct_share_of_beats": correct_points,
        "stable_after_beats": int(stable_after),
        "switches": switches,
        "phase": phase,
    }


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson interval, as the audit reported its own accuracies."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def phase_block(subset: list[dict], arm: str, key: str = "actual") -> dict:
    """F1 and F2 over the recordings where the arm produced a phase at all."""
    scores = [r[arm]["phase"][key] for r in subset
              if r[arm].get("phase", {}).get(key, {}).get("f1") is not None]
    if not scores:
        return {"f1": None, "f1_n": 0, "f1_ci": None,
                "precision": None, "recall": None, "phase_correct_share": None}
    f1 = np.asarray([s["f1"] for s in scores], dtype=np.float64)
    # A mean of per-recording F-measures, and a bootstrap over recordings for
    # its interval: Wilson is for a proportion and these are not one.
    boot = np.asarray([
        np.mean(f1[np.random.default_rng(seed).integers(0, len(f1), len(f1))])
        for seed in range(2000)
    ])
    return {
        "f1": float(f1.mean()),
        "f1_n": int(len(f1)),
        "f1_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "precision": float(np.mean([s["precision"] for s in scores])),
        "recall": float(np.mean([s["recall"] for s in scores])),
        "phase_correct_share": float(
            np.mean([s["phase_correct_share"] for s in scores])),
    }


def summarise(records: list[dict]) -> dict:
    """The five registered measurements, per arm, per corpus."""
    out: dict = {}
    corpora = sorted({r["corpus"] for r in records})
    for corpus in corpora + ["all"]:
        rows = [r for r in records
                if corpus == "all" or r["corpus"] == corpus]
        scored = [r for r in rows if r["true_metre"] in METRES]
        restricted = [r for r in scored if r["tracked_at_level"]]
        # The baseline the registered conditions did not name, and should have.
        # A resolver carrying a metre prior can score well on a corpus that is
        # almost all one metre without deciding anything, so the question "is
        # this better than a constant" has to be asked out loud. Reported for
        # the restricted set, which is what C1-C3 are judged on.
        counts: dict[int, int] = {}
        for r in restricted:
            counts[r["true_metre"]] = counts.get(r["true_metre"], 0) + 1
        majority = max(counts, key=lambda m: counts[m]) if counts else 0
        block: dict = {
            "records": len(rows),
            "scored": len(scored),
            "tracked_at_level": len(restricted),
            "majority_metre": majority,
            "always_majority_accuracy": (counts[majority] / len(restricted)
                                         if restricted else None),
            "true_metre_counts": {str(k): v for k, v in sorted(counts.items())},
        }
        for arm in ARMS:
            def accuracy(subset: list[dict]) -> dict:
                if not subset:
                    return {"n": 0, "accuracy": None, "ci": None}
                hits = sum(1 for r in subset
                           if r[arm]["final_metre"] == r["true_metre"])
                low, high = wilson(hits, len(subset))
                return {"n": len(subset),
                        "accuracy": hits / len(subset),
                        "ci": [low, high]}

            answering = [r for r in restricted if r[arm]["final_metre"] > 0]
            # Accuracy where the majority prior cannot help. A corpus that is
            # almost all one metre cannot discriminate a decoder anywhere else.
            off_majority = [r for r in restricted if r["true_metre"] != majority]
            block[arm] = {
                "accuracy_off_majority": accuracy(off_majority),
                # M1, the registered answer and its unrestricted twin.
                "m1_restricted": accuracy(restricted),
                "m1_all": accuracy(scored),
                "m1_among_answering": accuracy(answering),
                # M2.
                "m2_correct_share_of_beats": (
                    float(np.mean([r[arm]["correct_share_of_beats"]
                                   for r in restricted])) if restricted else None),
                # M3.
                "m3_coverage": (
                    float(np.mean([r[arm]["final_metre"] > 0 for r in scored]))
                    if scored else None),
                # M4, M5.
                "m4_median_stable_after_beats": (
                    float(np.median([r[arm]["stable_after_beats"]
                                     for r in restricted])) if restricted else None),
                "m5_median_switches": (
                    float(np.median([r[arm]["switches"] for r in restricted]))
                    if restricted else None),
                "beats_identical": (
                    int(sum(1 for r in rows if r[arm]["beats_identical"]))),
                **phase_block(restricted, arm),
            }
        block[PHASE_NULL] = phase_block(restricted, "beat_sync", key=PHASE_NULL)
        out[corpus] = block
    return out


def main(argv: list[str] | None = None) -> int:
    import concurrent.futures
    import os

    from eval.live_corpus_benchmark import load_corpus

    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth" / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path, default=repository / "music")
    parser.add_argument("--corpora", nargs="+", default=["gtzan"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)

    # Third positional is include_root_audio, not the corpus filter. Passing the
    # corpus list there reads as true, leaves the filter at None, and silently
    # runs the default set -- which includes Ballroom, in this model's training
    # set. Named, because the run it produced looked entirely healthy.
    items = load_corpus(args.manifest, args.music, False,
                        frozenset(args.corpora))
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("no recordings", file=sys.stderr)
        return 1
    seen = sorted({item["corpus"] for item in items})
    if seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1

    # Capture before the run: an output inside the tree would otherwise make
    # the run report itself as dirty.
    run_provenance = provenance(
        repository,
        {"manifest": args.manifest, "binary": args.binary,
         "model": args.model},
    )

    records: list[dict] = []
    failures: list[dict] = []
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(measure_one, item, args.binary, args.model, seed): item
            for seed, item in enumerate(items)
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            done_count += 1
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001
                failures.append({"name": item["name"], "error": str(error)[:300]})
            if done_count % 25 == 0 or done_count == len(items):
                print(f"{done_count}/{len(items)}", file=sys.stderr, flush=True)

    payload = {
        "provenance": run_provenance,
        "model": str(args.model),
        "corpora": args.corpora,
        "requested": len(items),
        "failures": failures,
        "summary": summarise(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
