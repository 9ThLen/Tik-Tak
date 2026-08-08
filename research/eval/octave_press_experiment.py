#!/usr/bin/env python3
"""How much of the octave ceiling does a person pressing a button recover?

`eval/PREREGISTERED_octave_press.md`. Development on RWC, transfer to Harmonix,
and no Harmonix run until RWC has fixed every free parameter.

**Nothing is re-implemented.** The listener is `SimulatedListener` inside
`dump_analysis`, online because its own presses change which stretches are at
the wrong level afterwards. Scoring is `live_corpus_benchmark._score_one`, the
same function every published live number came from, so `usable_rate` here means
what it means everywhere else in this repository.

The arms differ only in flags:

* `baseline` — nothing;
* `press` — the listener of the registration's sections 1 to 3;
* `press_random` — the times `press` actually realised, direction drawn from a
  seeded generator. **This is the arm the experiment stands on.**
  `setOctaveOffset` re-seeds the cloud and shifts the anchor, and any such
  disturbance perturbs a stuck tracker; without this the result cannot separate
  "the judgement helps" from "kicking the filter helps";
* `press_delayed` — the same times and the correct direction, one notice period
  later, which separates "the correction helped" from "it helped because it was
  early".
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
from eval.live_corpus_benchmark import (_score_one, load_corpus,  # noqa: E402
                                        load_reference_beats)

ARMS = ("baseline", "press", "press_random", "press_delayed")

# Registered primaries. `usable_rate` is the product criterion unmodified;
# `correct_share_of_eligible` is what a control that *ends* episodes can move,
# where the wrong-level clause of the first asks it to prevent them.
PRIMARY = "usable"
CO_PRIMARY = "correct_share_of_eligible"

# Everything is read at 50 Hz, including the baseline it is compared against.
# acquired_at off a 1 Hz confidence poll is wrong by seconds through aliasing,
# and the acquisition bar is one of usable_rate's four clauses.
SAMPLE_HZ = 50.0


def run_dump(binary: pathlib.Path, audio: pathlib.Path, model: pathlib.Path,
             extra: list[str]) -> dict:
    args = [str(binary), str(audio), "--live", "--live-model", str(model),
            "--live-sample-hz", repr(SAMPLE_HZ)] + extra
    done = subprocess.run(args, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"dump_analysis failed on {audio.name}: "
                           f"{done.stderr.strip()[:300]}")
    return json.loads(done.stdout)


def schedule_file(path: pathlib.Path, times: np.ndarray, ks: np.ndarray) -> None:
    """Pairs of time and k, one press per line, as --live-press-schedule reads."""
    rows = np.empty(2 * len(times), dtype=np.float64)
    rows[0::2] = times
    rows[1::2] = ks
    np.savetxt(path, rows, fmt="%.17g")


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path,
                notice_sec: float, max_presses: int, seed: int) -> dict:
    reference = load_reference_beats(item["annotation"])
    out: dict = {"name": item["name"], "corpus": item["corpus"]}

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        reference_path = root / "reference.txt"
        np.savetxt(reference_path, reference, fmt="%.17g")

        payloads: dict[str, dict] = {}
        payloads["baseline"] = run_dump(binary, item["audio"], model, [])
        payloads["press"] = run_dump(
            binary, item["audio"], model,
            ["--live-press", repr(float(notice_sec)), str(int(max_presses)),
             "--live-press-reference", str(reference_path)])

        times = np.asarray(payloads["press"].get("live_press_times", []),
                           dtype=np.float64)
        ks = np.asarray(payloads["press"].get("live_press_k", []), dtype=np.float64)

        if len(times) == 0:
            # No press to control against. Both controls are the baseline by
            # construction; running them would burn two model passes to
            # reproduce it, and recording that they are equal is more honest
            # than recording a number that came from somewhere else.
            payloads["press_random"] = payloads["baseline"]
            payloads["press_delayed"] = payloads["baseline"]
            out["controls_are_baseline"] = True
        else:
            out["controls_are_baseline"] = False
            rng = np.random.default_rng(seed)
            random_k = rng.choice(np.array([-1.0, 1.0]), size=len(times))
            random_path = root / "random.txt"
            schedule_file(random_path, times, random_k)
            payloads["press_random"] = run_dump(
                binary, item["audio"], model,
                ["--live-press-schedule", str(random_path)])

            delayed_path = root / "delayed.txt"
            schedule_file(delayed_path, times + notice_sec, ks)
            payloads["press_delayed"] = run_dump(
                binary, item["audio"], model,
                ["--live-press-schedule", str(delayed_path)])

    for arm in ARMS:
        payload = payloads[arm]
        scored = _score_one(item, "model", binary, model,
                            estimate=Estimate.from_json(payload))
        out[arm] = {
            "usable": bool(scored.get("usable", False)),
            "usable_any_octave": bool(scored.get("usable_any_octave", False)),
            "reasons": scored.get("reasons", []),
            "correct_share_of_eligible": scored.get("correct_share_of_eligible"),
            "switches": scored.get("switches"),
            "settled_at": scored.get("settled_at"),
            "acquired_at": scored.get("acquired_at"),
            "f_measure": scored.get("f_measure"),
            "p70": scored.get("p70"),
            "r70": scored.get("r70"),
            "presses": int(payload.get("live_press_count", 0)),
            "refusals": int(payload.get("live_press_refusals", 0)),
            "press_times": [float(t) for t in
                            payload.get("live_press_times", [])],
            "press_k": [int(k) for k in payload.get("live_press_k", [])],
            "press_accepted": [int(a) for a in
                               payload.get("live_press_accepted", [])],
            "octave_offset": int(payload.get("live_octave_offset", 0)),
        }

    # C4's population: recordings the baseline never had at the wrong level, on
    # which the listener should by construction never fire.
    out["baseline_level_clean"] = bool(
        out["baseline"]["correct_share_of_eligible"] is not None
        and out["baseline"]["correct_share_of_eligible"] >= 0.999)
    return out


def paired_bootstrap(records: list[dict], arm: str, against: str, key: str,
                     resamples: int = 10000, seed: int = 0) -> dict:
    """Difference of means with a cluster bootstrap over recordings.

    The recording is the cluster because two excerpts of one piece are not two
    observations, and the arms are paired within it.
    """
    a, b = [], []
    for r in records:
        x, y = r[arm].get(key), r[against].get(key)
        if isinstance(x, bool):
            x, y = float(x), float(y)
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        a.append(float(x))
        b.append(float(y))
    if not a:
        return {"n": 0, "difference": None, "ci": None, "p": None}
    a, b = np.asarray(a), np.asarray(b)
    observed = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(a), size=(resamples, len(a)))
    draws = a[index].mean(axis=1) - b[index].mean(axis=1)
    return {
        "n": int(len(a)),
        "difference": observed,
        "mean_arm": float(a.mean()),
        "mean_against": float(b.mean()),
        "ci": [float(np.percentile(draws, 2.5)),
               float(np.percentile(draws, 97.5))],
        # Two-sided, by how often a resample lands on the other side of zero.
        "p": float(2.0 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def holm(tests: dict[str, float]) -> dict[str, float]:
    """Holm over the three named tests, and only those three."""
    ordered = sorted((p, name) for name, p in tests.items() if p is not None)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (p, name) in enumerate(ordered):
        adjusted = min(1.0, p * (len(ordered) - rank))
        running = max(running, adjusted)
        out[name] = running
    return out


def summarise(records: list[dict]) -> dict:
    out: dict = {"records": len(records)}
    for arm in ARMS:
        usable = [r[arm]["usable"] for r in records]
        share = [r[arm]["correct_share_of_eligible"] for r in records
                 if r[arm]["correct_share_of_eligible"] is not None]
        out[arm] = {
            "usable_rate": float(np.mean(usable)) if usable else None,
            "correct_share_of_eligible": float(np.mean(share)) if share else None,
            "mean_f": float(np.mean([r[arm]["f_measure"] for r in records
                                     if r[arm]["f_measure"] is not None])),
            "presses_total": int(sum(r[arm]["presses"] for r in records)),
            "refusals_total": int(sum(r[arm]["refusals"] for r in records)),
        }

    a1 = paired_bootstrap(records, "press", "baseline", PRIMARY)
    a2 = paired_bootstrap(records, "press", "press_random", PRIMARY)
    a3 = paired_bootstrap(records, "press", "baseline", CO_PRIMARY)
    out["A1_press_vs_baseline_usable"] = a1
    out["A2_press_vs_random_usable"] = a2
    out["A3_press_vs_baseline_share"] = a3
    out["holm"] = holm({"A1": a1["p"], "A2": a2["p"], "A3": a3["p"]})
    out["press_delayed_vs_press_usable"] = paired_bootstrap(
        records, "press_delayed", "press", PRIMARY)

    # C4: the recordings the listener should never have touched.
    clean = [r for r in records if r["baseline_level_clean"]]
    out["C4_clean_recordings"] = len(clean)
    out["C4_presses_on_clean"] = int(sum(r["press"]["presses"] for r in clean))
    out["C4_usable_change_on_clean"] = (
        paired_bootstrap(clean, "press", "baseline", PRIMARY) if clean else None)

    # S2 and S3.
    fired = [r for r in records if r["press"]["presses"] > 0]
    out["S2_recordings_that_pressed"] = len(fired)
    out["S2_press_histogram"] = {
        str(n): int(sum(1 for r in records if r["press"]["presses"] == n))
        for n in range(0, 4)
    }
    attempts = sum(len(r["press"]["press_accepted"]) for r in records)
    refused = sum(r["press"]["refusals"] for r in records)
    out["S3_attempts"] = attempts
    out["S3_refusal_rate"] = (refused / attempts) if attempts else None
    out["S3_refused_by_direction"] = {
        "up": int(sum(1 for r in records
                      for k, ok in zip(r["press"]["press_k"],
                                       r["press"]["press_accepted"])
                      if k > 0 and not ok)),
        "down": int(sum(1 for r in records
                        for k, ok in zip(r["press"]["press_k"],
                                         r["press"]["press_accepted"])
                        if k < 0 and not ok)),
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
    parser.add_argument("--notice-sec", type=float, default=2.0)
    parser.add_argument("--max-presses", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False,
                        frozenset(args.corpora))
    if args.limit:
        items = items[:args.limit]
    seen = sorted({item["corpus"] for item in items})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=repository).stdout.strip()
    clean = not subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               cwd=repository).stdout.strip()

    records: list[dict] = []
    failures: list[dict] = []
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(measure_one, item, args.binary, args.model,
                        args.notice_sec, args.max_presses, seed): item
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
        "commit": commit,
        "clean": clean,
        "corpora": args.corpora,
        "notice_sec": args.notice_sec,
        "max_presses": args.max_presses,
        "requested": len(items),
        "failures": failures,
        "summary": summarise(records) if records else {},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
