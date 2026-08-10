#!/usr/bin/env python3
"""If the front end were perfect, would the product be usable?

`oracle_activation` measured recall under a perfect observation and found the
decoder costs 0.7 to 2.0 points on full-length songs, so the remaining loss is
in front of it. That says where to work. It does **not** say whether working
there is enough, because `usable` is four conditions and recall is one of them:
a recording can have every beat and still fail on precision, on acquisition, or
on the metrical level.

So this scores the *same* oracle arm through the *same* verdict every published
live number came from — `live_corpus_benchmark._score_one` — and reports what
still fails when the observation cannot be blamed. Nothing new is measured; a
run that already exists is put through the scorer it was never put through.

The oracle arm is `oracle_activation.synthesise`'s bump, imported rather than
re-written, so the observation here is byte-identical to the one that produced
the recall table.
"""

from __future__ import annotations

import argparse
import collections
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
from eval.oracle_activation import FPS, synthesise  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402

SAMPLE_HZ = 50.0
ARMS = ("real", "oracle")


def run_dump(binary: pathlib.Path, args: list[str]) -> dict:
    done = subprocess.run([str(binary), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip()[:300] or "dump_analysis failed")
    return json.loads(done.stdout)


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    reference = load_reference_beats(item["annotation"])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 8:
        raise RuntimeError("too few annotated beats")

    payloads = {}
    payloads["real"] = run_dump(binary, [
        str(item["audio"]), "--live", "--live-model", str(model),
        "--live-sample-hz", repr(SAMPLE_HZ)])

    duration = float(payloads["real"].get("duration_sec") or reference[-1] + 1.0)
    activation = synthesise(reference, duration, "bump")
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "oracle.txt"
        path.write_text("\n".join(f"{v:.4f}" for v in activation), encoding="utf-8")
        payloads["oracle"] = run_dump(binary, [
            str(item["audio"]), "--live-activation", str(path),
            "--activation-fps", str(FPS),
            "--live-sample-hz", repr(SAMPLE_HZ)])

    out: dict = {"name": item["name"], "corpus": item["corpus"]}
    for arm in ARMS:
        scored = _score_one(item, "model", binary, model,
                            estimate=Estimate.from_json(payloads[arm]))
        out[arm] = {
            "usable": bool(scored.get("usable", False)),
            "usable_any_octave": bool(scored.get("usable_any_octave", False)),
            "reasons": list(scored.get("reasons", [])),
            "p70": scored.get("p70"),
            "r70": scored.get("r70"),
            "acquired_at": scored.get("acquired_at"),
            "worst_wrong_octave_sec": scored.get("worst_wrong_octave_sec"),
            "f_measure": scored.get("f_measure"),
        }
    return out


def summarise(records: list[dict]) -> dict:
    out: dict = {}
    for corpus in sorted({r["corpus"] for r in records}) + ["all"]:
        rows = [r for r in records if corpus == "all" or r["corpus"] == corpus]
        block: dict = {"n": len(rows)}
        for arm in ARMS:
            reasons: collections.Counter = collections.Counter()
            for r in rows:
                if not r[arm]["usable"]:
                    reasons.update(r[arm]["reasons"])
            block[arm] = {
                "usable_rate": float(np.mean([r[arm]["usable"] for r in rows])),
                "usable_any_octave": float(
                    np.mean([r[arm]["usable_any_octave"] for r in rows])),
                "mean_p70": float(np.mean([r[arm]["p70"] for r in rows
                                           if r[arm]["p70"] is not None])),
                "mean_r70": float(np.mean([r[arm]["r70"] for r in rows
                                           if r[arm]["r70"] is not None])),
                # The point of the whole run: of the recordings that still fail
                # when the observation is perfect, what are they failing on.
                "failures": len([r for r in rows if not r[arm]["usable"]]),
                "reason_share_of_failures": {
                    key: value / max(1, len([r for r in rows
                                             if not r[arm]["usable"]]))
                    for key, value in reasons.most_common()
                },
            }
        out[corpus] = block
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False, frozenset(args.corpora))
    if args.limit:
        items = items[:args.limit]
    seen = sorted({item["corpus"] for item in items})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1

    run_provenance = provenance(
        repository,
        {"manifest": args.manifest, "binary": args.binary,
         "model": args.model},
    )

    records: list[dict] = []
    failures: list[dict] = []
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure_one, item, args.binary, args.model): item
                   for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            done_count += 1
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001
                failures.append({"name": item["name"], "error": str(error)[:300]})
            if done_count % 50 == 0 or done_count == len(items):
                print(f"{done_count}/{len(items)}", file=sys.stderr, flush=True)

    payload = {"provenance": run_provenance, "corpora": args.corpora,
               "requested": len(items), "failures": failures,
               "summary": summarise(records) if records else {},
               "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
