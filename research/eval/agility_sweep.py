#!/usr/bin/env python3
"""Does filter agility help or hurt, and does the observation decide?

Answers `eval/PREREGISTERED_agility.md`. Two existing measurements disagree
about the same knob — raising it helps under a perfect observation on RWC, and
hurts monotonically under the real one on ballroom, gtzan and smc — and they
were taken on different corpora, so neither corrects the other. RWC-Classical,
where the help was largest, is absent from the real sweep entirely.

So both arms are run over one set of corpora, one grid, one scorer.

    cd research
    .venv/Scripts/python -m eval.agility_sweep \
        --manifest <ground-truth>/manifest.csv --music <music> \
        --corpora rwc-pop rwc-genre rwc-jazz rwc-classical rwc-royalty-free \
        --binary <dump_analysis> --model <beatnet.ttw> \
        --output ../research/results/agility_sweep_rwc.json

The oracle activation is written once per recording and reused across every
setting, so the observation is byte-identical and the only thing differing
between settings is the filter. That is the whole point of the comparison and
re-synthesising per setting would quietly weaken it.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
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
from eval.oracle_usable import SAMPLE_HZ, run_dump  # noqa: E402

ARMS = ("real", "oracle")

# The grid, fixed in the registration. `shipped` is the core's own defaults --
# roughening 0.01 with the anchor on -- and is named rather than spelled so a
# change to those defaults cannot silently turn the baseline into an arm.
SETTINGS: dict[str, list[str]] = {
    "shipped": [],
    "r0.02": ["--live-roughening", "0.02"],
    "r0.04": ["--live-roughening", "0.04"],
    "r0.08": ["--live-roughening", "0.08"],
    "no_anchor": ["--live-no-anchor"],
}

FIELDS = ("usable", "usable_any_octave", "reasons", "p70", "r70", "f_measure",
          "coverage", "acquired_at", "worst_wrong_octave_sec")


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    reference = load_reference_beats(item["annotation"])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 8:
        raise RuntimeError("too few annotated beats")

    out: dict = {"name": item["name"], "corpus": item["corpus"]}

    # One real run first, only to learn the duration the oracle must cover.
    # Its settings are the shipped ones, so it doubles as that cell.
    base_real = [str(item["audio"]), "--live", "--live-model", str(model),
                 "--live-sample-hz", repr(SAMPLE_HZ)]
    first = run_dump(binary, base_real)
    duration = float(first.get("duration_sec") or reference[-1] + 1.0)

    activation = synthesise(reference, duration, "bump")
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = pathlib.Path(directory) / "oracle.txt"
        oracle_path.write_text("\n".join(f"{v:.4f}" for v in activation),
                               encoding="utf-8")
        base_oracle = [str(item["audio"]), "--live-activation", str(oracle_path),
                       "--activation-fps", str(FPS),
                       "--live-sample-hz", repr(SAMPLE_HZ)]

        for setting, flags in SETTINGS.items():
            payloads = {
                "real": first if setting == "shipped"
                else run_dump(binary, base_real + flags),
                "oracle": run_dump(binary, base_oracle + flags),
            }
            for arm in ARMS:
                scored = _score_one(item, "model", binary, model,
                                    estimate=Estimate.from_json(payloads[arm]))
                cell = {field: scored.get(field) for field in FIELDS}
                cell["usable"] = bool(cell["usable"])
                cell["usable_any_octave"] = bool(cell["usable_any_octave"])
                cell["reasons"] = list(cell["reasons"] or [])
                out[f"{arm}/{setting}"] = cell
    return out


def summarise(records: list[dict]) -> dict:
    out: dict = {}
    for corpus in sorted({r["corpus"] for r in records}) + ["all"]:
        rows = [r for r in records if corpus == "all" or r["corpus"] == corpus]
        block: dict = {"n": len(rows)}
        for arm in ARMS:
            shipped = float(np.mean([r[f"{arm}/shipped"]["usable"]
                                     for r in rows]))
            per_setting: dict = {}
            for setting in SETTINGS:
                key = f"{arm}/{setting}"
                reasons: collections.Counter = collections.Counter()
                for r in rows:
                    if not r[key]["usable"]:
                        reasons.update(r[key]["reasons"])
                usable = float(np.mean([r[key]["usable"] for r in rows]))

                def mean(field: str) -> float | None:
                    values = [r[key][field] for r in rows
                              if r[key][field] is not None
                              and np.isfinite(r[key][field])]
                    return float(np.mean(values)) if values else None

                per_setting[setting] = {
                    "usable_rate": usable,
                    # The primary readout, against the shipped setting of the
                    # same arm. Never against the other arm: the two are on
                    # different observations and a difference across them is a
                    # different question from the one registered.
                    "usable_delta_vs_shipped": usable - shipped,
                    "usable_any_octave": float(np.mean(
                        [r[key]["usable_any_octave"] for r in rows])),
                    "mean_p70": mean("p70"),
                    "mean_r70": mean("r70"),
                    "mean_f": mean("f_measure"),
                    "mean_coverage": mean("coverage"),
                    "mean_worst_wrong_octave_sec": mean("worst_wrong_octave_sec"),
                    "failures": len([r for r in rows if not r[key]["usable"]]),
                    "reason_share_of_failures": {
                        reason: count / max(1, len([r for r in rows
                                                    if not r[key]["usable"]]))
                        for reason, count in reasons.most_common()},
                }
            block[arm] = per_setting
        out[corpus] = block
    return out


def verdict(summary: dict) -> dict:
    """The registered prediction, applied as written.

    Raised settings are everything but `shipped`. The prediction is that the
    real arm never rises above its own shipped baseline and the oracle arm rises
    at least once; the third registered outcome -- a flip that reverses on some
    sub-corpus -- is reported by listing where each arm rose.
    """
    overall = summary["all"]
    raised = [s for s in SETTINGS if s != "shipped"]

    def rose(arm: str, block: dict) -> list[str]:
        return [s for s in raised
                if block[arm][s]["usable_delta_vs_shipped"] > 0.0]

    real_rose = rose("real", overall)
    oracle_rose = rose("oracle", overall)
    per_corpus = {
        corpus: {"real_rose_on": rose("real", block),
                 "oracle_rose_on": rose("oracle", block)}
        for corpus, block in summary.items() if corpus != "all"}
    return {
        "real_never_rises": not real_rose,
        "oracle_rises_at_least_once": bool(oracle_rose),
        "sign_flip_survives": (not real_rose) and bool(oracle_rose),
        "real_rose_on": real_rose,
        "oracle_rose_on": oracle_rose,
        "per_corpus": per_corpus,
    }


def main() -> int:
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
    args = parser.parse_args()

    items = load_corpus(args.manifest, args.music, False,
                        frozenset(args.corpora))
    if args.limit:
        items = items[:args.limit]
    seen = sorted({item["corpus"] for item in items})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1

    records, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure_one, item, args.binary, args.model): item
                   for item in items}
        for done in concurrent.futures.as_completed(futures):
            item = futures[done]
            try:
                records.append(done.result())
            except Exception as error:  # noqa: BLE001 - recorded, not raised
                failures.append({"name": item["name"], "error": str(error)[:300]})
            print(f"{len(records) + len(failures)}/{len(items)}", end="\r",
                  file=sys.stderr)
    print(file=sys.stderr)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    clean = not subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
    summary = summarise(records) if records else {}
    payload = {
        "commit": commit, "clean": clean,
        "registered_in": "research/eval/PREREGISTERED_agility.md",
        "corpora": args.corpora, "requested": len(items),
        "sample_hz": SAMPLE_HZ, "settings": SETTINGS,
        "failures": failures,
        "verdict": verdict(summary) if summary else {},
        "summary": summary,
        "records": sorted(records, key=lambda r: (r["corpus"], r["name"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if summary:
        for arm in ARMS:
            deltas = " ".join(
                f"{s}{summary['all'][arm][s]['usable_delta_vs_shipped']:+.3f}"
                for s in SETTINGS if s != "shipped")
            print(f"{arm:7s} shipped "
                  f"{summary['all'][arm]['shipped']['usable_rate']:.3f}  {deltas}")
        print("sign flip survives:", payload["verdict"]["sign_flip_survives"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
