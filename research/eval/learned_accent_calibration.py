"""Calibrating the learned path's accent gate on held-out GTZAN.

Registered in `PREREGISTERED_learned_accent_calibration.md`. One run of
`dump_analysis --learned` per recording, scored with `eval.downbeat.score`, a
fixed split by name hash, the threshold pair chosen on one half by the README's
own procedure and checked once on the other.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.analysis import DEFAULT_BINARY, Estimate  # noqa: E402
from eval.downbeat import choose_margins, evidence_gap, score, sweep  # noqa: E402
from eval.learned_file_path import modal_beats_per_bar  # noqa: E402
from eval.live_corpus_benchmark import (load_corpus,  # noqa: E402
                                        load_reference_beats,
                                        load_reference_downbeats)
from eval.provenance import experiment_provenance, provenance  # noqa: E402

PROVISIONAL = (0.25, 0.40)
MAX_WRONG_RATE = 0.05
MAX_CONDITIONAL_ERROR = 0.10


@dataclasses.dataclass
class Reference:
    beats: np.ndarray
    downbeats: np.ndarray
    beats_per_bar: int
    name: str


def half_of(name: str) -> str:
    return "validation" if hashlib.sha256(name.encode("utf-8")).digest()[0] % 2 == 0 \
        else "held_out"


def clip(item: dict[str, Any], binary: pathlib.Path, model: pathlib.Path):
    done = subprocess.run([str(binary), str(item["audio"]), "--learned", str(model)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{item['name']}: {done.stderr.strip()[:300]}")
    payload = json.loads(done.stdout)
    annotation = pathlib.Path(item["annotation"])
    beats = load_reference_beats(annotation)
    downbeats = load_reference_downbeats(annotation)
    reference = Reference(beats, downbeats, modal_beats_per_bar(beats, downbeats),
                          item["name"])
    return score(reference, Estimate.from_json(payload), item["name"]), \
        payload.get("grid_source", "")


def row_dict(row) -> dict[str, Any]:
    out = dataclasses.asdict(row)
    for key in ("coverage", "wrong_rate", "conditional_error", "wrong_rate_upper",
                "conditional_error_upper"):
        value = getattr(row, key, None)
        if value is not None:
            out[key] = float(value)
    return out


def at(scores, phase: float, meter: float):
    return sweep(scores, [phase], [meter])[0]


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth" / "manifest.csv")
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=pathlib.Path,
                        default=repository / "models" / "small0.onnx")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N recordings only; a smoke test, never a result")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.manifest.parent.parent, False, {"gtzan"})
    if args.limit > 0:
        items = items[:args.limit]
    stamp = provenance if args.limit > 0 else experiment_provenance
    record = stamp(repository, files={"binary": args.binary, "model": args.model},
                   limit=args.limit, smoke=args.limit > 0)

    scores = {"validation": [], "held_out": []}
    failures: list[str] = []
    not_learned = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(clip, item, args.binary, args.model): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                clip_score, source = future.result()
            except Exception as error:  # noqa: BLE001 - reported, not hidden
                failures.append(f"{item['name']}: {error}")
                continue
            not_learned += source != "learned"
            scores[half_of(item["name"])].append(clip_score)

    rows = sweep(scores["validation"])
    chosen = choose_margins(rows, max_wrong_rate=MAX_WRONG_RATE,
                            max_conditional_error=MAX_CONDITIONAL_ERROR)
    result: dict[str, Any] = {
        "provenance": record,
        "failures": failures,
        "not_learned": not_learned,
        "n": {half: len(s) for half, s in scores.items()},
        "budgets": {"max_wrong_rate": MAX_WRONG_RATE,
                    "max_conditional_error": MAX_CONDITIONAL_ERROR,
                    "bound": "upper 95% Wilson"},
        "provisional": {half: row_dict(at(s, *PROVISIONAL)) for half, s in scores.items()},
    }
    if chosen is None:
        result["chosen"] = None
        widest = max(rows, key=lambda r: r.coverage) if rows else None
        result["evidence_gap"] = evidence_gap(widest) if widest else None
    else:
        pair = (chosen.min_phase_margin, chosen.min_meter_margin)
        result["chosen"] = {"min_phase_margin": pair[0], "min_meter_margin": pair[1],
                            "validation": row_dict(chosen),
                            "held_out": row_dict(at(scores["held_out"], *pair))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=1,
                     default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
