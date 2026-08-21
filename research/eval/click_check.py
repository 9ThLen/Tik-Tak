#!/usr/bin/env python3
"""The click micro-check: does an audible click cost us the alignment?

`plan.md` permits this before the full P1-B0 pilot, and
`PREREGISTERED_P1B0.md` orders it first, because it is the one gate that can
reject the protocol for an hour of work rather than for a whole session. The
click-bleed condition has never been tested anywhere in this repository: every
published number comes from a harness that plays no click, so every one of them
is an upper bound for a shell with audible output.

**The primary observable is alignment, not F.** The registered consequence table
turns on whether the slate can still be found and the two slates still agree on
drift; how much F the click costs is the secondary "order of the effect" the
plan asks for, and it decides nothing here.

**Where the click comes from is a required field, not an assumption.** The plan
says the click is "physically reproduced by a loudspeaker", and in the same
breath that software mixing reproduces "neither room feedback nor AEC nor the
self-confirming loop". Those cannot both describe one setup: AEC and the
self-confirming loop only arise when the click comes from the *recording
device's own* speaker, while a click from the external music speaker reproduces
the room path alone. They are two different experiments and this harness refuses
to average them, so every capture must declare `click_source`.

The levels manifest must be written **before** the session and must say which
levels are product-plausible. A run that decides afterwards which levels counted
cannot return the third outcome, which is the only one that stops anything.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
import soundfile

from eval.live_corpus_benchmark import _score_one, load_corpus
from eval.provenance import experiment_provenance as provenance
from eval.room_session3 import align_one
from eval.slate import RATE

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "tiktak.click_check/v1"
CLICK_SOURCES = ("device", "speaker", "none")


def verdict(by_level: dict, plausible: set[str], order: list[str]) -> dict:
    """The registered three-way consequence, applied to alignment only.

    `plan.md` and PREREGISTERED_P1B0 fix these three outcomes before any capture
    exists. Nothing here inspects F: a protocol that cannot be aligned is
    unusable whatever it scores, and one that aligns is usable even if the click
    is expensive, because the cost is what the pilot exists to measure.
    """
    recovered = {level: rows["aligned"] == rows["captures"]
                 for level, rows in by_level.items()}
    if all(recovered.values()):
        return {"outcome": "proceed_with_click_bleed",
                "highest_recovered_level": order[-1] if order else None,
                "why": "every level tested recovered every capture"}

    # The highest level, in the declared order, at and below which everything
    # recovered. Declared order rather than sorted labels: "low/mid/high" does
    # not sort, and inferring an order from measured dBFS would let a session
    # reorder its own gate after the fact.
    ceiling = None
    for level in order:
        if recovered.get(level):
            ceiling = level
        else:
            break

    failing_plausible = [level for level in order
                         if level in plausible and not recovered.get(level, False)]
    if ceiling is None or (failing_plausible and ceiling is None):
        return {"outcome": "do_not_proceed_as_designed",
                "highest_recovered_level": None,
                "why": "no level recovered, including product-plausible ones"}
    if failing_plausible and order.index(failing_plausible[0]) <= order.index(ceiling):
        return {"outcome": "do_not_proceed_as_designed",
                "highest_recovered_level": ceiling,
                "why": f"product-plausible level {failing_plausible[0]} did not recover"}
    if failing_plausible:
        return {"outcome": "do_not_proceed_as_designed",
                "highest_recovered_level": ceiling,
                "why": ("product-plausible levels "
                        f"{', '.join(failing_plausible)} did not recover")}
    return {"outcome": "proceed_with_level_constraint",
            "highest_recovered_level": ceiling,
            "why": f"recovery stops above {ceiling}, which becomes a protocol constraint"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takes", type=pathlib.Path, required=True,
                        help="slate-wrapped takes and their layout json")
    parser.add_argument("--captures", type=pathlib.Path, required=True)
    parser.add_argument("--levels", type=pathlib.Path, required=True,
                        help="written before the session; see the docstring")
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--aligned", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--skip-scoring", action="store_true",
                        help="alignment only; the registered outcome needs no F")
    args = parser.parse_args(argv)

    declared = json.loads(args.levels.read_text(encoding="utf-8"))
    order = list(declared["order"])
    plausible = {level for level, spec in declared["levels"].items()
                 if spec["product_plausible"]}
    unknown = [level for level in declared["levels"] if level not in order]
    if unknown:
        raise ValueError(f"levels not placed in the declared order: {unknown}")
    sources = {level: spec["click_source"]
               for level, spec in declared["levels"].items()}
    bad = {level: value for level, value in sources.items()
           if value not in CLICK_SOURCES}
    if bad:
        raise ValueError(f"click_source must be one of {CLICK_SOURCES}: {bad}")
    distinct = {value for level, value in sources.items() if value != "none"}
    if len(distinct) > 1:
        raise ValueError(
            "one run may not mix click sources: a click from the device tests "
            f"AEC and the self-confirming loop, one from the speaker does not — {sources}")

    captures = declared["captures"]
    run_provenance = provenance(
        REPOSITORY,
        {"binary": args.binary, "model": args.model, "manifest": args.manifest,
         "levels": args.levels},
        check="click_micro_check", alignment="slate", rate=RATE,
        click_source=sorted(distinct) or ["none"])

    items = {item["name"]: item for item in load_corpus(
        args.manifest, args.music, False, frozenset({"harmonix"}))}
    args.aligned.mkdir(parents=True, exist_ok=True)

    records = []
    by_level = collections.defaultdict(
        lambda: {"captures": 0, "aligned": 0, "f_room": []})
    for entry in captures:
        track, level = entry["track"], entry["level"]
        if level not in declared["levels"]:
            raise ValueError(f"capture declares an undeclared level: {level}")
        layout = json.loads(
            (args.takes / f"{track}.layout.json").read_text(encoding="utf-8"))
        path = args.captures / entry["file"]
        row = {"track": track, "level": level, "file": entry["file"],
               "click_source": sources[level]}
        by_level[level]["captures"] += 1
        try:
            mono, found = align_one(path, layout)
        except RuntimeError as error:
            # A capture that will not align is the finding, not an error: the
            # whole check exists to see whether this happens.
            row.update({"aligned": False, "reason": str(error)})
            records.append(row)
            continue

        by_level[level]["aligned"] += 1
        row.update({"aligned": True,
                    "music_offset_sec": found["music_offset_sec"],
                    "drift_ppm": found.get("drift_ppm"),
                    "margin_db": found.get("margin_db")})

        if not args.skip_scoring:
            start = int(round(found["music_offset_sec"] * RATE))
            stop = start + int(round(layout["music_seconds"] * RATE))
            if stop > len(mono):
                row["reason"] = "capture ends before the music does"
                records.append(row)
                continue
            aligned_path = args.aligned / f"{track}__{level}.wav"
            soundfile.write(str(aligned_path), mono[start:stop], RATE)
            item = dict(items[track], audio=aligned_path)
            scored = _score_one(item, "model", args.binary, args.model)
            row["f_room"] = scored.get("f_measure")
            row["usable"] = bool(scored.get("usable", False))
            if row["f_room"] is not None:
                by_level[level]["f_room"].append(float(row["f_room"]))
        records.append(row)

    levels = {}
    for level in order:
        rows = by_level.get(level)
        if rows is None:
            continue
        scores = rows["f_room"]
        levels[level] = {
            "captures": rows["captures"], "aligned": rows["aligned"],
            "click_source": sources[level],
            "product_plausible": level in plausible,
            "mean_f_room": float(np.mean(scores)) if scores else None,
        }

    result = {
        "schema": SCHEMA, "research_only": True,
        "provenance": run_provenance,
        "declared_levels": declared,
        "levels": levels,
        "verdict": verdict(by_level, plausible, order),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["verdict"]))
    for level, row in levels.items():
        mean = row["mean_f_room"]
        print(f"  {level:>10}  aligned {row['aligned']}/{row['captures']}"
              f"  mean room F {'n/a' if mean is None else f'{mean:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
