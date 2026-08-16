#!/usr/bin/env python3
"""Is the octave residue real, or is it the flat pulse that measured it?

Answers `eval/PREREGISTERED_accented_oracle.md`. Both oracle runs end with
`wrong_octave` as almost the only surviving failure, and both record why that
may prove nothing: the oracle bump is the same height on every beat, so it
removes the amplitude difference that tells a level from its double.

    cd research
    .venv/Scripts/python -m eval.accented_oracle \
        --manifest ../music/rwc2/manifest.csv --music ../music \
        --corpora rwc-pop rwc-genre rwc-jazz rwc-classical rwc-royalty-free \
        --binary <dump_analysis> --model <beatnet.ttw> \
        --output ../research/results/accented_oracle_rwc.json

The shuffled arm is the point. If accents placed on the wrong beats help as much
as accents placed on the right ones, the gain is amplitude variation and not
metre — which is exactly how the downbeat audit's octave arm turned out to be
behind its own control.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]

from eval.analysis import Estimate  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402
from eval.live_corpus_benchmark import (_score_one, load_corpus,  # noqa: E402
                                        load_reference_beats,
                                        load_reference_downbeats)
from eval.oracle_activation import FPS  # noqa: E402
from eval.oracle_usable import SAMPLE_HZ, run_dump  # noqa: E402

# name -> (weak-beat height, shuffle the bar phase)
ARMS: dict[str, tuple[float, bool]] = {
    "flat": (1.0, False),
    "accent_0.5": (0.5, False),
    "accent_0.25": (0.25, False),
    "accent_0.5_shuffled": (0.5, True),
}


def downbeat_mask(beats: np.ndarray, downbeats: np.ndarray,
                  name: str, shuffle: bool) -> np.ndarray:
    """Which beats carry the accent.

    Unshuffled, that is the annotated bar lines. Shuffled, the same *number* of
    accents at the same spacing, rotated by a fixed non-zero offset — so the
    amplitude pattern survives and its alignment to the metre does not.

    The offset comes from the recording's own name by SHA-256 rather than from
    `hash()`, which Python salts per process: an unseeded control moved in the
    third digit between two runs of the same script, and a control that drifts
    when nothing drifted is worse than none.
    """
    mask = np.zeros(len(beats), dtype=bool)
    if len(downbeats) == 0:
        return mask
    for time in downbeats:
        mask[int(np.argmin(np.abs(beats - time)))] = True
    if not shuffle or not mask.any():
        return mask

    indices = np.flatnonzero(mask)
    period = int(np.median(np.diff(indices))) if len(indices) > 1 else 4
    period = max(2, period)
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = 1 + int.from_bytes(digest[:4], "little") % (period - 1)
    rotated = np.zeros(len(beats), dtype=bool)
    rotated[(indices + offset) % len(beats)] = True
    return rotated


def synthesise_accented(beats: np.ndarray, accents: np.ndarray,
                        duration_sec: float, weak: float) -> np.ndarray:
    """`oracle_activation.synthesise`'s bump, with a height per beat.

    The triangle is copied rather than imported because it now needs a
    per-beat scale; the offsets and heights are the same five values, so the
    `flat` arm reproduces the original exactly.
    """
    frames = int(duration_sec * FPS) + 1
    activation = np.zeros(frames, dtype=np.float64)
    indices = np.clip(np.round(beats * FPS).astype(np.int64), 0, frames - 1)
    scale = np.where(accents, 1.0, weak)
    for offset, height in ((-2, 0.33), (-1, 0.67), (0, 1.0), (1, 0.67), (2, 0.33)):
        moved = np.clip(indices + offset, 0, frames - 1)
        np.maximum.at(activation, moved, height * scale)
    return activation


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    beats = load_reference_beats(item["annotation"])
    beats = beats[np.isfinite(beats)]
    if len(beats) < 8:
        raise RuntimeError("too few annotated beats")
    downbeats = load_reference_downbeats(item["annotation"])
    downbeats = downbeats[np.isfinite(downbeats)]
    if len(downbeats) < 2:
        raise RuntimeError("no annotated bar lines")

    real = run_dump(binary, [str(item["audio"]), "--live", "--live-model",
                             str(model), "--live-sample-hz", repr(SAMPLE_HZ)])
    duration = float(real.get("duration_sec") or beats[-1] + 1.0)

    out: dict = {"name": item["name"], "corpus": item["corpus"],
                 "beats": int(len(beats)), "downbeats": int(len(downbeats))}
    with tempfile.TemporaryDirectory() as directory:
        for arm, (weak, shuffle) in ARMS.items():
            accents = downbeat_mask(beats, downbeats, item["name"], shuffle)
            activation = synthesise_accented(beats, accents, duration, weak)
            path = pathlib.Path(directory) / f"{arm}.txt"
            path.write_text("\n".join(f"{v:.4f}" for v in activation),
                            encoding="utf-8")
            payload = run_dump(binary, [
                str(item["audio"]), "--live-activation", str(path),
                "--activation-fps", str(FPS),
                "--live-sample-hz", repr(SAMPLE_HZ)])
            scored = _score_one(item, "model", binary, model,
                                estimate=Estimate.from_json(payload))
            out[arm] = {
                "usable": bool(scored.get("usable", False)),
                "usable_any_octave": bool(scored.get("usable_any_octave", False)),
                "reasons": list(scored.get("reasons", [])),
                "p70": scored.get("p70"), "r70": scored.get("r70"),
                "f_measure": scored.get("f_measure"),
                "worst_wrong_octave_sec": scored.get("worst_wrong_octave_sec"),
            }
    return out


def summarise(records: list[dict]) -> dict:
    out: dict = {}
    for corpus in sorted({r["corpus"] for r in records}) + ["all"]:
        rows = [r for r in records if corpus == "all" or r["corpus"] == corpus]
        block: dict = {"n": len(rows)}
        flat = float(np.mean([r["flat"]["usable"] for r in rows]))
        for arm in ARMS:
            failing = [r for r in rows if not r[arm]["usable"]]
            reasons: collections.Counter = collections.Counter()
            for r in failing:
                reasons.update(r[arm]["reasons"])
            usable = float(np.mean([r[arm]["usable"] for r in rows]))
            block[arm] = {
                "usable_rate": usable,
                "usable_delta_vs_flat": usable - flat,
                "usable_any_octave": float(np.mean(
                    [r[arm]["usable_any_octave"] for r in rows])),
                "wrong_octave_share_of_failures": (
                    reasons["wrong_octave"] / len(failing) if failing else 0.0),
                "failures": len(failing),
                "mean_f": float(np.mean([r[arm]["f_measure"] for r in rows
                                         if r[arm]["f_measure"] is not None])),
            }
        out[corpus] = block
    return out


def verdict(summary: dict) -> dict:
    """The registered prediction, applied as written."""
    overall = summary["all"]
    true_arms = ["accent_0.5", "accent_0.25"]
    best = max(true_arms, key=lambda a: overall[a]["usable_delta_vs_flat"])
    gain = overall[best]["usable_delta_vs_flat"]
    control = overall["accent_0.5_shuffled"]["usable_delta_vs_flat"]
    rises = gain >= 0.05
    controlled = control < gain / 2.0
    return {
        "best_true_accent": best,
        "gain_over_flat": gain,
        "shuffled_gain_over_flat": control,
        "rises_by_at_least_0.05": bool(rises),
        "control_rises_by_less_than_half": bool(controlled),
        "outcome": ("accent_carries_metre" if rises and controlled
                    else "amplitude_not_metre" if rises
                    else "residue_is_not_an_artefact"),
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

    # The shared wrapper rather than an inline `git status`: it raises on a
    # dirty or unreadable tree instead of recording a flag beside numbers that
    # are already written. An artifact that says `clean: false` is one a reader
    # has to notice; one that was never produced cannot be misread.
    sources = {"binary": args.binary, "model": args.model,
               "manifest": args.manifest}
    run_provenance = provenance(REPOSITORY, sources)
    commit = run_provenance["commit"]

    # A checkpoint, because the first attempt at this run was interrupted at
    # forty-five minutes and left nothing: the artifact is written once, at the
    # end. Each finished recording is appended here instead, and a rerun skips
    # what is already done.
    #
    # The commit is written into the sidecar and a mismatch refuses to resume
    # rather than resuming quietly. Half a result measured at one commit and
    # half at another is not a result, and it would carry a single `commit`
    # field in the artifact that named only one of them.
    partial_path = args.output.with_suffix(".partial.jsonl")
    records, failures, resumed = [], [], 0
    if partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("commit") != commit:
                print(f"refusing to resume: {partial_path.name} was written at "
                      f"{row.get('commit', '?')[:8]}, this is {commit[:8]}",
                      file=sys.stderr)
                return 1
            records.append(row["record"])
        resumed = len(records)
        if resumed:
            print(f"resuming: {resumed} recordings already done",
                  file=sys.stderr)
    done_names = {r["name"] for r in records}
    todo = [item for item in items if item["name"] not in done_names]

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as sidecar:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers) as pool:
            futures = {pool.submit(measure_one, item, args.binary,
                                   args.model): item for item in todo}
            for done in concurrent.futures.as_completed(futures):
                item = futures[done]
                try:
                    record = done.result()
                    records.append(record)
                    sidecar.write(json.dumps({"commit": commit,
                                              "record": record}) + "\n")
                    sidecar.flush()
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    failures.append({"name": item["name"],
                                     "error": str(error)[:300]})
                print(f"{len(records) + len(failures)}/{len(items)}", end="\r",
                      file=sys.stderr)
    print(file=sys.stderr)
    summary = summarise(records) if records else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "provenance": run_provenance,
        "registered_in": "research/eval/PREREGISTERED_accented_oracle.md",
        "corpora": args.corpora, "requested": len(items),
        "resumed_from_checkpoint": resumed,
        "sample_hz": SAMPLE_HZ,
        "arms": {k: {"weak_beat_height": v[0], "shuffled": v[1]}
                 for k, v in ARMS.items()},
        "failures": failures,
        "verdict": verdict(summary) if summary else {},
        "summary": summary,
        "records": sorted(records, key=lambda r: (r["corpus"], r["name"])),
    }, indent=2), encoding="utf-8")

    # Only once the artifact is safely on disk. A checkpoint removed any earlier
    # would reopen the window it exists to close.
    if summary and not failures:
        partial_path.unlink(missing_ok=True)

    if summary:
        for arm in ARMS:
            block = summary["all"][arm]
            print(f"{arm:20s} usable {block['usable_rate']:.3f} "
                  f"({block['usable_delta_vs_flat']:+.3f})  "
                  f"wrong_octave {block['wrong_octave_share_of_failures']:.3f}")
        print("outcome:", verdict(summary)["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
