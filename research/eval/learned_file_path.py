"""The learned file path, measured where it ships: through tt_offline's analyser.

Registered in `PREREGISTERED_learned_file_path.md`. Three runs of one
dump_analysis binary per recording:

* ``onsets``  — the shipped onset path, no flags;
* ``seam``    — ``--beat-this``: the model's beats and its own downbeat picks
  swapped in after the onset analysis, which is what measured +0.102 beat F;
* ``learned`` — ``--learned``: the model inside OfflineAnalyzer, bar lines
  from resolveMeter on the model's downbeat probability. The product path.

The beats of ``seam`` and ``learned`` must be identical — same features, same
chunking, same picker — and the run checks that before anything else is read.
What is new is the bar decision, scored against human annotation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.analysis import DEFAULT_BINARY  # noqa: E402
from eval.harness import evaluate, evaluate_downbeats  # noqa: E402
from eval.live_corpus_benchmark import (load_corpus,  # noqa: E402
                                        load_reference_beats,
                                        load_reference_downbeats)
from eval.provenance import experiment_provenance, provenance  # noqa: E402

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260930
ARMS = ("onsets", "seam", "learned")


def run(binary: pathlib.Path, audio: pathlib.Path, flags: list[str]) -> dict:
    done = subprocess.run([str(binary), str(audio), *flags], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{audio.name} {flags}: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout)


def rendered_downbeats(beats: np.ndarray, downbeats: np.ndarray,
                       beats_per_bar: int) -> np.ndarray:
    """What a player that knows one bar offset plays.

    The phase that holds the most of `downbeats`, ties to the first bar line's,
    extended over the whole grid — core `playbackDownbeatOffset`, transcribed so
    the scorer can say what is heard rather than what was listed.
    """
    if beats_per_bar <= 0 or len(downbeats) == 0 or len(beats) == 0:
        return np.zeros(0)
    index = np.searchsorted(beats, downbeats - 1e-9)
    inside = index < len(beats)
    index, bars = index[inside], downbeats[inside]
    index = index[np.abs(beats[index] - bars) <= 1e-6]   # bar lines are beats
    if len(index) == 0:
        return np.zeros(0)
    phases = index % beats_per_bar
    votes = np.bincount(phases, minlength=beats_per_bar)
    best = int(phases[0])
    for phase in range(beats_per_bar):
        if votes[phase] > votes[best]:
            best = phase
    return beats[best::beats_per_bar]


def modal_beats_per_bar(beats: np.ndarray, downbeats: np.ndarray) -> int:
    """The annotated metre: the commonest count of beats between bar lines."""
    if len(downbeats) < 2 or len(beats) < 2:
        return 0
    index = np.searchsorted(beats, downbeats - 0.035)
    index = index[index < len(beats)]
    lengths = np.diff(index)
    lengths = lengths[lengths > 0]
    return int(np.bincount(lengths).argmax()) if len(lengths) else 0


class Reference:
    def __init__(self, path: pathlib.Path) -> None:
        self.beats = load_reference_beats(path)
        self.downbeats = load_reference_downbeats(path)
        self.beats_per_bar = modal_beats_per_bar(self.beats, self.downbeats)


def score_item(item: dict[str, Any], binary: pathlib.Path,
               model: pathlib.Path) -> dict[str, Any]:
    reference = Reference(pathlib.Path(item["annotation"]))
    flags = {"onsets": [], "seam": ["--beat-this", str(model)],
             "learned": ["--learned", str(model)]}
    record: dict[str, Any] = {"name": item["name"],
                              "reference_beats_per_bar": int(reference.beats_per_bar),
                              "has_downbeats": bool(len(reference.downbeats) > 0)}
    beats_of: dict[str, list[float]] = {}
    for arm in ARMS:
        payload = run(binary, pathlib.Path(item["audio"]), flags[arm])
        beats = np.asarray(payload.get("beats", []), dtype=np.float64)
        downbeats = np.asarray(payload.get("downbeats", []), dtype=np.float64)
        per_bar = int(payload.get("beats_per_bar", 0))
        beats_of[arm] = payload.get("beats", [])
        scored = evaluate(reference.beats, beats)
        entry = {
            "f_measure": scored["f_measure"],
            "cmlt": scored["cmlt"],
            "amlt": scored["amlt"],
            "beats_per_bar": per_bar,
            "bpm": float(payload.get("bpm", 0.0)),
            "confidence": float(payload.get("confidence", 0.0)),
            "downbeat_confident": bool(payload.get("downbeat_confident", False)),
            "grid_source": payload.get("grid_source", ""),
            "beats_source": payload.get("beats_source", ""),
        }
        if record["has_downbeats"]:
            entry["downbeat_f"] = evaluate_downbeats(
                reference.downbeats, downbeats)["downbeat_f_measure"]
            entry["rendered_downbeat_f"] = evaluate_downbeats(
                reference.downbeats,
                rendered_downbeats(beats, downbeats, per_bar))["downbeat_f_measure"]
        record[arm] = entry
    record["seam_learned_beats_identical"] = beats_of["seam"] == beats_of["learned"]
    return record


def paired(records: list[dict], arm: str, against: str, key: str) -> dict:
    pairs = [(r[arm][key], r[against][key]) for r in records
             if key in r[arm] and key in r[against]
             and not math.isnan(r[arm][key]) and not math.isnan(r[against][key])]
    if not pairs:
        return {"n": 0}
    a, b = np.asarray(pairs).T
    delta = a - b
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(delta, size=(BOOTSTRAP_RESAMPLES, len(delta))).mean(axis=1)
    return {"n": len(delta), "arm": float(a.mean()), "against": float(b.mean()),
            "delta": float(delta.mean()),
            "interval": [float(np.percentile(means, 2.5)),
                         float(np.percentile(means, 97.5))],
            "better": int((delta > 0).sum()), "worse": int((delta < 0).sum())}


def summarise(records: list[dict]) -> dict:
    out: dict[str, Any] = {
        "n": len(records),
        "seam_learned_beats_identical": int(sum(r["seam_learned_beats_identical"]
                                                for r in records)),
        "learned_answered": int(sum(r["learned"]["grid_source"] == "learned"
                                    for r in records)),
    }
    for arm in ARMS:
        values = {k: [r[arm][k] for r in records
                      if k in r[arm] and not math.isnan(r[arm][k])]
                  for k in ("f_measure", "cmlt", "amlt", "downbeat_f",
                            "rendered_downbeat_f")}
        out[arm] = {k: (float(np.mean(v)) if v else None) for k, v in values.items()}
    out["replication_learned_vs_onsets"] = {
        k: paired(records, "learned", "onsets", k) for k in ("f_measure", "cmlt", "amlt")}
    out["bar_decision_learned_vs_seam"] = paired(records, "learned", "seam", "downbeat_f")
    out["rendered_learned_vs_listed"] = paired(
        [dict(r, rendered={"x": r["learned"].get("rendered_downbeat_f", math.nan)},
              listed={"x": r["learned"].get("downbeat_f", math.nan)}) for r in records],
        "rendered", "listed", "x")
    meters = [r for r in records if r["reference_beats_per_bar"] > 0]
    out["metre"] = {
        "n": len(meters),
        "always_4": float(np.mean([r["reference_beats_per_bar"] == 4 for r in meters])),
        "onsets": float(np.mean([r["onsets"]["beats_per_bar"] == r["reference_beats_per_bar"]
                                 for r in meters])),
        "learned": float(np.mean([r["learned"]["beats_per_bar"] == r["reference_beats_per_bar"]
                                  for r in meters])),
    }
    gated = [r for r in records if r["has_downbeats"]
             and not math.isnan(r["learned"].get("rendered_downbeat_f", math.nan))]
    accented = [r for r in gated if r["learned"]["downbeat_confident"]]
    wrong = [r for r in accented if r["learned"]["rendered_downbeat_f"] < 0.5]
    out["accent_gate_learned"] = {
        "n": len(gated), "coverage": len(accented) / len(gated) if gated else None,
        "conditional_error": len(wrong) / len(accented) if accented else None,
        "definition": "wrong = rendered downbeat F below 0.5 among recordings the "
                      "provisional gate would accent"}
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth" / "manifest.csv")
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=pathlib.Path,
                        default=repository / "models" / "small0.onnx")
    parser.add_argument("--corpora", nargs="+", default=["gtzan"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N recordings only; for a smoke test, never a result")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--records", type=pathlib.Path)
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.manifest.parent.parent, False,
                        set(args.corpora))
    if args.limit > 0:
        items = items[:args.limit]
    # A limited run is a smoke test of the harness, and says so: it records
    # provenance without demanding a clean tree, and is marked so that it can
    # never be read as the registered result.
    stamp = provenance if args.limit > 0 else experiment_provenance
    record = stamp(repository, files={"binary": args.binary, "model": args.model},
                   corpora=list(args.corpora), limit=args.limit,
                   smoke=args.limit > 0)

    records: list[dict] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_item, item, args.binary, args.model): item
                   for item in items}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001 - reported, not hidden
                failures.append(f"{futures[future]['name']}: {error}")
            if done % 50 == 0:
                print(json.dumps({"event": "progress", "done": done,
                                  "total": len(items)}), flush=True)

    records.sort(key=lambda r: r["name"])
    result = {"provenance": record, "failures": failures, "summary": summarise(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.records is not None:
        args.records.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
