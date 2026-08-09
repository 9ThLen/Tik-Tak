#!/usr/bin/env python3
"""Hand the causal decoder a perfect observation and see what it still loses.

This is the other half of ``activation_recall``. That one asks how much of the
beat is legible in what BeatNet produces; this one removes the front end
entirely and feeds the particle filter an activation that is right by
construction — a pulse at every annotated beat and nothing anywhere else, at the
same 50 fps and through the same ``LiveTracker.observe`` seam the swapped-front-
end experiments use.

There is no ambiguity about what the result means, which is the point:

* recall near 1.0 says the decoder is not what loses beats, and every remaining
  point of the live path's recall is in front of it;
* recall well below 1.0 says the decoder drops beats it was *told* about, and no
  improvement to the activation — BeatNet+, PLP's front end, Beat This! — can
  recover them, because they were already there.

Two shapes of pulse, because a single 20 ms frame is not what the network
produces and a decoder tuned to a smeared observation could fail on a spike for
reasons that have nothing to do with tracking. If both shapes agree, the answer
does not depend on the choice:

``impulse``   one frame at 1.0.
``bump``      a five-frame triangle, roughly 100 ms wide, which is about the
              width of a confident BeatNet peak.

The comparison run is the same tracker on the real activation, so the two
columns differ in the observation and in nothing else.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
sys.path.insert(0, str(RESEARCH))

from eval.analysis import DEFAULT_BINARY  # noqa: E402
from eval.live_corpus_benchmark import load_corpus  # noqa: E402
from eval.live_corpus_benchmark import load_reference_beats  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402
from eval.statistics import spearman  # noqa: E402

FPS = 50.0
WINDOW_SEC = 0.070
WARMUP_SEC = 5.0
MODEL = REPOSITORY / "models" / "beatnet_model_1.ttw"

# Overridden from the command line so a worktree can point at the main tree's
# weights and binary. Module-level because `one` runs in a thread pool and takes
# only the item; set once in main() before any work is submitted.
_binary = DEFAULT_BINARY
_model = MODEL

# The filter's tempo agility: how far every resampled particle's period is
# allowed to wander, in octaves. The core ships 0.01. Swept here because the
# oracle result made agility the question — if a knob we already have recovers
# the loss, no new decoder is needed, and if it does not, the limit is
# structural and that is worth knowing before porting one.
ROUGHENING_SWEEP = (0.02, 0.04, 0.08)


def sample(part: list, limit: int) -> list:
    """A subset that spans the corpus instead of stopping partway through it.

    The obvious spelling, ``part[::len(part) // limit][:limit]``, does not. With
    999 GTZAN recordings and a limit of 150 the stride is 6, which yields 167
    items, and truncating to 150 stops at ``reggae.00094`` — the whole rock
    genre absent from every number the run produces, on a corpus whose files are
    ordered by genre. Rounding the stride up instead makes the stride itself do
    the limiting, so the sample always reaches the end of the list.
    """
    if not limit or len(part) <= limit:
        return part
    return part[:: -(-len(part) // limit)]


def synthesise(beats: np.ndarray, duration_sec: float, shape: str) -> np.ndarray:
    frames = int(duration_sec * FPS) + 1
    activation = np.zeros(frames, dtype=np.float64)
    indices = np.clip(np.round(beats * FPS).astype(np.int64), 0, frames - 1)
    if shape == "impulse":
        activation[indices] = 1.0
        return activation
    # Triangle, peak 1.0, two frames either side.
    for offset, height in ((-2, 0.33), (-1, 0.67), (0, 1.0), (1, 0.67), (2, 0.33)):
        moved = np.clip(indices + offset, 0, frames - 1)
        np.maximum.at(activation, moved, height)
    return activation


def matched(reference: np.ndarray, found: np.ndarray) -> int:
    if len(reference) == 0 or len(found) == 0:
        return 0
    used = np.zeros(len(found), dtype=bool)
    hits = 0
    for beat in reference:
        near = np.flatnonzero((np.abs(found - beat) <= WINDOW_SEC) & ~used)
        if len(near):
            used[near[np.argmin(np.abs(found[near] - beat))]] = True
            hits += 1
    return hits


def run(arguments: list[str]) -> dict | None:
    done = subprocess.run([str(_binary), *arguments], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        return None


def one(item: dict) -> dict | None:
    reference = load_reference_beats(item["annotation"])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 8:
        return None

    real = run([str(item["audio"]), "--live", "--live-model", str(_model)])
    if real is None:
        return None
    duration = float(real.get("duration_sec") or reference[-1] + 1.0)

    scored = reference[reference >= WARMUP_SEC]
    if len(scored) < 8:
        return None

    # How much the annotated tempo moves across the recording, as the spread of
    # the beat intervals. Carried per recording because the oracle result on SMC
    # is far worse than on GTZAN, and "the decoder cannot follow a tempo that
    # changes" is a different diagnosis from "the decoder is weak", with a
    # different fix. Only a correlation, but a cheap one, and the two corpora
    # differ on exactly this axis.
    intervals = np.diff(reference)
    intervals = intervals[(intervals > 0.1) & (intervals < 3.0)]
    row = {"corpus": item["corpus"], "name": item["name"],
           "beats": int(len(scored)),
           "tempo_spread": (float(np.std(intervals) / np.mean(intervals))
                            if len(intervals) > 4 else float("nan"))}

    found = np.asarray(real.get("beats", []), dtype=np.float64)
    row["real"] = matched(scored, found[found >= WARMUP_SEC]) / len(scored)

    # The bump is written once and reused across the settings sweep, so every
    # arm is scored on a byte-identical observation and the only thing that
    # differs is the filter.
    for shape in ("impulse", "bump"):
        activation = synthesise(reference, duration, shape)
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        try:
            handle.write("\n".join(f"{value:.4f}" for value in activation))
            handle.close()
            base = [str(item["audio"]), "--live-activation", handle.name,
                    "--activation-fps", str(FPS)]
            got = run(base)
            if got is None:
                return None
            found = np.asarray(got.get("beats", []), dtype=np.float64)
            row[shape] = matched(scored, found[found >= WARMUP_SEC]) / len(scored)

            # The arm that separates the filter from the estimator in front of
            # it. `anchor_tempo` ships true, and `LiveTracker::submit` applies it
            # on every frame, so "the filter, told the beats" was never the
            # filter alone: it was the six-second autocorrelation and the filter
            # together, and either could be what fails.
            if shape == "bump":
                got = run(base + ["--live-no-anchor"])
                if got is None:
                    return None
                found = np.asarray(got.get("beats", []), dtype=np.float64)
                row["bump_no_anchor"] = (
                    matched(scored, found[found >= WARMUP_SEC]) / len(scored)
                )
                for roughening in ROUGHENING_SWEEP:
                    got = run(base + ["--live-roughening", repr(roughening)])
                    if got is None:
                        return None
                    found = np.asarray(got.get("beats", []), dtype=np.float64)
                    row[f"bump_r{roughening:g}"] = (
                        matched(scored, found[found >= WARMUP_SEC]) / len(scored)
                    )
        finally:
            pathlib.Path(handle.name).unlink(missing_ok=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="*", default=["gtzan", "smc"])
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=pathlib.Path)
    # RWC 2.0 arrived with its own manifest and its own audio root, and the
    # budget this script measures has only ever been read on thirty-second
    # excerpts and on SMC. Reaching the full-length corpora needs nothing but
    # these two paths.
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=REPOSITORY / "music" / "ground-truth" / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path,
                        default=REPOSITORY / "music")
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=pathlib.Path, default=MODEL)
    args = parser.parse_args()

    global _binary, _model
    _binary, _model = args.binary, args.model
    for label, path in (("binary", _binary), ("model", _model)):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 1

    # The corpus filter goes to load_corpus and not only to the check below.
    # Without it the loader falls back to DEFAULT_CORPORA, which is ballroom,
    # gtzan and smc -- so Harmonix, whose rows say `ok` rather than
    # `audio-aligned`, is silently dropped.
    items = load_corpus(args.manifest, args.music, False, frozenset(args.corpora))
    seen = sorted({item["corpus"] for item in items})
    missing = [c for c in args.corpora if c not in seen]
    if missing:
        print(f"{missing} not in {args.manifest}; it holds {seen}",
              file=sys.stderr)
        return 1
    report: dict = {
        "provenance": provenance(
            REPOSITORY, {"binary": _binary, "model": _model},
            corpora=args.corpora, limit=args.limit,
            roughening_sweep=list(ROUGHENING_SWEEP)),
        "fps": FPS, "window_sec": WINDOW_SEC, "warmup_sec": WARMUP_SEC,
        "by_corpus": {},
    }

    for corpus in args.corpora:
        part = sample([item for item in items if item["corpus"] == corpus],
                      args.limit)
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            rows = [row for row in pool.map(one, part) if row]
        if not rows:
            print(f"{corpus}: nothing scored")
            continue

        def mean(key: str) -> float:
            return float(np.mean([row[key] for row in rows]))

        def over80(key: str) -> float:
            return float(np.mean([row[key] >= 0.8 for row in rows]))

        # Does the oracle fail where the tempo moves? Spearman rather than
        # Pearson, because only the ordering is meant, and reported with its n
        # so a suggestive number on a small corpus cannot be read as a finding.
        spread = np.array([row["tempo_spread"] for row in rows])
        bump = np.array([row["bump"] for row in rows])
        usable = np.isfinite(spread) & np.isfinite(bump)
        correlation = spearman(spread, bump, min_samples=11)

        labels = [("real", "BeatNet activation      "),
                  ("impulse", "oracle, one frame       "),
                  ("bump", "oracle, bump, shipped   ")]
        if "bump_no_anchor" in rows[0]:
            labels.append(("bump_no_anchor", "oracle, bump, no anchor "))
        labels += [(f"bump_r{value:g}", f"oracle, bump, rough {value:<4g}")
                   for value in ROUGHENING_SWEEP
                   if f"bump_r{value:g}" in rows[0]]

        report["by_corpus"][corpus] = {
            "n": len(rows),
            **{key: mean(key) for key, _ in labels},
            **{f"share_over_80_{key}": over80(key) for key, _ in labels},
            "median_tempo_spread": float(np.median(spread[np.isfinite(spread)]))
            if np.isfinite(spread).any() else None,
            "spearman_tempo_spread_vs_oracle": correlation,
            "tracks": rows,
        }
        print(f"\n{corpus}: {len(rows)} recordings, recall at 70 ms")
        for key, label in labels:
            print(f"   {label} {mean(key):>6.1%}   "
                  f"(>=80% on {over80(key):.1%} of recordings)")
        if correlation is not None:
            print(f"   tempo spread vs oracle recall: rho {correlation:+.2f} "
                  f"over {int(usable.sum())} recordings, median spread "
                  f"{np.median(spread[np.isfinite(spread)]):.3f}")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
