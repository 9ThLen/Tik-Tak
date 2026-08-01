#!/usr/bin/env python3
"""Does the activation have a peak where the beat is, before any decoder looks?

The live benchmark's largest single failure is recall: on GTZAN, 54.3% of
recordings do not find 80% of the beats that were there. Every experiment so far
has asked which *decoder* loses them — the anchor, the particle filter, the
lock. None of those questions can be answered while nobody has checked whether
the evidence reaches the decoder at all, and that is what this measures. No
decoder is involved in the answer.

Every number here is reported against a chance baseline, and that is not
ceremony. The first version of this script asked only whether *any* local
maximum of the activation fell within the window of each beat and got 98.3% on
GTZAN, which reads as "the evidence is nearly always there". It is not what that
measures. The activation has 8.6 local maxima a second against 1.8 beats a
second, and the median peak is 0.0018 high — so the frames are dotted with noise
peaks, and the same test scores 86.8% against *random* times. A measurement whose
null hypothesis scores 87% cannot support a claim at 98%.

So, per recording, against the same ±70 ms window the score uses:

``top_n``
    Take exactly as many of the strongest local maxima as there are annotated
    beats, and match greedily one-to-one. Taking N fixes the density that made
    the naive test vacuous: a decoder is not allowed to win by guessing often.

    This is a **floor, not a ceiling**, and calling it a ceiling here was wrong.
    It is what a decoder gets by reading peak height and ignoring rhythm
    altogether, and ignoring rhythm is the one thing a beat tracker never does —
    ours duly beats it on GTZAN. Read it as the control a real decoder has to
    clear, not as a bound on what the activation contains. The bound is the
    oracle arm in ``oracle_activation.py``.

``top_n_chance``
    The same N strongest peaks, matched against N *random* times over the same
    span. Whatever ``top_n`` exceeds this by is the part that is about the music.

``tracker``
    What the shipping causal path found on the same file, `r70`. Compared with
    ``top_n``, this says whether the tracker is extracting more from the
    activation than a decoder that ignores rhythm altogether — which it should,
    since knowing that beats are periodic is the entire advantage a tracker has.

``peak_present`` and ``peak_present_chance``
    The naive test kept only so that its own null is visible beside it, because
    the number is quotable and would otherwise be quoted.

A local maximum is a frame strictly greater than its neighbours; at 50 fps that
is 20 ms, well inside the window, so peak-finding resolution cannot be what
loses a beat.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
sys.path.insert(0, str(RESEARCH))

from eval.analysis import DEFAULT_BINARY  # noqa: E402
from eval.live_corpus_benchmark import load_corpus  # noqa: E402
from eval.live_corpus_benchmark import load_reference_beats  # noqa: E402
from eval.provenance import provenance  # noqa: E402

WINDOW_SEC = 0.070
MODEL = REPOSITORY / "models" / "beatnet_model_1.ttw"


def local_maxima(values: np.ndarray) -> np.ndarray:
    """Indices of frames strictly above both neighbours."""
    if len(values) < 3:
        return np.empty(0, dtype=np.int64)
    inner = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    ) + 1
    return inner


def matched(reference: np.ndarray, candidates: np.ndarray) -> int:
    """How many reference beats have a candidate within the window.

    Greedy and one-to-one in both directions, as mir_eval is: a single peak
    cannot be credited with two beats, which is what makes a dense candidate set
    fail to score well simply for being dense.
    """
    if len(reference) == 0 or len(candidates) == 0:
        return 0
    used = np.zeros(len(candidates), dtype=bool)
    hits = 0
    for beat in reference:
        near = np.flatnonzero(
            (np.abs(candidates - beat) <= WINDOW_SEC) & ~used
        )
        if len(near):
            best = near[np.argmin(np.abs(candidates[near] - beat))]
            used[best] = True
            hits += 1
    return hits


def one(item: dict) -> dict | None:
    done = subprocess.run(
        [str(DEFAULT_BINARY), str(item["audio"]), "--live",
         "--live-model", str(MODEL), "--dump-activation"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if done.returncode != 0:
        return None
    raw = json.loads(done.stdout)
    times = np.asarray(raw.get("activation_times", []), dtype=np.float64)
    beat = np.asarray(raw.get("activation_beat", []), dtype=np.float64)
    if len(times) < 10 or len(times) != len(beat):
        return None

    reference = load_reference_beats(item["annotation"])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 8:
        return None
    # The tracker is not asked to find beats before it has heard any audio, and
    # neither is the activation. Same warm-up as the live benchmark.
    reference = reference[(reference >= 5.0) & (reference <= times[-1])]
    if len(reference) < 8:
        return None

    peaks = local_maxima(beat)
    if len(peaks) == 0:
        return None
    peak_times = times[peaks]
    peak_heights = beat[peaks]

    # Random times over the span the beats occupy, one per beat, as the null.
    # Seeded from the file's own name so the baseline is identical on every
    # rerun and cannot drift between arms of a comparison.
    #
    # Not `hash()`: Python salts string hashing per process, so the first two
    # runs of this script disagreed in the third digit — 78.7% against 79.1% —
    # for no reason but the interpreter's startup. A baseline that moves when
    # nothing moved is worse than no baseline, because the drift is invisible
    # and looks like signal.
    digest = hashlib.sha256(item["name"].encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    chance_times = np.sort(rng.uniform(reference[0], reference[-1], len(reference)))

    strongest = peaks[np.argsort(peak_heights)[::-1][: len(reference)]]
    top_times = np.sort(times[strongest])

    tracked = np.asarray(raw.get("beats", []), dtype=np.float64)
    tracked = tracked[tracked >= 5.0]

    n = len(reference)
    return {
        "corpus": item["corpus"],
        "name": item["name"],
        "beats": int(n),
        "peaks_per_sec": len(peaks) / float(times[-1] - times[0]),
        "peak_present": matched(reference, peak_times) / n,
        "peak_present_chance": matched(chance_times, peak_times) / n,
        "top_n": matched(reference, top_times) / n,
        "top_n_chance": matched(chance_times, top_times) / n,
        "tracker": matched(reference, tracked) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="*", default=["gtzan", "smc"])
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = every recording in the corpus")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    items = load_corpus(REPOSITORY / "music" / "ground-truth" / "manifest.csv",
                        REPOSITORY / "music", False)
    report: dict = {
        "provenance": provenance(
            REPOSITORY, {"binary": DEFAULT_BINARY, "model": MODEL},
            corpora=args.corpora, limit=args.limit),
        "window_sec": WINDOW_SEC, "warmup_sec": 5.0, "by_corpus": {},
    }

    # Taken from the manifest rather than an allow-list here, for the reason
    # `summarize` gives: a fixed list silently drops every dataset added after
    # it was written.
    available = sorted({item["corpus"] for item in items if item["annotated"]})
    for corpus in args.corpora:
        if corpus not in available:
            parser.error(f"unknown corpus {corpus}; the manifest has "
                         + ", ".join(available))
        part = [item for item in items if item["corpus"] == corpus]
        if args.limit:
            part = part[:: max(1, len(part) // args.limit)][: args.limit]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            rows = [row for row in pool.map(one, part) if row]
        if not rows:
            print(f"{corpus}: nothing scored")
            continue

        def mean(key: str) -> float:
            return float(np.mean([row[key] for row in rows]))

        def over80(key: str) -> float:
            return float(np.mean([row[key] >= 0.8 for row in rows]))

        summary = {
            "n": len(rows),
            "peaks_per_sec": mean("peaks_per_sec"),
            "top_n": mean("top_n"),
            "top_n_chance": mean("top_n_chance"),
            "tracker_r70": mean("tracker"),
            "peak_present": mean("peak_present"),
            "peak_present_chance": mean("peak_present_chance"),
            # The product's pass mark applied to the ceiling: a corpus where the
            # activation itself clears 80% on few recordings is a corpus no
            # decoder can rescue.
            "share_over_80_top_n": over80("top_n"),
            "share_over_80_tracker": over80("tracker"),
        }
        report["by_corpus"][corpus] = summary
        print(f"\n{corpus}: {len(rows)} recordings, "
              f"{mean('peaks_per_sec'):.1f} activation peaks a second")
        print(f"   strongest N peaks       {mean('top_n'):>6.1%}   "
              f"chance {mean('top_n_chance'):.1%}   "
              f"(>=80% on {over80('top_n'):.1%} of recordings)")
        print(f"   the shipping tracker    {mean('tracker'):>6.1%}   "
              f"{' ':13}(>=80% on {over80('tracker'):.1%})")
        print(f"   any peak in the window  {mean('peak_present'):>6.1%}   "
              f"chance {mean('peak_present_chance'):.1%}   <- near-vacuous, "
              f"shown with its null")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
