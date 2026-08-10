#!/usr/bin/env python3
"""When the live path is at the wrong metrical level, whose fault is it?

The causal path fails on the octave for 27% of ballroom and 38% of GTZAN even
driven by BeatNet, and being handed the true tempo of the whole file does not
fix it. There are two candidates and they need opposite work:

  * the **anchor** is wrong — the activation autocorrelation says half tempo,
    so the prior says half tempo, and the filter follows;
  * the **filter** is wrong — the anchor is right and the cloud sits somewhere
    else anyway, which would mean the prior is not doing what it was added for.

Only the product of the two is visible in the beat list, so `dump_analysis`
dumps both series. This attributes each second of each recording.

**What this cannot show, and what an earlier reading of it claimed.** The rate
at which the filter agrees with a correct anchor is not evidence that the filter
is sound. `LiveTracker::submit` applies `anchorTempo` on every submitted frame —
fifty a second under BeatNet — with a prior a tenth of an octave wide, so that
agreement is largely enforced rather than observed; only the estimate behind it
is refreshed once a second, which is what `update_interval_sec` bounds. What
these numbers do support is the other direction: where the anchor is wrong the
level is usually wrong with it. The residual is not negligible either — a third
of ballroom's wrong seconds happen while the anchor is right. Separating the
filter's own contribution needs it run without the anchor and against an oracle
level, not this statistic.

**Which seconds count.** The tracker's own lock/release hysteresis, matching
``live_corpus_benchmark``. This script used to keep every sample above the lock
threshold instead, which is a different and worse question: it samples the
tracker's most confident moments, and confidence is not independent of what is
being measured. Numbers taken before 2026-08-01 were computed that way and are
not comparable with these.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
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
from eval.provenance import experiment_provenance as provenance  # noqa: E402

MODEL = REPOSITORY / "models" / "beatnet_model_1.ttw"
GROUND_TRUTH = REPOSITORY / "music" / "ground-truth" / "manifest.csv"
TOLERANCE = math.log2(1.08)

# The same hysteresis the corpus benchmark uses, and for the same reason. An
# earlier version of this script kept every sample with `confidence >= 0.25`,
# which is not the tracker's notion of tracking at all: it is a sample of the
# tracker's most confident moments. That biases the very thing being measured,
# because confidence and correctness are not independent — dropping the shaky
# seconds drops the seconds where the anchor and the filter disagree. Once
# locked, the tracker keeps tracking down to `RELEASE`, and those seconds count.
LOCK = 0.25
RELEASE = 0.02
WARMUP_SEC = 5.0


def local_bpm(beats: np.ndarray, at: float) -> float:
    i = int(np.searchsorted(beats, at))
    window = np.diff(beats[max(0, i - 5): min(len(beats), i + 6)])
    window = window[(window > 0.1) & (window < 3.0)]
    return 60.0 / float(np.median(window)) if len(window) else 0.0


def level(bpm: float, reference: float) -> str:
    if not (bpm > 0 and reference > 0):
        return "none"
    r = math.log2(bpm / reference)
    for value, name in ((0.0, "same"), (1.0, "double"), (-1.0, "half")):
        if abs(r - value) <= TOLERANCE:
            return name
    return "other"


def one(item: dict) -> dict | None:
    done = subprocess.run([str(DEFAULT_BINARY), str(item["audio"]), "--live",
                           "--live-model", str(MODEL)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if done.returncode != 0:
        return None
    raw = json.loads(done.stdout)
    times = np.asarray(raw.get("live_times", []), dtype=np.float64)
    filt = np.asarray(raw.get("live_bpms", []), dtype=np.float64)
    anchor = np.asarray(raw.get("live_anchor_bpm", []), dtype=np.float64)
    conf = np.asarray(raw.get("live_confidences", []), dtype=np.float64)
    margin = np.asarray(raw.get("live_anchor_margin", []), dtype=np.float64)
    n = min(len(times), len(filt), len(anchor), len(conf), len(margin))
    if n < 10:
        return None
    reference = load_reference_beats(item["annotation"])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 8:
        return None

    rows = []
    locked = False
    for i in range(n):
        if not locked and conf[i] >= LOCK:
            locked = True
        elif locked and conf[i] < RELEASE:
            locked = False
        if times[i] < WARMUP_SEC or not locked:
            continue          # only while the tracker is actually tracking
        ref = local_bpm(reference, float(times[i]))
        rows.append((level(float(filt[i]), ref), level(float(anchor[i]), ref),
                     float(margin[i])))
    if not rows:
        return None

    # Per recording, and only from quantities the tracker holds at run time.
    # The blame statistic below says where the level goes wrong; it cannot say
    # *when* to act, because nothing in it is available without the annotation.
    # These four are: the estimator's own octave margin, how often it and the
    # filter disagree about the level, and how long the longest disagreement
    # lasts. If one of them separates the recordings that end at the right level
    # from the ones that do not, it is a candidate for making the anchor's
    # strength conditional instead of fixed.
    disagree = np.array([row[0] != row[1] for row in rows])
    longest = current = 0
    for flag in disagree:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {
        "name": item["name"],
        "corpus": item["corpus"],
        "rows": rows,
        "right_level_share": float(np.mean([row[0] == "same" for row in rows])),
        "median_margin": float(np.median([row[2] for row in rows])),
        "disagree_share": float(np.mean(disagree)),
        "longest_disagreement_sec": float(longest),  # one row per second
        "tracked_seconds": len(rows),
    }


def sample(part: list, limit: int) -> list:
    """A subset that spans the corpus rather than stopping partway through it.

    Same correction as `oracle_activation.sample`: the obvious stride-then-
    truncate drops the tail of a list, which on a corpus filed by genre means
    dropping whole genres.
    """
    if not limit or len(part) <= limit:
        return part
    return part[:: -(-len(part) // limit)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=GROUND_TRUTH)
    parser.add_argument("--corpora", nargs="*", default=["rwc-pop", "gtzan"])
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = every recording in the corpus")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    items = load_corpus(args.manifest, REPOSITORY / "music", False)
    report: dict = {
        "provenance": provenance(REPOSITORY,
                                 {"binary": DEFAULT_BINARY, "model": MODEL},
                                 manifest=str(args.manifest.name),
                                 lock=LOCK, release=RELEASE,
                                 warmup_sec=WARMUP_SEC),
        "by_corpus": {},
    }

    for corpus in args.corpora:
        part = sample([i for i in items if i["corpus"] == corpus], args.limit)
        if not part:
            print(f"{corpus}: not in {args.manifest}")
            continue
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            got = [g for g in pool.map(one, part) if g]
        rows = [r for g in got for r in g["rows"]]
        if not rows:
            print(f"{corpus}: nothing")
            continue

        filt = np.array([r[0] for r in rows])
        anchor = np.array([r[1] for r in rows])
        margin = np.array([r[2] for r in rows])
        wrong = filt != "same"
        good = anchor == "same"

        summary = {
            "recordings": len(got),
            "tracking_seconds": len(rows),
            "filter_at_annotated_level": float(np.mean(~wrong)),
            "anchor_at_annotated_level": float(np.mean(good)),
            # Both conditionals, because only the second is the causal one and
            # an earlier reading of this script quoted the first as if it were.
            "p_anchor_wrong_given_filter_wrong": float(
                np.mean(anchor[wrong] != "same")),
            "p_filter_wrong_given_anchor_wrong": float(
                np.mean(filt[~good] != "same")),
            "filter_follows_a_right_anchor": float(np.mean(filt[good] == "same")),
            "median_margin_anchor_right": float(np.median(margin[good])),
            "median_margin_anchor_wrong": float(np.median(margin[~good])),
        }

        # Sustained disagreement between the two is the run-time signal that
        # separates the recordings that hold the level from the ones that lose
        # it. A policy needs one more thing from it: during those seconds, which
        # of the two is right. "Relax the anchor" and "trust the anchor harder"
        # are opposite actions and this is the number that chooses between them.
        split = filt != anchor
        if split.any():
            summary["disagreement"] = {
                "share_of_tracked_seconds": float(np.mean(split)),
                "anchor_is_right": float(np.mean(anchor[split] == "same")),
                "filter_is_right": float(np.mean(filt[split] == "same")),
                "neither": float(np.mean((anchor[split] != "same")
                                         & (filt[split] != "same"))),
                "median_margin": float(np.median(margin[split])),
            }
            d = summary["disagreement"]
            print(f"   when they disagree ({d['share_of_tracked_seconds']:.1%} of "
                  f"seconds): anchor right {d['anchor_is_right']:.1%}, "
                  f"filter right {d['filter_is_right']:.1%}, "
                  f"neither {d['neither']:.1%}")

        print(f"\n{corpus}: {len(got)} recordings, {len(rows)} tracking seconds")
        print(f"   the filter is at the annotated level  "
              f"{summary['filter_at_annotated_level']:.1%}")
        print(f"   the anchor is                         "
              f"{summary['anchor_at_annotated_level']:.1%}")
        print(f"   P(anchor wrong | filter wrong)        "
              f"{summary['p_anchor_wrong_given_filter_wrong']:.1%}")
        print(f"   P(filter wrong | anchor wrong)        "
              f"{summary['p_filter_wrong_given_anchor_wrong']:.1%}   <- causal")
        print(f"   octave margin, anchor right {summary['median_margin_anchor_right']:.3f}"
              f"  wrong {summary['median_margin_anchor_wrong']:.3f}")

        # The discriminator question, per recording and from run-time
        # quantities only: does anything the tracker can see separate the
        # recordings that hold the right level from the ones that do not?
        held = np.array([g["right_level_share"] >= 0.8 for g in got])
        if 5 <= held.sum() <= len(got) - 5:
            print(f"   of {len(got)} recordings, {int(held.sum())} hold the right "
                  f"level for 80% of their tracked seconds:")
            for key, label in (("median_margin", "octave margin        "),
                               ("disagree_share", "anchor/filter disagree"),
                               ("longest_disagreement_sec", "longest disagreement ")):
                a = float(np.median([g[key] for g in np.array(got)[held]]))
                b = float(np.median([g[key] for g in np.array(got)[~held]]))
                summary[f"{key}_held"] = a
                summary[f"{key}_lost"] = b
                print(f"      {label}  held {a:8.3f}   lost {b:8.3f}")
        report["by_corpus"][corpus] = summary

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
