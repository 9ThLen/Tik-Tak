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
from eval.annotations import load_annotation  # noqa: E402

DATA = RESEARCH / "data"
MODEL = REPOSITORY / "models" / "beatnet_model_1.ttw"
TOLERANCE = math.log2(1.08)
LOCK = 0.25


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


def one(audio: pathlib.Path) -> dict | None:
    done = subprocess.run([str(DEFAULT_BINARY), str(audio), "--live",
                           "--live-model", str(MODEL)],
                          capture_output=True, text=True)
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
    reference = load_annotation(audio.with_suffix(".beats")).beats
    if len(reference) < 8:
        return None

    rows = []
    for i in range(n):
        if times[i] < 5.0 or conf[i] < LOCK:
            continue          # only while the tracker is actually tracking
        ref = local_bpm(reference, float(times[i]))
        rows.append((level(float(filt[i]), ref), level(float(anchor[i]), ref),
                     float(margin[i])))
    return {"name": audio.stem, "rows": rows} if rows else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=["ballroom", "gtzan"])
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for corpus in args.corpora:
        folder = DATA / corpus / corpus
        files = sorted(p for p in folder.rglob("*.wav")
                       if p.with_suffix(".beats").is_file())
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            got = [g for g in pool.map(one, files) if g]
        rows = [r for g in got for r in g["rows"]]
        if not rows:
            print(f"{corpus}: nothing")
            continue

        filt = np.array([r[0] for r in rows])
        anchor = np.array([r[1] for r in rows])
        margin = np.array([r[2] for r in rows])
        wrong = filt != "same"
        print(f"\n{corpus}: {len(got)} recordings, {len(rows)} tracking seconds")
        print(f"   the filter is at the annotated level  {np.mean(~wrong):.1%}")
        print(f"   the anchor is                          "
              f"{np.mean(anchor == 'same'):.1%}")
        print(f"\n   of the {int(wrong.sum())} seconds the filter gets wrong:")
        agrees = anchor[wrong] != "same"
        print(f"      the anchor was wrong too   {np.mean(agrees):.1%}  "
              f"-> the estimator chose the level")
        print(f"      the anchor was right       {np.mean(~agrees):.1%}  "
              f"-> the filter left an anchor that was right")
        good = anchor == "same"
        print(f"\n   of the {int(good.sum())} seconds the anchor gets right, "
              f"the filter follows {np.mean(filt[good] == 'same'):.1%}")
        print(f"   octave margin when the anchor is right {np.median(margin[good]):.3f}, "
              f"wrong {np.median(margin[~good]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
