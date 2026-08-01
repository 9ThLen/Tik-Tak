#!/usr/bin/env python3
"""What a period is worth to the causal tracker, supplied and self-found.

Every constant the live filter refused to be tuned further refused for the same
reason: it is what holds the period. ``beat_gain`` cannot go below 3.0, the
onset exponent cannot go above 3.0, and the publishing lock cannot go below
0.25 — each capped by a test that catches the period wandering rather than by
the corpus, which wanted all three looser. If the period came from somewhere
else, none of the three would be answerable for it any more. So this measures
the ceiling before anything is built.

Two ways to supply a tempo, and the difference between them is the whole point:

    seeded   ``--live-seeded`` concentrates the cloud on the right period once
             and lets the filter re-estimate from there
    pinned   ``--live-manual-bpm`` holds the period for the whole recording and
             asks the room only about phase

Neither ships. ``--live-seeded`` reads an offline analysis of the entire
recording, which a phone listening to a room does not have; ``--live-manual-bpm``
here is fed the annotation. They are used precisely because they cannot ship —
they are the only way to ask "what if the tempo were simply known", and they ask
it two different ways.

The reference tempo is 60 over the median annotated inter-beat interval, so it
is the same quantity on every corpus and needs no separate tempo file.

    ceiling, GTZAN 999            silent   F all   CMLt   AMLt  beats/ref
      offline, own tempo            0.0%   0.782  0.649  0.848       1.11
      offline, ORACLE tempo         0.0%   0.856  0.862  0.899       1.00
      live+beatnet, own tempo       0.4%   0.632  0.508  0.589       0.79
      live+beatnet, ORACLE seeded   0.6%   0.634  0.517  0.597       0.79
      live+beatnet, ORACLE pinned   4.1%   0.737  0.790  0.826       0.87
      live+flux,    ORACLE pinned  18.8%   0.528  0.669  0.704       0.77

    ceiling, Ballroom 698
      offline, own tempo            0.0%   0.763  0.579  0.848       1.04
      offline, ORACLE tempo         0.0%   0.860  0.853  0.896       1.01
      live+beatnet, own tempo       0.0%   0.700  0.584  0.600       0.75
      live+beatnet, ORACLE seeded   0.0%   0.708  0.596  0.610       0.75
      live+beatnet, ORACLE pinned   0.1%   0.881  0.873  0.889       0.95
      live+flux,    ORACLE pinned  14.2%   0.547  0.645  0.687       0.87

Three readings, and the third is the one that redirected the work.

*Seeding is nearly worthless and pinning is not.* +0.009 CMLt against +0.282 on
GTZAN, from the same number handed over two different ways. The filter walks
away from a seed within seconds. The tracker's difficulty is not finding the
tempo, it is holding it — which is the same conclusion the corpus-wide octave
telemetry reaches from the other side, where 2188 of 2263 octave switches happen
inside a continuous lock rather than after re-acquiring.

*The period outranks the observation.* Spectral flux with a pinned period
reaches CMLt 0.669 on GTZAN; BeatNet with a free period reaches 0.508. The
weaker front end wins by a wide margin when it is told the tempo, so tempo, not
salience, is what the causal path is short of.

*Pinning beats offline.* Live with BeatNet and a pinned period scores CMLt
0.873 on Ballroom against the offline path's own oracle 0.853. There is nothing
about being causal that costs this; what costs it is not knowing the period.

Then the shippable question, which is the second table. Pin to the tracker's
*own* estimate — nothing offline, nothing acausal, only live telemetry the phone
already has — either after a fixed wall clock or the first moment its own
confidence crosses a threshold.

    self-pinned, GTZAN 999      silent   F all   CMLt   AMLt   right
      free running                0.4%   0.632  0.508  0.589
      pinned after 5 s           18.8%   0.447  0.471  0.610   48.9%
      pinned after 10 s          13.7%   0.502  0.515  0.647   55.3%
      pinned after 15 s          12.9%   0.525  0.544  0.671   58.5%
      pinned after 20 s          12.9%   0.500  0.513  0.643   57.6%
      pinned at conf >= 0.5       6.7%   0.517  0.493  0.635   60.2%
      pinned at conf >= 0.7       1.0%   0.588  0.522  0.645   60.5%
      pinned at conf >= 0.85      0.4%   0.622  0.529  0.623   46.0%
      pinned to the answer        4.1%   0.737  0.790  0.826  100.0%

    self-pinned, Ballroom 698
      free running                0.0%   0.700  0.584  0.600
      pinned after 5 s           28.9%   0.382  0.497  0.577   46.3%
      pinned after 10 s          22.5%   0.500  0.617  0.680   60.0%
      pinned after 15 s          19.6%   0.567  0.677  0.735   66.6%
      pinned after 20 s          17.6%   0.557  0.657  0.718   68.6%
      pinned at conf >= 0.5      10.3%   0.558  0.590  0.667   70.5%
      pinned at conf >= 0.7       0.9%   0.653  0.601  0.653   70.8%
      pinned at conf >= 0.85      0.0%   0.686  0.597  0.624   50.4%
      pinned to the answer        0.1%   0.881  0.873  0.889  100.0%

**No self-pinned row beats free running on F, so hard pinning does not ship.**
A pinned wrong period is wrong for the rest of the recording, and the estimator
is right 46–71% of the time, so the losses swamp the wins. The ``conf >= 0.85``
row only approaches free running because it declines to pin on about half the
recordings and degenerates into it.

The two ends of that column bracket what a tempo estimator has to be worth
before pinning it is a good idea. At 100% right it buys +0.28 CMLt; at 60% it
buys nothing. Somewhere between is the threshold, and it has not been measured
directly — the rows here interpolate, they do not locate it.

The ``right`` column read against how often each gate fires is more useful than
the averages, and says the opposite of the averages. At ``conf >= 0.85`` the gate
fires on 51.2% of GTZAN and is right on 46.0% — 90% conditional accuracy — and on
Ballroom fires on 50.9% and is right on 50.4%, which is 99%. Confidence already
identifies a subset whose tempo can be trusted. What fails is not the selection
but the commitment: an exact BPM frozen forever, which no real performance
holds to anyway. Holding the *octave* while letting the BPM move is the next
thing to measure, and needs an octave-only oracle to bound it.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np

from eval.analysis import DEFAULT_BINARY
from eval.annotations import load_annotation
from eval.harness import evaluate

__all__ = ["reference_bpm", "octave_of", "CEILING_CONFIGS", "LISTEN_SECONDS",
           "CONFIDENCE_GATES"]

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "beatnet_model_1.ttw"

# The octave band matches the live corpus benchmark's, so the two files agree
# about what counts as the same metrical level.
OCTAVE_TOLERANCE = 0.08

LISTEN_SECONDS = (5.0, 10.0, 15.0, 20.0)
CONFIDENCE_GATES = (0.5, 0.7, 0.85)


def reference_bpm(beats: np.ndarray) -> float:
    """The annotated tempo, as 60 over the median inter-beat interval.

    The median rather than the mean because a single missing annotation
    doubles one interval and would drag a mean off the metrical level
    entirely, which is the one error this whole file is about.
    """
    if len(beats) < 2:
        return 0.0
    period = float(np.median(np.diff(beats)))
    return 60.0 / period if np.isfinite(period) and period > 0 else 0.0


def octave_of(estimate: float, reference: float) -> str:
    """Which metrical level an estimate landed on, or ``other``."""
    if not (estimate > 0 and reference > 0):
        return "none"
    ratio = estimate / reference
    for value, name in ((0.5, "half"), (1.0, "right"), (2.0, "double")):
        if abs(ratio / value - 1.0) < OCTAVE_TOLERANCE:
            return name
    return "other"


def _ceiling_configs(weights: str) -> dict[str, tuple[list[str], str]]:
    return {
        "offline, own tempo": ([], "none"),
        "offline, ORACLE tempo": ([], "hint"),
        "live+beatnet, own tempo": (["--live", "--live-model", weights], "none"),
        "live+beatnet, ORACLE seeded":
            (["--live-seeded", "--live-model", weights], "hint"),
        "live+beatnet, ORACLE pinned":
            (["--live", "--live-model", weights], "manual"),
        "live+flux, ORACLE pinned": (["--live"], "manual"),
    }


CEILING_CONFIGS = _ceiling_configs(str(DEFAULT_WEIGHTS))


def _run(binary, wav, flags):
    done = subprocess.run([str(binary), str(wav), *flags],
                          capture_output=True, text=True, timeout=300)
    return json.loads(done.stdout) if done.returncode == 0 else None


def _score(payload, ref):
    beats = np.asarray(payload.get("beats", []), dtype=np.float64)
    scores = evaluate(ref, beats)
    scores["emitted"] = len(beats)
    scores["ratio"] = len(beats) / max(len(ref), 1)
    return scores


def _reference(wav):
    for suffix in (".beats", ".txt"):
        path = wav.with_suffix(suffix)
        if path.is_file():
            beats = np.asarray(load_annotation(path).beats, dtype=np.float64)
            return beats if len(beats) >= 8 else None
    return None


def _ceiling_one(job):
    wav, binary, flags, oracle = job
    try:
        ref = _reference(wav)
        if ref is None:
            return None
        command = list(flags)
        if oracle != "none":
            bpm = reference_bpm(ref)
            if bpm <= 0:
                return None
            command += (["--live-manual-bpm", f"{bpm:.4f}"] if oracle == "manual"
                        else ["--bpm", f"{bpm:.4f}"])
        payload = _run(binary, wav, command)
        return _score(payload, ref) if payload else None
    except Exception:
        return None


def _self_pin_one(job):
    """One recording, free-running then pinned to its own live estimate."""
    wav, binary, weights = job
    try:
        ref = _reference(wav)
        if ref is None:
            return None
        target = reference_bpm(ref)
        live = ["--live", "--live-model", str(weights)]

        base = _run(binary, wav, live)
        if base is None:
            return None
        times = np.asarray(base.get("live_times", []), dtype=np.float64)
        bpms = np.asarray(base.get("live_bpms", []), dtype=np.float64)
        confidences = np.asarray(base.get("live_confidences", []), dtype=np.float64)

        out = {"free": _score(base, ref), "pinned": {}, "gated": {}}
        answer = _run(binary, wav, [*live, "--live-manual-bpm", f"{target:.4f}"])
        if answer is None:
            return None
        out["oracle"] = _score(answer, ref)

        for seconds in LISTEN_SECONDS:
            index = int(np.searchsorted(times, seconds))
            if index >= len(bpms) or not bpms[index] > 0:
                continue
            guess = float(bpms[index])
            payload = _run(binary, wav, [*live, "--live-manual-bpm", f"{guess:.4f}"])
            if payload is None:
                continue
            row = _score(payload, ref)
            row["octave"] = octave_of(guess, target)
            out["pinned"][seconds] = row

        # Pin on the tracker's own confidence rather than on a stopwatch, and
        # keep tracking when it never gets there. A wrong period held forever
        # is worse than one that can still be revised, so declining to pin has
        # to be a real outcome rather than a fallback that never fires.
        for gate in CONFIDENCE_GATES:
            ready = np.flatnonzero((confidences >= gate) & (bpms > 0))
            if len(ready) == 0:
                row = dict(out["free"])
                row["octave"] = "never pinned"
            else:
                guess = float(bpms[ready[0]])
                payload = _run(binary, wav,
                               [*live, "--live-manual-bpm", f"{guess:.4f}"])
                if payload is None:
                    continue
                row = _score(payload, ref)
                row["octave"] = octave_of(guess, target)
                row["waited"] = float(times[ready[0]])
            out["gated"][gate] = row
        return out
    except Exception:
        return None


def _line(label, scored, octaves=False, waits=None):
    spoke = [s for s in scored if s["emitted"] > 0] or scored
    text = (f"{label:<28}{1 - len([s for s in scored if s['emitted'] > 0]) / len(scored):>7.1%}"
            f"{statistics.mean(s['f_measure'] for s in scored):>8.3f}"
            f"{statistics.mean(s['cmlt'] for s in spoke):>7.3f}"
            f"{statistics.mean(s['amlt'] for s in spoke):>7.3f}")
    if octaves:
        counts = collections.Counter(s.get("octave", "none") for s in scored)
        text += (f"{counts['right'] / len(scored):>8.1%}"
                 f"{counts['half'] / len(scored):>7.1%}"
                 f"{counts['double'] / len(scored):>8.1%}"
                 f"{counts['never pinned'] / len(scored):>8.1%}")
        if waits:
            text += f"{statistics.median(waits):>7.0f}s"
    print(text, flush=True)


def _wavs(dataset, limit):
    wavs = sorted(dataset.rglob("*.wav"))
    if limit:
        wavs = wavs[:: max(1, len(wavs) // limit)][:limit]
    return wavs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--mode", choices=("ceiling", "self-pinned"),
                        default="ceiling")
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--weights", type=pathlib.Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    wavs = _wavs(args.dataset, args.limit)
    print(f"{args.dataset.name}: {len(wavs)} clips\n")
    header = f"{'configuration':<28}{'silent':>7}{'F all':>8}{'CMLt':>7}{'AMLt':>7}"
    if args.mode == "self-pinned":
        header += f"{'right':>8}{'half':>7}{'double':>8}{'never':>8}{'waited':>8}"
    print(header)
    print("-" * len(header))

    if args.mode == "ceiling":
        for name, (flags, oracle) in _ceiling_configs(str(args.weights)).items():
            jobs = [(w, args.binary, flags, oracle) for w in wavs]
            with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
                rows = [r for r in pool.map(_ceiling_one, jobs) if r]
            if rows:
                _line(name, rows)
        return 0

    jobs = [(w, args.binary, args.weights) for w in wavs]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(_self_pin_one, jobs) if r]
    if not rows:
        return 1
    _line("free running", [r["free"] for r in rows])
    for seconds in LISTEN_SECONDS:
        scored = [r["pinned"][seconds] for r in rows if seconds in r["pinned"]]
        if scored:
            _line(f"pinned after {int(seconds)} s", scored, octaves=True)
    for gate in CONFIDENCE_GATES:
        scored = [r["gated"][gate] for r in rows if gate in r["gated"]]
        if scored:
            _line(f"pinned at conf >= {gate}", scored, octaves=True,
                  waits=[s["waited"] for s in scored if "waited" in s])
    _line("pinned to the answer", [r["oracle"] for r in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
