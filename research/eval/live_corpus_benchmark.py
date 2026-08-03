#!/usr/bin/env python3
"""Corpus-wide benchmark for the causal live tracker.

The ground-truth bundle has a manifest because its four corpora keep audio in
different source folders.  This runner resolves that manifest, executes the
same ``dump_analysis --live`` path as the application, and reports both beat
quality and live tempo stability.

Octave switching is measured against a local reference tempo: the median of up
to ten annotated beat intervals around each one-second live observation.  A
state is half, normal, or double tempo when it is within eight percent of that
ratio.  The tracker's own lock/release hysteresis decides which observations
belong to an active tracking session.

Run from the research folder:

    .venv/Scripts/python -m eval.live_corpus_benchmark \
        --model ../models/beatnet_model_1.ttw --include-root-audio
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import pathlib
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Any

import mir_eval.beat
import mir_eval.util
import numpy as np

from eval.analysis import Analyser, DEFAULT_BINARY, Estimate
from eval.annotations import load_annotation
from eval.harness import evaluate

LOCK_CONFIDENCE = 0.25
RELEASE_CONFIDENCE = 0.02
OCTAVE_TOLERANCE = math.log2(1.08)
ROOT_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a"}

# The per-recording pass mark. See verdict(). These are product thresholds, not
# fitted ones: eight seconds is about the length of an intro, 80% precision is
# where a wrong click stops being an occasional annoyance, 80% coverage is
# where the metronome stops feeling like it is dropping out, and four seconds
# is two bars of 4/4 at 120 BPM -- long enough that a listener has certainly
# noticed. Nothing here was chosen by looking at the results.
MAX_ACQUISITION_SEC = 8.0
MIN_PRECISION = 0.80
MIN_RECALL = 0.80
MAX_WRONG_OCTAVE_SEC = 4.0
# How long a stretch at the annotated level has to last before it is called a
# settle rather than a lucky second. Same figure as the wrong-level limit, so
# the two criteria cannot disagree about what "briefly" means.
SETTLE_SEC = 4.0


def _audio_path(ground_truth: pathlib.Path, row: dict[str, str]) -> pathlib.Path:
    relative = pathlib.Path(row["audio_relpath"])
    if row["dataset"] == "ballroom":
        return ground_truth / "sources" / "ballroom_audio" / relative
    if row["dataset"] == "gtzan":
        return ground_truth / "audio" / "gtzan-ready" / relative
    return ground_truth / relative


def load_corpus(
    manifest: pathlib.Path,
    music: pathlib.Path,
    include_root_audio: bool,
) -> list[dict[str, Any]]:
    ground_truth = manifest.parent
    items: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            has_audio = (
                row["dataset"] in {"ballroom", "gtzan", "smc"}
                or row["status"] == "audio-aligned"
            )
            if not has_audio:
                continue
            items.append(
                {
                    "corpus": row["dataset"],
                    "name": row["track_id"],
                    "audio": _audio_path(ground_truth, row),
                    "annotation": ground_truth / row["annotation_relpath"],
                    "annotated": True,
                }
            )
    if include_root_audio:
        # Numbered, never named. This is whoever's music happens to be on the
        # machine, and it reaches at least three outputs — the aggregate report,
        # the per-track file and any traceback. Anonymising it here rather than
        # at each of those is the only version that stays true when a fourth
        # output is added. Nothing downstream needs the title: root audio is
        # unannotated, so it is scored for "does the binary survive it" and for
        # nothing else.
        items.extend(
            {
                "corpus": "root",
                "name": f"local-{index:04d}",
                "audio": path,
                "annotation": None,
                "annotated": False,
            }
            for index, path in enumerate(
                sorted(path for path in music.iterdir()
                       if path.is_file()
                       and path.suffix.lower() in ROOT_AUDIO_SUFFIXES)
            )
        )
    return items


def load_reference_beats(path: pathlib.Path) -> np.ndarray:
    """Read either the normalized bundle CSV or a regular beat annotation."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "time_seconds" in reader.fieldnames:
            return np.asarray(
                [float(row["time_seconds"]) for row in reader],
                dtype=np.float64,
            )
    return load_annotation(path).beats


def local_reference_bpm(beats: np.ndarray, time_sec: float) -> float:
    beats = np.asarray(beats, dtype=np.float64)
    index = int(np.searchsorted(beats, time_sec))
    start = max(0, index - 5)
    stop = min(len(beats), index + 6)
    intervals = np.diff(beats[start:stop])
    intervals = intervals[
        np.isfinite(intervals) & (intervals > 0.1) & (intervals < 3.0)
    ]
    if len(intervals) == 0:
        return 0.0
    return 60.0 / float(np.median(intervals))


def _midpoints(beats: np.ndarray) -> np.ndarray:
    """The same grid read at twice its rate: every beat plus every gap's centre.

    Interpolated rather than re-tracked, because the question is what the
    *metrical level* costs and not what a differently configured tracker would
    have found. If the tracker is a clean half of the reference, this recovers
    the reference exactly.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 2:
        return beats
    return np.sort(np.concatenate([beats, 0.5 * (beats[:-1] + beats[1:])]))


def tempo_state(bpm: float, reference_bpm: float) -> str:
    if not (
        bpm > 0.0
        and reference_bpm > 0.0
        and math.isfinite(bpm)
        and math.isfinite(reference_bpm)
    ):
        return "zero"
    ratio = math.log2(bpm / reference_bpm)
    for octave, name in ((-1, "half"), (0, "same"), (1, "double")):
        if abs(ratio - octave) <= OCTAVE_TOLERANCE:
            return name
    return "other"


def octave_statistics(estimate: Estimate, beats: np.ndarray) -> dict[str, Any]:
    columns = (
        estimate.live_times,
        estimate.live_bpms,
        estimate.live_confidences,
        estimate.live_tempo_spreads_octaves,
    )
    count = min(map(len, columns))
    locked = False
    previous_state: str | None = None
    last_session_state: str | None = None
    new_session = False
    within_switches = 0
    reacquire_switches = 0
    sessions = 0
    active_samples = 0
    eligible_samples = 0
    states: Counter[str] = Counter()
    active_spreads: list[float] = []
    # When the tracker first believed it had the tempo, and the longest
    # uninterrupted stretch it then spent at the wrong metrical level. Both are
    # per-recording facts that no average over recordings can reconstruct.
    #
    # `acquired_at` is the first confidence lock and nothing more — it does not
    # ask whether the level was right. That is not only a labelling problem, it
    # can change a verdict: lock at 2 s on half tempo, release, re-lock
    # correctly at 10 s, and if the wrong stretch was under the four-second
    # limit the recording passes the eight-second acquisition criterion on the
    # strength of a lock that was wrong. An earlier note here claimed nothing
    # was lost by this, only misattributed, and that was too strong.
    #
    # `settled_at` is the honest one: the **start** of the first locked stretch
    # at the annotated level that then lasts `SETTLE_SEC`, not the moment that
    # stretch is confirmed. Those differ by `SETTLE_SEC`, and the choice is
    # deliberate — the question the criterion asks is "when did the tracker
    # begin doing the right thing", and a listener hears the first correct beat,
    # not the fourth second of correctness. The consequence has to be stated
    # rather than left implicit: `settled_at <= 8 s` admits a recording whose
    # correctness is only *established* at twelve seconds. Anyone comparing this
    # against a latency figure defined the other way will be out by four
    # seconds, in our favour.
    #
    # It is reported beside `acquired_at` rather than replacing it in the pass
    # criterion, so the rates stay comparable with everything measured before it
    # existed — the difference between the two columns is the size of the
    # problem.
    acquired_at: float | None = None
    settled_at: float | None = None
    same_since: float | None = None
    worst_wrong_run = 0.0
    wrong_since: float | None = None

    observations = zip(*(np.asarray(column)[:count] for column in columns))
    for time_sec, bpm, confidence, spread in observations:
        if not locked and confidence >= LOCK_CONFIDENCE:
            locked = True
            new_session = True
            sessions += 1
            if acquired_at is None:
                acquired_at = float(time_sec)
        elif locked and confidence < RELEASE_CONFIDENCE:
            locked = False
            previous_state = None
            new_session = False

        if not locked and wrong_since is not None:
            worst_wrong_run = max(worst_wrong_run, float(time_sec) - wrong_since)
            wrong_since = None
        if not locked:
            same_since = None  # a release ends the stretch, however right it was

        if time_sec < 5.0:
            continue
        eligible_samples += 1
        if not locked:
            continue

        active_samples += 1
        active_spreads.append(float(spread))
        state = tempo_state(
            float(bpm), local_reference_bpm(beats, float(time_sec))
        )
        states[state] += 1
        # A stretch at any level other than the annotated one, however it is
        # reached. "other" and "zero" count: from the player's side an
        # unrelated tempo is not better news than a clean half.
        if state == "same":
            if wrong_since is not None:
                worst_wrong_run = max(worst_wrong_run,
                                      float(time_sec) - wrong_since)
                wrong_since = None
            if same_since is None:
                same_since = float(time_sec)
            elif (settled_at is None
                  and float(time_sec) - same_since >= SETTLE_SEC):
                settled_at = same_since
        else:
            same_since = None
            if wrong_since is None:
                wrong_since = float(time_sec)

        if state not in {"half", "same", "double"}:
            continue
        if previous_state is not None and state != previous_state:
            within_switches += 1
        elif (
            previous_state is None
            and new_session
            and last_session_state is not None
            and state != last_session_state
        ):
            reacquire_switches += 1
        previous_state = state
        last_session_state = state
        new_session = False

    final_time = (
        float(estimate.live_times[count - 1]) if count else float(beats[-1])
    )
    if wrong_since is not None:
        worst_wrong_run = max(worst_wrong_run, final_time - wrong_since)
    return {
        "switches": within_switches + reacquire_switches,
        "within_switches": within_switches,
        "reacquire_switches": reacquire_switches,
        "sessions": sessions,
        "active_samples": active_samples,
        "eligible_samples": eligible_samples,
        "states": dict(states),
        "acquired_at": acquired_at,
        "settled_at": settled_at,
        "worst_wrong_octave_sec": worst_wrong_run,
        "final_ref_bpm": local_reference_bpm(beats, final_time),
        "final_state": tempo_state(
            float(estimate.live_bpm), local_reference_bpm(beats, final_time)
        ),
        "final_active": locked,
        "median_active_spread": (
            float(np.median(active_spreads)) if active_spreads else 0.0
        ),
    }


def verdict(result: dict[str, Any]) -> dict[str, Any]:
    """Is this recording usable, and if not, which way did it fail.

    An average F-measure cannot answer the question the product asks. Two
    recordings with the same CMLt are not the same experience: one takes four
    seconds to start clicking and then holds, the other starts instantly and
    jumps an octave twice in the middle. The user notices the difference and
    the mean does not, so the verdict is per recording and the headline is the
    share of recordings that pass it.

    Four conditions, and every one of them is a thing a user would complain
    about rather than a threshold chosen to make a number look good:

      * it starts within `MAX_ACQUISITION_SEC` -- a metronome that needs longer
        than the first phrase is not a metronome;
      * the beats it does emit are mostly right (`MIN_PRECISION` of them land
        within 70 ms of an annotated beat), because a wrong click is worse
        than no click;
      * it emits most of the beats there were (`MIN_COVERAGE`), because
        silence is also a failure, just a quiet one;
      * it never spends longer than `MAX_WRONG_OCTAVE_SEC` at the wrong
        metrical level, which is the failure an F-measure most readily hides.

    Reasons are reported for every condition that failed, not just the first,
    because a run that fails three of four needs different work from one that
    fails on latency alone.
    """
    reasons = []
    acquired = result.get("acquired_at")
    if acquired is None or not math.isfinite(acquired):
        reasons.append("never_acquired")
    elif acquired > MAX_ACQUISITION_SEC:
        reasons.append("slow_acquisition")
    precision = result.get("p70")
    if precision is None or not math.isfinite(precision) or precision < MIN_PRECISION:
        reasons.append("wrong_beats")
    # Matched recall, not the ratio of the two counts. `coverage` says how many
    # beats were emitted against how many there were, and a tracker can emit
    # exactly the right number of them in all the wrong places: precision 0.80
    # with coverage 0.80 bounds the beats actually found at only 64% of the
    # reference. r70 is the quantity the criterion was always meant to be.
    recall = result.get("r70")
    if recall is None or not math.isfinite(recall) or recall < MIN_RECALL:
        reasons.append("too_few_beats")
    if result.get("worst_wrong_octave_sec", 0.0) > MAX_WRONG_OCTAVE_SEC:
        reasons.append("wrong_octave")

    # The same verdict with the acquisition criterion read strictly: not "when
    # did confidence first cross the lock threshold" but "when did the tracker
    # first settle at the annotated level and stay there". The two differ far
    # more often than the caveat suggested — on 43% of RWC-Pop and 62% of
    # RWC-Jazz the recording acquires inside the limit on the strength of a lock
    # that was at the wrong level or did not hold. Reported beside the headline
    # rather than replacing it, so every earlier number stays comparable and the
    # gap between the two columns is visible instead of being absorbed.
    strict = list(reasons)
    settled = result.get("settled_at")
    if settled is None or not math.isfinite(settled):
        if "never_acquired" not in strict:
            strict.append("never_settled")
    elif settled > MAX_ACQUISITION_SEC:
        if "slow_acquisition" not in strict:
            strict.append("slow_settle")

    # The same verdict with the metrical level forgiven: the grid is allowed to
    # be read at half or twice its rate, whichever agrees with the reference
    # better, and only then judged. Not a claim that a wrong level is
    # acceptable — it is the size of the prize for a product that lets the
    # player halve or double the click with one tap, against one that has to be
    # right unaided. Those are different amounts of work and the gap between
    # these two numbers is what decides which is worth doing.
    # Each reading is judged whole. Taking the best precision from one and the
    # best recall from another describes no grid that exists: thinning a
    # correct grid raises no precision and halves its recall, so a
    # mix-and-match rule would force the thinned arm whenever it scored highest
    # and then fail it for the recall thinning had just destroyed.
    #
    # And this is an **oracle** correction, chosen with the annotation in hand
    # and applied to the whole recording at once. It is therefore an upper
    # bound on a control the player could be given — one press, one level, for
    # the track — and strictly better than that wherever the level wanders
    # mid-recording, which no single press could follow.
    readings = [
        (precision, recall),
        (result.get("p70_thinned"), result.get("r70_thinned")),
        (result.get("p70_thinned_odd"), result.get("r70_thinned_odd")),
        (result.get("p70_doubled"), result.get("r70_doubled")),
    ]
    forgiving: list[str] = []
    if acquired is None or not math.isfinite(acquired):
        forgiving.append("never_acquired")
    elif acquired > MAX_ACQUISITION_SEC:
        forgiving.append("slow_acquisition")
    passes = any(
        p is not None and math.isfinite(p) and p >= MIN_PRECISION
        and r is not None and math.isfinite(r) and r >= MIN_RECALL
        for p, r in readings)
    if not passes:
        # Which of the two it failed on, at whichever reading came closest, so
        # the reason still points somewhere.
        best = max((x for x in readings
                    if x[0] is not None and math.isfinite(x[0])),
                   key=lambda x: x[0], default=(0.0, 0.0))
        if best[0] < MIN_PRECISION:
            forgiving.append("wrong_beats")
        if best[1] is None or not math.isfinite(best[1]) or best[1] < MIN_RECALL:
            forgiving.append("too_few_beats")

    return {"usable": not reasons, "reasons": reasons,
            "usable_strict": not strict, "reasons_strict": strict,
            "usable_any_octave": not forgiving,
            "reasons_any_octave": forgiving}


def _score_one(
    item: dict[str, Any],
    mode: str,
    binary: pathlib.Path,
    model: pathlib.Path | None,
    seeded: bool = False,
    extra: tuple[str, ...] = (),
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        estimate = Analyser(binary).analyse_live_file(
            item["audio"], model=model if mode == "model" else None,
            seeded=seeded, extra=extra,
        )
        result: dict[str, Any] = {
            "ok": True,
            "mode": mode,
            "corpus": item["corpus"],
            "name": item["name"],
            "annotated": item["annotated"],
            "duration": float(estimate.duration_sec),
            "wall": time.perf_counter() - started,
            "live_bpm": float(estimate.live_bpm),
            "live_confidence": float(estimate.live_confidence),
            "live_spread": float(estimate.live_tempo_spread_octaves),
            "beats": len(estimate.beats),
            "late": int(estimate.live_beats_late),
        }
        if not item["annotated"]:
            return result

        reference = load_reference_beats(item["annotation"])
        result.update(evaluate(reference, estimate.beats, trim=True))
        trimmed_reference = mir_eval.beat.trim_beats(
            np.asarray(reference), min_beat_time=5.0
        )
        trimmed_estimate = mir_eval.beat.trim_beats(
            np.asarray(estimate.beats), min_beat_time=5.0
        )
        matches = (
            mir_eval.util.match_events(
                trimmed_reference, trimmed_estimate, window=0.07
            )
            if len(trimmed_reference) and len(trimmed_estimate)
            else []
        )
        result["p70"] = (
            len(matches) / len(trimmed_estimate) if len(trimmed_estimate) else 0.0
        )
        result["r70"] = (
            len(matches) / len(trimmed_reference)
            if len(trimmed_reference)
            else float("nan")
        )
        result["coverage"] = (
            len(trimmed_estimate) / len(trimmed_reference)
            if len(trimmed_reference)
            else float("nan")
        )
        # The same counts with the grid moved one metrical level either way, so
        # a run can ask what the level alone costs without a second pass over
        # the corpus. A tracker at double time hits every reference beat and
        # twice as many besides — full recall, half the precision — and no
        # single number tells that apart from a tracker that is simply wrong.
        #
        # Both thinning phases, because reading a doubled grid at half its rate
        # can start on either of its beats and only one of them is the music's.
        # Testing one would report half the doubled trackers as unrecoverable.
        for name, folded in (("thinned", trimmed_estimate[::2]),
                             ("thinned_odd", trimmed_estimate[1::2]),
                             ("doubled", _midpoints(trimmed_estimate))):
            hits = (
                mir_eval.util.match_events(trimmed_reference, folded, window=0.07)
                if len(trimmed_reference) and len(folded)
                else []
            )
            result[f"p70_{name}"] = len(hits) / len(folded) if len(folded) else 0.0
            result[f"r70_{name}"] = (
                len(hits) / len(trimmed_reference)
                if len(trimmed_reference) else float("nan")
            )
        result.update(octave_statistics(estimate, reference))
        result.update(verdict(result))
        return result
    except Exception as error:  # keep a bad file from hiding the corpus result
        return {
            "ok": False,
            "mode": mode,
            "corpus": item["corpus"],
            "name": item["name"],
            "annotated": item["annotated"],
            "error": _without_local_paths(str(error)),
            "wall": time.perf_counter() - started,
        }


# The report is meant to be committed beside the numbers that cite it, so it
# must not carry this machine's directory layout. Exception text is the one
# place a path gets in, because the decoder names the file it could not read.
def _without_local_paths(text: str) -> str:
    cleaned = re.sub(r"[A-Za-z]:[\\/][^\s,;]*[\\/]([^\s\\/,;]+)", r"<audio>/\1", text)
    return re.sub(r"(?<![\w<])/(?:[^\s/,;]+/)+([^\s/,;]+)", r"<audio>/\1", cleaned)


# What a number needs beside it to still mean something in six months. A rate
# quoted without the commit, the weights and the corpus it was measured on is
# not reproducible and not falsifiable either: any later disagreement is
# unresolvable, because nobody can tell whether the code moved or the corpus did.
def _digest(path: pathlib.Path) -> dict | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return {"name": path.name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _provenance(binary: pathlib.Path, model: pathlib.Path | None,
                items: list[dict[str, Any]], repository: pathlib.Path) -> dict:
    def git(*command: str) -> str | None:
        try:
            done = subprocess.run(("git", "-C", str(repository)) + command,
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    corpora: Counter[str] = Counter(item["corpus"] for item in items)
    annotated: Counter[str] = Counter(
        item["corpus"] for item in items if item["annotated"])
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git("rev-parse", "HEAD"),
        "tree_clean": git("status", "--porcelain") == "",
        "binary": _digest(binary),
        "model": _digest(model) if model else None,
        "python": platform.python_version(),
        "platform": platform.system(),
        "corpora": {name: {"files": corpora[name], "annotated": annotated[name]}
                    for name in sorted(corpora)},
    }


def _finite_stat(values: list[float], function: Any) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(function(array)) if len(array) else None


def summarize(mode: str, results: list[dict[str, Any]], wall: float) -> dict:
    good = [result for result in results if result["ok"]]
    scored = [result for result in good if result["annotated"]]
    quality = {
        key: _finite_stat([result[key] for result in scored], np.mean)
        for key in ("f_measure", "cmlt", "amlt", "p70", "r70", "coverage")
    }
    state_counts: Counter[str] = Counter()
    for result in scored:
        state_counts.update(result["states"])
    total_states = sum(state_counts.values())
    final_states = Counter(result["final_state"] for result in scored)
    switching = [result for result in scored if result["switches"] > 0]
    scored_hours = sum(result["duration"] for result in scored) / 3600.0

    # The manifest defines the annotated corpora. Keeping a fixed allow-list
    # here silently dropped every newly prepared dataset from the per-corpus
    # report; RWC 2.0 was the first one to expose that. Root audio is excluded
    # because it returns before scoring and therefore is absent from `scored`.
    corpus_names = sorted({result["corpus"] for result in scored})
    by_corpus = {}
    for corpus in corpus_names:
        part = [result for result in scored if result["corpus"] == corpus]
        switches = sum(result["switches"] for result in part)
        hours = sum(result["duration"] for result in part) / 3600.0
        finals = Counter(result["final_state"] for result in part)
        failures: Counter[str] = Counter()
        for result in part:
            failures.update(result.get("reasons", ()))
        by_corpus[corpus] = {
            "n": len(part),
            "usable_rate": (
                sum(result["usable"] for result in part) / len(part)
                if part else None
            ),
            # The oracle-level ceiling beside it, because the gap between the
            # two is the whole argument about whether the metrical level is
            # worth engineering around, and a summary that omits it forces
            # every reader back to the per-track file.
            "usable_rate_any_octave": (
                sum(result["usable_any_octave"] for result in part) / len(part)
                if part else None
            ),
            # The headline read strictly: acquisition means settling at the
            # right level, not merely locking. The gap between this and
            # `usable_rate` is how much of the headline rests on locks that were
            # not right.
            "usable_rate_strict": (
                sum(result["usable_strict"] for result in part) / len(part)
                if part else None
            ),
            # Every reason a recording failed for, so a corpus that fails on
            # latency is never confused with one that fails on octaves.
            "failure_reasons": {
                key: count / len(part) for key, count in failures.most_common()
            } if part else {},
            "median_acquisition_sec": _finite_stat(
                [result["acquired_at"] for result in part
                 if result.get("acquired_at") is not None], np.median),
            # Beside it, and deliberately not instead of it: the first lock that
            # was at the right level and held. The share that never settle at
            # all is the part `median_acquisition_sec` cannot see, because a
            # recording that locks fast and wrongly still contributes a small
            # number to it.
            "median_settle_sec": _finite_stat(
                [result["settled_at"] for result in part
                 if result.get("settled_at") is not None], np.median),
            "never_settled_fraction": (
                sum(result.get("settled_at") is None for result in part)
                / len(part) if part else None
            ),
            "acquisition_was_a_false_start_fraction": (
                sum(
                    result.get("acquired_at") is not None
                    and result["acquired_at"] <= MAX_ACQUISITION_SEC
                    and (result.get("settled_at") is None
                         or result["settled_at"] > MAX_ACQUISITION_SEC)
                    for result in part
                ) / len(part) if part else None
            ),
            "f_measure": _finite_stat(
                [result["f_measure"] for result in part], np.mean
            ),
            "cmlt": _finite_stat([result["cmlt"] for result in part], np.mean),
            "switch_tracks": sum(result["switches"] > 0 for result in part),
            "switches": switches,
            "switches_per_hour": switches / hours if hours else 0.0,
            "final_same_fraction": finals["same"] / len(part) if part else None,
            "final_half": finals["half"],
            "final_double": finals["double"],
            "final_other_or_zero": finals["other"] + finals["zero"],
        }

    wanted = {
        "0038_bringmetolife",
        "0439_lovethewayyoulie",
        "0446_midnightcity",
        "0471_paradise",
    }
    harmonix_four = [
        {
            key: result[key]
            for key in (
                "name",
                "live_bpm",
                "live_confidence",
                "live_spread",
                "final_ref_bpm",
                "final_state",
                "final_active",
                "switches",
                "within_switches",
                "reacquire_switches",
            )
        }
        for result in scored
        if result["name"] in wanted
    ]
    top = sorted(
        switching,
        key=lambda result: (result["switches"], result["within_switches"]),
        reverse=True,
    )[:12]
    duration = sum(result["duration"] for result in good)
    # Macro, not pooled. GTZAN is five times the size of SMC, so a pooled rate
    # is mostly a statement about GTZAN and would let the corpus this product
    # finds hardest disappear into the corpus it finds easiest.
    big = [c for c in by_corpus
           if by_corpus[c]["usable_rate"] is not None and by_corpus[c]["n"] >= 30]
    rates = [by_corpus[c]["usable_rate"] for c in big]
    rates_any = [by_corpus[c]["usable_rate_any_octave"] for c in big]
    rates_strict = [by_corpus[c]["usable_rate_strict"] for c in big]
    pooled_failures: Counter[str] = Counter()
    for result in scored:
        pooled_failures.update(result.get("reasons", ()))

    return {
        "mode": mode,
        "attempted": len(results),
        "success": len(good),
        "scored": len(scored),
        "usable_rate_macro": float(np.mean(rates)) if rates else None,
        "usable_rate_any_octave_macro": (
            float(np.mean(rates_any)) if rates_any else None),
        "usable_rate_pooled": (
            sum(result["usable"] for result in scored) / len(scored)
            if scored else None
        ),
        "usable_rate_any_octave_pooled": (
            sum(result["usable_any_octave"] for result in scored) / len(scored)
            if scored else None
        ),
        # Aggregated alongside the other two rather than left per corpus, so a
        # reader who quotes the headline can see in the same object how much of
        # it rests on locks that were never at the right level.
        "usable_rate_strict_macro": (
            float(np.mean(rates_strict)) if rates_strict else None),
        "usable_rate_strict_pooled": (
            sum(result["usable_strict"] for result in scored) / len(scored)
            if scored else None
        ),
        "failure_reasons": {
            key: count / len(scored) for key, count in pooled_failures.most_common()
        } if scored else {},
        # Named only for the annotated corpora. `root` is whatever audio the
        # machine happens to hold — a person's own music — and this report is
        # meant to be committed, so those titles do not belong in it. The count
        # stays, because a run that silently stopped reading half its input has
        # to be visible.
        "failures": [
            {
                "corpus": result["corpus"],
                "name": (result["name"] if result["annotated"]
                         else "<local audio>"),
                "error": (result["error"] if result["annotated"]
                          else "not decodable"),
            }
            for result in results
            if not result["ok"]
        ],
        "quality": quality,
        "silent_tracks": sum(result["beats"] == 0 for result in scored),
        "median_final_confidence": _finite_stat(
            [result["live_confidence"] for result in scored], np.median
        ),
        "median_final_spread_octaves": _finite_stat(
            [result["live_spread"] for result in scored], np.median
        ),
        "active_fraction": (
            sum(result["active_samples"] for result in scored)
            / max(1, sum(result["eligible_samples"] for result in scored))
        ),
        "octave": {
            "tracks_with_switch": len(switching),
            "track_fraction": len(switching) / len(scored) if scored else 0.0,
            "total_switches": sum(result["switches"] for result in scored),
            "within_lock_switches": sum(
                result["within_switches"] for result in scored
            ),
            "reacquire_switches": sum(
                result["reacquire_switches"] for result in scored
            ),
            "switches_per_audio_hour": (
                sum(result["switches"] for result in scored) / scored_hours
            ),
            "active_state_shares": {
                state: state_counts[state] / total_states if total_states else 0.0
                for state in ("same", "half", "double", "other", "zero")
            },
            "final_states": dict(final_states),
        },
        "by_corpus": by_corpus,
        "top_switching": [
            {
                "corpus": result["corpus"],
                "name": result["name"],
                "switches": result["switches"],
                "within": result["within_switches"],
                "reacquire": result["reacquire_switches"],
                "final_bpm": result["live_bpm"],
                "final_state": result["final_state"],
            }
            for result in top
        ],
        "harmonix_four": sorted(
            harmonix_four, key=lambda result: result["name"]
        ),
        "audio_hours": duration / 3600.0,
        "wall_seconds": wall,
        "rtf": wall / max(1.0, duration),
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=repository / "music" / "ground-truth" / "manifest.csv",
    )
    parser.add_argument(
        "--music", type=pathlib.Path, default=repository / "music"
    )
    parser.add_argument(
        "--binary", type=pathlib.Path, default=DEFAULT_BINARY
    )
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument(
        "--mode", choices=("baseline", "model", "both"), default="both"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-root-audio", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--seeded", action="store_true",
                        help="hand the tracker the tempo an offline analysis "
                             "of the whole file found. Not shippable — it is "
                             "the ceiling on any proposal to listen first "
                             "before answering, since a few seconds of buffer "
                             "cannot beat the whole recording")
    # The arm that matters enough to be a flag of its own. `--extra` can express
    # it too, but only as `--extra=--live-no-anchor`: argparse refuses a
    # separate value that begins with a dash, so the obvious spelling with a
    # space fails outright — and an arm of the central experiment should not
    # depend on remembering that.
    parser.add_argument("--no-anchor", action="store_true",
                        help="run the particle filter with the activation-tempo "
                             "anchor off. The anchor is applied on every frame, "
                             "so the filter agreeing with a correct anchor says "
                             "nothing about the filter; this is the only arm "
                             "that measures what the filter contributes alone")
    # One string rather than nargs, because argparse hands leading-dash values
    # in an nargs list straight to its own parser and eats the flag.
    parser.add_argument("--extra", default="",
                        help="further flags passed to dump_analysis unchanged. "
                             "Must be spelled --extra=--flag, not --extra "
                             "'--flag'")
    parser.add_argument("--per-track", type=pathlib.Path,
                        help="write one file per mode with every recording's "
                             "metrics and verdict")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode in {"model", "both"} and args.model is None:
        parser.error("--model is required for model mode")

    extra_flags = tuple(args.extra.split())
    if args.no_anchor:
        extra_flags = ("--live-no-anchor",) + extra_flags

    items = load_corpus(args.manifest, args.music, args.include_root_audio)
    missing_audio = [str(item["audio"]) for item in items if not item["audio"].is_file()]
    missing_annotations = [
        str(item["annotation"])
        for item in items
        if item["annotated"] and not item["annotation"].is_file()
    ]
    if missing_audio or missing_annotations:
        print(
            json.dumps(
                {
                    "missing_audio": missing_audio,
                    "missing_annotations": missing_annotations,
                }
            )
        )
        return 2

    modes = ("baseline", "model") if args.mode == "both" else (args.mode,)
    print(
        json.dumps(
            {
                "event": "start",
                "canonical": len(items),
                "annotated": sum(item["annotated"] for item in items),
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    summaries = []
    for mode in modes:
        started = time.perf_counter()
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = [
                pool.submit(_score_one, item, mode, args.binary, args.model,
                            args.seeded, extra_flags)
                for item in items
            ]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                results.append(future.result())
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "mode": mode,
                                "done": completed,
                                "total": len(futures),
                                "elapsed_sec": round(
                                    time.perf_counter() - started, 1
                                ),
                            }
                        ),
                        flush=True,
                    )
        summaries.append(
            summarize(mode, results, time.perf_counter() - started)
        )
        # Per recording, not only the aggregate. A verdict is only useful if the
        # recordings that failed can be listened to, and an aggregate cannot be
        # re-cut when the pass mark is questioned.
        if args.per_track:
            path = args.per_track.with_name(
                f"{args.per_track.stem}.{mode}{args.per_track.suffix}")
            path.write_text(json.dumps(
                [{key: value for key, value in result.items()
                  if not isinstance(value, np.ndarray)} for result in results],
                ensure_ascii=False), encoding="utf-8")

    report = {
        "provenance": _provenance(args.binary, args.model, items, repository),
        "protocol": {
            "causal": True,
            "callback_samples": 512,
            "poll_seconds": 1.0,
            "warmup_seconds": 5.0,
            "lock_confidence": LOCK_CONFIDENCE,
            "release_confidence": RELEASE_CONFIDENCE,
            "octave_tolerance_percent": 8.0,
            "seeded": args.seeded,
            # The arm this run is, spelled exactly as it reached the binary.
            "tracker_flags": list(extra_flags),
            "usable": {
                "max_acquisition_sec": MAX_ACQUISITION_SEC,
                "min_precision_70ms": MIN_PRECISION,
                "min_recall_70ms": MIN_RECALL,
                "max_wrong_octave_sec": MAX_WRONG_OCTAVE_SEC,
                "settle_sec": SETTLE_SEC,
                "acquisition": (
                    "the pass criterion uses acquired_at, the first confidence "
                    "lock, which does not ask whether the level was right — a "
                    "brief wrong lock inside the limit can carry a recording "
                    "past it. settled_at is reported beside it and is the "
                    "first locked stretch at the annotated level lasting "
                    "settle_sec; acquisition_was_a_false_start_fraction is how "
                    "often the two disagree in the direction that flatters us"
                ),
                "aggregate": "macro-average over corpora with n >= 30",
                # Stated because it is easy to read these as shares of the
                # failures and be wrong by the failure rate.
                "failure_reasons": (
                    "share of ALL recordings in the corpus failing for each "
                    "reason; one recording can fail for several, so they do "
                    "not sum to the failure rate"
                ),
                "any_octave": (
                    "oracle: the grid is read at half (both phases) or twice "
                    "its rate and judged at whichever agrees best, over the "
                    "whole recording — an upper bound on any control the "
                    "player could be given"
                ),
            },
            "reference_bpm": (
                "local median of up to 10 neighbouring annotated intervals"
            ),
            "switch_definition": (
                "half/normal/double state change while hysteresis-active; "
                "reacquisition changes reported separately"
            ),
        },
        "summaries": summaries,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        "FINAL_JSON="
        + json.dumps(report, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
