#!/usr/bin/env python3
"""Run the preregistered M0e paired non-oracle decoder regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
import wave
from collections import Counter, defaultdict

import numpy as np

from eval.live_corpus_benchmark import _without_local_paths
from eval.m0a_oracle import (
    BLOCK_SAMPLES,
    oracle_channel,
    reference_emit,
    replay_bar,
    require_reference_within_audio,
    visible_indices,
)
from eval.m0b_oracle import (
    PAUSED_EXIT_CODE,
    _atomic_write_json,
    _checkpoint_state,
    _match,
    _require_outside_repository,
    load_annotation,
    load_manifest,
    prepare_checkpoint,
    run_checkpointed,
    score_dynamic,
)
from eval.m0c_transition import (
    _assert_parity,
    transition_traces,
    validate_source_artifact as validate_m0b_artifact,
)
from eval.m0d_reacquisition import (
    _file_sha256,
    _state_change_indices,
    validate_source_artifact as validate_m0c_artifact,
)
from eval.octave_veto_replay import run
from eval.provenance import digest, experiment_provenance
from eval.s0_reset import InvariantError, paired_bootstrap


SOURCE_M0D_SHA256 = (
    "668b1890e5055ffb4db9d2ecba00fa03f1c4bfdfd80c656e1764b8a95f419991")
SOURCE_M0D_COMMIT = "b5376267e70a5d98daefb0cf9365e25f60e3cbca"
SOURCE_M0C_SHA256 = (
    "88d7ecc2e2ef655faf475081768218a0dc2467d29d048bfa4b1eafc5d23f74fa")
SOURCE_M0B_SHA256 = (
    "142580478abfe0734bc91ac8fdd20c605a392f9ad7334cb63047d98e5135e921")
SOURCE_MANIFEST_SHA256 = (
    "484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd")
SOURCE_MODEL_SHA256 = (
    "812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1")

CHECKPOINT_SCHEMA = "tiktak.m0e_checkpoint/v1"
ARTIFACT_SCHEMA = "tiktak.m0e_non_oracle/v1"
BOOTSTRAP_DRAWS = 2000
PARITY_TOLERANCE = 1e-12
MATCH_TOLERANCE_SEC = 0.07
FIXED_RECORDS = 980
FIXED_WORKS = 414
FIXED_TRANSITION_RECORDS = 34
FIXED_TRANSITIONS = 123
FIXED_FULLY_OBSERVABLE = 61
FIXED_EFFICACY_WORKS = 31
MIN_EFFICACY_WORKS = 30
EFFICACY_MIN_GAIN = 0.10
STABLE_MAX_LOSS = 0.03
FALSE_SWITCH_MAX_INCREASE_5MIN = 1.0
LONG_EPISODE_MAX_INCREASE_5MIN = 0.25

ARMS = {
    "B64_opening": {"latest": False, "switch_cost": 64.0},
    "L2_latest": {"latest": True, "switch_cost": 2.0},
}
BASELINE = "B64_opening"
CANDIDATE = "L2_latest"

FIXED_CORPUS_COUNTS = {
    "bpsd": (122, 31, 20),
    "candombe": (35, 35, 0),
    "kraisler": (20, 20, 1),
    "rubato": (489, 14, 56),
    "rwc2": (314, 314, 123),
}


def _flags(arm: str) -> list[str]:
    spec = ARMS[arm]
    flags = ["--bar-phase-switch-cost", repr(spec["switch_cost"])]
    if spec["latest"]:
        flags.append("--bar-latest-path-phase")
    return flags


def _array_sha256(*arrays: np.ndarray) -> str:
    value = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        value.update(str(contiguous.dtype).encode("ascii"))
        value.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        value.update(contiguous.tobytes())
    return value.hexdigest()


def validate_m0d_artifact(path: pathlib.Path) -> dict:
    if _file_sha256(path) != SOURCE_M0D_SHA256:
        raise ValueError("source M0d artifact digest changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {})
    if payload.get("schema") != "tiktak.m0d_reacquisition/v1":
        raise ValueError("source artifact is not M0d schema v1")
    if (provenance.get("experiment") != "M0d"
            or provenance.get("commit") != SOURCE_M0D_COMMIT
            or provenance.get("tree_clean") is not True):
        raise ValueError("source M0d provenance changed")
    if (payload.get("selected") != 34 or payload.get("scored") != 34
            or payload.get("technical_exclusions") not in ([], None)):
        raise ValueError("source M0d accounting changed")
    summary = payload.get("summary", {})
    if (summary.get("selected_candidate") != CANDIDATE
            or summary.get("interpretation") != "phase_hysteresis_bottleneck"):
        raise ValueError("source M0d no longer selects L2_latest")
    if payload.get("source_m0c", {}).get("sha256") != SOURCE_M0C_SHA256:
        raise ValueError("source M0d is bound to another M0c artifact")
    return payload


def validate_sources(m0b_path: pathlib.Path, m0c_path: pathlib.Path,
                     m0d_path: pathlib.Path) -> tuple[dict, dict, dict]:
    m0b = validate_m0b_artifact(m0b_path)
    m0c = validate_m0c_artifact(m0c_path)
    m0d = validate_m0d_artifact(m0d_path)
    if _file_sha256(m0b_path) != SOURCE_M0B_SHA256:
        raise ValueError("source M0b artifact digest changed")
    if _file_sha256(m0c_path) != SOURCE_M0C_SHA256:
        raise ValueError("source M0c artifact digest changed")
    primary = [row for row in m0b["records"]
               if row.get("primary_eligible") is True]
    if (m0b.get("selected") != 1005 or m0b.get("scored") != 998
            or len(m0b.get("technical_exclusions", [])) != 7
            or len(primary) != FIXED_RECORDS
            or len({row["work_id"] for row in primary}) != FIXED_WORKS):
        raise ValueError("source M0b primary population changed")
    return m0b, m0c, m0d


def select_population(items: list[dict], m0b: dict, m0c: dict,
                      m0d: dict) -> list[dict]:
    manifest = {}
    for item in items:
        identity = (item["corpus"], item["name"])
        if identity in manifest:
            raise ValueError(f"duplicate manifest identity: {identity}")
        manifest[identity] = item
    m0c_rows = {(row["corpus"], row["name"]): row
                for row in m0c["records"]}
    m0d_rows = {(row["corpus"], row["name"]): row
                for row in m0d["records"]}
    if len(m0c_rows) != FIXED_TRANSITION_RECORDS or set(m0c_rows) != set(m0d_rows):
        raise ValueError("M0c/M0d transition populations differ")

    selected = []
    for source in m0b["records"]:
        if source.get("primary_eligible") is not True:
            continue
        identity = (source["corpus"], source["name"])
        item = manifest.get(identity)
        if item is None:
            raise ValueError(f"source M0b record missing from manifest: {identity}")
        if item.get("work_id") != source.get("work_id"):
            raise ValueError(f"source/manifest work mismatch: {identity}")
        selected.append({**item, "source_m0b": source,
                         "source_m0c": m0c_rows.get(identity)})
    selected.sort(key=lambda row: (row["corpus"], row["name"]))
    if len(selected) != FIXED_RECORDS:
        raise ValueError(f"expected {FIXED_RECORDS} primary records")

    profile = {}
    for corpus in FIXED_CORPUS_COUNTS:
        rows = [row for row in selected if row["corpus"] == corpus]
        profile[corpus] = (
            len(rows), len({row["work_id"] for row in rows}),
            sum(row["source_m0b"]["arms"]["A1"]["changes"]["total"]
                for row in rows))
    if profile != FIXED_CORPUS_COUNTS:
        raise ValueError(f"source corpus profile changed: {profile}")
    if sum(row["source_m0c"] is not None for row in selected) != 34:
        raise ValueError("expected 34 transition records")
    return selected


def _map_to_reference(beats: np.ndarray, positions: np.ndarray,
                      meters: np.ndarray, confident: np.ndarray,
                      reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    match = _match(reference["times"], beats, tolerance=MATCH_TOLERANCE_SEC)
    mapped_positions = np.full(len(match), -1, dtype=np.int64)
    mapped_meters = np.zeros(len(match), dtype=np.int64)
    mapped_confident = np.zeros(len(match), dtype=bool)
    found = match >= 0
    mapped_positions[found] = positions[match[found]].astype(np.int64)
    mapped_meters[found] = meters[match[found]].astype(np.int64)
    mapped_confident[found] = confident[match[found]] > 0.0
    return {"match": match, "positions": mapped_positions,
            "meters": mapped_meters, "confident": mapped_confident}


def _change_indices(reference: dict[str, np.ndarray], start_sec: float) -> list[int]:
    supported = reference["supported"] & (reference["times"] >= start_sec)
    downbeats = np.flatnonzero((reference["positions"] == 1) & supported)
    return [int(current) for previous, current in zip(downbeats, downbeats[1:])
            if reference["groupings"][current]
            != reference["groupings"][previous]]


def adaptation_mask(reference: dict[str, np.ndarray], start_sec: float) -> np.ndarray:
    mask = np.zeros(len(reference["times"]), dtype=bool)
    changes = _change_indices(reference, start_sec)
    segments = reference["segments"]
    for ordinal, change in enumerate(changes):
        stop = changes[ordinal + 1] if ordinal + 1 < len(changes) else len(mask)
        segment = segments[change]
        later_segment = np.flatnonzero(segments[change:] != segment)
        segment_end = (change + int(later_segment[0])
                       if len(later_segment) else len(mask))
        stop = min(stop, segment_end,
                   change + 2 * int(reference["groupings"][change]))
        mask[change:stop] = True
    return mask


def static_safety(mapped: dict[str, np.ndarray],
                  reference: dict[str, np.ndarray], start_sec: float) -> dict:
    positions = mapped["positions"]
    meters = mapped["meters"]
    eligible = (reference["supported"]
                & (reference["times"] >= start_sec)
                & ~adaptation_mask(reference, start_sec))
    exact = ((meters == reference["groupings"])
             & (positions + 1 == reference["positions"]))
    indices = np.flatnonzero(eligible)
    duration = 0.0
    for index in indices:
        following = index + 1
        if (following < len(eligible) and eligible[following]
                and reference["segments"][following]
                == reference["segments"][index]):
            duration += float(reference["times"][following]
                              - reference["times"][index])

    false_switches = 0
    previous_answered = None
    for index in indices:
        if meters[index] <= 0 or positions[index] < 0:
            previous_answered = None
            continue
        if previous_answered is not None:
            previous = previous_answered
            reference_anchor = (
                int(reference["groupings"][index]),
                (int(index) - (int(reference["positions"][index]) - 1))
                % int(reference["groupings"][index]))
            previous_reference_anchor = (
                int(reference["groupings"][previous]),
                (int(previous) - (int(reference["positions"][previous]) - 1))
                % int(reference["groupings"][previous]))
            state = (int(meters[index]),
                     (int(index) - int(positions[index])) % int(meters[index]))
            previous_state = (
                int(meters[previous]),
                (int(previous) - int(positions[previous]))
                % int(meters[previous]))
            if (index == previous + 1
                    and reference["segments"][index]
                    == reference["segments"][previous]
                    and reference_anchor == previous_reference_anchor
                    and state != previous_state):
                false_switches += 1
        previous_answered = int(index)

    episodes = []
    active: list[int] = []

    def finish() -> None:
        if active:
            episodes.append(tuple(active))
            active.clear()

    previous = None
    for index in indices:
        contiguous = (previous is not None and index == previous + 1
                      and reference["segments"][index]
                      == reference["segments"][previous])
        if not contiguous:
            finish()
        if exact[index]:
            finish()
        else:
            active.append(int(index))
        previous = int(index)
    finish()

    long_one = 0
    long_two = 0
    longest_events = 0
    longest_sec = 0.0
    for episode in episodes:
        grouping = int(reference["groupings"][episode[0]])
        count = len(episode)
        long_one += int(count >= grouping)
        long_two += int(count >= 2 * grouping)
        longest_events = max(longest_events, count)
        if count > 1:
            longest_sec = max(
                longest_sec,
                float(reference["times"][episode[-1]]
                      - reference["times"][episode[0]]))
    events = int(np.sum(eligible))
    correct = int(np.sum(exact & eligible))
    return {
        "correct": correct, "events": events,
        "accuracy": correct / events if events else None,
        "eligible_duration_sec": duration,
        "false_switches": false_switches,
        "wrong_episodes": len(episodes),
        "long_wrong_episodes_1bar": long_one,
        "long_wrong_episodes_2bar": long_two,
        "longest_wrong_episode_events": longest_events,
        "longest_wrong_episode_sec": longest_sec,
    }


def _assert_live_baseline_parity(initial: dict, positions: np.ndarray,
                                 meters: np.ndarray,
                                 confident: np.ndarray,
                                 name: str) -> None:
    for expected_key, actual in (
        ("live_bar_positions_all", positions),
        ("live_bar_meters_all", meters),
        ("live_bar_confident_all", confident),
    ):
        expected = np.asarray(initial[expected_key], dtype=np.float64)
        if not np.array_equal(actual, expected):
            raise InvariantError(
                f"{name}: live/replay {expected_key} parity failed")


def _measure_arm(item: dict, arm: str, binary: pathlib.Path,
                 reference: dict[str, np.ndarray], initial: dict) -> dict:
    frame_times = np.asarray(initial["activation_times"], dtype=np.float64)
    frame_emit = np.asarray(initial["activation_emit"], dtype=np.float64)
    beat_activation = np.asarray(initial["activation_beat"], dtype=np.float64)
    downbeat_activation = np.asarray(
        initial["activation_downbeat"], dtype=np.float64)
    all_beats = np.asarray(initial["live_bar_beats_all"], dtype=np.float64)
    all_emit = np.asarray(initial["live_bar_emit_all"], dtype=np.float64)
    visible_beats = np.asarray(initial["beats"], dtype=np.float64)
    visible = visible_indices(all_beats, visible_beats)
    payload = replay_bar(
        binary, item["audio"], beat_activation, downbeat_activation,
        frame_emit, frame_times, all_beats, all_emit, extra=_flags(arm))
    all_positions = np.asarray(
        payload["bar_replay_positions"], dtype=np.float64)
    path_positions = np.asarray(
        payload["bar_replay_path_positions"], dtype=np.float64)
    all_meters = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
    all_confident = np.asarray(
        payload["bar_replay_confident"], dtype=np.float64)
    if not (len(all_positions) == len(path_positions) == len(all_meters)
            == len(all_confident) == len(all_beats)):
        raise InvariantError(f"{item['name']} {arm}: incomplete replay")

    positions = all_positions[visible]
    meters = all_meters[visible]
    confident = all_confident[visible]
    common_start = float(item["source_m0b"]["common_start_sec"])
    score = score_dynamic(
        visible_beats, positions, meters, reference, common_start)
    mapped = _map_to_reference(
        visible_beats, positions, meters, confident, reference)
    if arm == BASELINE:
        _assert_live_baseline_parity(
            initial, all_positions, all_meters, all_confident, item["name"])
    static = static_safety(mapped, reference, common_start)
    transitions = []
    source_m0c = item.get("source_m0c")
    if source_m0c is not None:
        transitions = transition_traces(
            mapped["positions"], mapped["meters"], reference, common_start,
            name=item["name"], work_id=item["work_id"])
    supported = np.ones(len(all_beats), dtype=bool)
    return {
        "score": score,
        "transitions": transitions,
        "static": static,
        "diagnostics": {
            "resolver_path_state_changes": len(_state_change_indices(
                path_positions, all_meters, supported)),
            "held_state_changes": len(_state_change_indices(
                all_positions, all_meters, supported)),
            "path_held_disagreements": int(np.sum(
                (path_positions >= 0) & (all_positions != path_positions))),
            "internal_beats": int(len(all_beats)),
            "published_beats": int(len(visible_beats)),
            "matched_reference_events": int(np.sum(mapped["match"] >= 0)),
            "reference_match_sha256": _array_sha256(mapped["match"]),
        },
    }


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    reference = load_annotation(item["annotation"])
    initial = run(binary, item["audio"], model, extra=["--live-bars"])
    require_reference_within_audio(
        reference["times"], float(initial["duration_sec"]), item["name"])
    arms = {arm: _measure_arm(item, arm, binary, reference, initial)
            for arm in ARMS}

    baseline = arms[BASELINE]
    if (arms[BASELINE]["diagnostics"]["reference_match_sha256"]
            != arms[CANDIDATE]["diagnostics"]["reference_match_sha256"]):
        raise InvariantError(f"{item['name']}: arm reference matching changed")

    _assert_parity(
        baseline["score"], item["source_m0b"]["arms"]["A4"],
        path=f"{item['name']}.A4")
    source_m0c = item.get("source_m0c")
    if source_m0c is not None:
        source_ids = [row["transition_id"] for row in source_m0c["transitions"]]
        for arm in ARMS:
            actual = arms[arm]["transitions"]
            if [row["transition_id"] for row in actual] != source_ids:
                raise InvariantError(
                    f"{item['name']} {arm}: transition IDs changed")
            for got, fixed in zip(actual, source_m0c["transitions"], strict=True):
                for field in ("reference_index", "next_change_or_end_index",
                              "two_bar_latency_observable",
                              "previous_grouping", "new_grouping"):
                    if got[field] != fixed[field]:
                        raise InvariantError(
                            f"{item['name']} {arm}: fixed {field} changed")
    return {
        "name": item["name"], "corpus": item["corpus"],
        "work_id": item["work_id"],
        "annotation": digest(item["annotation"]),
        "common_start_sec": float(item["source_m0b"]["common_start_sec"]),
        "transition_cohort": source_m0c is not None,
        "evidence_sha256": _array_sha256(
            np.asarray(initial["activation_times"], dtype=np.float64),
            np.asarray(initial["activation_emit"], dtype=np.float64),
            np.asarray(initial["activation_beat"], dtype=np.float64),
            np.asarray(initial["activation_downbeat"], dtype=np.float64),
            np.asarray(initial["live_bar_beats_all"], dtype=np.float64),
            np.asarray(initial["live_bar_emit_all"], dtype=np.float64),
            np.asarray(initial["beats"], dtype=np.float64)),
        "arms": arms,
        "invariants": {
            "source_M0b_A4_parity": True,
            "arm_evidence_identical_by_construction": True,
            "reference_matching_identical_by_construction": True,
            "transition_population_fixed": True,
        },
    }


def measure_outcome(item: dict, binary: pathlib.Path,
                    model: pathlib.Path) -> tuple[str, dict]:
    try:
        return "record", measure_one(item, binary, model)
    except InvariantError:
        raise
    except Exception as error:
        return "exclusion", {
            "name": item["name"], "corpus": item["corpus"],
            "error_type": type(error).__name__,
            "reason": _without_local_paths(str(error)),
            "annotation": digest(item.get("annotation")),
        }


def _metric(values: list[float]) -> dict:
    array = np.asarray([value for value in values if math.isfinite(value)],
                       dtype=np.float64)
    if not len(array):
        return {"mean": None, "ci": [None, None], "n": 0}
    return {"mean": float(np.mean(array)),
            "ci": paired_bootstrap(array, draws=BOOTSTRAP_DRAWS),
            "n": int(len(array))}


def _paired(candidate: dict[str, float], baseline: dict[str, float]) -> dict:
    works = sorted(set(candidate) & set(baseline))
    return _metric([candidate[work] - baseline[work] for work in works])


def _transition_values(records: list[dict], arm: str, predicate,
                       eligible=lambda _row: True) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        for row in record["arms"][arm]["transitions"]:
            if eligible(row):
                grouped[record["work_id"]].append(float(predicate(row)))
    return {work: float(np.mean(values)) for work, values in grouped.items()}


def _work_static(records: list[dict], arm: str, numerator: str,
                 denominator: str) -> dict[str, float]:
    grouped = defaultdict(lambda: [0.0, 0.0])
    for record in records:
        block = record["arms"][arm]["static"]
        grouped[record["work_id"]][0] += float(block[numerator])
        grouped[record["work_id"]][1] += float(block[denominator])
    return {work: top / bottom for work, (top, bottom) in grouped.items()
            if bottom > 0.0}


def _work_score(records: list[dict], arm: str, metric: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["work_id"]].append(
            float(record["arms"][arm]["score"][metric]))
    return {work: float(np.mean(values)) for work, values in grouped.items()}


def summarise(records: list[dict], exclusions: list[dict]) -> dict:
    arm_summaries = {}
    efficacy_values = {}
    stable_values = {}
    switch_values = {}
    episode_values = {}
    for arm in ARMS:
        observable = _transition_values(
            records, arm, lambda row: row["acquired_within_two_bars"],
            lambda row: row["two_bar_latency_observable"])
        intention = _transition_values(
            records, arm, lambda row: row["acquired_within_two_bars"])
        first = _transition_values(
            records, arm, lambda row: row["acquired_first_bar"])
        stable = _work_static(records, arm, "correct", "events")
        switches = _work_static(
            records, arm, "false_switches", "eligible_duration_sec")
        long_episodes = _work_static(
            records, arm, "long_wrong_episodes_1bar", "eligible_duration_sec")
        switches = {work: value * 300.0 for work, value in switches.items()}
        long_episodes = {
            work: value * 300.0 for work, value in long_episodes.items()}
        efficacy_values[arm] = observable
        stable_values[arm] = stable
        switch_values[arm] = switches
        episode_values[arm] = long_episodes
        transitions = [row for record in records
                       for row in record["arms"][arm]["transitions"]]
        latency = [float(row["acquisition_latency_bars"])
                   for row in transitions
                   if row["acquisition_latency_bars"] is not None]
        diagnostics = {}
        for metric in ("phase_f1", "grouping_balanced_accuracy",
                       "position_accuracy", "coverage",
                       "false_confident_share", "unnecessary_unknown_share"):
            diagnostics[metric] = _metric(list(
                _work_score(records, arm, metric).values()))
        arm_summaries[arm] = {
            "fully_observable_within_two_bars": _metric(
                list(observable.values())),
            "intention_to_treat_within_two_bars": _metric(
                list(intention.values())),
            "first_bar_acquisition": _metric(list(first.values())),
            "stable_exact_position": _metric(list(stable.values())),
            "false_switches_per_5min": _metric(list(switches.values())),
            "long_wrong_episodes_per_5min": _metric(
                list(long_episodes.values())),
            "diagnostic_scores": diagnostics,
            "raw_outcome_counts": dict(sorted(Counter(
                row["outcome_class"] for row in transitions).items())),
            "acquired_latency_bars": {
                "n": len(latency),
                "median": float(np.median(latency)) if latency else None,
                "iqr": ([float(value) for value in np.percentile(
                    latency, [25, 75])] if latency else [None, None]),
            },
            "resolver_path_state_changes": sum(
                row["arms"][arm]["diagnostics"][
                    "resolver_path_state_changes"] for row in records),
            "held_state_changes": sum(
                row["arms"][arm]["diagnostics"]["held_state_changes"]
                for row in records),
            "wrong_episodes": sum(
                row["arms"][arm]["static"]["wrong_episodes"]
                for row in records),
            "long_wrong_episodes_1bar": sum(
                row["arms"][arm]["static"]["long_wrong_episodes_1bar"]
                for row in records),
            "long_wrong_episodes_2bar": sum(
                row["arms"][arm]["static"]["long_wrong_episodes_2bar"]
                for row in records),
        }

    efficacy = _paired(efficacy_values[CANDIDATE], efficacy_values[BASELINE])
    stable = _paired(stable_values[CANDIDATE], stable_values[BASELINE])
    switches = _paired(switch_values[CANDIDATE], switch_values[BASELINE])
    episodes = _paired(episode_values[CANDIDATE], episode_values[BASELINE])
    efficacy_passed = bool(
        efficacy["mean"] is not None
        and efficacy["mean"] >= EFFICACY_MIN_GAIN
        and efficacy["ci"][0] > 0.0)
    stable_passed = bool(
        stable["mean"] is not None and stable["ci"][0] >= -STABLE_MAX_LOSS)
    switch_passed = bool(
        switches["mean"] is not None
        and switches["ci"][1] <= FALSE_SWITCH_MAX_INCREASE_5MIN)
    episode_passed = bool(
        episodes["mean"] is not None
        and episodes["ci"][1] <= LONG_EPISODE_MAX_INCREASE_5MIN)

    transition_records = sum(row.get("transition_cohort") is True
                             for row in records)
    transitions = sum(len(row["arms"][BASELINE]["transitions"])
                      for row in records)
    fully_observable = sum(
        item["two_bar_latency_observable"]
        for row in records for item in row["arms"][BASELINE]["transitions"])
    complete = (
        not exclusions and len(records) == FIXED_RECORDS
        and len({row["work_id"] for row in records}) == FIXED_WORKS
        and transition_records == FIXED_TRANSITION_RECORDS
        and transitions == FIXED_TRANSITIONS
        and fully_observable == FIXED_FULLY_OBSERVABLE
        and len(efficacy_values[BASELINE]) == FIXED_EFFICACY_WORKS
        and len(efficacy_values[BASELINE]) >= MIN_EFFICACY_WORKS
        and len(stable_values[BASELINE]) == FIXED_WORKS
        and len(switch_values[BASELINE]) == FIXED_WORKS
        and len(episode_values[BASELINE]) == FIXED_WORKS)
    safety_passed = stable_passed and switch_passed and episode_passed
    if not complete:
        interpretation = "inconclusive"
    elif efficacy_passed and safety_passed:
        interpretation = "non_oracle_decoder_candidate_pass"
    elif efficacy_passed:
        interpretation = "non_oracle_gain_static_cost"
    elif safety_passed:
        interpretation = "oracle_gain_does_not_transfer"
    else:
        interpretation = "non_oracle_candidate_regression"
    return {
        "records": len(records),
        "independent_works": len({row["work_id"] for row in records}),
        "transition_records": transition_records,
        "transitions": transitions,
        "fully_observable_transitions": fully_observable,
        "efficacy_works": len(efficacy_values[BASELINE]),
        "arms": arm_summaries,
        "paired_effects": {
            "within_two_bars_gain": efficacy,
            "stable_exact_difference": stable,
            "false_switch_rate_difference_per_5min": switches,
            "long_wrong_episode_rate_difference_per_5min": episodes,
        },
        "gates": {
            "complete": complete,
            "efficacy": efficacy_passed,
            "stable_exact": stable_passed,
            "false_switch_rate": switch_passed,
            "long_wrong_episode_rate": episode_passed,
        },
        "interpretation": interpretation,
        "thresholds": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "parity_tolerance": PARITY_TOLERANCE,
            "match_tolerance_sec": MATCH_TOLERANCE_SEC,
            "efficacy_min_gain": EFFICACY_MIN_GAIN,
            "stable_max_loss": STABLE_MAX_LOSS,
            "false_switch_max_increase_per_5min": (
                FALSE_SWITCH_MAX_INCREASE_5MIN),
            "long_episode_max_increase_per_5min": (
                LONG_EPISODE_MAX_INCREASE_5MIN),
        },
    }


def synthetic_preflight(binary: pathlib.Path) -> dict:
    sample_rate = 48000.0
    duration_sec = 48
    frame_times = np.arange(0.0, duration_sec, 0.02, dtype=np.float64)
    frame_emit = np.maximum(1.0, np.ceil(
        (frame_times + 0.064) * sample_rate / BLOCK_SAMPLES))
    beat_activation = np.zeros(len(frame_times), dtype=np.float64)
    beats = np.arange(1.0, duration_sec - 1.0, 0.5, dtype=np.float64)
    beat_emit = reference_emit(beats, sample_rate)
    change = 32
    downbeats = beats[np.concatenate([
        np.arange(0, change, 4),
        np.arange(change + 2, len(beats), 4),
    ])]
    with tempfile.TemporaryDirectory() as directory:
        audio = pathlib.Path(directory) / "m0e_phase_shift.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(b"\0\0" * int(sample_rate * duration_sec))
        outputs = {}
        for arm in ARMS:
            payload = replay_bar(
                binary, audio, beat_activation,
                oracle_channel(frame_times, downbeats), frame_emit,
                frame_times, beats, beat_emit, extra=_flags(arm))
            outputs[arm] = np.asarray(
                payload["bar_replay_positions"], dtype=np.int64)
    expected = (np.arange(len(beats)) - (change + 2)) % 4
    window = np.arange(change + 2, min(change + 10, len(beats)))
    candidate_exact = outputs[CANDIDATE][window] == expected[window]
    acquired = next((int(index) for index in range(len(window))
                     if np.all(candidate_exact[index:index + 4])), None)
    differences = int(np.sum(outputs[BASELINE][window]
                             != outputs[CANDIDATE][window]))
    if acquired is None or acquired > 8 or differences == 0:
        raise InvariantError(
            "M0e synthetic L2 preflight failed: "
            f"acquired={acquired}, differences={differences}")
    return {"passed": True, "meter": 4, "phase_shift_tactus": 2,
            "candidate_acquisition_events": acquired,
            "baseline_candidate_differences": differences,
            "neural_state_used": False}


def checkpoint_identity(provenance: dict, items: list[dict], *, limit: int,
                        skip_audio_verification: bool) -> dict:
    selection = [{
        "corpus": item["corpus"], "name": item["name"],
        "work_id": item["work_id"],
        "audio_sha256": item.get("audio_sha256"),
        "annotation_sha256": item.get("annotation_sha256"),
        "transition_ids": ([row["transition_id"]
                            for row in item["source_m0c"]["transitions"]]
                           if item.get("source_m0c") is not None else []),
    } for item in items]
    selection_sha = hashlib.sha256(json.dumps(
        selection, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "commit": provenance["commit"],
        "binary": provenance["binary"], "model": provenance["model"],
        "manifest": provenance["manifest"],
        "source_m0b": provenance["source_m0b"],
        "source_m0c": provenance["source_m0c"],
        "source_m0d": provenance["source_m0d"],
        "artifact_schema": ARTIFACT_SCHEMA,
        "arms": ARMS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "parity_tolerance": PARITY_TOLERANCE,
        "match_tolerance_sec": MATCH_TOLERANCE_SEC,
        "fixed_records": FIXED_RECORDS, "fixed_works": FIXED_WORKS,
        "fixed_transition_records": FIXED_TRANSITION_RECORDS,
        "fixed_transitions": FIXED_TRANSITIONS,
        "fixed_fully_observable": FIXED_FULLY_OBSERVABLE,
        "fixed_efficacy_works": FIXED_EFFICACY_WORKS,
        "efficacy_min_gain": EFFICACY_MIN_GAIN,
        "stable_max_loss": STABLE_MAX_LOSS,
        "false_switch_max_increase_per_5min": (
            FALSE_SWITCH_MAX_INCREASE_5MIN),
        "long_episode_max_increase_per_5min": (
            LONG_EPISODE_MAX_INCREASE_5MIN),
        "limit": limit,
        "skip_audio_verification": skip_audio_verification,
        "selected": len(items), "selection_sha256": selection_sha,
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-m0b", type=pathlib.Path, required=True)
    parser.add_argument("--source-m0c", type=pathlib.Path, required=True)
    parser.add_argument("--source-m0d", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-file", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-audio-verification", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    checkpoint = (args.checkpoint if args.checkpoint is not None else
                  pathlib.Path(f"{args.output}.checkpoint"))
    pause_file = (args.pause_file if args.pause_file is not None else
                  pathlib.Path(f"{checkpoint}.pause"))
    for path, label in ((args.output, "output"),
                        (checkpoint, "checkpoint"),
                        (pause_file, "pause file")):
        try:
            _require_outside_repository(path, repository, label)
        except ValueError as error:
            parser.error(str(error))
    if pause_file.exists():
        parser.error(f"remove the pause file before starting: {pause_file}")

    m0b, m0c, m0d = validate_sources(
        args.source_m0b, args.source_m0c, args.source_m0d)
    items, manifest = load_manifest(
        args.manifest, args.music_root,
        verify_audio=not args.skip_audio_verification)
    items = select_population(items, m0b, m0c, m0d)
    if args.limit:
        items = items[:args.limit]
    provenance = experiment_provenance(
        repository,
        files={"binary": args.binary, "model": args.model,
               "manifest": args.manifest, "source_m0b": args.source_m0b,
               "source_m0c": args.source_m0c, "source_m0d": args.source_m0d},
        experiment="M0e", arms=list(ARMS),
        bootstrap_draws=BOOTSTRAP_DRAWS, workers=args.workers)
    preflight = synthetic_preflight(args.binary)
    identity = checkpoint_identity(
        provenance, items, limit=args.limit,
        skip_audio_verification=args.skip_audio_verification)
    ordered, state, resumed = prepare_checkpoint(
        checkpoint, identity, provenance, items, resume=args.resume,
        workers=args.workers, schema=CHECKPOINT_SCHEMA)
    print(json.dumps({"event": "start", "recordings": len(items),
                      "workers": args.workers, "resumed": resumed}), flush=True)
    ordered, paused = run_checkpointed(
        items, args.binary, args.model, workers=args.workers,
        checkpoint=checkpoint, state=state, ordered=ordered,
        pause_file=pause_file, measure=measure_outcome,
        checkpoint_schema=CHECKPOINT_SCHEMA)
    if paused:
        return PAUSED_EXIT_CODE
    records = [payload for kind, payload in ordered if kind == "record"]
    exclusions = [payload for kind, payload in ordered if kind == "exclusion"]
    summary = summarise(records, exclusions)
    if args.limit or args.skip_audio_verification:
        summary["interpretation"] = "inconclusive"
        summary["diagnostic_only"] = True
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "provenance": provenance,
        "sources": {
            "m0b": {"sha256": SOURCE_M0B_SHA256},
            "m0c": {"sha256": SOURCE_M0C_SHA256},
            "m0d": {"sha256": SOURCE_M0D_SHA256,
                     "commit": SOURCE_M0D_COMMIT},
        },
        "source_profile": manifest.get("profile", {}),
        "synthetic_preflight": preflight,
        "selected": len(items), "scored": len(records),
        "technical_exclusions": exclusions,
        "records": records,
        "checkpoint": {"schema": CHECKPOINT_SCHEMA,
                       "resumed_outcomes": resumed,
                       "sessions": state.get("sessions", [])},
        "summary": summary,
    }
    _atomic_write_json(args.output, artifact)
    _checkpoint_state(checkpoint, state, status="artifact_written",
                      completed=len(ordered), total=len(items))
    print(json.dumps({"event": "complete", "records": len(records),
                      "exclusions": len(exclusions),
                      "interpretation": summary["interpretation"],
                      "output": args.output.name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
