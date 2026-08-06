"""Stateful replay and event metrics for the pre-registered octave veto.

The decoder stays in :mod:`eval.octave_veto`; this module gives its decisions
the consequences they would have in the shipping live tracker.  A replay emits
a small schedule of ``(onset, close, committed BPM)`` veto intervals, the C++
core applies those intervals at its existing anchor seam, and the schedule is
recomputed from the resulting live state.  Convergence means the schedule and
the state it produces agree -- a fixed point of the actual core, not a Python
copy of the particle filter.

No corpus constants are tuned here.  Threshold and comparison-policy grids are
the values fixed in ``PREREGISTERED_octave_veto.md``.
"""

from __future__ import annotations

import dataclasses
import concurrent.futures
import argparse
import json
import math
import pathlib
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from eval.live_level import local_reference_bpm, tempo_state
from eval.octave_veto import (Decision, Proposal, WindowTrack, committed_grid,
                              extract_proposals, judge)
from eval.octave_veto_replay import (Replay, from_payload, run, run_activation,
                                     same_live_series)

TAU_CANDIDATES = tuple(float(x) for x in np.arange(0.0, 5.0 + 0.25, 0.25))
DEBOUNCE_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MARGIN_CANDIDATES = tuple(float(x) for x in np.arange(0.0, 1.0, 0.1))
RATE_LIMIT_CANDIDATES = (5.0, 10.0, 20.0, 30.0, 60.0, 120.0)

MAX_FIXED_POINT_PASSES = 8
SCHEDULE_TIME_DIGITS = 6
SCHEDULE_BPM_DIGITS = 9
BOOTSTRAP_SEED = 20260806
BOOTSTRAP_REPLICATES = 10_000

HARMONIX_EPISODE_GATE = 0.479
HARMONIX_COST_GATES = {
    "usable_rate_strict": (">=", 0.30),
    "mean_correct_share_of_eligible": (">=", 0.75),
    "switches_per_five_minutes": ("<=", 4.21),
    "p90_settle_sec": ("<=", 36.61),
    "f_measure": (">=", 0.785),
}

CORRECT_TO_WRONG = "correct_to_wrong"
WRONG_TO_CORRECT = "wrong_to_correct"
AMBIGUOUS = "ambiguous"


@dataclasses.dataclass(frozen=True)
class VetoInterval:
    onset_sec: float
    close_sec: float
    committed_bpm: float


@dataclasses.dataclass(frozen=True)
class EventResult:
    proposal: Proposal
    decision: Decision
    label: str

    @property
    def null_delta(self) -> float:
        """The shifted-track control statistic, on the decoder's own scale."""
        return self.decision.null_committed - self.decision.null_proposed

    def veto(self, tau: float, *, control: bool = False) -> bool:
        if not self.decision.answered:
            return False
        value = self.null_delta if control else self.decision.delta
        return bool(value > tau)


@dataclasses.dataclass(frozen=True)
class TrackEvents:
    name: str
    events: tuple[EventResult, ...]
    meter: str | None = None


@dataclasses.dataclass(frozen=True)
class ConvergedReplay:
    payload: dict
    replay: Replay
    events: TrackEvents
    schedule: tuple[VetoInterval, ...]
    passes: int


@dataclasses.dataclass
class CorpusArm:
    name: str
    parameter: float | None
    corpus: str
    scores: list[dict[str, Any]]
    tracks: dict[str, TrackEvents]
    summary: dict[str, Any]
    max_passes: int

    @property
    def metrics(self) -> dict[str, Any]:
        return self.summary["by_corpus"][self.corpus]


def event_label(proposal: Proposal, reference_beats: np.ndarray) -> str:
    """Label one proposal from the annotation, at the existing 8% tolerance."""
    reference = local_reference_bpm(reference_beats, proposal.onset_sec)
    if not (reference > 0.0):
        return AMBIGUOUS
    committed_correct = tempo_state(proposal.committed_bpm, reference) == "same"
    proposed_correct = tempo_state(proposal.measured_bpm, reference) == "same"
    if committed_correct and not proposed_correct:
        return CORRECT_TO_WRONG
    if not committed_correct and proposed_correct:
        return WRONG_TO_CORRECT
    return AMBIGUOUS


def evaluate_events(name: str, replay: Replay,
                    reference_beats: np.ndarray,
                    meter: str | None = None) -> TrackEvents:
    """Extract, label and judge every proposal in one replay.

    The raw activation track is clipped to the fixed decision window before it
    is handed to ``judge``.  That makes the circular null a shift *inside that
    window* and prevents frames after the proposal from entering through the
    back of the circle.
    """
    proposals = extract_proposals(
        replay.times, replay.bpm, replay.measured_bpm,
        replay.answered, replay.locked,
    )
    out: list[EventResult] = []
    for proposal in proposals:
        grid = committed_grid(replay.beats, replay.times, replay.bpm,
                              proposal.onset_sec)
        if len(grid):
            available = ((replay.activation_times >= grid[0])
                         & (replay.activation_times <= proposal.onset_sec))
            track = WindowTrack(replay.activation_times[available],
                                replay.downbeat[available])
        else:
            track = WindowTrack(np.zeros(0, dtype=np.float64),
                                np.zeros(0, dtype=np.float64))
        out.append(EventResult(
            proposal=proposal,
            decision=judge(track, grid, proposal.k),
            label=event_label(proposal, reference_beats),
        ))
    return TrackEvents(name=name, events=tuple(out), meter=meter)


def event_signature(events: TrackEvents) -> tuple[tuple[float, int], ...]:
    """The fifth registered parity sequence: proposal onset and sign."""
    return tuple((event.proposal.onset_sec, int(math.copysign(1, event.proposal.k)))
                 for event in events.events)


def verify_cached_parity(name: str, initial: dict, cached: dict,
                         reference_beats: np.ndarray) -> None:
    if not same_live_series(initial, cached):
        raise RuntimeError(f"cached activation replay diverged for {name}")
    initial_events = evaluate_events(name, from_payload(initial), reference_beats)
    cached_events = evaluate_events(name, from_payload(cached), reference_beats)
    if event_signature(initial_events) != event_signature(cached_events):
        raise RuntimeError(f"cached activation event list diverged for {name}")


def decoder_schedule(events: TrackEvents, tau: float, *,
                     control: bool = False) -> tuple[VetoInterval, ...]:
    return tuple(
        VetoInterval(event.proposal.onset_sec, event.proposal.close_sec,
                     event.proposal.committed_bpm)
        for event in events.events if event.veto(tau, control=control)
    )


def debounce_schedule(events: TrackEvents, seconds: float) -> tuple[VetoInterval, ...]:
    """Delay each proposed octave change by the registered debounce duration."""
    return tuple(
        VetoInterval(event.proposal.onset_sec,
                     min(event.proposal.close_sec,
                         event.proposal.onset_sec + seconds),
                     event.proposal.committed_bpm)
        for event in events.events
        if event.proposal.close_sec > event.proposal.onset_sec
    )


def rate_limit_schedule(events: TrackEvents, seconds: float) -> tuple[VetoInterval, ...]:
    """Allow the first proposal, then at most one further change per interval."""
    last_allowed = -math.inf
    vetoed: list[VetoInterval] = []
    for event in events.events:
        proposal = event.proposal
        if proposal.onset_sec - last_allowed >= seconds:
            last_allowed = proposal.onset_sec
        else:
            vetoed.append(VetoInterval(proposal.onset_sec, proposal.close_sec,
                                       proposal.committed_bpm))
    return tuple(vetoed)


def total_ban_schedule(events: TrackEvents) -> tuple[VetoInterval, ...]:
    return tuple(
        VetoInterval(event.proposal.onset_sec, event.proposal.close_sec,
                     event.proposal.committed_bpm)
        for event in events.events
    )


def schedule_signature(schedule: Sequence[VetoInterval]) -> tuple[tuple[float, ...], ...]:
    """Stable identity across JSON/text round trips, not a policy tolerance."""
    return tuple((round(row.onset_sec, SCHEDULE_TIME_DIGITS),
                  round(row.close_sec, SCHEDULE_TIME_DIGITS),
                  round(row.committed_bpm, SCHEDULE_BPM_DIGITS))
                 for row in schedule)


def write_schedule(path: pathlib.Path, schedule: Sequence[VetoInterval]) -> None:
    lines = ["# onset_sec close_sec committed_bpm"]
    lines.extend(f"{row.onset_sec:.9f} {row.close_sec:.9f} "
                 f"{row.committed_bpm:.12g}" for row in schedule)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ScheduleBuilder = Callable[[TrackEvents], tuple[VetoInterval, ...]]
ReplayRunner = Callable[[tuple[VetoInterval, ...]], dict]


def converge_replay(initial_payload: dict, name: str,
                    reference_beats: np.ndarray, rerun: ReplayRunner,
                    build_schedule: ScheduleBuilder,
                    max_passes: int = MAX_FIXED_POINT_PASSES,
                    meter: str | None = None) -> ConvergedReplay:
    """Recompute and apply schedules until policy and live state agree."""
    payload = initial_payload
    applied: tuple[VetoInterval, ...] | None = None
    for pass_index in range(max_passes + 1):
        replay = from_payload(payload)
        events = evaluate_events(name, replay, reference_beats, meter)
        wanted = build_schedule(events)
        if applied is None and not wanted:
            return ConvergedReplay(payload, replay, events, wanted, pass_index)
        if applied is not None and schedule_signature(wanted) == schedule_signature(applied):
            return ConvergedReplay(payload, replay, events, wanted, pass_index)
        if pass_index == max_passes:
            break
        payload = rerun(wanted)
        applied = wanted
    raise RuntimeError(f"anchor policy did not converge for {name} in "
                       f"{max_passes} passes")


def run_track(binary: pathlib.Path, audio: pathlib.Path, weights: pathlib.Path,
              reference_beats: np.ndarray, build_schedule: ScheduleBuilder,
              *, name: str | None = None,
              meter: str | None = None,
              extra: Sequence[str] = ()) -> ConvergedReplay:
    """Run one audio file to a fixed point through the C++ schedule seam."""
    return next(result for _, result in iter_track_policies(
        binary, audio, weights, reference_beats, (("policy", build_schedule),),
        name=name, meter=meter, extra=extra))


def iter_track_policies(
    binary: pathlib.Path,
    audio: pathlib.Path,
    weights: pathlib.Path,
    reference_beats: np.ndarray,
    policies: Sequence[tuple[str, ScheduleBuilder]],
    *,
    name: str | None = None,
    meter: str | None = None,
    extra: Sequence[str] = (),
):
    """Yield many policy replays after one model pass and one parity pass."""
    track_name = name or audio.stem
    initial = run(binary, audio, weights, extra=list(extra))
    beat_activation = np.asarray(initial.get("activation_beat", []),
                                 dtype=np.float64)
    if not len(beat_activation):
        raise RuntimeError(f"model run produced no beat activation for {track_name}")
    with tempfile.TemporaryDirectory(prefix="tiktak-octave-veto-") as directory:
        activation_path = pathlib.Path(directory) / "beat_activation.txt"
        schedule_path = pathlib.Path(directory) / "schedule.txt"
        np.savetxt(activation_path, beat_activation, fmt="%.9g")

        activation_columns = {
            key: initial[key] for key in
            ("activation_times", "activation_beat", "activation_downbeat")
        }
        cached_baseline = run_activation(binary, audio, activation_path,
                                         extra=list(extra))
        cached_baseline.update(activation_columns)
        verify_cached_parity(track_name, initial, cached_baseline,
                             reference_beats)

        def rerun(schedule: tuple[VetoInterval, ...]) -> dict:
            write_schedule(schedule_path, schedule)
            flags = [*extra, "--live-anchor-veto", str(schedule_path)]
            payload = run_activation(binary, audio, activation_path, extra=flags)
            payload.update(activation_columns)
            return payload

        for policy_name, build_schedule in policies:
            yield policy_name, converge_replay(
                cached_baseline, track_name, reference_beats, rerun,
                build_schedule, meter=meter)


def _class_rates(tracks: Iterable[TrackEvents], *, control: bool,
                 tau: float) -> tuple[list[float], list[float]]:
    veto_rates: list[float] = []
    allow_rates: list[float] = []
    for track in tracks:
        answered = [event for event in track.events if event.decision.answered]
        positives = [event for event in answered if event.label == CORRECT_TO_WRONG]
        negatives = [event for event in answered if event.label == WRONG_TO_CORRECT]
        if positives:
            veto_rates.append(float(np.mean([
                event.veto(tau, control=control) for event in positives
            ])))
        if negatives:
            allow_rates.append(float(np.mean([
                not event.veto(tau, control=control) for event in negatives
            ])))
    return veto_rates, allow_rates


def balanced_accuracy(tracks: Iterable[TrackEvents], tau: float, *,
                      control: bool = False) -> float:
    """Recording-balanced mean of veto sensitivity and escape specificity."""
    veto_rates, allow_rates = _class_rates(tracks, control=control, tau=tau)
    if not veto_rates or not allow_rates:
        return float("nan")
    return 0.5 * (float(np.mean(veto_rates)) + float(np.mean(allow_rates)))


def false_veto_rate(tracks: Iterable[TrackEvents], tau: float, *,
                    control: bool = False) -> float:
    """A3: recording-balanced veto rate on wrong-to-correct escapes."""
    _, allow_rates = _class_rates(tracks, control=control, tau=tau)
    if not allow_rates:
        return float("nan")
    return 1.0 - float(np.mean(allow_rates))


def veto_rate(tracks: Iterable[TrackEvents], tau: float, *,
              control: bool = False) -> float:
    rates = []
    for track in tracks:
        answered = [event for event in track.events if event.decision.answered]
        if answered:
            rates.append(float(np.mean([
                event.veto(tau, control=control) for event in answered
            ])))
    return float(np.mean(rates)) if rates else float("nan")


def event_counts(tracks: Iterable[TrackEvents]) -> dict[str, int]:
    rows = [event for track in tracks for event in track.events]
    return {
        "events": len(rows),
        "answered": sum(event.decision.answered for event in rows),
        CORRECT_TO_WRONG: sum(event.label == CORRECT_TO_WRONG for event in rows),
        WRONG_TO_CORRECT: sum(event.label == WRONG_TO_CORRECT for event in rows),
        AMBIGUOUS: sum(event.label == AMBIGUOUS for event in rows),
    }


def raw_sign_agreement(tracks: Iterable[TrackEvents]) -> dict[str, Any]:
    """P2: how often the cheap raw contrast and registered delta agree."""
    rows = [event for track in tracks for event in track.events
            if event.decision.answered
            and math.isfinite(event.decision.delta)
            and math.isfinite(event.decision.delta_raw)]
    agreements = sum(np.sign(event.decision.delta)
                     == np.sign(event.decision.delta_raw) for event in rows)
    return {
        "events": len(rows),
        "agreements": int(agreements),
        "fraction": agreements / len(rows) if rows else None,
    }


def ambiguity_diagnostic(tracks: Iterable[TrackEvents]) -> dict[str, Any]:
    counts = event_counts(tracks)
    labelled = counts[CORRECT_TO_WRONG] + counts[WRONG_TO_CORRECT]
    return {
        "ambiguous": counts[AMBIGUOUS],
        "labelled": labelled,
        "dominates": counts[AMBIGUOUS] > labelled,
    }


def d1_zero_committed_score(tracks: Iterable[TrackEvents]) -> dict[str, Any]:
    """D1: correct-to-wrong doubling events with no committed contrast."""
    rows = [event for track in tracks for event in track.events
            if event.decision.answered
            and event.label == CORRECT_TO_WRONG
            and event.proposal.k > 0]
    zeros = sum(event.decision.score_committed == 0.0 for event in rows)
    return {"events": len(rows), "zero_score": zeros,
            "fraction": zeros / len(rows) if rows else None}


def event_coverage_by_meter(tracks: Iterable[TrackEvents]) -> dict[str, Any]:
    """Registered answered-event coverage, grouped by annotated metre."""
    buckets: dict[str, list[EventResult]] = {}
    for track in tracks:
        key = str(track.meter).strip() if track.meter is not None else "unknown"
        if not key:
            key = "unknown"
        buckets.setdefault(key, []).extend(track.events)
    return {
        meter: {
            "events": len(rows),
            "answered": sum(event.decision.answered for event in rows),
            "coverage": (sum(event.decision.answered for event in rows) / len(rows)
                         if rows else None),
        }
        for meter, rows in sorted(buckets.items())
    }


def cluster_bootstrap_difference(decoder: dict[str, TrackEvents],
                                 control: dict[str, TrackEvents], tau: float,
                                 *, replicates: int = BOOTSTRAP_REPLICATES,
                                 seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """Paired recording-cluster bootstrap for A2's balanced-accuracy gain."""
    names = sorted(set(decoder) & set(control))
    if not names:
        raise ValueError("decoder and control have no recordings in common")
    point = (balanced_accuracy((decoder[name] for name in names), tau)
             - balanced_accuracy((control[name] for name in names), tau,
                                 control=True))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(replicates):
        selected = rng.integers(0, len(names), size=len(names))
        left = [decoder[names[index]] for index in selected]
        right = [control[names[index]] for index in selected]
        difference = (balanced_accuracy(left, tau)
                      - balanced_accuracy(right, tau, control=True))
        if math.isfinite(difference):
            draws.append(difference)
    if not draws:
        raise ValueError("bootstrap has no resample containing both event classes")
    nonpositive = sum(value <= 0.0 for value in draws)
    nonnegative = sum(value >= 0.0 for value in draws)
    p = min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1.0)
            / (len(draws) + 1.0))
    return {
        "difference": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p": p,
        "replicates": len(draws),
    }


def score_converged(item: dict[str, Any], replayed: ConvergedReplay) -> dict[str, Any]:
    """Use the existing live benchmark, without another run of the audio."""
    # Imported only for a corpus run. Pure decoder/replay tests intentionally do
    # not need mir_eval installed.
    from eval.analysis import Estimate
    from eval.live_corpus_benchmark import score_estimate

    return score_estimate(item, Estimate.from_json(replayed.payload), mode="model")


def _bounded_results(pool: concurrent.futures.Executor, function: Callable,
                     items: Iterable[Any], limit: int):
    """Yield results while retaining at most ``limit`` submitted futures.

    ``as_completed`` snapshots its complete input collection internally, so
    removing futures from the caller's set does not release their results.  A
    bounded submission window is required when each result briefly owns a
    full-resolution live replay.
    """
    source = iter(items)
    pending: set[concurrent.futures.Future] = set()
    for _ in range(limit):
        try:
            item = next(source)
        except StopIteration:
            break
        pending.add(pool.submit(function, item))

    while pending:
        done, pending = concurrent.futures.wait(
            pending, return_when=concurrent.futures.FIRST_COMPLETED)
        while done:
            future = done.pop()
            result = future.result()
            try:
                item = next(source)
            except StopIteration:
                pass
            else:
                pending.add(pool.submit(function, item))
            yield result


def run_policy_grid(
    items: Sequence[dict[str, Any]],
    binary: pathlib.Path,
    weights: pathlib.Path,
    policies: Sequence[tuple[str, float | None, ScheduleBuilder]],
    *,
    corpus: str,
    workers: int = 8,
    extra: Sequence[str] = (),
) -> dict[str, CorpusArm]:
    """Run a whole policy grid with one model pass per recording.

    ``corpus`` is the pooled label used for the registered decision (``rwc`` or
    ``harmonix``). Original sub-corpus names remain in per-track rows, while a
    copy is relabelled for the canonical summary so RWC's five parts are pooled
    rather than macro-averaged.
    """
    from eval.live_corpus_benchmark import load_reference_beats, summarize

    if workers < 1:
        raise ValueError("workers must be positive")
    names = [name for name, _, _ in policies]
    if len(set(names)) != len(names):
        raise ValueError("policy names must be unique")
    scores = {name: [] for name in names}
    tracks = {name: {} for name in names}
    passes = {name: [] for name in names}

    def one(item: dict[str, Any]):
        reference = load_reference_beats(item["annotation"])
        builders = tuple((name, builder) for name, _, builder in policies)
        rows = []
        for name, replayed in iter_track_policies(
                binary, item["audio"], weights, reference, builders,
                name=item["name"], meter=item.get("meter"), extra=extra):
            # Score while this is the sole full-resolution policy payload held
            # by the worker. Returning a list of ConvergedReplay objects kept
            # all ~36 50-fps payloads for a long recording alive together.
            rows.append((name, score_converged(item, replayed),
                         replayed.events, replayed.passes))
        return item, rows

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for item, rows in _bounded_results(pool, one, items, workers):
            for name, score, events, pass_count in rows:
                scores[name].append(score)
                tracks[name][item["name"]] = events
                passes[name].append(pass_count)

    wall = time.perf_counter() - started
    out: dict[str, CorpusArm] = {}
    parameters = {name: parameter for name, parameter, _ in policies}
    for name in names:
        failed = [row.get("name", "<unknown>") for row in scores[name]
                  if not row.get("ok")]
        if failed:
            raise RuntimeError(f"{name} failed to score {len(failed)} tracks: "
                               + ", ".join(failed[:10]))
        pooled = [dict(row, corpus=corpus) for row in scores[name]]
        out[name] = CorpusArm(
            name=name,
            parameter=parameters[name],
            corpus=corpus,
            scores=scores[name],
            tracks=tracks[name],
            summary=summarize("model", pooled, wall),
            max_passes=max(passes[name], default=0),
        )
    return out


def run_direct_flag_grid(
    items: Sequence[dict[str, Any]],
    binary: pathlib.Path,
    weights: pathlib.Path,
    policies: Sequence[tuple[str, float | None, Sequence[str]]],
    *,
    corpus: str,
    workers: int = 8,
) -> dict[str, CorpusArm]:
    """Replay policies expressible by existing live-core flags (margin sweep)."""
    from eval.live_corpus_benchmark import load_reference_beats, summarize

    names = [name for name, _, _ in policies]
    scores = {name: [] for name in names}
    tracks = {name: {} for name in names}

    def one(item: dict[str, Any]):
        reference = load_reference_beats(item["annotation"])
        initial = run(binary, item["audio"], weights)
        beat_activation = np.asarray(initial.get("activation_beat", []),
                                     dtype=np.float64)
        if not len(beat_activation):
            raise RuntimeError(f"model run produced no activation for {item['name']}")
        with tempfile.TemporaryDirectory(
                prefix="tiktak-octave-veto-direct-") as directory:
            activation_path = pathlib.Path(directory) / "beat_activation.txt"
            np.savetxt(activation_path, beat_activation, fmt="%.9g")
            columns = {key: initial[key] for key in
                       ("activation_times", "activation_beat", "activation_downbeat")}
            baseline = run_activation(binary, item["audio"], activation_path)
            baseline.update(columns)
            verify_cached_parity(item["name"], initial, baseline, reference)
            rows = []
            for name, _, flags in policies:
                payload = run_activation(binary, item["audio"], activation_path,
                                         extra=list(flags))
                payload.update(columns)
                replay = from_payload(payload)
                events = evaluate_events(item["name"], replay, reference,
                                         item.get("meter"))
                converged = ConvergedReplay(payload, replay, events, (), 1)
                rows.append((name, score_converged(item, converged), events))
            return item, rows

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for item, rows in _bounded_results(pool, one, items, workers):
            for name, score, events in rows:
                scores[name].append(score)
                tracks[name][item["name"]] = events

    wall = time.perf_counter() - started
    parameters = {name: parameter for name, parameter, _ in policies}
    for name in names:
        failed = [row.get("name", "<unknown>") for row in scores[name]
                  if not row.get("ok")]
        if failed:
            raise RuntimeError(f"{name} failed to score {len(failed)} tracks: "
                               + ", ".join(failed[:10]))
    return {
        name: CorpusArm(
            name=name, parameter=parameters[name], corpus=corpus,
            scores=scores[name], tracks=tracks[name],
            summary=summarize(
                "model", [dict(row, corpus=corpus) for row in scores[name]], wall),
            max_passes=1,
        ) for name in names
    }


def _cost_gates_hold(arm: CorpusArm, baseline: CorpusArm) -> bool:
    """The RWC development gates fixed before the corpus is opened.

    RWC's absolute baselines differ sharply from Harmonix, so development gates
    are no-regression gates against the same RWC baseline. Beat F retains the
    standing one-point tolerance used on Harmonix.
    """
    got, base = arm.metrics, baseline.metrics
    comparisons = (
        (got["usable_rate_strict"], base["usable_rate_strict"], ">="),
        (got["mean_correct_share_of_eligible"],
         base["mean_correct_share_of_eligible"], ">="),
        (got["switches_per_five_minutes"], base["switches_per_five_minutes"], "<="),
        (got["p90_settle_sec"], base["p90_settle_sec"], "<="),
        (got["f_measure"], base["f_measure"] - 0.01, ">="),
    )
    for value, bound, direction in comparisons:
        if value is None or bound is None:
            return False
        if direction == ">=" and value < bound:
            return False
        if direction == "<=" and value > bound:
            return False
    return True


def select_tau(arms: Sequence[CorpusArm], baseline: CorpusArm) -> CorpusArm:
    """Apply the registered RWC objective, constraints and tie-breaks."""
    eligible = []
    for arm in arms:
        if arm.parameter is None:
            continue
        tau = float(arm.parameter)
        a3 = false_veto_rate(arm.tracks.values(), tau)
        if math.isfinite(a3) and a3 <= 0.05 and _cost_gates_hold(arm, baseline):
            eligible.append(arm)
    if not eligible:
        raise RuntimeError("no tau candidate satisfies A3 and the RWC cost gates")

    def key(arm: CorpusArm):
        tau = float(arm.parameter)
        return (
            arm.metrics["no_wrong_level_episode_fraction"],
            arm.metrics["mean_correct_share_of_eligible"],
            -veto_rate(arm.tracks.values(), tau),
            -tau,
        )

    return max(eligible, key=key)


def select_matched_policy(arms: Sequence[CorpusArm], decoder: CorpusArm,
                          tolerance: float = 0.005) -> CorpusArm:
    """Best simple policy whose retained correct locked-time matches decoder."""
    target = decoder.metrics["mean_correct_share_of_eligible"]
    matched = [
        arm for arm in arms
        if abs(arm.metrics["mean_correct_share_of_eligible"] - target) <= tolerance
    ]
    if not matched:
        raise RuntimeError("no simple policy matches retained correct locked-time")
    # Candidate grids are registered in ascending order. Python's stable max
    # therefore makes an exact objective tie choose the first registered value.
    return max(matched,
               key=lambda arm: arm.metrics["no_wrong_level_episode_fraction"])


def registered_tau_policies() -> tuple[tuple[str, float, ScheduleBuilder], ...]:
    return tuple(
        (f"decoder_tau_{tau:g}", tau,
         lambda events, value=tau: decoder_schedule(events, value))
        for tau in TAU_CANDIDATES
    )


def registered_simple_schedule_policies(
) -> tuple[tuple[str, float | None, ScheduleBuilder], ...]:
    rows: list[tuple[str, float | None, ScheduleBuilder]] = []
    rows.extend(
        (f"debounce_{seconds:g}", seconds,
         lambda events, value=seconds: debounce_schedule(events, value))
        for seconds in DEBOUNCE_CANDIDATES
    )
    rows.extend(
        (f"rate_limit_{seconds:g}", seconds,
         lambda events, value=seconds: rate_limit_schedule(events, value))
        for seconds in RATE_LIMIT_CANDIDATES
    )
    rows.append(("total_ban", None, total_ban_schedule))
    return tuple(rows)


def registered_margin_policies(
) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    return tuple(
        (f"margin_{margin:g}", margin,
         ("--live-anchor-margin", repr(margin)))
        for margin in MARGIN_CANDIDATES
    )


def matched_policy_families(arms: Iterable[CorpusArm], decoder: CorpusArm,
                            tolerance: float = 0.005
                            ) -> dict[str, CorpusArm | None]:
    rows = list(arms)
    families = {
        "debounce": [arm for arm in rows if arm.name.startswith("debounce_")],
        "margin": [arm for arm in rows if arm.name.startswith("margin_")],
        "rate_limit": [arm for arm in rows if arm.name.startswith("rate_limit_")],
        "total_ban": [arm for arm in rows if arm.name == "total_ban"],
    }
    out: dict[str, CorpusArm | None] = {}
    for family, candidates in families.items():
        try:
            out[family] = select_matched_policy(candidates, decoder, tolerance)
        except RuntimeError:
            out[family] = None
    return out


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test on discordant recording pairs."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2 ** total)


def paired_episode_test(left: CorpusArm, right: CorpusArm) -> dict[str, Any]:
    def rows(arm: CorpusArm) -> dict[str, bool]:
        return {
            row["name"]: bool(row["worst_wrong_octave_sec"] <= 4.0)
            for row in arm.scores if row.get("ok") and row.get("annotated")
        }

    a, b = rows(left), rows(right)
    shared = sorted(set(a) & set(b))
    wins = sum(a[name] and not b[name] for name in shared)
    losses = sum(b[name] and not a[name] for name in shared)
    return {"n": len(shared), "wins": wins, "losses": losses,
            "p": sign_test(wins, losses)}


def holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Holm-Bonferroni for the registered family of exactly three tests."""
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running,
                      min(1.0, rows[index]["p"] * (len(rows) - rank)))
        rows[index]["p_holm"] = running
        rows[index]["significant_at_05"] = running < 0.05
    return rows


def transfer_verdict(baseline: CorpusArm, decoder: CorpusArm,
                     matched: CorpusArm, control: CorpusArm,
                     tau: float) -> dict[str, Any]:
    """Evaluate A1--A4 and the standing Harmonix gates without retuning."""
    if decoder.corpus != "harmonix" or any(
            arm.corpus != "harmonix" for arm in (baseline, matched, control)):
        raise ValueError("transfer verdict is defined only for Harmonix arms")
    a2 = cluster_bootstrap_difference(decoder.tracks, control.tracks, tau)
    a3 = false_veto_rate(decoder.tracks.values(), tau)
    decoder_rate = decoder.metrics["no_wrong_level_episode_fraction"]
    matched_rate = matched.metrics["no_wrong_level_episode_fraction"]
    cost_drift = abs(
        matched.metrics["mean_correct_share_of_eligible"]
        - decoder.metrics["mean_correct_share_of_eligible"])

    family = holm([
        {"test": "decoder_vs_baseline_episode",
         **paired_episode_test(decoder, baseline)},
        {"test": "decoder_vs_matched_episode",
         **paired_episode_test(decoder, matched)},
        {"test": "decoder_vs_shift_control_balanced_accuracy", "p": a2["p"]},
    ])
    significant = {row["test"]: row["significant_at_05"] for row in family}
    a1 = bool(
        decoder_rate >= HARMONIX_EPISODE_GATE
        and decoder_rate > matched_rate
        and significant["decoder_vs_baseline_episode"]
        and significant["decoder_vs_matched_episode"]
    )
    a2_passed = bool(
        a2["difference"] >= 0.15 and a2["ci_low"] > 0.0
        and significant["decoder_vs_shift_control_balanced_accuracy"]
    )
    a3_passed = bool(math.isfinite(a3) and a3 <= 0.05)
    cost_rows = []
    for metric, (direction, bound) in HARMONIX_COST_GATES.items():
        value = decoder.metrics[metric]
        passed = (value >= bound if direction == ">=" else value <= bound)
        cost_rows.append({"metric": metric, "value": value, "direction": direction,
                          "bound": bound, "passed": bool(passed)})
    matched_cost = cost_drift <= 0.005
    a4 = bool(math.isfinite(tau) and a1 and a2_passed and a3_passed)
    accepted = bool(a4 and matched_cost
                    and all(row["passed"] for row in cost_rows))
    return {
        "tau": tau,
        "A1": a1,
        "A2": a2_passed,
        "A3": a3_passed,
        "A4": a4,
        "accepted": accepted,
        "episode_rate": decoder_rate,
        "matched_episode_rate": matched_rate,
        "a2": a2,
        "false_veto_rate": a3,
        "matched_cost_drift": cost_drift,
        "matched_cost_passed": matched_cost,
        "cost_gates": cost_rows,
        "holm_family": family,
    }


def apply_protocol_diagnostics(verdict: dict[str, Any],
                               tracks: Iterable[TrackEvents],
                               rwc_ambiguity: dict[str, Any]) -> dict[str, Any]:
    """Apply the two conditional sinks registered outside A1--A4."""
    tracks = list(tracks)
    harmonix_sign = raw_sign_agreement(tracks)
    harmonix_ambiguity = ambiguity_diagnostic(tracks)
    sign_failed_after_win = bool(
        verdict["A1"]
        and (harmonix_sign["fraction"] is None
             or harmonix_sign["fraction"] <= 0.90))
    ambiguity_failed = bool(
        rwc_ambiguity.get("dominates", False)
        and harmonix_ambiguity["dominates"])
    if sign_failed_after_win or ambiguity_failed:
        verdict["accepted"] = False
    verdict["protocol_diagnostics"] = {
        "P2_raw_sign_agreement": harmonix_sign,
        "P2_sink_triggered": sign_failed_after_win,
        "rwc_ambiguity": rwc_ambiguity,
        "harmonix_ambiguity": harmonix_ambiguity,
        "ambiguity_sink_triggered": ambiguity_failed,
        "D1": d1_zero_committed_score(tracks),
    }
    return verdict


def _arm_digest(arm: CorpusArm, tau: float | None = None) -> dict[str, Any]:
    use_tau = tau if tau is not None else (
        float(arm.parameter) if arm.parameter is not None else 0.0)
    is_decoder = arm.name.startswith("decoder_tau_")
    is_control = arm.name.startswith("shift_control_tau_")
    return {
        "name": arm.name,
        "parameter": arm.parameter,
        "metrics": arm.metrics,
        "events": event_counts(arm.tracks.values()),
        "event_coverage_by_meter": event_coverage_by_meter(arm.tracks.values()),
        "raw_sign_agreement": raw_sign_agreement(arm.tracks.values()),
        "d1_zero_committed_score": d1_zero_committed_score(arm.tracks.values()),
        "balanced_accuracy": (
            balanced_accuracy(arm.tracks.values(), use_tau, control=is_control)
            if is_decoder or is_control else None),
        "false_veto_rate": (
            false_veto_rate(arm.tracks.values(), use_tau, control=is_control)
            if is_decoder or is_control else None),
        "veto_rate": (
            veto_rate(arm.tracks.values(), use_tau, control=is_control)
            if is_decoder or is_control else None),
        "max_fixed_point_passes": arm.max_passes,
    }


def _event_rows(arm: CorpusArm) -> list[dict[str, Any]]:
    rows = []
    for name, track in sorted(arm.tracks.items()):
        for event in track.events:
            rows.append({
                "track": name,
                "label": event.label,
                "proposal": dataclasses.asdict(event.proposal),
                "decision": dataclasses.asdict(event.decision),
                "null_delta": event.null_delta,
            })
    return rows


def _git(repository: pathlib.Path, *args: str) -> str:
    done = subprocess.run(("git", "-C", str(repository), *args),
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "git command failed")
    return done.stdout.strip()


def _json_default(value: Any):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")


def _load_items(manifest: pathlib.Path, music: pathlib.Path, corpus: str):
    from eval.live_corpus_benchmark import load_corpus

    names = ({"rwc-classical", "rwc-genre", "rwc-jazz", "rwc-pop",
              "rwc-royalty-free"} if corpus == "rwc" else {corpus})
    items = load_corpus(manifest, music, False, corpora=names)
    missing = [str(item["audio"]) for item in items if not item["audio"].is_file()]
    missing += [str(item["annotation"]) for item in items
                if not item["annotation"].is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} corpus files are missing; first: "
                           + ", ".join(missing[:5]))
    if not items:
        raise RuntimeError(f"manifest contains no {corpus} recordings")
    return items


def run_rwc(args) -> dict[str, Any]:
    repository = pathlib.Path(__file__).resolve().parents[2]
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("RWC selection requires a clean implementation commit")
    source_commit = _git(repository, "rev-parse", "HEAD")
    items = _load_items(args.manifest, args.music, "rwc")
    policies = (("baseline", None, lambda _: ()),
                *registered_tau_policies(),
                *registered_simple_schedule_policies())
    schedule_arms = run_policy_grid(
        items, args.binary, args.model, policies, corpus="rwc",
        workers=args.workers)
    margin_arms = run_direct_flag_grid(
        items, args.binary, args.model, registered_margin_policies(),
        corpus="rwc", workers=args.workers)

    baseline = schedule_arms["baseline"]
    tau_arms = [schedule_arms[name] for name, _, _ in registered_tau_policies()]
    decoder = select_tau(tau_arms, baseline)
    tau = float(decoder.parameter)

    control_name = f"shift_control_tau_{tau:g}"
    control = run_policy_grid(
        items, args.binary, args.model,
        ((control_name, tau,
          lambda events: decoder_schedule(events, tau, control=True)),),
        corpus="rwc", workers=args.workers)[control_name]

    simple_names = [name for name, _, _ in registered_simple_schedule_policies()]
    simple = [schedule_arms[name] for name in simple_names] + list(margin_arms.values())
    families = matched_policy_families(simple, decoder)
    available = [arm for arm in families.values() if arm is not None]
    if not available:
        raise RuntimeError("no matched-cost policy family has a candidate")
    best = select_matched_policy(available, decoder)
    a2 = cluster_bootstrap_difference(decoder.tracks, control.tracks, tau)
    a2_passed = a2["difference"] >= 0.15 and a2["ci_low"] > 0.0
    a3 = false_veto_rate(decoder.tracks.values(), tau)
    a3_passed = a3 <= 0.05
    sign_agreement = raw_sign_agreement(decoder.tracks.values())
    ambiguity = ambiguity_diagnostic(decoder.tracks.values())
    ready_for_transfer = (all(arm is not None for arm in families.values())
                          and a2_passed and a3_passed)

    selected_families = {
        family: (None if arm is None else {
            "name": arm.name,
            "parameter": arm.parameter,
            "retained_correct_locked_time":
                arm.metrics["mean_correct_share_of_eligible"],
            "episode_rate": arm.metrics["no_wrong_level_episode_fraction"],
        }) for family, arm in families.items()
    }
    payload = {
        "stage": "rwc-development",
        "tree_clean_before_output": True,
        "code_commit_before_selection_commit": source_commit,
        "corpus": {"name": "rwc", "tracks": len(items)},
        "selection": {
            "tau": tau,
            "ready_for_transfer": ready_for_transfer,
            "best_matched_family": next(
                family for family, arm in families.items() if arm is best),
            "best_matched_name": best.name,
            "matched_families": selected_families,
        },
        "rwc_checks": {
            "A2": a2,
            "A2_passed": a2_passed,
            "A3": a3,
            "A3_passed": a3_passed,
            "P2_raw_sign_agreement": sign_agreement,
            "ambiguity": ambiguity,
            "D1": d1_zero_committed_score(decoder.tracks.values()),
        },
        "arms": {
            name: _arm_digest(arm, tau if name == control_name else None)
            for name, arm in {**schedule_arms, **margin_arms,
                              control_name: control}.items()
        },
        "selected_decoder_events": _event_rows(decoder),
        "selected_control_events": _event_rows(control),
    }
    _write_json(args.output, payload)
    return payload


def _require_frozen_selection(repository: pathlib.Path,
                              selection_path: pathlib.Path,
                              source_commit: str) -> str:
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("Harmonix requires a clean worktree")
    try:
        relative = selection_path.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise RuntimeError("selection file must be inside the repository") from error
    _git(repository, "ls-files", "--error-unmatch", str(relative))
    commit = _git(repository, "rev-parse", "HEAD")
    if _git(repository, "log", "-1", "--format=%H", "--",
            str(relative)) != commit:
        raise RuntimeError("selection file must be committed at the current HEAD")
    if _git(repository, "rev-parse", "HEAD^") != source_commit:
        raise RuntimeError("selection commit must directly follow the RWC code commit")
    changed = _git(repository, "diff", "--name-only", source_commit, commit)
    changed_paths = {pathlib.PurePosixPath(row).as_posix()
                     for row in changed.splitlines() if row}
    expected = pathlib.PurePosixPath(relative).as_posix()
    if changed_paths != {expected}:
        raise RuntimeError("selection commit must contain only the RWC selection file")
    return commit


def _selected_schedule_policy(selection: dict[str, Any], family: str,
                              tau: float) -> tuple[str, float | None,
                                                    ScheduleBuilder] | None:
    row = selection["matched_families"].get(family)
    if row is None:
        return None
    parameter = row["parameter"]
    name = row["name"]
    if family == "debounce":
        return name, parameter, lambda events: debounce_schedule(events, parameter)
    if family == "rate_limit":
        return name, parameter, lambda events: rate_limit_schedule(events, parameter)
    if family == "total_ban":
        return name, parameter, total_ban_schedule
    if family == "margin":
        return None
    raise ValueError(f"unknown policy family {family}")


def run_harmonix(args) -> dict[str, Any]:
    repository = pathlib.Path(__file__).resolve().parents[2]
    development = json.loads(args.selection.read_text(encoding="utf-8"))
    source_commit = development.get("code_commit_before_selection_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise RuntimeError("selection does not name its implementation commit")
    frozen_commit = _require_frozen_selection(
        repository, args.selection, source_commit)
    selection = development["selection"]
    if not selection.get("ready_for_transfer", False):
        raise RuntimeError("RWC did not license opening Harmonix")
    tau = float(selection["tau"])
    items = _load_items(args.manifest, args.music, "harmonix")

    policies: list[tuple[str, float | None, ScheduleBuilder]] = [
        ("baseline", None, lambda _: ()),
        (f"decoder_tau_{tau:g}", tau,
         lambda events: decoder_schedule(events, tau)),
        (f"shift_control_tau_{tau:g}", tau,
         lambda events: decoder_schedule(events, tau, control=True)),
    ]
    for family in ("debounce", "rate_limit", "total_ban"):
        policy = _selected_schedule_policy(selection, family, tau)
        if policy is not None:
            policies.append(policy)
    arms = run_policy_grid(items, args.binary, args.model, tuple(policies),
                           corpus="harmonix", workers=args.workers)

    margin_row = selection["matched_families"].get("margin")
    if margin_row is not None:
        margin = float(margin_row["parameter"])
        arms.update(run_direct_flag_grid(
            items, args.binary, args.model,
            ((margin_row["name"], margin,
              ("--live-anchor-margin", repr(margin))),),
            corpus="harmonix", workers=args.workers))

    baseline = arms["baseline"]
    decoder = arms[f"decoder_tau_{tau:g}"]
    control = arms[f"shift_control_tau_{tau:g}"]
    best = arms[selection["best_matched_name"]]
    verdict = transfer_verdict(baseline, decoder, best, control, tau)

    rwc_ambiguity = development.get("rwc_checks", {}).get("ambiguity", {})
    apply_protocol_diagnostics(verdict, decoder.tracks.values(), rwc_ambiguity)

    family_costs = {}
    for family, row in selection["matched_families"].items():
        if row is None:
            family_costs[family] = None
            continue
        arm = arms[row["name"]]
        drift = abs(arm.metrics["mean_correct_share_of_eligible"]
                    - decoder.metrics["mean_correct_share_of_eligible"])
        family_costs[family] = {"name": arm.name, "drift": drift,
                                "passed": drift <= 0.005}
    if any(row is not None and not row["passed"] for row in family_costs.values()):
        verdict["accepted"] = False

    payload = {
        "stage": "harmonix-transfer",
        "frozen_commit": frozen_commit,
        "corpus": {"name": "harmonix", "tracks": len(items)},
        "selection": selection,
        "verdict": verdict,
        "all_family_cost_transfer": family_costs,
        "arms": {name: _arm_digest(arm, tau if "control" in name else None)
                 for name, arm in arms.items()},
        "decoder_events": _event_rows(decoder),
        "control_events": _event_rows(control),
    }
    _write_json(args.output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("rwc", "harmonix"):
        command = sub.add_parser(stage)
        command.add_argument("--binary", type=pathlib.Path, required=True)
        command.add_argument("--model", type=pathlib.Path, required=True)
        command.add_argument("--manifest", type=pathlib.Path,
                             default=repository / "music" / "ground-truth" /
                             "manifest.csv")
        command.add_argument("--music", type=pathlib.Path,
                             default=repository / "music")
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--output", type=pathlib.Path, required=True)
    sub.choices["harmonix"].add_argument("--selection", type=pathlib.Path,
                                         required=True)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    payload = run_rwc(args) if args.stage == "rwc" else run_harmonix(args)
    print(json.dumps({"stage": payload["stage"], "output": str(args.output),
                      "accepted": payload.get("verdict", {}).get("accepted")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
