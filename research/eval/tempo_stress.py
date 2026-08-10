#!/usr/bin/env python3
"""Deterministic tempo-dynamics stress test for the shipping live tracker.

The input is a clean 50-fps activation with a five-frame beat bump.  It enters
through ``LiveTracker.observe()`` and beats leave through ``takeBeat()``, so the
test exercises the same anchor, particle filter, lock/release hysteresis and
publication rules as the application without asking an audio front end to
solve a different problem first.

Two arms are reported from byte-identical activations:

``shipping``
    The current live path, including its six-second activation-tempo anchor.

``no_anchor``
    The same particle filter and publisher with only the anchor disabled.  The
    difference identifies work done by the anchor rather than attributing all
    tempo behaviour to the filter.

This is a diagnostic, not a calibration corpus.  It characterises limits of
the shipped settings; it does not choose new settings from synthetic data.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any

import mir_eval.util
import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
sys.path.insert(0, str(RESEARCH))

from eval.analysis import DEFAULT_BINARY, Estimate  # noqa: E402
from eval.live_corpus_benchmark import (  # noqa: E402
    MAX_ACQUISITION_SEC,
    MAX_WRONG_OCTAVE_SEC,
    MIN_PRECISION,
    MIN_RECALL,
    octave_statistics,
)
from eval.provenance import experiment_provenance as provenance  # noqa: E402

FPS = 50.0
WARMUP_SEC = 5.0
WINDOW_SEC = 0.070
TEMPO_TOLERANCE = 0.03
STABLE_SAMPLES = 3
SEED = 20260801
INPUT_AUDIO = REPOSITORY / "core" / "tests" / "data" / "tone_mono.wav"


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    duration_sec: float
    start_bpm: float
    end_bpm: float | None = None
    change_start_sec: float | None = None
    change_end_sec: float | None = None
    jitter_std_sec: float = 0.0
    drop_probability: float = 0.0
    burst_drop_beats: int = 0
    replicate: int = 0

    @property
    def final_bpm(self) -> float:
        return self.start_bpm if self.end_bpm is None else self.end_bpm


def scenarios() -> list[Scenario]:
    """The preregistered matrix requested for the first stress run."""
    out = [
        Scenario(f"steady_{bpm}", "steady", 60.0, float(bpm))
        for bpm in (60, 90, 120, 180)
    ]
    for ramp_sec in (15, 45):
        for percent in (2, 5, 10):
            for direction in (-1, 1):
                end = 120.0 * (1.0 + direction * percent / 100.0)
                sign = "up" if direction > 0 else "down"
                out.append(Scenario(
                    f"ramp_{sign}_{percent}pct_over_{ramp_sec}s", "ramp",
                    30.0 + ramp_sec, 120.0, end,
                    change_start_sec=15.0, change_end_sec=15.0 + ramp_sec,
                ))
    for start, end in ((120, 90), (120, 150), (120, 180), (180, 120)):
        out.append(Scenario(
            f"step_{start}_to_{end}", "step", 70.0, float(start), float(end),
            change_start_sec=25.0, change_end_sec=25.0,
        ))
    for milliseconds in (10, 20, 40, 70):
        for replicate in range(5):
            out.append(Scenario(
                f"jitter_{milliseconds}ms", "jitter", 60.0, 120.0,
                jitter_std_sec=milliseconds / 1000.0, replicate=replicate,
            ))
    for percent in (10, 20, 40):
        for replicate in range(5):
            out.append(Scenario(
                f"drop_random_{percent}pct", "drop_random", 60.0, 120.0,
                drop_probability=percent / 100.0, replicate=replicate,
            ))
    for count in (1, 2, 4, 8):
        out.append(Scenario(
            f"drop_burst_{count}", "drop_burst", 60.0, 120.0,
            burst_drop_beats=count,
        ))
    return out


def tempo_at(scenario: Scenario, time_sec: float) -> float:
    """Instantaneous reference tempo, with phase kept continuous elsewhere."""
    if scenario.end_bpm is None or scenario.change_start_sec is None:
        return scenario.start_bpm
    if scenario.family == "step":
        return scenario.start_bpm if time_sec < scenario.change_start_sec else scenario.final_bpm
    assert scenario.change_end_sec is not None
    if time_sec <= scenario.change_start_sec:
        return scenario.start_bpm
    if time_sec >= scenario.change_end_sec:
        return scenario.final_bpm
    share = ((time_sec - scenario.change_start_sec)
             / (scenario.change_end_sec - scenario.change_start_sec))
    return scenario.start_bpm + share * (scenario.final_bpm - scenario.start_bpm)


def reference_beats(scenario: Scenario, integration_step_sec: float = 0.001) -> np.ndarray:
    """Integrate instantaneous BPM into a phase-continuous reference grid."""
    beats = [0.0]
    phase = 0.0
    time_sec = 0.0
    while time_sec < scenario.duration_sec:
        step = min(integration_step_sec, scenario.duration_sec - time_sec)
        rate = tempo_at(scenario, time_sec + 0.5 * step) / 60.0
        advance = rate * step
        if phase + advance >= 1.0:
            fraction = (1.0 - phase) / advance
            beats.append(time_sec + fraction * step)
            phase = phase + advance - 1.0
        else:
            phase += advance
        time_sec += step
    return np.asarray(beats, dtype=np.float64)


def _scenario_rng(scenario: Scenario, seed: int) -> np.random.Generator:
    identity = f"{scenario.name}:{scenario.replicate}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return np.random.default_rng(seed ^ int.from_bytes(digest[:8], "little"))


def observed_pulses(scenario: Scenario, reference: np.ndarray,
                    seed: int = SEED) -> np.ndarray:
    """Apply observation jitter or omissions without changing ground truth."""
    observed = np.asarray(reference, dtype=np.float64).copy()
    rng = _scenario_rng(scenario, seed)
    if scenario.jitter_std_sec:
        observed += rng.normal(0.0, scenario.jitter_std_sec, len(observed))
        observed = np.sort(observed[(observed >= 0.0)
                                    & (observed <= scenario.duration_sec)])
    if scenario.drop_probability:
        keep = rng.random(len(observed)) >= scenario.drop_probability
        observed = observed[keep]
    if scenario.burst_drop_beats:
        centre = int(np.argmin(np.abs(observed - 35.0)))
        first = max(0, centre - scenario.burst_drop_beats // 2)
        last = min(len(observed), first + scenario.burst_drop_beats)
        observed = np.delete(observed, np.arange(first, last))
    return observed


def activation_from_pulses(pulses: np.ndarray, duration_sec: float,
                           fps: float = FPS) -> np.ndarray:
    """A realistic-scale 100-ms beat bump over a small activation floor."""
    frames = int(math.ceil(duration_sec * fps)) + 1
    activation = np.full(frames, 0.02, dtype=np.float64)
    indices = np.round(np.asarray(pulses) * fps).astype(np.int64)
    indices = indices[(indices >= 0) & (indices < frames)]
    for offset, height in ((-2, 0.33), (-1, 0.67), (0, 0.95),
                           (1, 0.67), (2, 0.33)):
        moved = indices + offset
        moved = moved[(moved >= 0) & (moved < frames)]
        np.maximum.at(activation, moved, height)
    return activation


def _trim(values: np.ndarray, duration_sec: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[(values >= WARMUP_SEC) & (values <= duration_sec)]


def _match_quality(reference: np.ndarray, found: np.ndarray,
                   duration_sec: float) -> tuple[float, float, float]:
    ref = _trim(reference, duration_sec)
    est = _trim(found, duration_sec)
    matches = (mir_eval.util.match_events(ref, est, window=WINDOW_SEC)
               if len(ref) and len(est) else [])
    precision = len(matches) / len(est) if len(est) else 0.0
    recall = len(matches) / len(ref) if len(ref) else float("nan")
    f_measure = (2.0 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)
    return float(precision), float(recall), float(f_measure)


def _first_stable(times: np.ndarray, good: np.ndarray, after_sec: float) -> float | None:
    eligible = np.flatnonzero(times >= after_sec)
    for index in eligible:
        end = index + STABLE_SAMPLES
        if end <= len(good) and np.all(good[index:end]):
            # Live telemetry is nominally one hertz. Refuse to call samples
            # separated by a discontinuity one continuous stable stretch.
            if np.all(np.diff(times[index:end]) <= 1.5):
                return float(times[index] - after_sec)
    return None


def score(payload: dict[str, Any], scenario: Scenario,
          reference: np.ndarray) -> dict[str, Any]:
    estimate = Estimate.from_json(payload)
    found = np.asarray(estimate.beats, dtype=np.float64)
    p70, r70, f70 = _match_quality(reference, found, scenario.duration_sec)
    octave = octave_statistics(estimate, reference)

    count = min(len(estimate.live_times), len(estimate.live_bpms))
    times = np.asarray(estimate.live_times[:count], dtype=np.float64)
    bpms = np.asarray(estimate.live_bpms[:count], dtype=np.float64)
    valid = (times >= WARMUP_SEC) & np.isfinite(bpms) & (bpms > 0.0)
    true_bpms = np.asarray([tempo_at(scenario, value) for value in times])
    errors = np.full(count, np.nan, dtype=np.float64)
    errors[valid] = np.abs(bpms[valid] / true_bpms[valid] - 1.0)
    good = valid & (errors <= TEMPO_TOLERANCE)
    finite_errors = errors[np.isfinite(errors)]

    acquisition_3pct = _first_stable(times, good, WARMUP_SEC)
    transition_lag = None
    during_change = None
    if scenario.change_start_sec is not None:
        recovery_from = (scenario.change_start_sec if scenario.family == "step"
                         else float(scenario.change_end_sec))
        transition_lag = _first_stable(times, good, recovery_from)
        if scenario.family == "ramp":
            changing = ((times >= scenario.change_start_sec)
                        & (times <= float(scenario.change_end_sec)))
            during_change = (float(np.mean(good[changing]))
                             if np.any(changing) else None)

    settled = octave.get("settled_at")
    usable_v2 = (
        settled is not None
        and settled <= MAX_ACQUISITION_SEC
        and p70 >= MIN_PRECISION
        and r70 >= MIN_RECALL
        and octave["worst_wrong_octave_sec"] <= MAX_WRONG_OCTAVE_SEC
    )
    return {
        "p70": p70,
        "r70": r70,
        "f70": f70,
        "emitted": int(len(_trim(found, scenario.duration_sec))),
        "reference": int(len(_trim(reference, scenario.duration_sec))),
        "mean_abs_tempo_error_pct": (
            float(np.mean(finite_errors) * 100.0) if len(finite_errors) else None
        ),
        "p95_abs_tempo_error_pct": (
            float(np.quantile(finite_errors, 0.95) * 100.0)
            if len(finite_errors) else None
        ),
        "within_3pct_fraction": float(np.mean(good[times >= WARMUP_SEC]))
        if np.any(times >= WARMUP_SEC) else None,
        "acquisition_3pct_sec": acquisition_3pct,
        "transition_lag_sec": transition_lag,
        "ramp_within_3pct_fraction": during_change,
        "confidence_acquired_at": octave.get("acquired_at"),
        "settled_at": settled,
        "worst_wrong_octave_sec": octave["worst_wrong_octave_sec"],
        "final_bpm": float(estimate.live_bpm),
        "final_reference_bpm": scenario.final_bpm,
        "usable_v2": bool(usable_v2),
    }


def _run(binary: pathlib.Path, activation_path: pathlib.Path,
         mode: str) -> dict[str, Any]:
    command = [
        str(binary), str(INPUT_AUDIO), "--live-activation", str(activation_path),
        "--activation-fps", repr(FPS),
    ]
    if mode == "no_anchor":
        command.append("--live-no-anchor")
    done = subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or f"exit {done.returncode}")
    return json.loads(done.stdout)


def _provenance(binary: pathlib.Path) -> dict[str, Any]:
    return provenance(REPOSITORY, {"binary": binary})


def summarize(rows: list[dict[str, Any]], group_by: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['mode']}:{row[group_by]}"].append(row)
    report: dict[str, Any] = {}
    for key, part in sorted(grouped.items()):
        lags = [row["transition_lag_sec"] for row in part
                if row["transition_lag_sec"] is not None]
        report[key] = {
            "n": len(part),
            "mean_p70": float(np.mean([row["p70"] for row in part])),
            "mean_r70": float(np.mean([row["r70"] for row in part])),
            "mean_f70": float(np.mean([row["f70"] for row in part])),
            "mean_within_3pct_fraction": float(np.mean([
                row["within_3pct_fraction"] for row in part
                if row["within_3pct_fraction"] is not None
            ])),
            "usable_v2_rate": float(np.mean([row["usable_v2"] for row in part])),
            "median_transition_lag_sec": float(np.median(lags)) if lags else None,
            "never_recovered": sum(
                row["transition_lag_sec"] is None
                for row in part if row["family"] in {"ramp", "step"}
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=pathlib.Path,
                        default=RESEARCH / "results" / "tempo_stress.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--modes", nargs="+", choices=("shipping", "no_anchor"),
                        default=("shipping", "no_anchor"))
    args = parser.parse_args(argv)

    if not args.binary.is_file():
        parser.error(f"{args.binary} does not exist; build tools/eval first")
    if not INPUT_AUDIO.is_file():
        parser.error(f"fixture is missing: {INPUT_AUDIO}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    matrix = scenarios()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="tempo-stress-", dir=args.output.parent
    ) as temporary:
        activation_path = pathlib.Path(temporary) / "activation.txt"
        for index, scenario in enumerate(matrix, start=1):
            reference = reference_beats(scenario)
            pulses = observed_pulses(scenario, reference, args.seed)
            activation = activation_from_pulses(pulses, scenario.duration_sec)
            activation_path.write_text(
                "\n".join(f"{value:.5f}" for value in activation),
                encoding="utf-8",
            )
            for mode in args.modes:
                result = score(_run(args.binary, activation_path, mode),
                               scenario, reference)
                row = {
                    "scenario": scenario.name,
                    "replicate": scenario.replicate,
                    "family": scenario.family,
                    "mode": mode,
                    "duration_sec": scenario.duration_sec,
                    "start_bpm": scenario.start_bpm,
                    "end_bpm": scenario.final_bpm,
                    "input_pulses": int(len(pulses)),
                    **result,
                }
                rows.append(row)
                lag = ("--" if row["transition_lag_sec"] is None
                       else f"{row['transition_lag_sec']:.1f}s")
                label = (scenario.name if scenario.replicate == 0
                         else f"{scenario.name}#{scenario.replicate + 1}")
                print(f"[{index:02d}/{len(matrix)}] {mode:<9} "
                      f"{label:<32} p/r/f {row['p70']:.3f}/"
                      f"{row['r70']:.3f}/{row['f70']:.3f}  "
                      f"tempo3 {row['within_3pct_fraction']:.1%}  lag {lag}",
                      flush=True)

    report = {
        "schema_version": 1,
        "description": "Synthetic tempo dynamics through LiveTracker.observe/takeBeat",
        "configuration": {
            "fps": FPS,
            "warmup_sec": WARMUP_SEC,
            "match_window_sec": WINDOW_SEC,
            "tempo_tolerance": TEMPO_TOLERANCE,
            "stable_samples": STABLE_SAMPLES,
            "seed": args.seed,
            "stochastic_replicates": 5,
            "ramp_durations_sec": [15, 45],
            "modes": list(args.modes),
            "activation": {"floor": 0.02, "peak": 0.95,
                           "shape": "five-frame triangle"},
        },
        "provenance": _provenance(args.binary),
        "summary_by_family": summarize(rows, "family"),
        "summary_by_condition": summarize(rows, "scenario"),
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
