"""Run the pre-registered S0 recurrent-state reset-horizon experiment.

The model pass and the policy pass are deliberately separate.  The first pass
produces BeatNet activations while resetting only its recurrent state.  The
second replays those exact values, frame times and release blocks through the
unchanged LiveTracker and BarTracker.  No corpus run should be accepted before
``PREREGISTERED_M0a_S0.md`` has received independent review.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import tempfile
import time
from collections import defaultdict

import numpy as np

from eval.causal_metre import score_phase
from eval.analysis import Estimate
from eval.harness import evaluate, evaluate_downbeats
from eval.live_corpus_benchmark import (
    _without_local_paths,
    load_corpus,
    load_reference_beats,
    load_reference_downbeats,
    score_estimate,
)
from eval.octave_veto_replay import run, run_activation
from eval.provenance import digest, experiment_provenance


HORIZONS: tuple[float | None, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, None)
REQUIRED_CORPORA = frozenset({"gtzan", "harmonix"})
INITIAL_CUT_SEC = 2.0
TRANSIENT_SEC = 2.0
BOOTSTRAP_DRAWS = 2000


class InvariantError(RuntimeError):
    """A harness mismatch that voids the run instead of excluding a file."""


def finite_or_none(value: float) -> float | None:
    """Keep unscorable secondary metrics valid in strict JSON artifacts."""
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def finite_mean(values: list[float | None]) -> tuple[float | None, int]:
    finite = np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ], dtype=np.float64)
    return ((float(np.mean(finite)) if len(finite) else None), int(len(finite)))


def arm_name(horizon: float | None) -> str:
    return "Rinf" if horizon is None else f"R{int(horizon)}"


def _write(path: pathlib.Path, values: list[float], fmt: str = "%.17g") -> None:
    np.savetxt(path, np.asarray(values, dtype=np.float64), fmt=fmt)


def _activation_bundle(payload: dict, root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    paths = tuple(root / name for name in
                  ("beat.txt", "downbeat.txt", "emit.txt", "times.txt"))
    _write(paths[0], payload["activation_beat"])
    _write(paths[1], payload["activation_downbeat"])
    _write(paths[2], payload["activation_emit"], "%.0f")
    _write(paths[3], payload["activation_times"])
    return paths


def _score(item: dict, payload: dict, reference: np.ndarray,
           reference_downbeats: np.ndarray) -> dict:
    beats = np.asarray(payload.get("beats", []), dtype=np.float64)
    meters = np.asarray(payload.get("live_bar_meters", []), dtype=np.float64)
    positions = np.asarray(payload.get("live_bar_positions", []), dtype=np.float64)
    phase = score_phase(beats, positions, meters, reference_downbeats,
                        start_sec=INITIAL_CUT_SEC)
    claimed = beats[(positions == 0) & (beats >= INITIAL_CUT_SEC)]
    beat = evaluate(reference[reference >= INITIAL_CUT_SEC],
                    beats[beats >= INITIAL_CUT_SEC], trim=False)
    downbeat = evaluate_downbeats(
        reference_downbeats[reference_downbeats >= INITIAL_CUT_SEC],
        claimed, trim=False)
    if phase["f1"] is None:
        phase = {**phase, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                 "phase_correct_share": 0.0}
    canonical = score_estimate(item, Estimate.from_json(payload), mode="S0")
    return {
        "phase": phase,
        "beat_f": finite_or_none(beat["f_measure"]),
        "downbeat_f": finite_or_none(downbeat["downbeat_f_measure"]),
        "final_meter": int(payload.get("live_beats_per_bar", 0)),
        "usable_strict": bool(canonical.get("usable_strict", False)),
        "resets": [float(x) for x in payload.get("activation_model_resets", [])],
    }


def expected_resets(times: np.ndarray, horizon: float) -> np.ndarray:
    if len(times) == 0:
        return np.zeros(0, dtype=np.float64)
    thresholds = np.arange(horizon, times[-1] + horizon, horizon)
    indices = np.searchsorted(times, thresholds, side="left")
    return times[indices[indices < len(times)]]


def require_replay_parity(initial: dict, replayed: dict, name: str) -> None:
    for key in ("beats", "live_bar_positions", "live_bar_meters"):
        left = np.asarray(initial.get(key, []), dtype=np.float64)
        right = np.asarray(replayed.get(key, []), dtype=np.float64)
        if not np.array_equal(left, right):
            raise InvariantError(f"{name}: Rinf replay parity failed for {key}")


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    reference = load_reference_beats(item["annotation"])
    reference_downbeats = load_reference_downbeats(item["annotation"])
    arms: dict[str, dict] = {}
    channels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for horizon in HORIZONS:
        extra = ["--live-bars"]
        if horizon is not None:
            extra += ["--activation-reset-horizon", repr(horizon)]
        activation = run(binary, item["audio"], model, extra=extra)
        name = arm_name(horizon)
        channels[name] = (
            np.asarray(activation["activation_times"], dtype=np.float64),
            np.asarray(activation["activation_beat"], dtype=np.float64),
            np.asarray(activation["activation_downbeat"], dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as directory:
            beat, downbeat, emit, times = _activation_bundle(
                activation, pathlib.Path(directory))
            replayed = run_activation(
                binary, item["audio"], beat,
                extra=["--live-bars", "--live-downbeat", str(downbeat)],
                emit_path=emit, times_path=times)
        # Reset metadata belongs to the model pass; carry it across the policy
        # replay so the accepted artifact proves the requested schedule ran.
        replayed["activation_model_resets"] = activation.get(
            "activation_model_resets", [])
        if horizon is None:
            require_replay_parity(activation, replayed, item["name"])
        arms[name] = _score(item, replayed, reference, reference_downbeats)

    inf_times, inf_beat, inf_downbeat = channels["Rinf"]
    for horizon in HORIZONS:
        name = arm_name(horizon)
        times, beat, downbeat = channels[name]
        if not (np.array_equal(times, inf_times)
                and len(beat) == len(inf_beat)
                and len(downbeat) == len(inf_downbeat)):
            raise InvariantError(
                f"{item['name']} {name}: activation clocks diverged")
        transient = np.zeros(len(times), dtype=bool)
        resets = arms[name]["resets"]
        if horizon is not None and not np.array_equal(
                np.asarray(resets, dtype=np.float64),
                expected_resets(times, horizon)):
            raise InvariantError(
                f"{item['name']} {name}: model reset schedule diverged")
        for i, start in enumerate(resets):
            stop = min(start + TRANSIENT_SEC,
                       resets[i + 1] if i + 1 < len(resets) else np.inf)
            transient |= (times >= start) & (times < stop)
        steady = (times >= INITIAL_CUT_SEC) & ~transient
        active = (times >= INITIAL_CUT_SEC) & transient

        def mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
            return float(np.mean(np.abs(left[mask] - right[mask]))) if mask.any() else None

        arms[name]["activation_diagnostic"] = {
            "frames": int(len(times)),
            "transient_frames": int(np.sum(active)),
            "steady_frames": int(np.sum(steady)),
            "beat_mae_vs_Rinf_transient": mae(beat, inf_beat, active),
            "beat_mae_vs_Rinf_steady": mae(beat, inf_beat, steady),
            "downbeat_mae_vs_Rinf_transient": mae(
                downbeat, inf_downbeat, active),
            "downbeat_mae_vs_Rinf_steady": mae(
                downbeat, inf_downbeat, steady),
        }
    return {
        "name": item["name"],
        "corpus": item["corpus"],
        "annotation": digest(item["annotation"]),
        "arms": arms,
        "invariants": {"Rinf_replay_parity": True,
                       "reset_schedules_exact": True},
    }


def paired_bootstrap(values: np.ndarray,
                     draws: int = BOOTSTRAP_DRAWS) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    boot = np.asarray([
        np.mean(values[np.random.default_rng(seed).integers(
            0, len(values), len(values))])
        for seed in range(draws)
    ])
    return [float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5))]


def summarise(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["corpus"]].append(record)
    out: dict[str, dict] = {}
    names = [arm_name(horizon) for horizon in HORIZONS]
    for corpus, rows in sorted(grouped.items()):
        means = {
            arm: float(np.mean([r["arms"][arm]["phase"]["f1"] for r in rows]))
            for arm in names
        }
        secondary = {}
        for arm in names:
            beat_f, beat_n = finite_mean([
                r["arms"][arm].get("beat_f") for r in rows])
            downbeat_f, downbeat_n = finite_mean([
                r["arms"][arm].get("downbeat_f") for r in rows])
            secondary[arm] = {
                "beat_f": beat_f,
                "beat_f_n_scored": beat_n,
                "downbeat_f": downbeat_f,
                "downbeat_f_n_scored": downbeat_n,
                "usable_strict": float(np.mean([
                    r["arms"][arm].get("usable_strict", False) for r in rows])),
            }
        adjacent = {}
        for left, right in zip(names, names[1:]):
            delta = np.asarray([
                r["arms"][right]["phase"]["f1"]
                - r["arms"][left]["phase"]["f1"] for r in rows
            ])
            adjacent[f"{right}-{left}"] = {
                "mean": float(np.mean(delta)), "ci": paired_bootstrap(delta)}
        extreme = np.asarray([
            r["arms"]["Rinf"]["phase"]["f1"]
            - r["arms"]["R2"]["phase"]["f1"] for r in rows
        ])
        out[corpus] = {
            "records": len(rows),
            "phase_f1": means,
            "secondary": secondary,
            "Rinf-R2": {"mean": float(np.mean(extreme)),
                         "ci": paired_bootstrap(extreme)},
            "adjacent": adjacent,
        }
    if out:
        present = frozenset(out)
        required_present = REQUIRED_CORPORA.issubset(present)
        blocks = [out[name] for name in sorted(REQUIRED_CORPORA & present)]
        negative = (required_present
                    and any(block["Rinf-R2"]["ci"][1] < 0.05
                            for block in blocks))
        positive_margin = (required_present
                           and all(block["Rinf-R2"]["ci"][0] >= 0.05
                                   for block in blocks))
        monotone = True
        for block in blocks:
            lower = [step["ci"][0] for step in block["adjacent"].values()]
            violations = sum(value < -0.01 for value in lower)
            if violations > 1 or any(value < -0.03 for value in lower):
                monotone = False
        out["decision"] = {
            "verdict": ("negative" if negative else
                        "positive" if positive_margin and monotone else
                        "inconclusive"),
            "positive_margin": positive_margin,
            "monotonicity_passed": monotone,
            "required_corpora_present": required_present,
        }
    return out


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth" / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path,
                        default=repository / "music")
    parser.add_argument("--corpora", nargs="+", default=["gtzan", "harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    items = load_corpus(args.manifest, args.music, False,
                        frozenset(args.corpora))
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("no recordings")

    provenance = experiment_provenance(
        repository,
        files={"binary": args.binary, "model": args.model,
               "manifest": args.manifest},
        experiment="S0", horizons=[arm_name(h) for h in HORIZONS],
        initial_cut_sec=INITIAL_CUT_SEC, transient_sec=TRANSIENT_SEC,
        bootstrap_draws=BOOTSTRAP_DRAWS, workers=args.workers)
    print(json.dumps({"event": "start", "recordings": len(items),
                      "workers": args.workers}), flush=True)
    started = time.perf_counter()
    ordered: list[tuple[str, dict] | None] = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        futures = {
            pool.submit(measure_outcome, item, args.binary, args.model): index
            for index, item in enumerate(items)
        }
        for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1):
            ordered[futures[future]] = future.result()
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"event": "progress", "done": completed,
                                  "total": len(futures), "elapsed_sec": round(
                                      time.perf_counter() - started, 1)}),
                      flush=True)
    outcomes = [outcome for outcome in ordered if outcome is not None]
    records = [payload for kind, payload in outcomes if kind == "record"]
    exclusions = [payload for kind, payload in outcomes if kind == "exclusion"]
    artifact = {"provenance": provenance,
                "selected": len(items), "scored": len(records),
                "technical_exclusions": exclusions, "records": records,
                "summary": summarise(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
