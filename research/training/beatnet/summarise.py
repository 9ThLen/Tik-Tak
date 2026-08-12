"""Summarise six completed S1 runs without trusting per-run means."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from eval.provenance import digest, experiment_provenance

from .cache import _atomic_json, _outside_repository
from .data import file_sha256
from .trainer import ARMS


SCHEMA = "tiktak.s1_summary/v1"
SEEDS = (17, 29, 43)
GATED_METRICS = (
    "phase_f1", "beat_f1", "downbeat_f1", "stable_exact_position",
    "false_switches_per_5min", "long_wrong_episodes_per_5min",
)
DIAGNOSTIC_METRICS = (
    "beat_precision", "beat_recall", "downbeat_precision", "downbeat_recall",
    "usable_strict", "position_accuracy", "grouping_balanced_accuracy",
    "coverage", "false_confident_share", "unnecessary_unknown_share",
    "wrong_episodes_per_5min", "resolver_state_changes_per_5min",
    "held_state_changes_per_5min",
)
METRICS = GATED_METRICS + DIAGNOSTIC_METRICS


def bootstrap(values: dict[str, float], draws: int = 2000) -> dict:
    works = sorted(values)
    array = np.asarray([values[work] for work in works], dtype=np.float64)
    if not len(array):
        return {"mean": None, "ci": [None, None], "n": 0}
    sampled = np.empty(draws, dtype=np.float64)
    for seed in range(draws):
        indices = np.random.default_rng(seed).integers(0, len(array), len(array))
        sampled[seed] = float(np.mean(array[indices]))
    return {"mean": float(np.mean(array)),
            "ci": [float(value) for value in np.percentile(sampled, [2.5, 97.5])],
            "n": len(array)}


def _load_run(path: pathlib.Path) -> tuple[dict, dict | None]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if (run.get("schema") != "tiktak.s1_training/v1"
            or run.get("complete") is not True
            or run.get("provenance", {}).get("tree_clean") is not True):
        raise ValueError(f"incomplete or unprovenanced S1 run: {path.name}")
    if run.get("best") is None:
        return run, None
    evaluation_path = path.parent / run["best"]["evaluation"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    history = {int(row["epoch"]): row for row in run["history"]}
    selected = history.get(int(run["best"]["epoch"]), {})
    if selected.get("evaluation_sha256") != file_sha256(evaluation_path):
        raise ValueError(f"S1 selected evaluation digest changed: {path.name}")
    if (evaluation.get("schema") != "tiktak.s1_evaluation/v1"
            or evaluation.get("arm") != run["arm"]
            or evaluation.get("seed") != run["seed"]
            or evaluation.get("dev_works") != 84):
        raise ValueError(f"S1 best evaluation mismatch: {path.name}")
    return run, evaluation


def summarise(paths: list[pathlib.Path]) -> dict:
    if len(paths) != 6:
        raise ValueError("S1 requires exactly six arm/seed results")
    loaded = {}
    sources = []
    common = None
    ineligible = []
    for path in paths:
        run, evaluation = _load_run(path)
        key = (run["arm"], int(run["seed"]))
        if key in loaded:
            raise ValueError(f"duplicate S1 arm/seed: {key}")
        if run["arm"] not in ARMS or run["seed"] not in SEEDS:
            raise ValueError(f"unregistered S1 arm/seed: {key}")
        identity = {name: value for name, value in run["identity"].items()
                    if name not in {"arm", "seed"}}
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("S1 run identities differ")
        loaded[key] = evaluation
        source = {"run": digest(path), "evaluation": None}
        if evaluation is None:
            ineligible.append({"arm": run["arm"], "seed": run["seed"]})
        else:
            source["evaluation"] = digest(
                path.parent / run["best"]["evaluation"])
        sources.append(source)
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(loaded) != expected:
        raise ValueError("S1 arm/seed matrix is incomplete")
    if ineligible:
        return {
            "schema": SCHEMA, "research_only": True, "complete": False,
            "sources": sources, "identity": common,
            "ineligible_runs": ineligible, "interpretation": "inconclusive",
            "reason": "one or more arms had no beat-noninferior checkpoint",
        }

    per_seed = {}
    by_metric = {metric: {} for metric in METRICS}
    work_corpora = None
    for seed in SEEDS:
        reset_evaluation = loaded[("A3_reset", seed)]
        stateful_evaluation = loaded[("A3_stateful", seed)]
        reset = reset_evaluation["work_metrics"]
        stateful = stateful_evaluation["work_metrics"]
        if set(reset) != set(stateful) or len(reset) != 84:
            raise ValueError(f"S1 work pairing changed for seed {seed}")
        if reset_evaluation.get("work_corpora") != stateful_evaluation.get(
                "work_corpora"):
            raise ValueError(f"S1 work/corpus mapping changed for seed {seed}")
        if work_corpora is None:
            work_corpora = reset_evaluation["work_corpora"]
        elif work_corpora != reset_evaluation["work_corpora"]:
            raise ValueError("S1 work/corpus mapping differs across seeds")
        per_seed[str(seed)] = {}
        for metric in METRICS:
            differences = {
                work: float(stateful[work][metric] - reset[work][metric])
                for work in reset
            }
            per_seed[str(seed)][metric] = float(np.mean(list(differences.values())))
            for work, difference in differences.items():
                by_metric[metric].setdefault(work, []).append(difference)

    effects = {}
    averaged_effects = {}
    for metric, works in by_metric.items():
        averaged = {work: float(np.mean(values)) for work, values in works.items()}
        if any(len(values) != len(SEEDS) for values in works.values()):
            raise ValueError(f"S1 seed pairing incomplete for {metric}")
        effects[metric] = bootstrap(averaged)
        averaged_effects[metric] = averaged
    per_corpus = {}
    for corpus in sorted(set(work_corpora.values())):
        selected = {work for work, value in work_corpora.items()
                    if value == corpus}
        per_corpus[corpus] = {
            metric: bootstrap({work: values[work] for work in selected})
            for metric, values in averaged_effects.items()
        }
    efficacy = bool(
        effects["phase_f1"]["mean"] >= 0.03
        and effects["phase_f1"]["ci"][0] > 0.0
        and min(per_seed[str(seed)]["phase_f1"] for seed in SEEDS) >= -0.01)
    safety = {
        "beat": effects["beat_f1"]["ci"][0] >= -0.01,
        "downbeat": effects["downbeat_f1"]["ci"][0] >= -0.01,
        "stable_exact": effects["stable_exact_position"]["ci"][0] >= -0.03,
        "false_switch_rate": effects["false_switches_per_5min"]["ci"][1] <= 1.0,
        "long_wrong_episode_rate": effects[
            "long_wrong_episodes_per_5min"]["ci"][1] <= 0.25,
    }
    safe = all(safety.values())
    if efficacy and safe:
        interpretation = "stateful_training_positive"
    elif efficacy:
        interpretation = "stateful_gain_with_regression"
    elif safe:
        interpretation = "stateful_training_no_material_gain"
    else:
        interpretation = "stateful_training_negative"
    return {
        "schema": SCHEMA, "research_only": True, "complete": True,
        "sources": sources, "identity": common, "independent_works": 84,
        "per_seed_effects": per_seed, "paired_effects": effects,
        "per_corpus_effects": per_corpus,
        "gates": {"efficacy": efficacy, **safety},
        "interpretation": interpretation,
        "thresholds": {
            "bootstrap_draws": 2000, "phase_min_gain": 0.03,
            "phase_ci_lower": 0.0, "seed_phase_min": -0.01,
            "beat_ci_lower": -0.01, "downbeat_ci_lower": -0.01,
            "stable_ci_lower": -0.03, "false_switch_ci_upper": 1.0,
            "long_wrong_episode_ci_upper": 0.25,
        },
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    try:
        _outside_repository(args.output, repository)
        result = summarise(args.run)
        result["provenance"] = experiment_provenance(
            repository, files={f"run_{index}": path
                               for index, path in enumerate(args.run)},
            experiment="S1 summary")
        _atomic_json(args.output, result)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": "complete",
                      "interpretation": result["interpretation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
