"""Summarise six completed S1 runs without trusting per-run means."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from eval.provenance import digest, experiment_provenance

from .cache import _atomic_json, _outside_repository
from .data import file_sha256
from .arms import ARMS


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
A0_METRICS = ("phase_f1", "beat_f1", "downbeat_f1")


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


def _load_evaluation(path: pathlib.Path, expected_sha256: str,
                     run: dict) -> dict:
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"S1 evaluation digest changed: {path.name}")
    if (evaluation.get("schema") != "tiktak.s1_evaluation/v1"
            or evaluation.get("arm") != run["arm"]
            or evaluation.get("seed") != run["seed"]
            or evaluation.get("dev_works") != 84):
        raise ValueError(f"S1 evaluation mismatch: {path.name}")
    return evaluation


def _load_run(path: pathlib.Path) -> tuple[dict, dict | None, dict[int, tuple]]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if (run.get("schema") != "tiktak.s1_training/v1"
            or run.get("complete") is not True
            or run.get("provenance", {}).get("tree_clean") is not True):
        raise ValueError(f"incomplete or unprovenanced S1 run: {path.name}")
    history = {int(row["epoch"]): row for row in run["history"]}
    if len(history) != len(run["history"]):
        raise ValueError(f"duplicate S1 history epoch: {path.name}")
    evaluation_refs = {}
    for epoch, row in history.items():
        if "evaluation_sha256" not in row:
            continue
        if not isinstance(row.get("eligible"), bool):
            raise ValueError(f"S1 eligibility missing at epoch {epoch}")
        evaluation_path = (path.parent / "candidates" / f"epoch-{epoch:03d}"
                           / "evaluation.json")
        evaluation_refs[epoch] = (evaluation_path, row["evaluation_sha256"])
    if run.get("best") is None:
        return run, None, evaluation_refs
    best_epoch = int(run["best"]["epoch"])
    evaluation_path = path.parent / run["best"]["evaluation"]
    selected = history.get(int(run["best"]["epoch"]), {})
    if selected.get("eligible") is not True or best_epoch not in evaluation_refs:
        raise ValueError(f"S1 selected checkpoint is not eligible: {path.name}")
    if evaluation_path.resolve() != (
            path.parent / "candidates" / f"epoch-{best_epoch:03d}"
            / "evaluation.json").resolve():
        raise ValueError(f"S1 selected evaluation path changed: {path.name}")
    evaluation = _load_evaluation(
        *evaluation_refs[best_epoch], run=run)
    return run, evaluation, evaluation_refs


def _paired_arm_effects(pairs: dict[int, tuple[dict, dict]]) -> dict:
    per_seed = {}
    by_metric = {metric: {} for metric in METRICS}
    work_corpora = None
    for seed in SEEDS:
        reset_evaluation, stateful_evaluation = pairs[seed]
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
            per_seed[str(seed)][metric] = float(
                np.mean(list(differences.values())))
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
    return {
        "per_seed_effects": per_seed, "paired_effects": effects,
        "per_corpus_effects": per_corpus, "work_corpora": work_corpora,
    }


def _a0_diagnostics(selected: dict[tuple[str, int], dict],
                    baseline: dict, work_corpora: dict[str, str]) -> dict:
    baseline_metrics = baseline.get("work_metrics", {})
    if (len(baseline_metrics) != 84
            or baseline.get("work_corpora") != work_corpora):
        raise ValueError("S1 frozen A0 work population changed")
    per_seed = {arm: {} for arm in ARMS}
    by_arm = {arm: {metric: {} for metric in A0_METRICS} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            candidate = selected[(arm, seed)]["work_metrics"]
            if set(candidate) != set(baseline_metrics):
                raise ValueError(f"S1 A0 pairing changed for {arm}/{seed}")
            per_seed[arm][str(seed)] = {}
            for metric in A0_METRICS:
                differences = {
                    work: float(candidate[work][metric]
                                - baseline_metrics[work][metric])
                    for work in baseline_metrics
                }
                per_seed[arm][str(seed)][metric] = float(
                    np.mean(list(differences.values())))
                for work, difference in differences.items():
                    by_arm[arm][metric].setdefault(work, []).append(difference)
    effects = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for metric in A0_METRICS:
            works = by_arm[arm][metric]
            if any(len(values) != len(SEEDS) for values in works.values()):
                raise ValueError(
                    f"S1 A0 seed pairing incomplete for {arm}/{metric}")
            effects[arm][metric] = bootstrap({
                work: float(np.mean(values)) for work, values in works.items()
            })
    return {"per_seed_effects": per_seed, "paired_effects": effects}


def summarise(paths: list[pathlib.Path], baseline_path: pathlib.Path) -> dict:
    if len(paths) != 6:
        raise ValueError("S1 requires exactly six arm/seed results")
    loaded = {}
    runs = {}
    evaluation_refs = {}
    sources = []
    common = None
    ineligible = []
    for path in paths:
        run, evaluation, run_evaluations = _load_run(path)
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
        runs[key] = run
        evaluation_refs[key] = run_evaluations
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
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (baseline.get("schema") != "tiktak.s1_evaluation/v1"
            or baseline.get("dev_works") != 84
            or baseline.get("provenance", {}).get("tree_clean") is not True
            or file_sha256(baseline_path) != common.get("baseline_sha256")):
        raise ValueError("S1 frozen A0 baseline identity changed")
    selection = {arm: {} for arm in ARMS}
    common_epochs = {}
    common_pairs = {}
    for arm in ARMS:
        for seed in SEEDS:
            run = runs[(arm, seed)]
            validation_rows = [row for row in run["history"]
                               if "evaluation_sha256" in row]
            selection[arm][str(seed)] = {
                "validation_points": len(validation_rows),
                "eligible_points": sum(
                    row["eligible"] for row in validation_rows),
                "selected_epoch": (None if run.get("best") is None
                                   else int(run["best"]["epoch"])),
            }
    for seed in SEEDS:
        shared = set(evaluation_refs[("A3_reset", seed)]) & set(
            evaluation_refs[("A3_stateful", seed)])
        if not shared:
            return {
                "schema": SCHEMA, "research_only": True, "complete": False,
                "sources": sources, "identity": common,
                "selection_diagnostics": selection,
                "interpretation": "inconclusive",
                "reason": "one or more seeds had no common validation epoch",
            }
        epoch = max(shared)
        common_epochs[str(seed)] = epoch
        reset_run = runs[("A3_reset", seed)]
        stateful_run = runs[("A3_stateful", seed)]
        common_pairs[seed] = (
            _load_evaluation(
                *evaluation_refs[("A3_reset", seed)][epoch], run=reset_run),
            _load_evaluation(
                *evaluation_refs[("A3_stateful", seed)][epoch],
                run=stateful_run),
        )
    common_endpoint = _paired_arm_effects(common_pairs)
    common_endpoint.pop("work_corpora")
    common_endpoint.update({"non_gating": True,
                            "last_common_epoch_by_seed": common_epochs})
    if ineligible:
        return {
            "schema": SCHEMA, "research_only": True, "complete": False,
            "sources": sources, "identity": common,
            "selection_diagnostics": selection,
            "common_epoch_endpoint": common_endpoint,
            "ineligible_runs": ineligible, "interpretation": "inconclusive",
            "reason": "one or more arms had no beat-noninferior checkpoint",
        }

    primary = _paired_arm_effects({
        seed: (loaded[("A3_reset", seed)], loaded[("A3_stateful", seed)])
        for seed in SEEDS
    })
    per_seed = primary["per_seed_effects"]
    effects = primary["paired_effects"]
    per_corpus = primary["per_corpus_effects"]
    a0 = _a0_diagnostics(loaded, baseline, primary["work_corpora"])
    a0["baseline"] = digest(baseline_path)
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
        "selection_diagnostics": selection,
        "per_seed_effects": per_seed, "paired_effects": effects,
        "per_corpus_effects": per_corpus,
        "common_epoch_endpoint": common_endpoint,
        "a0_diagnostics": a0,
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
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    try:
        _outside_repository(args.output, repository)
        result = summarise(args.run, args.baseline)
        result["provenance"] = experiment_provenance(
            repository, files={
                **{f"run_{index}": path for index, path in enumerate(args.run)},
                "baseline": args.baseline,
            },
            experiment="S1 summary")
        _atomic_json(args.output, result)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": "complete",
                      "interpretation": result["interpretation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
