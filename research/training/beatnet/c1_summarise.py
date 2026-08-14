#!/usr/bin/env python3
"""Summarise the C1 curve: two axes, two bootstraps, and a selection override.

The primary interval resamples **works and training seeds**, because S1's
work-only scheme produces an interval conditional on the three models that were
actually trained -- it answers "how would this differ on other works" when a
curve is asked "how would this differ on another training run". The work-only
interval is kept as an S1-comparable secondary and decides nothing.

The deciding slope is the **all-except-Candombe** one. The overall slope is
computed and reported and does not gate: it is the single quantity one genre can
contaminate -- Candombe supplied 72.4% of S1's pooled gain -- and it is
insensitive in exactly that case, because an effect confined to seven works of
eighty-four moves the mean by 7/84 of its size while resampling works widens the
interval around it. Candombe's own slope chooses between the two saturated names
and gates nothing, since seven works are exploratory by registration.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .cache import _atomic_json, _outside_repository
from .data import file_sha256
from .summarise import bootstrap as work_bootstrap

SCHEMA = "tiktak.c1_summary/v1"
SEEDS = (17, 29, 43)
FRACTIONS = ("0.25", "0.50", "1.00")
DRAWS = 2000
MCID = 0.03
METRIC = "phase_f1"
CANDOMBE = "candombe"
# Fixed before the run: how much per-corpus evidence each corpus can carry.
CORPUS_STATUS = {"rwc2": "interval", "candombe": "exploratory",
                 "bpsd": "exploratory", "kraisler": "descriptive",
                 "rubato": "descriptive"}


def two_way_bootstrap(per_seed: dict[int, dict[str, float]]) -> dict:
    """The registered draw, in the registered order.

    Works ascend as byte strings, seeds keep their registered order, and one
    generator per draw yields the seed index *before* the work index. Written
    out rather than described because two summarisers that both look correct
    would otherwise report different intervals.
    """
    works = sorted(per_seed[SEEDS[0]], key=lambda name: name.encode("utf-8"))
    matrix = np.asarray(
        [[per_seed[seed][work] for work in works] for seed in SEEDS],
        dtype=np.float64)
    if matrix.shape != (len(SEEDS), len(works)):
        raise ValueError("C1 two-way bootstrap needs a full seed/work matrix")
    drawn = np.empty(DRAWS, dtype=np.float64)
    for draw in range(DRAWS):
        rng = np.random.default_rng(draw)
        seed_index = rng.integers(0, len(SEEDS), len(SEEDS))
        work_index = rng.integers(0, len(works), len(works))
        drawn[draw] = float(
            np.mean(np.mean(matrix[seed_index, :], axis=0)[work_index]))
    mean = float(np.mean(matrix))
    return {"mean": mean,
            "ci": [float(value) for value in np.percentile(drawn, [2.5, 97.5])],
            "works": len(works), "seeds": len(SEEDS), "draws": DRAWS}


def classify(interval: dict) -> str:
    """Non-overlapping by construction: an interval spanning the MCID is neither."""
    if interval["ci"][0] >= MCID:
        return "material"
    if interval["ci"][1] < MCID:
        return "saturated"
    return "inconclusive"


def _work_values(evaluation: dict, metric: str) -> dict[str, float]:
    return {work: float(values[metric])
            for work, values in evaluation["work_metrics"].items()}


def slopes(high: dict[int, dict], low: dict[int, dict], works: set[str] | None
           ) -> tuple[dict, dict, dict]:
    """Paired per-seed differences, then both intervals over the same values."""
    per_seed = {}
    for seed in SEEDS:
        top = _work_values(high[seed], METRIC)
        bottom = _work_values(low[seed], METRIC)
        if set(top) != set(bottom):
            raise ValueError(f"seed {seed}: work pairing differs between fractions")
        names = set(top) if works is None else set(top) & works
        per_seed[seed] = {work: top[work] - bottom[work] for work in names}
    averaged = {work: float(np.mean([per_seed[s][work] for s in SEEDS]))
                for work in per_seed[SEEDS[0]]}
    return (two_way_bootstrap(per_seed), work_bootstrap(averaged, draws=DRAWS),
            {str(s): float(np.mean(list(per_seed[s].values()))) for s in SEEDS})


def summarise(runs: dict[tuple[str, int], dict],
              evaluations: dict[tuple[str, int, str], dict]) -> dict:
    """`runs` and `evaluations` are keyed (fraction, seed) and (fraction, seed, kind)."""
    selected = {f: {s: evaluations[(f, s, "selected")] for s in SEEDS}
                for f in FRACTIONS}
    common = {f: {s: evaluations[(f, s, "common")] for s in SEEDS}
              for f in FRACTIONS}
    corpora = selected["1.00"][SEEDS[0]]["work_corpora"]
    non_candombe = {work for work, corpus in corpora.items()
                    if corpus != CANDOMBE}

    axes = {}
    for label, works in (("overall", None), ("non_candombe", non_candombe)):
        primary, work_only, per_seed = slopes(
            selected["1.00"], selected["0.50"], works)
        axes[label] = {
            "two_way": primary, "work_only": work_only,
            "per_seed": per_seed, "class": classify(primary),
        }
    shape = {label: slopes(selected["0.50"], selected["0.25"],
                           None if label == "overall" else non_candombe)[0]
             for label in ("overall", "non_candombe")}

    by_corpus = {}
    for corpus, status in sorted(CORPUS_STATUS.items()):
        works = {work for work, name in corpora.items() if name == corpus}
        primary, work_only, per_seed = slopes(
            selected["1.00"], selected["0.50"], works)
        by_corpus[corpus] = {
            "status": status, "works": len(works), "per_seed": per_seed,
            "two_way": primary if status != "descriptive" else None,
            "mean": primary["mean"],
        }

    selection = {f: {str(s): {
        "validation_points": sum(
            1 for row in runs[(f, s)]["history"] if "evaluation_sha256" in row),
        "eligible_points": sum(
            1 for row in runs[(f, s)]["history"] if row.get("eligible")),
        "selected_epoch": (None if runs[(f, s)].get("best") is None
                           else int(runs[(f, s)]["best"]["epoch"])),
    } for s in SEEDS} for f in FRACTIONS}

    # The deciding slope is the all-except-Candombe one. The overall slope is
    # still computed and reported and no longer gates: it is the one quantity a
    # single genre can contaminate, and it is insensitive in exactly that case,
    # because an effect confined to seven works of eighty-four moves the mean by
    # 7/84 of its size while resampling works widens the interval around it.
    common_primary, _, _ = slopes(common["1.00"], common["0.50"], non_candombe)
    deciding = axes["non_candombe"]["class"]
    selection_sensitive = classify(common_primary) != deciding

    # Candombe's own term is a label and not a gate: its slope is over seven
    # works, which the registration calls exploratory, so it may not turn a
    # saturated result into a growth one. It chooses which of the two saturated
    # names is used, and both carry the same consequence.
    candombe_climbing = by_corpus[CANDOMBE]["mean"] >= MCID
    if deciding == "inconclusive":
        verdict = "inconclusive"
    elif deciding == "material":
        verdict = "data_limited_under_fixed_recipe"
    elif candombe_climbing:
        verdict = "candombe_localized_growth"
    else:
        verdict = "saturated_at_mcid"
    if selection_sensitive:
        verdict = "selection_sensitive/inconclusive"

    # Never dropped: at a fixed epoch cap 100% receives about four times the
    # updates of 25%, so data volume and update count are confounded. Early
    # stopping is reported as a diagnostic and does not remove the suffix --
    # that needs a longer-schedule or compute-matched arm C1 does not run.
    reached_cap = {f: [s for s in SEEDS if not runs[(f, s)].get("stopped_early")]
                   for f in FRACTIONS}
    return {
        "schema": SCHEMA, "research_only": True,
        "mcid": MCID, "metric": METRIC,
        "axes": axes, "shape_50_minus_25": shape, "by_corpus": by_corpus,
        "selection_diagnostics": selection,
        "last_common_epoch": {"two_way": common_primary,
                              "class": classify(common_primary)},
        "selection_sensitive": selection_sensitive,
        "deciding_axis": "non_candombe",
        "candombe_label_only": {"mean": by_corpus[CANDOMBE]["mean"],
                                "climbing": candombe_climbing,
                                "gates": False},
        "update_confound": {"suffix": "under_fixed_recipe",
                            "unconditional": True,
                            "seeds_reaching_epoch_cap": reached_cap},
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        metavar="FRACTION=PATH",
                        help="nine times: fraction and its run output root")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        _outside_repository(args.output, repository)
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        roots: dict[tuple[str, int], pathlib.Path] = {}
        runs: dict[tuple[str, int], dict] = {}
        for item in args.run:
            fraction, _, raw = item.partition("=")
            if fraction not in FRACTIONS:
                raise ValueError(f"unregistered fraction {fraction!r}")
            root = pathlib.Path(raw)
            run = json.loads((root / "result.json").read_text(encoding="utf-8"))
            if run.get("provenance", {}).get("tree_clean") is not True:
                raise ValueError(f"{root.name}: run provenance is not clean")
            key = (fraction, int(run["seed"]))
            if key in runs:
                raise ValueError(f"duplicate run for {key}")
            roots[key], runs[key] = root, run
        if set(runs) != {(f, s) for f in FRACTIONS for s in SEEDS}:
            raise ValueError("C1 needs every fraction at every seed")

        def evaluated(key: tuple[str, int]) -> list[int]:
            return sorted(int(row["epoch"]) for row in runs[key]["history"]
                          if "evaluation_sha256" in row)

        def load(key: tuple[str, int], epoch: int) -> dict:
            path = (roots[key] / "candidates" / f"epoch-{epoch:03d}"
                    / "evaluation.json")
            return json.loads(path.read_text(encoding="utf-8"))

        ineligible = [key for key, run in runs.items()
                      if run.get("best") is None
                      or run.get("eligible_checkpoint") is False]
        if ineligible:
            _atomic_json(args.output, {
                "schema": SCHEMA, "research_only": True, "complete": False,
                "verdict": "inconclusive",
                "reason": "one or more runs had no beat-noninferior checkpoint",
                "ineligible_runs": [{"fraction": f, "seed": s}
                                    for f, s in sorted(ineligible)]})
            print(json.dumps({"verdict": "inconclusive",
                              "reason": "ineligible checkpoint"}))
            return 0

        evaluations = {}
        for key, run in runs.items():
            evaluations[(*key, "selected")] = load(
                key, int(run["best"]["epoch"]))
        for seed in SEEDS:
            # The last epoch *every* fraction evaluated at this seed. Patience
            # truncates unequally, which is the whole reason this endpoint is
            # registered.
            common_epochs = set(evaluated((FRACTIONS[0], seed)))
            for fraction in FRACTIONS[1:]:
                common_epochs &= set(evaluated((fraction, seed)))
            if not common_epochs:
                raise ValueError(f"seed {seed}: no shared validation epoch")
            shared = max(common_epochs)
            for fraction in FRACTIONS:
                evaluations[(fraction, seed, "common")] = load(
                    (fraction, seed), shared)

        result = summarise(runs, evaluations)
        result["sources"] = {
            f"{fraction}:{seed}": file_sha256(
                roots[(fraction, seed)] / "result.json")
            for fraction, seed in sorted(roots)}
        _atomic_json(args.output, result)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"verdict": result["verdict"],
                      "overall": result["axes"]["overall"]["class"],
                      "non_candombe": result["axes"]["non_candombe"]["class"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
