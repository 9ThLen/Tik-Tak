"""The C1 verdict: two axes, a specified bootstrap, and a selection override.

Every test here is about a way the summary could look right and decide wrong.
"""
import numpy as np
import pytest

from training.beatnet import c1_summarise as c1s

WORKS = {f"rwc2::w{i:03d}": "rwc2" for i in range(63)}
WORKS.update({f"candombe::w{i}": "candombe" for i in range(7)})
WORKS.update({f"bpsd::w{i}": "bpsd" for i in range(7)})
WORKS.update({f"kraisler::w{i}": "kraisler" for i in range(4)})
WORKS.update({f"rubato::w{i}": "rubato" for i in range(3)})


def _evaluation(level, *, candombe_level=None, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    metrics = {}
    for work, corpus in WORKS.items():
        base = candombe_level if (corpus == "candombe"
                                  and candombe_level is not None) else level
        metrics[work] = {"phase_f1": base + rng.normal(0.0, jitter)}
    return {"work_metrics": metrics, "work_corpora": dict(WORKS),
            "dev_works": len(WORKS)}


def _runs(points=(5, 10), best=10):
    history = [{"epoch": e, "evaluation_sha256": f"d{e}", "eligible": True}
               for e in points]
    return {"history": history, "best": {"epoch": best}, "seed": 0,
            "stopped_early": True}


def _bundle(levels, *, candombe=None, jitter=0.002):
    runs, evaluations = {}, {}
    for index, (fraction, level) in enumerate(sorted(levels.items())):
        for seed in c1s.SEEDS:
            runs[(fraction, seed)] = _runs()
            # A distinct stream per (fraction, seed). Reusing one per seed made
            # the noise cancel in the difference, so both intervals collapsed to
            # zero width and the fixture proved nothing.
            payload = _evaluation(
                level, candombe_level=(None if candombe is None
                                       else candombe[fraction]),
                jitter=jitter, seed=1000 * index + seed)
            evaluations[(fraction, seed, "selected")] = payload
            evaluations[(fraction, seed, "common")] = payload
    return runs, evaluations


def test_the_two_way_draw_is_the_registered_one():
    """Seed index before work index, one generator per draw, fixed work order."""
    per_seed = {seed: {work: 0.1 * (index + seed)
                       for index, work in enumerate(sorted(WORKS))}
                for seed in c1s.SEEDS}
    got = c1s.two_way_bootstrap(per_seed)

    works = sorted(per_seed[c1s.SEEDS[0]], key=lambda n: n.encode("utf-8"))
    matrix = np.asarray([[per_seed[s][w] for w in works] for s in c1s.SEEDS])
    expected = np.empty(c1s.DRAWS)
    for draw in range(c1s.DRAWS):
        rng = np.random.default_rng(draw)
        seed_index = rng.integers(0, len(c1s.SEEDS), len(c1s.SEEDS))
        work_index = rng.integers(0, len(works), len(works))
        expected[draw] = np.mean(np.mean(matrix[seed_index, :], 0)[work_index])
    assert got["ci"] == [float(v) for v in np.percentile(expected, [2.5, 97.5])]

    # Drawing works before seeds is a different, equally plausible reading.
    other = np.empty(c1s.DRAWS)
    for draw in range(c1s.DRAWS):
        rng = np.random.default_rng(draw)
        work_index = rng.integers(0, len(works), len(works))
        seed_index = rng.integers(0, len(c1s.SEEDS), len(c1s.SEEDS))
        other[draw] = np.mean(np.mean(matrix[seed_index, :], 0)[work_index])
    assert got["ci"] != [float(v) for v in np.percentile(other, [2.5, 97.5])]


def test_the_two_way_interval_is_wider_than_the_work_only_one():
    """Which is the reason for it: work-only is conditional on three models."""
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.30, "1.00": 0.34})
    result = c1s.summarise(runs, evaluations)
    axis = result["axes"]["overall"]
    two_way = axis["two_way"]["ci"][1] - axis["two_way"]["ci"][0]
    work_only = axis["work_only"]["ci"][1] - axis["work_only"]["ci"][0]
    assert two_way > work_only


def test_classes_do_not_overlap_at_the_mcid():
    assert c1s.classify({"ci": [0.03, 0.09]}) == "material"
    assert c1s.classify({"ci": [0.031, 0.09]}) == "material"
    assert c1s.classify({"ci": [-0.01, 0.0299]}) == "saturated"
    assert c1s.classify({"ci": [0.01, 0.05]}) == "inconclusive"
    assert c1s.classify({"ci": [-0.2, 0.03]}) == "inconclusive"


def test_candombe_can_carry_the_overall_slope_and_is_named_for_it():
    """The outcome the second axis exists to separate.

    Everything but Candombe is flat; Candombe alone still climbs. The overall
    slope is dragged up by seven works of eighty-four, and a verdict on that
    slope alone would read as though more data of this distribution helps.
    """
    runs, evaluations = _bundle(
        {"0.25": 0.30, "0.50": 0.30, "1.00": 0.30},
        candombe={"0.25": 0.03, "0.50": 0.05, "1.00": 0.95}, jitter=0.001)
    result = c1s.summarise(runs, evaluations)
    assert result["axes"]["overall"]["class"] == "material"
    assert result["axes"]["non_candombe"]["class"] == "saturated"
    assert result["verdict"] == "candombe_localized_growth"


def test_a_moderate_candombe_only_effect_is_now_named_not_swallowed():
    """What changing the deciding axis bought.

    Under the earlier table the overall slope gated, and an effect confined to
    seven works of eighty-four could not push its lower bound past the MCID
    unless the step approached 0.90 -- so a moderate Candombe-only signal came
    out `inconclusive`, which is honest but says nothing about one genre still
    climbing while the rest has stopped. The deciding slope is now the
    all-except-Candombe one, and the same input is named.
    """
    runs, evaluations = _bundle(
        {"0.25": 0.30, "0.50": 0.30, "1.00": 0.30},
        candombe={"0.25": 0.05, "0.50": 0.20, "1.00": 0.95}, jitter=0.001)
    result = c1s.summarise(runs, evaluations)
    assert result["axes"]["overall"]["class"] == "inconclusive"
    assert result["axes"]["non_candombe"]["class"] == "saturated"
    assert result["verdict"] == "candombe_localized_growth"
    assert result["deciding_axis"] == "non_candombe"


def test_candombe_labels_the_result_and_cannot_gate_it():
    """Seven works may choose a name; they may not change a consequence.

    Both saturated names carry the same action -- neither justifies sizing
    P1-B1 to extend this distribution -- so a Candombe slope that is exploratory
    by registration is allowed to distinguish them and nothing more.
    """
    flat = c1s.summarise(*_bundle(
        {"0.25": 0.30, "0.50": 0.30, "1.00": 0.30},
        candombe={"0.25": 0.90, "0.50": 0.95, "1.00": 0.95}, jitter=0.001))
    assert flat["verdict"] == "saturated_at_mcid"
    assert flat["candombe_label_only"]["gates"] is False

    climbing = c1s.summarise(*_bundle(
        {"0.25": 0.30, "0.50": 0.30, "1.00": 0.30},
        candombe={"0.25": 0.05, "0.50": 0.20, "1.00": 0.95}, jitter=0.001))
    assert climbing["verdict"] == "candombe_localized_growth"
    # Candombe climbing hard cannot turn a saturated rest into growth.
    assert climbing["axes"]["non_candombe"]["class"] == "saturated"


def test_growth_everywhere_is_data_limited_and_keeps_its_suffix():
    runs, evaluations = _bundle(
        {"0.25": 0.10, "0.50": 0.20, "1.00": 0.30}, jitter=0.001)
    result = c1s.summarise(runs, evaluations)
    assert result["verdict"] == "data_limited_under_fixed_recipe"
    assert result["update_confound"]["unconditional"] is True


def test_a_flat_curve_saturates_and_a_noisy_one_does_not_decide():
    flat = c1s.summarise(*_bundle(
        {"0.25": 0.30, "0.50": 0.30, "1.00": 0.30}, jitter=0.001))
    assert flat["verdict"] == "saturated_at_mcid"
    noisy = c1s.summarise(*_bundle(
        {"0.25": 0.10, "0.50": 0.20, "1.00": 0.23}, jitter=0.30))
    assert noisy["verdict"] == "inconclusive"


def test_selection_disagreement_overrides_any_verdict():
    """Checkpoint choice may not be what decides a curve."""
    runs, evaluations = _bundle(
        {"0.25": 0.10, "0.50": 0.20, "1.00": 0.30}, jitter=0.001)
    for seed in c1s.SEEDS:
        # At the last common epoch the fractions are level; only the selected
        # checkpoints differ.
        evaluations[("1.00", seed, "common")] = _evaluation(
            0.20, jitter=0.001, seed=seed)
    result = c1s.summarise(runs, evaluations)
    assert result["selection_sensitive"] is True
    assert result["verdict"] == "selection_sensitive/inconclusive"
    assert result["last_common_epoch"]["class"] == "saturated"


def test_descriptive_corpora_get_no_interval():
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.25, "1.00": 0.30})
    result = c1s.summarise(runs, evaluations)
    assert result["by_corpus"]["rwc2"]["status"] == "interval"
    assert result["by_corpus"]["rwc2"]["two_way"] is not None
    for corpus in ("rubato", "kraisler"):
        assert result["by_corpus"][corpus]["status"] == "descriptive"
        assert result["by_corpus"][corpus]["two_way"] is None


def test_selection_diagnostics_and_epoch_cap_are_reported():
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.25, "1.00": 0.30})
    runs[("0.25", 17)]["stopped_early"] = False
    result = c1s.summarise(runs, evaluations)
    assert result["selection_diagnostics"]["1.00"]["17"]["validation_points"] == 2
    assert result["selection_diagnostics"]["1.00"]["17"]["selected_epoch"] == 10
    assert result["update_confound"]["seeds_reaching_epoch_cap"]["0.25"] == [17]


def test_a_missing_work_in_one_fraction_is_refused():
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.25, "1.00": 0.30})
    dropped = dict(evaluations[("1.00", 17, "selected")])
    dropped["work_metrics"] = {k: v for k, v in
                               dropped["work_metrics"].items()
                               if k != "rwc2::w000"}
    evaluations[("1.00", 17, "selected")] = dropped
    with pytest.raises(ValueError, match="work pairing differs"):
        c1s.summarise(runs, evaluations)
