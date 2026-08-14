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
        value = base + rng.normal(0.0, jitter)
        # Real evaluations carry the whole registered set; a fixture with only
        # phase_f1 would have hidden that the secondary block reads the others.
        metrics[work] = {"phase_f1": value}
        metrics[work].update({name: value * 0.9
                              for name in c1s.DIAGNOSTIC_METRICS})
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


def _run(fraction, seed, *, arm=c1s.C1_ARM, c1_block="auto", clean=True,
         commit="c1-commit"):
    identity = {"schema": "tiktak.s1_checkpoint/v1", "source_sha256": "s",
                "split_sha256": "p", "cache_sha256": "c",
                "baseline_sha256": "b", "config": {"k": 1}, "commit": commit}
    if c1_block == "auto":
        c1_block = (None if fraction == "1.00" else
                    {"fraction": float(fraction),
                     "identity_sha256": f"id{fraction}",
                     "subset_sha256": "subset-sha"})
    if c1_block is not None:
        identity["c1"] = c1_block
    return {"schema": "tiktak.s1_training/v1", "complete": True, "arm": arm,
            "seed": seed, "provenance": {"tree_clean": clean},
            "identity": identity, "history": [], "best": {"epoch": 10}}


def _auth_inputs(**overrides):
    runs = {(f, s): _run(f, s) for f in c1s.FRACTIONS for s in c1s.SEEDS}
    runs.update(overrides.get("runs", {}))
    digests = {key: f"sha-{key[0]}-{key[1]}" for key in runs}
    subset = {"fractions": {f: {"identity_sha256": f"id{f}"}
                            for f in c1s.FRACTIONS}}
    anchor = {digests[("1.00", s)] for s in c1s.SEEDS}
    return runs, digests, subset, overrides.get("s1_sources", anchor)


SUBSET_SHA = "subset-sha"


def test_authentication_accepts_only_the_nine_registered_runs():
    c1s.authenticate(*_auth_inputs(), SUBSET_SHA)


def test_the_six_new_runs_must_share_one_commit():
    """The anchor's commit differs by design; the six are one experiment.

    Excluding `commit` from the shared block answered the anchor half and left
    this half unguarded, so six runs from six different trees would have passed.
    """
    runs, digests, subset, sources = _auth_inputs()
    identity = c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)
    assert identity["c1_commit"] == "c1-commit"

    runs[("0.50", 29)] = _run("0.50", 29, commit="another-commit")
    with pytest.raises(ValueError, match="must share one commit"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_a_run_trained_from_another_subset_artifact_is_refused():
    runs, digests, subset, sources = _auth_inputs()
    runs[("0.25", 17)]["identity"]["c1"]["subset_sha256"] = "elsewhere"
    with pytest.raises(ValueError, match="different subset artifact"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_an_arbitrary_clean_run_cannot_pose_as_the_anchor():
    """The gap that mattered: C1 never trains 1.00, so it cannot detect a fake
    anchor from its own outputs. Only the registered S1 summary can."""
    runs, digests, subset, _ = _auth_inputs()
    with pytest.raises(ValueError, match="registered S1 summary"):
        c1s.authenticate(runs, digests, subset, {"some-other-run"}, SUBSET_SHA)


def test_a_reset_arm_is_refused_at_any_fraction():
    runs, digests, subset, sources = _auth_inputs()
    runs[("0.50", 29)] = _run("0.50", 29, arm="A3_reset")
    with pytest.raises(ValueError, match="is not A3_stateful"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_the_anchor_must_not_carry_a_subset_and_a_fraction_must():
    runs, digests, subset, sources = _auth_inputs()
    runs[("1.00", 17)] = _run("1.00", 17, c1_block={"fraction": 1.0})
    with pytest.raises(ValueError, match="anchor must be an S1 run"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)

    runs, digests, subset, sources = _auth_inputs()
    runs[("0.25", 17)] = _run("0.25", 17, c1_block=None)
    with pytest.raises(ValueError, match="must record its subset"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_a_run_that_trained_another_subset_is_refused():
    runs, digests, subset, sources = _auth_inputs()
    runs[("0.25", 43)] = _run("0.25", 43, c1_block={
        "fraction": 0.25, "identity_sha256": "somethingelse"})
    with pytest.raises(ValueError, match="subset identity does not match"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_a_differing_recipe_is_refused_but_a_differing_commit_is_not():
    """The anchor ran at b12eea82 and the fractions run later, so requiring the
    commit to match would refuse the reuse the whole design rests on."""
    runs, digests, subset, sources = _auth_inputs()
    for seed in c1s.SEEDS:
        runs[("1.00", seed)]["identity"]["commit"] = "b12eea82"
    c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)

    runs[("0.50", 17)]["identity"]["cache_sha256"] = "another-cache"
    with pytest.raises(ValueError, match="identity differs"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_an_incomplete_or_dirty_run_is_refused():
    for field, value, message in (("complete", False, "not complete"),
                                  ("schema", "other", "not an S1 training")):
        runs, digests, subset, sources = _auth_inputs()
        runs[("0.25", 17)][field] = value
        with pytest.raises(ValueError, match=message):
            c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)
    runs, digests, subset, sources = _auth_inputs()
    runs[("0.50", 43)] = _run("0.50", 43, clean=False)
    with pytest.raises(ValueError, match="not clean"):
        c1s.authenticate(runs, digests, subset, sources, SUBSET_SHA)


def test_the_registered_secondary_set_is_reported_at_every_fraction():
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.25, "1.00": 0.30})
    result = c1s.summarise(runs, evaluations)
    assert set(result["secondary"]) == set(c1s.DIAGNOSTIC_METRICS)
    for metric, block in result["secondary"].items():
        assert set(block["level"]) == set(c1s.FRACTIONS), metric
        assert block["1.00-0.50"]["mean"] > 0, metric
        assert "vs_a0" not in block
    assert result["secondary_against_a0"] is False


def test_the_secondary_set_reaches_a0_when_the_baseline_is_given():
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.25, "1.00": 0.30})
    baseline = _evaluation(0.10, jitter=0.0, seed=99)
    result = c1s.summarise(runs, evaluations, baseline)
    assert result["secondary_against_a0"] is True
    assert result["secondary"]["beat_f1"]["vs_a0"]["1.00"] > 0


def test_a_missing_registered_metric_is_a_failure_not_a_silent_gap():
    runs, evaluations = _bundle({"0.25": 0.20, "0.50": 0.25, "1.00": 0.30})
    for seed in c1s.SEEDS:
        for values in evaluations[("1.00", seed, "selected")]["work_metrics"].values():
            values.pop("coverage")
    with pytest.raises(KeyError):
        c1s.summarise(runs, evaluations)
