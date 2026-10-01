"""The C1 subsets, and the ordering trap they exist to avoid.

The load-bearing test is `test_hash_order_would_have_changed_the_schedule`. It
does not check that the implementation is right; it checks that the mistake the
implementation avoids is a real mistake, so the invariant cannot later be
"simplified" away by someone who cannot see what it was for.
"""
import json

import pytest

from training.beatnet import c1_subsets as c1


def _cache(records_per_work=2):
    """A cache shaped like the real one, with the same corpus work counts."""
    records = []
    for corpus, works in sorted(c1.CORPUS_WORKS.items()):
        for index in range(works):
            work = f"{corpus}-work-{index:03d}"
            for take in range(records_per_work if corpus == "rubato" else 1):
                records.append({
                    "corpus": corpus, "name": f"{work}-take{take}",
                    "work_id": work, "split": "train",
                    "frames": 1000 + 37 * index + 11 * take,
                })
    # Pad to the registered record count with extra rubato performances, which
    # is where the real corpus's duplicates live.
    while len(records) < c1.TRAIN_RECORDS:
        index = len(records)
        records.append({
            "corpus": "rubato", "name": f"rubato-extra-{index}",
            "work_id": "rubato-work-000", "split": "train",
            "frames": 900 + index,
        })
    records.append({"corpus": "rwc2", "name": "held", "work_id": "dev-work",
                    "split": "dev", "frames": 1234})
    return {"records": records}


SUBSET_SHA = "b" * 64


def _valid_subset(cache, **overrides):
    """What the generator produces on the registered corpus, for filter tests."""
    payload = dict(c1.build(cache), total_frames=c1.REAL_TOTAL_FRAMES,
                   registered_corpus=True, frame_fraction_deviations={},
                   cache_sha256=c1.REGISTERED_CACHE_SHA256,
                   preflight={
                       "passed": True,
                       "seeds": list(c1.PREFLIGHT_SEEDS),
                       "blocks_compared": {
                           str(seed): c1.PREFLIGHT_BLOCKS_PER_SEED
                           for seed in c1.PREFLIGHT_SEEDS},
                   },
                   provenance={"tree_clean": True, "commit": "abc"})
    payload.update(overrides)
    return payload


def test_fractions_are_nested_stratified_and_close():
    cache = _cache()
    built = c1.build(cache)
    order = c1.work_order(c1.training_rows(cache))
    small = c1.members(order, 0.25)
    medium = c1.members(order, 0.50)
    full = c1.members(order, 1.00)
    assert small < medium < full
    for corpus, works in order.items():
        for fraction in (0.25, 0.50):
            got = len([w for w in c1.members(order, fraction) if w in works])
            assert got == c1.PREFIX[fraction][corpus], (corpus, fraction)
    assert built["fractions"]["1.00"]["records"] == c1.TRAIN_RECORDS
    assert built["fractions"]["1.00"]["works"] == c1.TRAIN_WORKS
    assert built["fractions"]["1.00"]["frame_fraction"] == 1.0


def test_the_registered_prefix_table_matches_a_half_up_reading():
    """The table is the specification; this only shows it is not eccentric.

    0.5 * 11 and 0.5 * 251 are exact ties, where Python's round is ties-to-even
    and most other languages are half-up. Tabulating removes the question, and
    pinning the totals here keeps the table honest to its own arithmetic.
    """
    import math
    for fraction, lengths in c1.PREFIX.items():
        for corpus, take in lengths.items():
            expected = math.floor(fraction * c1.CORPUS_WORKS[corpus] + 0.5)
            assert take == expected, (fraction, corpus)
    assert sum(c1.PREFIX[0.25].values()) == 83
    assert sum(c1.PREFIX[0.50].values()) == 166
    assert sum(c1.PREFIX[1.00].values()) == c1.TRAIN_WORKS


def test_the_membership_key_is_byte_exact():
    """Pins the construction the registration fixes, separator included."""
    import hashlib
    digest, tie = c1._order_key("rwc2", "RM-P001")
    raw = (b"tiktak-c1-v1" + bytes([0]) + b"rwc2" + bytes([0])
           + b"RM-P001")
    # The NUL separator is load-bearing: without it these two collide.
    assert digest == hashlib.sha256(raw).hexdigest()
    assert tie == b"RM-P001"
    assert c1._order_key("rwc2", "a")[0] != c1._order_key("rwc", "2a")[0]


def test_the_frame_axis_is_reported_and_is_not_the_work_axis():
    """Works differ in length, so 25% of works is not 25% of audio."""
    built = c1.build(_cache())
    quarter = built["fractions"]["0.25"]
    assert quarter["nominal_work_fraction"] == 0.25
    assert quarter["frame_fraction"] != pytest.approx(0.25, abs=1e-6)
    assert 0.0 < quarter["frame_fraction"] < 1.0


def test_selection_preserves_manifest_order_at_every_fraction():
    cache = _cache()
    rows = c1.training_rows(cache)
    order = c1.work_order(rows)
    for fraction in c1.FRACTIONS:
        selected = c1.subset_rows(rows, c1.members(order, fraction))
        positions = [rows.index(row) for row in selected]
        assert positions == sorted(positions), fraction


def test_full_selection_is_the_original_rows_and_schedule():
    cache = _cache()
    rows = c1.training_rows(cache)
    selected = c1.subset_rows(rows, c1.members(c1.work_order(rows), 1.00))
    assert [id(row) for row in selected] == [id(row) for row in rows]
    report = c1.assert_anchor_schedule(cache, seeds=(17, 18))
    assert report["passed"] and report["seeds"] == [17, 18]
    assert all(count > 0 for count in report["blocks_compared"].values())


def test_hash_order_would_have_changed_the_schedule():
    """Why membership order and emission order have to be different things.

    `contiguous_batches` permutes *positional* indices, so the same seed over a
    resorted list of the same recordings produces a different batch schedule.
    Sorting the rows by the membership hash -- the obvious simplification --
    would therefore break the 100% anchor while leaving every count, digest and
    corpus total looking correct.
    """
    cache = _cache()
    rows = c1.training_rows(cache)
    resorted = sorted(rows, key=lambda row: c1._order_key(
        row["corpus"], row["work_id"]))

    assert {(r["corpus"], r["name"]) for r in resorted} == {
        (r["corpus"], r["name"]) for r in rows}
    assert resorted != rows

    honest = c1.schedule_signature(rows, seed=17)
    sorted_by_hash = c1.schedule_signature(resorted, seed=17)
    assert len(honest) == len(sorted_by_hash)
    assert honest != sorted_by_hash


def test_a_changed_corpus_mix_is_refused():
    cache = _cache()
    for row in cache["records"]:
        if row["corpus"] == "kraisler":
            row["corpus"] = "rwc2"
    with pytest.raises(ValueError, match="corpus work counts"):
        c1.work_order(c1.training_rows(cache))


def test_the_binding_path_refuses_a_corpus_that_is_not_the_registered_one():
    """The guard a first version disarmed by trying to be convenient.

    Skipping the frame-share comparison whenever the total moved meant that
    changing a recording's length -- the case it exists for -- switched it off.
    `build` now always reports, and `require_registered_corpus` is what refuses,
    so a fixture can still be built while no binding artifact can be written.
    """
    fixture = c1.build(_cache())
    assert fixture["registered_corpus"] is False
    with pytest.raises(ValueError, match="is not the registered"):
        c1.require_registered_corpus(fixture)

    real = dict(fixture, total_frames=c1.REAL_TOTAL_FRAMES,
                registered_corpus=True, frame_fraction_deviations={})
    c1.require_registered_corpus(real)
    drifted = dict(real, frame_fraction_deviations={"0.25": 0.31})
    with pytest.raises(ValueError, match="frame shares differ"):
        c1.require_registered_corpus(drifted)


def test_build_is_json_serialisable_and_carries_digests():
    built = c1.build(_cache())
    json.dumps(built)
    digests = {block["identity_sha256"]
               for block in built["fractions"].values()}
    assert len(digests) == len(c1.FRACTIONS)


def test_an_s1_run_identity_is_unchanged_when_no_subset_is_given():
    """C1's anchor argument in one assertion.

    The 100% arm is the S1 runs reused, not repeated, so `run_training` without
    a subset must build exactly the identity S1 built. A field added
    unconditionally would have made every existing S1 checkpoint unresumable and
    quietly falsified the reuse.
    """
    # The only test in this file that genuinely needs the trainer, and so the
    # only one that may be skipped where torch is absent. The rest guard the
    # subset rules and now run everywhere.
    pytest.importorskip("torch")
    from training.beatnet.trainer import checkpoint_identity

    identity = checkpoint_identity(
        {"a": 1}, source_sha256="s", split_sha256="p", cache_sha256="c",
        commit="k")
    identity["arm"] = "A3_stateful"
    identity["seed"] = 17
    identity["baseline_sha256"] = "b"
    assert "c1" not in identity


def test_the_runner_filter_refuses_an_unregistered_fraction():
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    subset = _valid_subset(cache)
    rows = c1.training_rows(cache)
    for fraction in (0.75, 0.0):
        with pytest.raises((ValueError, KeyError)):
            c1_training_rows(subset, rows, fraction, arm=c1.C1_ARM,
                             cache_sha256=c1.REGISTERED_CACHE_SHA256, subset_sha256=SUBSET_SHA)
    with pytest.raises(ValueError, match="explicit --fraction"):
        c1_training_rows(subset, rows, None, arm=c1.C1_ARM, cache_sha256="x",
                         subset_sha256="s")


def test_a_subset_without_a_cache_digest_or_provenance_is_refused():
    """`not in (None, ...)` was fail-open: a subset that never recorded which
    cache it came from passed the check that exists to catch exactly that."""
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    rows = c1.training_rows(cache)
    bare = dict(c1.build(cache), total_frames=c1.REAL_TOTAL_FRAMES,
                registered_corpus=True, frame_fraction_deviations={})
    with pytest.raises(ValueError, match="no cache digest"):
        c1_training_rows(bare, rows, 0.25, arm=c1.C1_ARM,
                         cache_sha256=c1.REGISTERED_CACHE_SHA256,
                         subset_sha256=SUBSET_SHA)
    unprovenanced = dict(bare, cache_sha256=c1.REGISTERED_CACHE_SHA256)
    with pytest.raises(ValueError, match="no clean provenance"):
        c1_training_rows(unprovenanced, rows, 0.25, arm=c1.C1_ARM,
                         cache_sha256=c1.REGISTERED_CACHE_SHA256, subset_sha256=SUBSET_SHA)


def test_the_subset_digest_travels_into_run_identity():
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    _, identity = c1_training_rows(
        _valid_subset(cache), c1.training_rows(cache), 0.25,
        arm=c1.C1_ARM, cache_sha256=c1.REGISTERED_CACHE_SHA256,
        subset_sha256=SUBSET_SHA)
    assert identity["subset_sha256"] == SUBSET_SHA
    assert identity["cache_sha256"] == c1.REGISTERED_CACHE_SHA256


def test_the_registered_matrix_is_closed_by_the_runner():
    """Six runs, one arm. Nothing stopped a subset run training the anchor.

    Retraining 1.00 would quietly replace the S1 runs the six-run design exists
    to reuse, and A3_reset is not a C1 arm at all.
    """
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    subset = _valid_subset(cache)
    rows = c1.training_rows(cache)
    with pytest.raises(ValueError, match="registers only A3_stateful"):
        c1_training_rows(subset, rows, 0.25, arm="A3_reset", cache_sha256="x",
                         subset_sha256="s")
    with pytest.raises(ValueError, match="is the S1 anchor and is reused"):
        c1_training_rows(subset, rows, 1.00, arm=c1.C1_ARM, cache_sha256="x",
                         subset_sha256="s")


def test_a_subset_from_another_cache_is_refused():
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    subset = dict(_valid_subset(cache), cache_sha256="a" * 64)
    with pytest.raises(ValueError, match="cache digest"):
        c1_training_rows(subset, c1.training_rows(cache), 0.25,
                         arm=c1.C1_ARM, cache_sha256="b" * 64,
                         subset_sha256="s")


def test_the_runner_filter_catches_a_digest_that_does_not_match():
    """A filter returning the right works in the wrong order would pass counts."""
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    subset = _valid_subset(cache)
    subset["fractions"]["0.25"]["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match the subset artifact"):
        c1_training_rows(subset, c1.training_rows(cache), 0.25,
                         arm=c1.C1_ARM, cache_sha256=c1.REGISTERED_CACHE_SHA256, subset_sha256=SUBSET_SHA)


def test_a_cache_that_is_not_the_registered_one_is_refused_at_training_time():
    """The digest was pinned where subsets are built, not where they are used.

    The runner required the subset and the cache to agree with each other and
    with nothing else, so another cache plus a subset generated from it agreed
    and passed.
    """
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    other = "c" * 64
    subset = dict(_valid_subset(cache), cache_sha256=other)
    with pytest.raises(ValueError, match="is not the registered"):
        c1_training_rows(subset, c1.training_rows(cache), 0.25,
                         arm=c1.C1_ARM, cache_sha256=other,
                         subset_sha256=SUBSET_SHA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("schema", None, "schema"),
     ("schema", "tiktak.c1_subsets/v0", "schema"),
     ("salt", None, "membership salt"),
     ("salt", "another-order", "membership salt")),
)
def test_subset_schema_and_membership_salt_are_identity(field, value, message):
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    subset = _valid_subset(cache)
    if value is None:
        subset.pop(field)
    else:
        subset[field] = value
    with pytest.raises(ValueError, match=message):
        c1_training_rows(
            subset, c1.training_rows(cache), 0.25, arm=c1.C1_ARM,
            cache_sha256=c1.REGISTERED_CACHE_SHA256,
            subset_sha256=SUBSET_SHA)


@pytest.mark.parametrize("digest", [None, "", "not-a-digest", "a" * 63])
def test_a_run_without_the_subsets_own_digest_never_starts(digest):
    """`or ""` let it train for hours and be caught only at summary time.

    Worse, identity carried the empty value, so resume stayed self-consistent
    and the run would have completed.
    """
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    with pytest.raises(ValueError, match="subset artifact's own digest"):
        c1_training_rows(_valid_subset(cache), c1.training_rows(cache), 0.25,
                         arm=c1.C1_ARM,
                         cache_sha256=c1.REGISTERED_CACHE_SHA256,
                         subset_sha256=digest)


def test_the_preflight_covers_every_training_seed_not_only_the_first():
    """`run.py` advances the scheduler seed as `seed + epoch`.

    The seeds actually used are 17..66, 29..78 and 43..92, so checking 17, 18,
    19 exercised the first training seed and left the other two unvisited.
    """
    assert set(c1.PREFLIGHT_SEEDS) >= {17, 29, 43}
    for seed in (17, 29, 43):
        assert seed + 1 in c1.PREFLIGHT_SEEDS


def test_training_requires_evidence_that_the_preflight_ran():
    """The old check sat under `if fraction == 1.00`, which stopped being
    reachable the moment 1.00 was refused -- a guard deleted into silence.

    The comparison still runs, in the generator; the training path needs the
    artifact to carry its result rather than repeat 26,000 blocks per job.
    """
    from training.beatnet.c1_subsets import c1_training_rows

    cache = _cache()
    rows = c1.training_rows(cache)
    valid = _valid_subset(cache)["preflight"]
    for broken in (
            {},
            {"passed": False, "seeds": list(c1.PREFLIGHT_SEEDS),
             "blocks_compared": valid["blocks_compared"]},
            {"passed": True, "seeds": [17],
             "blocks_compared": {"17": c1.PREFLIGHT_BLOCKS_PER_SEED}},
            {**valid, "blocks_compared": {
                **valid["blocks_compared"], "17": 1}},
    ):
        subset = _valid_subset(cache, preflight=broken)
        with pytest.raises(
                ValueError, match="no passing registered schedule preflight"):
            c1_training_rows(subset, rows, 0.25, arm=c1.C1_ARM,
                             cache_sha256=c1.REGISTERED_CACHE_SHA256,
                             subset_sha256=SUBSET_SHA)


def test_a_run_says_which_experiment_it_is():
    """Every C1 artifact shipped labelled `S1`, because the label was a literal.

    Nothing was ambiguous -- identity, digests and the `c1` block were right, so
    the run stood -- but the artifact made a provenance claim that was false and
    happened not to matter. The label now follows the same condition that adds
    the `c1` identity block, so the two cannot disagree.
    """
    pytest.importorskip("torch")
    from training.beatnet.run import experiment_label

    assert experiment_label(None) == "S1"
    assert experiment_label({"fraction": 0.25}) == "C1"
    assert experiment_label({}) == "C1"
