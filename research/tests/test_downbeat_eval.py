import numpy as np
import pytest

import eval.downbeat_benchmark as benchmark
from eval.analysis import Estimate
from eval.annotations import Reference
from eval.downbeat import (
    SweepRow,
    Verdict,
    choose_margins,
    eligible,
    evidence_gap,
    grouped_verdicts,
    score,
    sweep,
    thresholds_from,
    wilson_upper,
)
from eval.downbeat_benchmark import (
    group_manifest_path,
    held_out_error_report,
    load_groups,
    split_of,
)


def reference(beats_per_bar=4, bars=8, bpm=120.0, phase=0):
    """Ground truth for an exact grid, optionally starting mid-bar."""
    gap = 60.0 / bpm
    beats = np.arange(bars * beats_per_bar) * gap
    downbeats = beats[(np.arange(len(beats)) + phase) % beats_per_bar == 0]
    return Reference(
        beats=beats,
        downbeats=downbeats,
        beats_per_bar=beats_per_bar,
        name="clip",
        bar_lengths=(beats_per_bar,) * max(len(downbeats) - 1, 0),
    )


def estimate_from(ref, beats_per_bar=None, phase=0, margin=1.0, meter_margin=None,
                  strength=1.0, shift=0.0, confident=True):
    beats_per_bar = beats_per_bar or ref.beats_per_bar
    beats = ref.beats + shift
    downbeats = beats[(np.arange(len(beats)) - phase) % beats_per_bar == 0]
    return Estimate(
        beats=beats,
        downbeats=downbeats,
        beats_per_bar=beats_per_bar,
        downbeat_strength=strength,
        downbeat_phase_margin=margin,
        downbeat_meter_margin=margin if meter_margin is None else meter_margin,
        downbeat_confident=confident,
    )


def test_a_perfect_answer_scores_perfectly():
    ref = reference()
    result = score(ref, estimate_from(ref))

    assert result.beat_f == pytest.approx(1.0)
    assert result.downbeat_f == pytest.approx(1.0)
    assert result.meter_correct is True
    assert result.phase_correct is True
    assert result.verdict(0.25, 0.25) == Verdict.CORRECT


def test_the_bar_line_on_the_wrong_beat_is_a_phase_error():
    # The error a listener notices immediately, and the one the whole margin
    # threshold exists to avoid showing.
    ref = reference()
    result = score(ref, estimate_from(ref, phase=2))

    assert result.meter_correct is True
    assert result.phase_correct is False
    assert result.verdict(0.25, 0.25) == Verdict.WRONG_PHASE


def test_calling_four_four_two_four_is_caught_even_though_the_f_measure_looks_fine():
    # This is why the metre is scored separately. Every real bar line is hit, so
    # recall is perfect and only precision suffers — the F-measure lands around
    # 0.67, which reads like a passable result for an answer that is wrong.
    ref = reference(beats_per_bar=4)
    result = score(ref, estimate_from(ref, beats_per_bar=2))

    assert result.downbeat_f > 0.6
    assert result.meter_correct is False
    assert result.verdict(0.25, 0.25) == Verdict.WRONG_METER


def test_phase_is_not_reported_when_the_metre_is_already_wrong():
    ref = reference(beats_per_bar=4)
    result = score(ref, estimate_from(ref, beats_per_bar=3))
    assert result.meter_correct is False
    assert result.phase_correct is None


def test_a_bar_line_within_the_tolerance_still_counts():
    ref = reference()
    assert score(ref, estimate_from(ref, shift=0.05)).phase_correct is True
    assert score(ref, estimate_from(ref, shift=0.15)).phase_correct is False


def test_finding_nothing_is_not_the_same_as_being_wrong():
    ref = reference()
    nothing = Estimate(beats=ref.beats, downbeats=np.zeros(0), beats_per_bar=0,
                       downbeat_strength=0.0, downbeat_phase_margin=0.0,
                       downbeat_meter_margin=0.0)
    result = score(ref, nothing)

    assert result.meter_correct is None
    assert result.verdict(0.25, 0.25) == Verdict.NO_ANSWER
    assert result.verdict(0.0, 0.0) == Verdict.NO_ANSWER


def test_a_low_margin_withholds_rather_than_answers_wrong():
    ref = reference()
    wrong = estimate_from(ref, phase=1, margin=0.1)

    assert score(ref, wrong).verdict(0.25, 0.25) == Verdict.WITHHELD
    assert score(ref, wrong).verdict(0.05, 0.05) == Verdict.WRONG_PHASE


def test_the_shipped_verdict_uses_the_cores_decision_not_a_python_threshold_copy():
    ref = reference()
    estimate = estimate_from(ref, margin=9.0, meter_margin=9.0, confident=False)
    result = score(ref, estimate)

    assert result.verdict(0.0, 0.0) == Verdict.CORRECT
    assert result.shipped_verdict() == Verdict.WITHHELD


def test_a_reference_without_bar_lines_is_scored_on_beats_only():
    ref = Reference(beats=np.arange(32) * 0.5, downbeats=np.zeros(0),
                    beats_per_bar=0, name="tapped")
    result = score(ref, estimate_from(reference()))

    assert not result.scorable
    assert np.isnan(result.downbeat_f)
    assert result.meter_correct is None
    assert result.beat_f == pytest.approx(1.0)


def test_unscorable_clips_are_left_out_of_the_sweep():
    ref = reference()
    good = score(ref, estimate_from(ref))
    tapped = score(
        Reference(beats=ref.beats, downbeats=np.zeros(0), beats_per_bar=0, name="tapped"),
        estimate_from(ref),
    )
    rows = sweep([good, tapped], [0.0], [0.0])
    assert rows[0].n == 1


# ------------------------------------------------------------------- sweep --

def _scores(spec):
    """spec: list of (margin, phase_offset, metre) tuples."""
    out = []
    for i, (margin, phase, metre) in enumerate(spec):
        ref = reference()
        est = estimate_from(ref, beats_per_bar=metre, phase=phase, margin=margin)
        result = score(ref, est, name=f"clip{i}")
        out.append(result)
    return out


def test_raising_the_threshold_trades_coverage_for_safety():
    scores = _scores([
        (1.0, 0, 4),    # right, confident
        (0.8, 0, 4),    # right
        (0.3, 2, 4),    # wrong phase, and not very confident about it
        (0.1, 0, 2),    # wrong metre, barely confident at all
    ])
    low, high = sweep(scores, [0.0, 0.5], [0.0])

    assert low.shown == 4 and low.wrong_rate == pytest.approx(0.5)
    assert high.shown == 2 and high.wrong_rate == pytest.approx(0.0)
    assert high.coverage < low.coverage
    assert high.precision > low.precision


def test_the_counts_account_for_every_clip():
    scores = _scores([(1.0, 0, 4), (0.3, 2, 4), (0.1, 0, 2)])
    for row in sweep(scores, [0.0, 0.2, 0.5, 1.5], [0.0]):
        total = (row.correct + row.wrong_meter + row.wrong_phase
                 + row.withheld + row.no_answer)
        assert total == row.n
        assert row.shown == row.correct + row.wrong_meter + row.wrong_phase


def test_many_excerpts_from_one_group_are_one_wilson_trial():
    scores = _scores([(1.0, 0, 4)] * 50)
    for clip in scores:
        clip.group_id = "same-song-and-session"

    row = sweep(scores, [0.0], [0.0])[0]

    assert row.n == 1
    assert row.shown == 1
    assert row.correct == 1


def test_any_wrong_excerpt_makes_the_independent_group_wrong():
    scores = _scores([
        (1.0, 0, 4),
        (1.0, 2, 4),
    ])
    for clip in scores:
        clip.group_id = "same-backing-track"

    row = sweep(scores, [0.0], [0.0])[0]

    assert row.n == 1
    assert row.shown == 1
    assert row.correct == 0
    assert row.wrong_phase == 1


def test_held_out_accounting_uses_the_same_independent_group_outcomes():
    scores = _scores([
        (1.0, 0, 4),
        (1.0, 2, 4),
    ])
    for clip in scores:
        clip.group_id = "one-held-out-session"

    verdicts = grouped_verdicts(scores, 0.0, 0.0)
    shown = sum(verdict in Verdict.SHOWN for verdict in verdicts)
    wrong = sum(verdict in Verdict.WRONG for verdict in verdicts)
    report = held_out_error_report(wrong, len(verdicts), shown)

    assert verdicts == [Verdict.WRONG_PHASE]
    assert "wrong rate 100.00%" in report


def test_wrong_metre_is_the_diagnostic_for_a_group_with_both_wrong_kinds():
    scores = _scores([
        (1.0, 2, 4),
        (1.0, 0, 3),
    ])
    for clip in scores:
        clip.group_id = "same-song"

    row = sweep(scores, [0.0], [0.0])[0]

    assert row.n == 1
    assert row.wrong_meter == 1
    assert row.wrong_phase == 0


def test_the_threshold_is_the_most_generous_one_inside_the_budget():
    # Two thresholds are safe; the useful one is the lower, because among
    # equally safe options the one that answers more often is the better product.
    # bounded=False because this is about which safe row wins, not about
    # whether three clips can demonstrate anything. They cannot, and the
    # bounded chooser correctly says so; see the tests below.
    scores = _scores([(1.0, 0, 4), (0.9, 0, 4), (0.2, 2, 4)])
    rows = sweep(scores, [0.0, 0.5, 0.95], [0.0])
    chosen = choose_margins(rows, max_wrong_rate=0.0, bounded=False)

    assert chosen is not None
    assert chosen.min_phase_margin == pytest.approx(0.5)
    assert chosen.coverage == pytest.approx(2 / 3)


def test_no_acceptable_threshold_is_an_answer_and_not_a_crash():
    # Every clip is wrong with full confidence: there is no threshold that makes
    # the feature safe, and saying so beats raising the budget until it passes.
    scores = _scores([(1.0, 2, 4), (1.0, 1, 4)])
    assert choose_margins(sweep(scores, [0.0, 0.5, 1.5], [0.0]), 0.05) is None


def test_a_threshold_that_never_fires_is_not_a_calibration():
    # It has a perfect wrong rate because it shows nothing, and would otherwise
    # win the budget outright while describing the feature being switched off.
    scores = _scores([(0.2, 0, 4), (0.2, 2, 4)])
    rows = sweep(scores, [0.9], [0.0])
    assert rows[0].shown == 0
    assert choose_margins(rows, max_wrong_rate=0.0) is None


def test_choosing_on_an_empty_split_returns_nothing():
    assert choose_margins(sweep([], [0.0, 0.5], [0.0]), 0.05) is None


def test_precision_can_be_bought_with_coverage_which_is_why_both_are_reported():
    scores = _scores([(1.0, 0, 4)] + [(0.2, 2, 4)] * 5)
    rows = {r.min_phase_margin: r for r in sweep(scores, [0.0, 0.5], [0.0])}

    assert rows[0.5].precision == pytest.approx(1.0)
    assert rows[0.5].coverage == pytest.approx(1 / 6)
    assert rows[0.0].wrong_rate == pytest.approx(5 / 6)


# ------------------------------------------------------------------ splits --

def test_a_clip_stays_in_its_split_when_the_set_grows():
    # The point of hashing the name instead of shuffling: a threshold chosen
    # last month and a test score computed today must still be about disjoint
    # material, even though ten recordings have been added since.
    before = {name: split_of(name, 0.35) for name in ("a", "b", "c")}
    after = {name: split_of(name, 0.35)
             for name in ("a", "b", "c", "d", "e", "f", "g")}
    assert all(after[name] == split_of(name, 0.35) for name in before)


def test_the_split_is_roughly_the_fraction_asked_for():
    names = [f"clip-{i:03d}" for i in range(1000)]
    held = sum(split_of(name, 0.35) == "test" for name in names)
    assert 0.30 < held / len(names) < 0.40


def test_everything_is_one_split_or_the_other():
    assert all(split_of(f"c{i}", 0.5) in ("test", "validation") for i in range(50))


def test_related_excerpts_cannot_leak_across_splits():
    names = [f"clip-{i}" for i in range(100)]
    test_name = next(name for name in names if split_of(name, 0.35) == "test")
    validation_name = next(
        name for name in names if split_of(name, 0.35) == "validation"
    )
    assert split_of(test_name, 0.35) != split_of(validation_name, 0.35)

    assert (
        split_of(test_name, 0.35, group_id="same-song-and-session")
        == split_of(validation_name, 0.35, group_id="same-song-and-session")
    )


def test_group_manifest_is_path_stable_and_backwards_compatible(tmp_path):
    manifest = tmp_path / "groups.json"
    manifest.write_text(
        '{"set\\\\take-1": "song-a", "set/take-2": "song-a"}',
        encoding="utf-8",
    )

    groups = load_groups(manifest)

    assert groups["set/take-1"] == groups["set/take-2"] == "song-a"
    assert split_of("set\\take-1", 0.35) == split_of("set/take-1", 0.35)
    assert split_of("clip", 0.35) == split_of(
        "clip", 0.35, group_id=None
    )


def test_a_real_held_out_split_requires_independent_groups(tmp_path):
    with pytest.raises(ValueError, match="group manifest is required"):
        group_manifest_path(tmp_path, requested=None, no_split=False)

    assert group_manifest_path(
        tmp_path, requested=None, no_split=True
    ) is None

    manifest = tmp_path / "groups.json"
    manifest.write_text("{}", encoding="utf-8")
    assert group_manifest_path(
        tmp_path, requested=None, no_split=False
    ) == manifest


@pytest.mark.parametrize("fraction", ["0", "-0.1", "1", "nan", "inf"])
def test_test_fraction_must_be_finite_and_strictly_inside_unit_interval(
        fraction, capsys):
    with pytest.raises(SystemExit) as stopped:
        benchmark.main(["--test-fraction", fraction])

    assert stopped.value.code == 2
    assert "finite and strictly between 0 and 1" in capsys.readouterr().err


def test_a_realized_empty_split_refuses_calibration(monkeypatch, capsys):
    ref = reference()

    class FakeAnalyser:
        def __init__(self, _binary):
            self.available = True

        def analyse_audio(self, _audio, _sample_rate):
            return estimate_from(ref)

    monkeypatch.setattr(benchmark, "Analyser", FakeAnalyser)
    monkeypatch.setattr(
        benchmark,
        "synthetic_cases",
        lambda: [(ref, np.zeros(32, dtype=np.float32), 48_000)],
    )

    assert benchmark.main([]) == 2
    output = capsys.readouterr().out
    assert "cannot calibrate" in output
    assert "0 validation and 1 test group(s)" in output or (
        "1 validation and 0 test group(s)" in output
    )


def test_no_split_reports_the_frontier_but_never_calibrates(
        monkeypatch, capsys):
    ref = reference()

    class FakeAnalyser:
        def __init__(self, _binary):
            self.available = True

        def analyse_audio(self, _audio, _sample_rate):
            return estimate_from(ref)

    monkeypatch.setattr(benchmark, "Analyser", FakeAnalyser)
    monkeypatch.setattr(
        benchmark,
        "synthetic_cases",
        lambda: [(ref, np.zeros(32, dtype=np.float32), 48_000)],
    )

    assert benchmark.main(["--no-split"]) == 0
    output = capsys.readouterr().out
    assert "--- all clips" in output
    assert "exploratory --no-split run" in output
    assert "no threshold is calibrated" in output
    assert "chosen on" not in output


def test_group_id_is_propagated_from_reference_to_score():
    ref = reference()
    ref.group_id = "song/session"

    result = score(ref, estimate_from(ref))

    assert result.group_id == "song/session"


# ------------------------------------------------------- the second doubt --

def test_a_settled_phase_inside_the_wrong_metre_is_still_withheld():
    # The failure that motivated splitting the two: a 4/4 track read as three,
    # with the phase inside that wrong metre entirely unambiguous. One threshold
    # on the phase margin cannot express the doubt, because there is none — the
    # doubt is about the bar length.
    ref = reference(beats_per_bar=4)
    est = estimate_from(ref, beats_per_bar=3, margin=0.69, meter_margin=0.05)
    result = score(ref, est)

    assert result.verdict(0.25, 0.0) == Verdict.WRONG_METER, "phase alone lets it through"
    assert result.verdict(0.25, 0.40) == Verdict.WITHHELD, "the metre margin catches it"


def test_both_thresholds_have_to_be_cleared():
    ref = reference()
    good = estimate_from(ref, margin=1.0, meter_margin=1.0)
    shaky_phase = estimate_from(ref, margin=0.05, meter_margin=1.0)
    shaky_metre = estimate_from(ref, margin=1.0, meter_margin=0.05)

    assert score(ref, good).verdict(0.25, 0.25) == Verdict.CORRECT
    assert score(ref, shaky_phase).verdict(0.25, 0.25) == Verdict.WITHHELD
    assert score(ref, shaky_metre).verdict(0.25, 0.25) == Verdict.WITHHELD


def test_the_conditional_error_bound_rejects_a_narrow_but_unreliable_threshold():
    # Nineteen clips withheld, one accented and wrong. The wrong rate is 5% and
    # passes a 5% budget, but every accent the user ever sees is wrong.
    scores = _scores([(1.0, 2, 4)] + [(0.1, 0, 4)] * 19)
    rows = sweep(scores, [0.5], [0.0])

    assert rows[0].wrong_rate == pytest.approx(0.05)
    assert rows[0].conditional_error == pytest.approx(1.0)
    assert choose_margins(rows, max_wrong_rate=0.05, max_conditional_error=0.10,
                          bounded=False) is None
    assert choose_margins(rows, max_wrong_rate=0.05, max_conditional_error=1.0,
                          bounded=False) is not None


# -------------------------------------------------------------- thresholds --

def test_candidate_thresholds_come_from_the_margins_actually_seen():
    # A fixed 0…1 grid wasted most of its rows below the data and truncated
    # everything above 1.0 into one bucket, hiding the top of the curve.
    candidates = thresholds_from([0.02, 0.04, 1.2, 2.6, 3.57])
    assert min(candidates) == 0.0
    assert max(candidates) > 3.57, "there is a threshold that withholds everything"
    assert any(c > 1.0 for c in candidates), "the range above 1.0 is represented"


def test_every_observed_margin_gets_its_exact_decision_boundary():
    values = np.linspace(0.01, 3.57, 37)
    candidates = thresholds_from(values)

    assert len(candidates) == len(values) + 1  # zero plus one boundary per value
    for value in values:
        assert float(np.nextafter(value, np.inf)) in candidates


def _row(n, shown, correct):
    return SweepRow(min_phase_margin=0.0, min_meter_margin=0.3, n=n, shown=shown,
                    correct=correct, wrong_meter=0, wrong_phase=0,
                    withheld=n - shown, no_answer=0)


def test_nothing_observed_rules_nothing_out():
    assert wilson_upper(0, 0) == 1.0


def test_a_clean_run_bounds_tighter_the_longer_it_stays_clean():
    assert wilson_upper(0, 4) == pytest.approx(0.49, abs=0.02)
    assert wilson_upper(0, 35) == pytest.approx(0.10, abs=0.02)
    assert wilson_upper(0, 200) < wilson_upper(0, 100) < wilson_upper(0, 35)


def test_the_bound_covers_a_run_that_did_see_failures():
    # The rule of three cannot speak here at all; this is why Wilson.
    assert wilson_upper(1, 80) > 1 / 80
    assert wilson_upper(1, 80) < 0.10


def test_the_chooser_refuses_the_clean_handful_that_fooled_us_twice():
    # Six clips, four accented, none wrong — the exact shape that reported a 0%
    # wrong rate on validation and then missed on the held-out split. The
    # observed rates pass the budget; the interval does not.
    scores = _scores([(1.0, 0, 4)] * 4 + [(0.1, 0, 4)] * 2)
    rows = sweep(scores, [0.5], [0.0])

    assert rows[0].wrong_rate == pytest.approx(0.0)
    assert choose_margins(rows, bounded=False) is not None
    assert choose_margins(rows) is None


def test_enough_clean_results_do_qualify():
    # Nothing about the bound makes it unsatisfiable — it asks for material.
    scores = _scores([(1.0, 0, 4)] * 90 + [(0.1, 0, 4)] * 10)
    rows = sweep(scores, [0.5], [0.0])

    assert choose_margins(rows) is not None


def test_a_clean_sweep_of_a_handful_of_clips_is_not_a_demonstrated_error_rate():
    # The exact shape of every run so far: no wrong answers, and nowhere near
    # enough results for that to mean the error rate is under five percent.
    complaint = evidence_gap(_row(n=6, shown=4, correct=4))

    assert complaint is not None
    assert "73 independent groups" in complaint
    assert "35 shown groups" in complaint


def test_enough_clips_and_enough_shown_accents_carry_the_claim():
    assert evidence_gap(_row(n=60, shown=30, correct=30)) is not None
    assert evidence_gap(_row(n=73, shown=35, correct=35)) is None


def test_evidence_requirement_includes_failures_already_observed():
    row = SweepRow(
        min_phase_margin=0.0,
        min_meter_margin=0.3,
        n=80,
        shown=80,
        correct=79,
        wrong_meter=1,
        wrong_phase=0,
        withheld=0,
        no_answer=0,
    )

    complaint = evidence_gap(row)

    assert complaint is not None
    assert "110 independent groups" in complaint
    assert "53 shown groups" in complaint


def test_coverage_too_thin_fails_even_with_the_clips_to_spare():
    # A hundred clips that only ever accent five of them says nothing about the
    # conditional error, which is the number a player actually meets.
    assert evidence_gap(_row(n=100, shown=5, correct=5)) is not None


def test_a_looser_budget_needs_correspondingly_less_evidence():
    row = _row(n=16, shown=10, correct=10)

    assert evidence_gap(row, max_wrong_rate=0.05, max_conditional_error=0.10)
    assert evidence_gap(row, max_wrong_rate=0.20, max_conditional_error=0.30) is None


def test_held_out_report_carries_both_wilson_upper_bounds():
    report = held_out_error_report(wrong=0, total=73, shown=35)

    assert "wrong rate 0.00% (95% upper 5.00%)" in report
    assert "conditional error 0.00% (95% upper 9.89%)" in report
    assert "both budgets DEMONSTRATED" in report


def test_no_held_out_accents_reports_no_conditional_point_estimate():
    report = held_out_error_report(wrong=0, total=10, shown=0)

    assert "conditional error n/a" in report
    assert "95% upper 100.00%" in report
    assert "both budgets NOT DEMONSTRATED" in report


def test_a_small_clean_held_out_split_is_explicitly_not_demonstrated():
    report = held_out_error_report(wrong=0, total=10, shown=10)

    assert "wrong rate 0.00%" in report
    assert "both budgets NOT DEMONSTRATED" in report


def test_a_bound_just_over_budget_is_not_rounded_down_to_look_compliant():
    report = held_out_error_report(wrong=0, total=72, shown=72)
    gap = evidence_gap(_row(n=72, shown=72, correct=72))

    assert "95% upper 5.07%" in report
    assert "both budgets NOT DEMONSTRATED" in report
    assert gap is not None
    assert "upper bounds here: 5.07%" in gap


def test_one_failure_just_below_the_required_n_is_visibly_over_budget():
    row = SweepRow(
        min_phase_margin=0.0,
        min_meter_margin=0.0,
        n=109,
        shown=109,
        correct=108,
        wrong_meter=1,
        wrong_phase=0,
        withheld=0,
        no_answer=0,
    )
    report = held_out_error_report(wrong=1, total=109, shown=109)
    gap = evidence_gap(row)

    assert "95% upper 5.01%" in report
    assert "both budgets NOT DEMONSTRATED" in report
    assert gap is not None
    assert "upper bounds here: 5.01%" in gap
    assert "at least 110 independent groups" in gap


def test_thresholds_survive_an_empty_set():
    assert thresholds_from([]) == [0.0]


def test_thresholds_ignore_infinities():
    candidates = thresholds_from([0.5, float("nan"), float("inf")])
    assert all(np.isfinite(c) for c in candidates)


# ---------------------------------------------------- who may be calibrated --

def test_a_clip_that_changes_metre_is_reported_but_not_calibrated_on():
    # No single answer is right for the whole take, so counting it either way
    # would move the threshold on a question the analysis was never asked.
    ref = reference()
    ref.bar_lengths = (4, 4, 3)
    changing = score(ref, estimate_from(ref), name="changes")
    steady = score(reference(), estimate_from(reference()), name="steady")

    assert not changing.meter_is_stable
    assert [s.name for s in eligible([changing, steady])] == ["steady"]
    assert sweep([changing, steady], [0.0], [0.0])[0].n == 1
