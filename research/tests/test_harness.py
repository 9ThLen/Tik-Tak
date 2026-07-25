import numpy as np

from eval.harness import (
    METRIC_NAMES,
    evaluate,
    evaluate_downbeats,
    evaluate_many,
    format_report,
)
from tiktak.synth import make_clip


REF = np.arange(0.0, 60.0, 0.5)  # 120 BPM, 60 s


class TestEvaluatePerfect:
    def test_perfect_estimate_is_perfect(self):
        r = evaluate(REF, REF.copy())
        assert set(METRIC_NAMES) <= set(r)
        assert r["f_measure"] == 1.0
        assert r["cmlc"] == 1.0
        assert r["cmlt"] == 1.0
        assert r["amlc"] == 1.0
        assert r["amlt"] == 1.0
        assert r["p_score"] == 1.0
        assert r["information_gain"] > 0.9

    def test_trim_convention(self):
        # Estimate wrong only in the first 5 s: with trim=True (MIREX) that
        # region is ignored, with trim=False it costs us.
        est = REF.copy()
        est[est < 5.0] += 0.2  # 200 ms off, outside the 70 ms tolerance
        assert evaluate(REF, est, trim=True)["f_measure"] == 1.0
        assert evaluate(REF, est, trim=False)["f_measure"] < 1.0


class TestEvaluateDegradation:
    def test_small_jitter_within_tolerance(self):
        rng = np.random.default_rng(0)
        est = REF + rng.uniform(-0.03, 0.03, size=len(REF))  # < 70 ms tol
        r = evaluate(REF, est)
        assert r["f_measure"] == 1.0
        # continuity has a tighter (17.5%-of-IBI) window, so it can dip but
        # should stay high for 30 ms jitter on a 500 ms IBI
        assert r["cmlt"] > 0.7

    def test_larger_jitter_degrades_monotonically(self):
        rng = np.random.default_rng(1)
        f_prev = 1.01
        for jitter in (0.02, 0.06, 0.12, 0.25):
            est = REF + rng.uniform(-jitter, jitter, size=len(REF))
            f = evaluate(REF, est)["f_measure"]
            assert f <= f_prev + 1e-9
            f_prev = f
        assert f_prev < 0.7  # 250 ms jitter must hurt badly

    def test_random_times_near_zero(self):
        rng = np.random.default_rng(2)
        est = np.sort(rng.uniform(0.0, 60.0, size=len(REF)))
        r = evaluate(REF, est)
        assert r["f_measure"] < 0.35
        assert r["cmlt"] < 0.1
        assert r["amlt"] < 0.1


class TestOctaveErrors:
    """AMLt must forgive double/half tempo where CMLt collapses — this is
    how we will tell octave errors apart from real failures later."""

    def test_double_tempo(self):
        est = np.arange(0.0, 60.0, 0.25)  # 240 BPM, phase-aligned
        r = evaluate(REF, est)
        assert r["cmlt"] < 0.1
        assert r["amlt"] > 0.9

    def test_half_tempo(self):
        est = REF[::2]  # 60 BPM
        r = evaluate(REF, est)
        assert r["cmlt"] < 0.1
        assert r["amlt"] > 0.9

    def test_offbeat_phase(self):
        est = REF + 0.25  # right tempo, wrong phase (off-beat)
        r = evaluate(REF, est)
        assert r["cmlt"] < 0.1
        assert r["amlt"] > 0.9


class TestEdgeCases:
    def test_empty_estimate_scores_zero(self):
        r = evaluate(REF, np.array([]))
        for name in METRIC_NAMES:
            assert r[name] == 0.0

    def test_single_estimated_beat_scores_low(self):
        r = evaluate(REF, np.array([10.0]))
        assert r["cmlt"] == 0.0
        assert r["amlt"] == 0.0
        assert r["f_measure"] < 0.1

    def test_unscorable_reference_is_nan(self):
        # everything before 5 s gets trimmed away -> no reference left
        short_ref = np.array([1.0, 1.5, 2.0])
        r = evaluate(short_ref, short_ref)
        assert all(np.isnan(r[name]) for name in METRIC_NAMES)
        # but with trim=False the same clip is perfectly scorable
        r2 = evaluate(short_ref, short_ref, trim=False)
        assert r2["f_measure"] == 1.0

    def test_unsorted_and_duplicated_input_tolerated(self):
        est = np.concatenate([REF[::-1], REF[:5]])  # reversed + duplicates
        r = evaluate(REF, est)
        assert r["f_measure"] == 1.0


class TestSynthIntegration:
    def test_ground_truth_scores_perfect_against_itself(self):
        c = make_clip(bpm=97.0, duration_sec=20.0, tempo_drift=15.0, seed=11)
        r = evaluate(c.beats, c.beats)
        assert r["f_measure"] == 1.0
        assert r["cmlt"] == 1.0


class TestDownbeats:
    def test_perfect_and_shifted(self):
        c = make_clip(bpm=120, duration_sec=30.0, seed=12)
        r = evaluate_downbeats(c.downbeats, c.downbeats)
        assert r["downbeat_f_measure"] == 1.0
        # estimating beat 2 as the downbeat (classic failure) -> F = 0
        wrong = c.beats[1 :: c.beats_per_bar]
        r2 = evaluate_downbeats(c.downbeats, wrong)
        assert r2["downbeat_f_measure"] == 0.0

    def test_empty_reference_nan(self):
        r = evaluate_downbeats(np.array([]), np.array([1.0]), trim=False)
        assert np.isnan(r["downbeat_f_measure"])


class TestAggregation:
    def test_evaluate_many_and_report(self):
        rng = np.random.default_rng(3)
        pairs = [
            (REF, REF.copy()),                                        # perfect
            (REF, REF + 0.25),                                        # off-beat
            (REF, np.sort(rng.uniform(0, 60, size=len(REF)))),        # junk
        ]
        agg = evaluate_many(pairs)
        assert agg["n_clips"] == 3
        assert agg["n_scored"] == 3
        assert len(agg["per_clip"]) == 3
        assert agg["per_clip"][0]["f_measure"] == 1.0
        assert agg["mean"]["f_measure"] < 1.0
        assert agg["median"]["amlt"] > 0.95  # perfect + off-beat both AML-pass
        assert agg["std"]["f_measure"] > 0.0

        report = format_report(agg)
        assert "f_measure" in report
        assert "amlt" in report
        assert "mean" in report
        assert "3 clip(s)" in report

    def test_evaluate_many_skips_nan_in_aggregate(self):
        short = np.array([1.0, 2.0])  # trimmed to nothing
        agg = evaluate_many([(REF, REF.copy()), (short, short)])
        assert agg["n_clips"] == 2
        assert agg["n_scored"] == 1
        assert agg["mean"]["f_measure"] == 1.0  # NaN clip excluded

    def test_format_report_single_clip(self):
        report = format_report(evaluate(REF, REF.copy()))
        assert "f_measure" in report
        assert "1.000" in report
