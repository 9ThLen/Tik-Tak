"""Protocol tests for the synthetic live tempo stress stand."""

import numpy as np
import pytest

from eval.tempo_stress import (
    SEED,
    Scenario,
    activation_from_pulses,
    observed_pulses,
    reference_beats,
    scenarios,
    tempo_at,
)


def test_matrix_contains_every_preregistered_family():
    matrix = scenarios()
    assert {case.start_bpm for case in matrix if case.family == "steady"} == {
        60.0, 90.0, 120.0, 180.0,
    }
    ramp_changes = sorted({
        case.final_bpm / case.start_bpm - 1.0
        for case in matrix if case.family == "ramp"
    })
    assert ramp_changes == pytest.approx(
        [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]
    )
    assert {case.family for case in matrix} >= {
        "steady", "ramp", "step", "jitter", "drop_random", "drop_burst",
    }
    assert {case.change_end_sec - case.change_start_sec for case in matrix
            if case.family == "ramp"} == {15.0, 45.0}
    assert len([case for case in matrix if case.name == "jitter_40ms"]) == 5
    assert len([case for case in matrix if case.name == "drop_random_40pct"]) == 5


def test_steady_grid_has_the_requested_period():
    case = Scenario("steady", "steady", 20.0, 120.0)
    beats = reference_beats(case)
    np.testing.assert_allclose(np.diff(beats), 0.5, atol=1e-6)


def test_ramp_is_phase_continuous_and_monotonic():
    case = Scenario("ramp", "ramp", 30.0, 120.0, 132.0, 5.0, 25.0)
    beats = reference_beats(case)
    assert np.all(np.diff(beats) > 0.0)
    assert np.all(np.diff(np.diff(beats[(beats >= 5.0) & (beats <= 25.0)])) < 1e-5)
    assert tempo_at(case, 0.0) == 120.0
    assert tempo_at(case, 15.0) == pytest.approx(126.0)
    assert tempo_at(case, 30.0) == 132.0


def test_jitter_changes_only_observations_and_is_reproducible():
    case = Scenario("jitter", "jitter", 20.0, 120.0, jitter_std_sec=0.04)
    reference = reference_beats(case)
    first = observed_pulses(case, reference, SEED)
    second = observed_pulses(case, reference, SEED)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, reference)
    np.testing.assert_allclose(reference, reference_beats(case))


def test_burst_removes_exactly_the_requested_number_of_pulses():
    case = Scenario("burst", "drop_burst", 60.0, 120.0, burst_drop_beats=8)
    reference = reference_beats(case)
    observed = observed_pulses(case, reference)
    assert len(reference) - len(observed) == 8


def test_activation_has_fixed_floor_and_peak():
    activation = activation_from_pulses(np.array([1.0]), 2.0)
    assert activation.min() == pytest.approx(0.02)
    assert activation.max() == pytest.approx(0.95)
    assert np.count_nonzero(activation > 0.02) == 5
