"""What the phase-instability evaluator must do before its numbers mean anything.

The measured verdict is in the module docstring and needs annotated recordings.
What is testable here is the part that made the first pass of it wrong: whether
the signals are causal, whether each is scored on its own availability rather
than on another's, and whether the windows are counted the way the report says
they are.
"""

import numpy as np
import pytest

from eval.phase_instability import (
    DETECT_WINDOW, FEATURES, NEGATIVE_WIDTH, PREDICT_WINDOW, Episode, Timeline,
    _period_at, operating_curve, windows,
)

FPS = 100.0


def timeline(n: int = 2000, **overrides) -> Timeline:
    """A timeline with every signal flat and fully available."""
    times = np.arange(n) / FPS
    fields = {
        "times": times,
        "g": np.zeros(n),
        "rho_min": np.zeros(n),
        "balance": np.zeros(n),
        "anchor_margin": np.zeros(n),
        "available": {name: np.ones(n, dtype=bool) for name in FEATURES},
    }
    fields.update(overrides)
    return Timeline(**fields)


# --------------------------------------------------------------- causality

def test_period_is_nan_before_the_first_poll():
    """No frame may borrow a tempo the tracker has not reported yet.

    This is the regression the first pass of this module shipped: the index was
    clipped to zero before it was tested, so every frame ahead of the first
    poll silently took that poll's BPM out of the future.
    """
    times = np.linspace(0.0, 5.0, 51)
    period = _period_at(times, np.asarray([2.0, 3.0]), np.asarray([120.0, 60.0]))

    assert np.all(np.isnan(period[times < 2.0]))
    assert period[times >= 2.0][0] == pytest.approx(0.5)


def test_period_forward_fills_but_never_backwards():
    times = np.asarray([0.5, 1.5, 2.5, 3.5])
    period = _period_at(times, np.asarray([1.0, 3.0]), np.asarray([120.0, 0.0]))

    assert np.isnan(period[0])
    # The unusable second poll is held over from the first, not filled from
    # anything later.
    assert period[1] == pytest.approx(0.5)
    assert period[3] == pytest.approx(0.5)


def test_a_later_tempo_cannot_change_an_earlier_period():
    times = np.linspace(0.0, 10.0, 101)
    early = _period_at(times, np.asarray([1.0]), np.asarray([120.0]))
    late = _period_at(times, np.asarray([1.0, 6.0]),
                      np.asarray([120.0, 200.0]))

    before = times < 6.0
    assert np.allclose(early[before], late[before], equal_nan=True)


# ------------------------------------------------- availability is per signal

def test_each_signal_is_scored_on_its_own_availability():
    """One signal being unavailable must not remove an episode from another.

    The bug this replaced shared the phase feature's mask with every other
    signal, so `margin` was scored only where a *phase* had settled — which
    quietly shrank the denominator of the headline number.
    """
    n = 2000
    available = {name: np.ones(n, dtype=bool) for name in FEATURES}
    available["phase"] = np.zeros(n, dtype=bool)
    line = timeline(n, available=available)
    correct = np.ones(n, dtype=bool)

    out = windows(line, [Episode(10.0, 16.0)], correct)

    assert out["phase"]["episodes_scored"] == 0
    assert out["phase"]["predict"] == []
    assert out["margin"]["episodes_scored"] == 1
    assert len(out["margin"]["predict"]) == 1


def test_every_feature_reports_the_same_episode_total():
    line = timeline()
    out = windows(line, [Episode(10.0, 16.0)], np.ones(2000, dtype=bool))
    assert {out[name]["episodes_total"] for name in FEATURES} == {1}


# ------------------------------------------------------------ window counting

def test_the_predict_window_ends_before_the_onset():
    """A predictor may not see the transition it is predicting."""
    n = 2000
    g = np.zeros(n)
    line = timeline(n, g=g)
    onset = 12.0
    # A spike exactly at the onset must not reach the predict window.
    g[int(onset * FPS)] = 1.0

    out = windows(line, [Episode(onset, onset + 6.0)],
                  np.ones(n, dtype=bool))

    assert out["phase"]["predict"] == [0.0]
    assert out["phase"]["detect"] == [1.0]
    assert PREDICT_WINDOW[1] < DETECT_WINDOW[0] or PREDICT_WINDOW[1] <= 0.0


def test_negatives_are_tiled_and_exclude_the_guarded_neighbourhood():
    n = 6000  # 60 seconds
    line = timeline(n)
    correct = np.ones(n, dtype=bool)

    without = windows(line, [], correct)["margin"]["negative"]
    with_one = windows(line, [Episode(30.0, 34.0)], correct)["margin"]["negative"]

    assert len(without) == pytest.approx(60.0 / NEGATIVE_WIDTH, abs=2)
    # 8 s of guard either side of a 4 s episode removes about 20 s of the 60.
    assert len(with_one) < len(without)
    assert len(with_one) == pytest.approx(40.0 / NEGATIVE_WIDTH, abs=3)


def test_a_window_is_scored_by_its_maximum():
    n = 2000
    g = np.zeros(n)
    g[int(9.5 * FPS)] = 0.75
    line = timeline(n, g=g)

    out = windows(line, [Episode(12.0, 20.0)], np.ones(n, dtype=bool))

    assert out["phase"]["predict"] == [0.75]


# ------------------------------------------------------------- the cost columns

def test_the_two_cost_columns_are_not_the_same_number():
    """A window triggers on one frame in many, so it must read higher.

    Reporting the windowed figure as "share of correct time" is what the first
    pass did, and it flatters the gate by the ratio between these two.
    """
    negatives = [1.0] * 10 + [0.0] * 10
    frames = [1.0] * 10 + [0.0] * 190
    rows = operating_curve([1.0] * 8 + [0.0] * 2, negatives, frames,
                           episodes_total=10)

    hit = [row for row in rows if row["threshold"] == pytest.approx(1.0)][0]
    assert hit["negative_windows_triggered"] == pytest.approx(0.5)
    assert hit["correct_frames_triggered"] == pytest.approx(0.05)


def test_unscored_episodes_count_as_missed_in_warned_of_all():
    rows = operating_curve([1.0, 1.0], [0.0, 0.0], [0.0],
                           episodes_total=4)
    row = rows[0]
    assert row["warned_of_scored"] == pytest.approx(1.0)
    assert row["warned_of_all"] == pytest.approx(0.5)
