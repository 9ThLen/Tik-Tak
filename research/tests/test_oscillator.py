"""What the oscillator bank must do before any measurement of it means anything.

The interesting claim — that nonlinear resonance decides the octave where
autocorrelation cannot — is not testable here; it needs annotated recordings and
is measured by the sweep. What is testable here is that the bank resonates where
it is driven, stays silent when it is not, and does not diverge.
"""

import numpy as np
import pytest

from eval.oscillator import (OscillatorBank, OscillatorConfig, Resonance,
                             estimate_tempo)

FPS = 100.0


def impulse_odf(bpm: float, seconds: float = 20.0, fps: float = FPS,
                jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """An onset function that is one at each beat and zero between them."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    odf = np.zeros(n)
    period = 60.0 / bpm * fps
    position = period
    while position < n - 1:
        index = int(round(position + (rng.normal(0.0, jitter * period) if jitter else 0.0)))
        if 0 <= index < n:
            odf[index] = 1.0
        position += period
    return odf


# A quick bank: the default 240 oscillators are for measurement, and the tests
# only need enough resolution to tell one tempo from another.
def quick(**kw) -> OscillatorConfig:
    return OscillatorConfig(count=60, **kw)


def test_the_bank_resonates_at_the_tempo_it_is_driven_with():
    for bpm in (80.0, 120.0, 150.0):
        estimate = estimate_tempo(impulse_odf(bpm), FPS, quick())
        assert abs(estimate - bpm) / bpm < 0.06, f"{bpm} came back as {estimate}"


def test_silence_resonates_nowhere():
    bank = OscillatorBank(quick())
    result = bank.run(np.zeros(1000), FPS)
    assert np.all(result.amplitude == 0.0)
    assert result.peak_bpm == 0.0


def test_a_constant_drive_carries_no_rhythm_and_is_refused():
    # Not an error and not a tempo: a direct current has no period, and the
    # estimator must not divide by a zero spread to find that out.
    bank = OscillatorBank(quick())
    assert np.all(bank.run(np.ones(1000), FPS).amplitude == 0.0)


def test_amplitudes_stay_finite_under_a_drive_that_never_stops():
    # beta < 0 is what bounds this. If saturation were ever dropped the bank
    # would run away, and an infinite amplitude wins every argmax it enters.
    bank = OscillatorBank(quick(drive=25.0))
    result = bank.run(np.abs(np.sin(np.linspace(0, 400, 4000))), FPS)
    assert np.all(np.isfinite(result.amplitude))
    assert float(np.max(result.amplitude)) < 1e6


def test_it_survives_input_too_short_to_hold_a_beat():
    bank = OscillatorBank(quick())
    for odf in (np.zeros(0), np.ones(1), np.array([1.0, 0.0])):
        result = bank.run(odf, FPS)
        assert len(result.amplitude) == len(result.bpm)
        assert np.all(np.isfinite(result.amplitude))


def test_a_bad_frame_rate_is_refused_rather_than_guessed_at():
    bank = OscillatorBank(quick())
    for fps in (0.0, -1.0, float("nan")):
        assert np.all(bank.run(impulse_odf(120.0), fps).amplitude == 0.0)


def test_the_resonance_curve_is_reported_whole():
    # The peak alone would hide the thing this module exists to look at: how
    # much the octave neighbour resonates compared with the winner.
    bank = OscillatorBank(quick())
    result = bank.run(impulse_odf(120.0), FPS)
    assert len(result.bpm) == 60
    assert result.bpm[0] == pytest.approx(40.0)
    assert result.bpm[-1] == pytest.approx(220.0)
    assert float(np.max(result.normalised())) == pytest.approx(1.0)


def test_normalising_an_empty_or_dead_bank_does_not_divide_by_zero():
    assert len(Resonance(np.zeros(0), np.zeros(0)).normalised()) == 0
    dead = Resonance(np.array([100.0, 120.0]), np.zeros(2))
    assert np.all(dead.normalised() == 0.0)


def test_jitter_does_not_move_the_answer_off_the_tempo():
    # Real onsets are not on a grid. A few per cent of the period is ordinary
    # and must not change which oscillator wins.
    estimate = estimate_tempo(impulse_odf(120.0, jitter=0.03, seed=7), FPS, quick())
    assert abs(estimate - 120.0) / 120.0 < 0.08


@pytest.mark.parametrize("bad", [
    dict(min_bpm=0.0), dict(min_bpm=200.0, max_bpm=100.0), dict(count=1),
    dict(alpha=0.1), dict(beta=1.0), dict(settle=1.0),
])
def test_a_configuration_that_cannot_work_is_refused(bad):
    with pytest.raises(ValueError):
        OscillatorConfig(**bad).validate()
