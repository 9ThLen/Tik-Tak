"""The AM hierarchy's properties, separately from whether it scores well.

Whether bar-rate modulation carries the bar line on real music is what the
benchmark answers. What is checked here is that the band is placed where the
beat grid says it should be, that the output is in the unit a calibration
threshold assumes, and that degenerate input produces no number at all rather
than a confident wrong one.
"""

import numpy as np
import pytest

from eval.am_hierarchy import AmConfig, bar_band_envelope, bar_salience
from eval.backends import am_hierarchy_backend

RATE = 22050.0


def bar_marked_audio(bpm: float = 120.0, beats_per_bar: int = 4,
                     seconds: float = 12.0, downbeat_gain: float = 4.0,
                     rate: float = RATE) -> np.ndarray:
    """Clicks on every beat, louder on the first of each bar."""
    n = int(seconds * rate)
    audio = np.zeros(n)
    period = 60.0 / bpm
    click = np.exp(-np.arange(int(0.03 * rate)) / (0.004 * rate))
    click *= np.sin(2.0 * np.pi * 1200.0 * np.arange(len(click)) / rate)
    for index in range(int(seconds / period)):
        start = int(index * period * rate)
        gain = downbeat_gain if index % beats_per_bar == 0 else 1.0
        end = min(n, start + len(click))
        audio[start:end] += gain * click[:end - start]
    return audio


def beat_grid(bpm: float = 120.0, seconds: float = 12.0) -> np.ndarray:
    period = 60.0 / bpm
    return np.arange(0.0, seconds, period)


def test_the_salience_is_scaled_into_the_unit_a_threshold_assumes():
    audio = bar_marked_audio()
    values = bar_salience(audio, RATE, beat_grid())
    assert len(values) == len(beat_grid())
    assert float(np.min(values)) == pytest.approx(0.0)
    assert float(np.max(values)) == pytest.approx(1.0)


def test_the_band_follows_the_beat_grid_rather_than_the_clock():
    # The same music at half the tempo must move the band, or "bar rate" would
    # mean a fixed frequency and the hierarchy would be a fixed filter.
    audio = bar_marked_audio()
    slow, _ = bar_band_envelope(audio, RATE, beat_period_sec=1.0)
    fast, _ = bar_band_envelope(audio, RATE, beat_period_sec=0.25)
    assert len(slow) == len(fast)
    assert not np.allclose(slow, fast)


def test_a_grid_with_fewer_than_two_beats_yields_no_opinion():
    audio = bar_marked_audio()
    assert len(bar_salience(audio, RATE, np.zeros(0))) == 0
    assert np.all(bar_salience(audio, RATE, np.array([1.0])) == 0.0)


def test_silence_produces_no_salience_rather_than_a_flat_guess():
    values = bar_salience(np.zeros(int(8 * RATE)), RATE, beat_grid(seconds=8.0))
    assert np.all(values == 0.0)


def test_a_nonsense_beat_period_is_refused():
    audio = bar_marked_audio()
    for period in (0.0, -1.0, float("nan")):
        envelope, times = bar_band_envelope(audio, RATE, period)
        assert len(envelope) == 0 and len(times) == 0


def test_the_envelope_is_rectified():
    # The negative half of a band-pass says "quieter than the local average",
    # which is not evidence for a bar line somewhere else and must not be fed
    # to a resolver as though it were.
    envelope, _ = bar_band_envelope(bar_marked_audio(), RATE, 0.5)
    assert np.all(envelope >= 0.0)


def test_the_backend_is_a_salience_source_the_resolver_can_drive():
    backend = am_hierarchy_backend()
    assert not backend.is_builtin
    assert backend.name == "am_hierarchy"
    values = backend.salience(bar_marked_audio(), RATE, beat_grid())
    assert len(values) == len(beat_grid())
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("bad", [
    dict(lowest_bar_in_beats=1.0, highest_bar_in_beats=2.0),
    dict(highest_bar_in_beats=0.0),
    dict(use_bands=()),
])
def test_a_configuration_that_cannot_work_is_refused(bad):
    with pytest.raises(ValueError):
        AmConfig(**bad).validate()
