"""The peak map is causal, deterministic, and bounded — asserted, not assumed.

Every one of these is a place where a wrong implementation produces a plausible
number rather than an error, which is why they are tested before any result is
read from the module.

The first test is the important one. `scipy`'s `origin` sign is the opposite of
the obvious reading, and a leading window would be future context that looks
exactly like the experiment succeeding.
"""
import numpy as np
import pytest

from eval.peaks import (PeakParams, apply_refractory, collapse, dense_signals,
                        local_maxima, peak_map, windowed_max)


def brute_force_max(values, past, future, radius):
    frames, bands = values.shape
    out = np.empty_like(values)
    for t in range(frames):
        for b in range(bands):
            patch = values[max(0, t - past):t + future + 1,
                           max(0, b - radius):b + radius + 1]
            out[t, b] = patch.max()
    return out


@pytest.mark.parametrize("past,future,radius",
                         [(6, 0, 2), (3, 0, 0), (0, 0, 3), (4, 2, 1), (10, 5, 4)])
def test_the_window_is_where_it_says_it_is(past, future, radius):
    rng = np.random.default_rng(20260809)
    values = rng.random((40, 12))
    assert np.allclose(windowed_max(values, past, future, radius),
                       brute_force_max(values, past, future, radius))


def test_a_causal_map_cannot_see_the_future():
    """Change a frame; nothing at or before it may move.

    This is the property the whole `causal` arm rests on. A leading window, or
    an origin off by one, fails here and passes every other test in the file.
    """
    rng = np.random.default_rng(7)
    values = rng.random((60, 8))
    params = PeakParams(band_radius=2, past_frames=6, future_frames=0,
                        refractory_frames=0)
    before = local_maxima(values, params.band_radius, params.past_frames,
                          params.future_frames)

    changed = values.copy()
    changed[41:] = rng.random((19, 8)) + 5.0     # a much louder future
    after = local_maxima(changed, params.band_radius, params.past_frames,
                         params.future_frames)

    # Strictly before the edit: frame 41 is itself changed input, so it is
    # allowed to move and comparing it would test nothing.
    assert np.array_equal(before[:41], after[:41])


def test_a_symmetric_map_does_see_the_future():
    """The control for the test above: it must be able to fail."""
    rng = np.random.default_rng(7)
    values = rng.random((60, 8))
    before = local_maxima(values, 2, 6, 6)
    changed = values.copy()
    changed[41:] = rng.random((19, 8)) + 5.0
    after = local_maxima(changed, 2, 6, 6)
    # The same slice the causal test protects, so the two are exact mirrors.
    assert not np.array_equal(before[:41], after[:41])


def test_a_plateau_yields_one_peak():
    values = np.zeros((6, 5))
    values[2:4, 1:3] = 1.0        # a 2x2 flat block
    mask = local_maxima(values, band_radius=2, past_frames=3, future_frames=0)
    assert mask.sum() == 1
    assert mask[2, 1]             # earliest frame, then lowest band


def test_separated_equal_maxima_use_the_full_neighbourhood_for_ties():
    values = np.zeros((5, 5))
    values[1, 3] = 2.0
    values[3, 1] = 2.0
    mask = local_maxima(values, band_radius=2, past_frames=3, future_frames=0)
    assert np.argwhere(mask).tolist() == [[1, 3]]

    same_frame = np.zeros((1, 5))
    same_frame[0, 0] = same_frame[0, 2] = 2.0
    mask = local_maxima(same_frame, band_radius=2, past_frames=0,
                        future_frames=0)
    assert np.argwhere(mask).tolist() == [[0, 0]]


def test_zero_and_negative_never_peak():
    values = np.zeros((10, 4))
    assert local_maxima(values, 1, 3, 0).sum() == 0
    values[:] = -1.0
    assert local_maxima(values, 1, 3, 0).sum() == 0


def test_edges_are_truncated_not_padded():
    """A rise at frame 0 is a rise, and a window off the start invents nothing."""
    values = np.zeros((5, 1))
    values[0, 0] = 1.0
    mask = local_maxima(values, band_radius=0, past_frames=4, future_frames=0)
    assert mask[0, 0]
    assert mask.sum() == 1


def test_refractory_closes_a_band_and_only_that_band():
    mask = np.zeros((10, 2), dtype=bool)
    mask[:, 0] = True
    mask[3, 1] = True
    out = apply_refractory(mask, refractory_frames=2)
    assert np.flatnonzero(out[:, 0]).tolist() == [0, 3, 6, 9]
    assert np.flatnonzero(out[:, 1]).tolist() == [3]


def test_refractory_uses_only_the_past():
    """Extending a recording cannot change what fired before the extension."""
    rng = np.random.default_rng(11)
    mask = rng.random((50, 6)) > 0.7
    short = apply_refractory(mask[:30], 4)
    long = apply_refractory(mask, 4)
    assert np.array_equal(short, long[:30])


def test_collapse_rules_are_bounded_and_sane():
    mask = np.zeros((4, 3), dtype=bool)
    mask[1, 0] = mask[1, 2] = mask[3, 0] = True
    heights = np.full((4, 3), 2.0)
    params = PeakParams(novelty_frames=1)

    assert collapse(mask, heights, "count", params).tolist() == [0, 2, 0, 1]
    assert collapse(mask, heights, "weighted", params).tolist() == [0, 4, 0, 2]
    # Band 0 fires at 1 and again at 3, two frames later, which is outside a
    # one-frame horizon, so both are new.
    assert collapse(mask, heights, "novelty", params).tolist() == [0, 2, 0, 1]


def test_novelty_horizon_suppresses_a_repeat():
    mask = np.zeros((4, 1), dtype=bool)
    mask[1, 0] = mask[2, 0] = True
    heights = np.ones((4, 1))
    out = collapse(mask, heights, "novelty", PeakParams(novelty_frames=5))
    assert out.tolist() == [0, 1, 0, 0]


def test_unknown_rules_are_refused():
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="collapse rule"):
        collapse(mask, np.zeros((2, 2)), "median", PeakParams())
    with pytest.raises(ValueError, match="merge rule"):
        peak_map(np.zeros((2, 2)), np.zeros((2, 2)), PeakParams(merge="both"))


def test_the_map_is_deterministic():
    rng = np.random.default_rng(3)
    filterbank = rng.random((80, 10))
    difference = rng.random((80, 10))
    params = PeakParams(merge="union")
    first = peak_map(filterbank, difference, params)
    second = peak_map(filterbank, difference, params)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_sum_picks_each_channel_before_merging():
    """A raw channel sum can invent a peak that neither channel contains."""
    filterbank = np.array([[10.0], [0.0], [6.0]])
    difference = np.array([[0.0], [10.0], [6.0]])
    params = PeakParams(band_radius=0, past_frames=2, future_frames=0,
                        refractory_frames=0, merge="sum")

    mask, heights = peak_map(filterbank, difference, params)

    assert np.flatnonzero(mask[:, 0]).tolist() == [0, 1]
    assert heights[:, 0].tolist() == [10.0, 10.0, 0.0]


def test_union_uses_filterbank_height_for_a_filterbank_only_peak():
    filterbank = np.array([[0.0], [7.0]])
    difference = np.zeros_like(filterbank)
    params = PeakParams(band_radius=0, past_frames=1, future_frames=0,
                        refractory_frames=0, merge="union")

    mask, heights = peak_map(filterbank, difference, params)

    assert mask[1, 0]
    assert collapse(mask, heights, "weighted", params).tolist() == [0.0, 7.0]


def test_refractory_is_applied_once_after_channels_are_merged():
    filterbank = np.array([[4.0], [0.0], [4.0], [0.0]])
    difference = np.array([[0.0], [4.0], [0.0], [4.0]])
    params = PeakParams(band_radius=0, past_frames=1, future_frames=0,
                        refractory_frames=1, merge="union")

    mask, _ = peak_map(filterbank, difference, params)

    assert np.flatnonzero(mask[:, 0]).tolist() == [0, 2]


def test_dense_control_is_not_a_constant():
    """`count` over a dense map would be the band count in every frame.

    The control has to vary or it cannot have a floor-to-peak ratio at all,
    which is the hole this definition exists to close.
    """
    rng = np.random.default_rng(5)
    signals = dense_signals(rng.random((30, 9)), rng.random((30, 9)))
    for name, series in signals.items():
        assert series.shape == (30,)
        assert series.std() > 0.0, name
