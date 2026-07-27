"""CMLt and AMLt, held to grids whose right answer is known by construction.

These metrics exist to tell an octave error apart from a lost grid, and that
separation is the whole reason to prefer them over an F-measure. A bug here
would not crash: it would quietly merge the two failure modes again and send
the next piece of work at the wrong one.
"""

import numpy as np
import pytest

from eval.continuity import amlt, cmlt, continuity


def grid(bpm, seconds=60.0, offset=0.0):
    period = 60.0 / bpm
    return np.arange(offset, seconds, period)


def test_a_grid_scores_perfectly_against_itself():
    beats = grid(120)
    assert cmlt(beats, beats) == pytest.approx(1.0)
    assert amlt(beats, beats) == pytest.approx(1.0)


def test_half_tempo_is_the_signature_the_metrics_exist_to_show():
    reference = grid(120)
    half = grid(60)
    # Right music, wrong level: AMLt sees it, CMLt does not. Reading only one
    # of these numbers is how an octave error gets mistaken for a lost tracker.
    assert cmlt(half, reference) < 0.05
    assert amlt(half, reference) > 0.95


def test_double_tempo_reads_the_same_way():
    reference = grid(120)
    assert cmlt(grid(240), reference) < 0.05
    assert amlt(grid(240), reference) > 0.95


def test_the_offbeat_is_allowed_but_not_correct():
    reference = grid(120)
    offbeat = grid(120, offset=0.25)      # half a beat over at 120 BPM
    assert cmlt(offbeat, reference) < 0.05
    assert amlt(offbeat, reference) > 0.95


def test_a_genuinely_lost_grid_scores_low_on_both():
    # An unrelated tempo, not a metrical relative of the reference. This is the
    # case that must look different from the three above, or the metrics are
    # not doing their job.
    reference = grid(120)
    assert cmlt(grid(97), reference) < 0.2
    assert amlt(grid(97), reference) < 0.4


def test_small_jitter_is_tolerated_and_large_jitter_is_not():
    reference = grid(120)
    rng = np.random.default_rng(0)
    period = 0.5
    # Well inside the 17.5% window, then well outside it.
    tight = reference + rng.uniform(-0.05, 0.05, len(reference)) * period
    loose = reference + rng.uniform(-0.6, 0.6, len(reference)) * period
    assert cmlt(tight, reference) > 0.8
    assert cmlt(loose, reference) < 0.3


def test_continuity_separates_a_broken_run_from_a_whole_one():
    reference = grid(120, seconds=60.0)
    # Right for the first half, then a phase jump for the rest. The total stays
    # near a half either way; what changes is the longest unbroken run.
    broken = np.concatenate([reference[:60], reference[60:] + 0.25])
    longest, total = continuity(broken, reference)
    assert total > 0.4
    assert longest < total + 0.05
    whole, whole_total = continuity(reference, reference)
    assert whole == pytest.approx(whole_total)


def test_degenerate_input_is_zero_rather_than_a_crash():
    reference = grid(120)
    assert continuity([], reference) == (0.0, 0.0)
    assert continuity(reference, []) == (0.0, 0.0)
    assert continuity([1.0], reference) == (0.0, 0.0)
    assert cmlt([], []) == 0.0
    assert amlt([], []) == 0.0
