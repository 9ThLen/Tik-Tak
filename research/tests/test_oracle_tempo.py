"""The tempo oracle's two judgements: what the answer was, and which octave."""

import numpy as np
import pytest

from eval.oracle_tempo import CEILING_CONFIGS, octave_of, reference_bpm


def test_reference_bpm_reads_a_steady_grid():
    assert reference_bpm(np.arange(0.0, 20.0, 0.5)) == pytest.approx(120.0)


def test_reference_bpm_survives_a_dropped_annotation():
    """One missing beat doubles an interval; a mean would follow it, a median
    does not. Landing on the wrong octave is the error this file exists to
    measure, so the measurement must not commit it itself."""
    beats = np.delete(np.arange(0.0, 20.0, 0.5), 12)
    assert reference_bpm(beats) == pytest.approx(120.0)


def test_reference_bpm_declines_when_there_is_no_grid():
    assert reference_bpm(np.zeros(0)) == 0.0
    assert reference_bpm(np.array([1.0])) == 0.0


@pytest.mark.parametrize("estimate, expected", [
    (120.0, "right"),
    (127.0, "right"),      # inside the eight percent band
    (60.0, "half"),
    (240.0, "double"),
    (160.0, "other"),      # three against two, a real error and not an octave
    (0.0, "none"),
])
def test_octave_of_names_the_metrical_level(estimate, expected):
    assert octave_of(estimate, 120.0) == expected


def test_octave_bands_do_not_overlap():
    """A band wide enough to touch its neighbour would make the count
    ambiguous exactly where the tracker's errors live."""
    for reference in (60.0, 90.0, 120.0, 180.0):
        assert octave_of(reference * 0.75, reference) == "other"
        assert octave_of(reference * 1.5, reference) == "other"


def test_seeded_and_pinned_are_separate_configurations():
    """The gap between them is the result; collapsing them loses it."""
    seeded = CEILING_CONFIGS["live+beatnet, ORACLE seeded"]
    pinned = CEILING_CONFIGS["live+beatnet, ORACLE pinned"]
    assert seeded[1] == "hint" and "--live-seeded" in seeded[0]
    assert pinned[1] == "manual" and "--live" in pinned[0]
