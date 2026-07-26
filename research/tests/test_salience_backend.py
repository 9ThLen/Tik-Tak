"""The seam, crossed from outside: a model's salience through the shipping resolver.

These tests run the actual dump_analysis binary, because the thing they verify
is precisely that an *external* per-beat salience reaches the same resolveMeter
the app runs. A Python-side mock would verify the mock.

Skipped when the binary is missing so a source-only checkout can still run the
rest of the suite; CI builds the tool before pytest, so there they always run.
"""

import numpy as np
import pytest

from eval.analysis import Analyser, DEFAULT_BINARY
from tiktak.synth import make_clip

pytestmark = pytest.mark.skipif(
    not Analyser(DEFAULT_BINARY).available,
    reason="dump_analysis is not built — see eval/analysis.py",
)


@pytest.fixture(scope="module")
def clip():
    return make_clip(duration_sec=25, bpm=120, beats_per_bar=4, seed=1)


@pytest.fixture(scope="module")
def analyser():
    return Analyser(DEFAULT_BINARY)


@pytest.fixture(scope="module")
def grid(analyser, clip):
    """The two-pass workflow's first pass: learn the beat times."""
    return analyser.analyse_audio(clip.audio, clip.sample_rate)


def test_the_cues_backend_reports_itself(grid):
    assert grid.salience_source == "cues"
    assert grid.beats_per_bar == 4


def test_an_injected_salience_overrules_the_cues_through_the_same_resolver(
        analyser, clip, grid):
    # The audio is an unambiguous 4/4; the file paints a waltz starting on the
    # second beat. If the resolver answers 3 with that phase, the salience came
    # from the file and the decision came from the shipping code — which is the
    # entire claim the seam makes.
    salience = np.where(np.arange(len(grid.beats)) % 3 == 1, 1.0, 0.05)
    result = analyser.analyse_audio(clip.audio, clip.sample_rate, salience=salience)

    assert result.salience_source == "file"
    assert result.beats_per_bar == 3
    assert result.downbeats[0] == pytest.approx(grid.beats[1])


def test_the_beat_grid_is_not_touched_by_the_swap(analyser, clip, grid):
    # The seam replaces the scorer and nothing upstream of it. A backend that
    # moved the beats would be a different tracker, not a different scorer.
    salience = np.ones(len(grid.beats))
    result = analyser.analyse_audio(clip.audio, clip.sample_rate, salience=salience)

    np.testing.assert_allclose(result.beats, grid.beats)
    assert result.bpm == pytest.approx(grid.bpm)


def test_a_flat_salience_is_no_answer_whatever_the_audio_says(analyser, clip, grid):
    # The cues would find this clip's bars easily; a backend that sees nothing
    # must produce nothing, not inherit the cues' answer.
    result = analyser.analyse_audio(clip.audio, clip.sample_rate,
                                    salience=np.ones(len(grid.beats)))
    assert not result.downbeat_confident


def test_a_nearly_flat_periodic_salience_is_not_normalised_into_an_answer(
        analyser, clip, grid):
    salience = np.where(np.arange(len(grid.beats)) % 4 == 0,
                        0.500003, 0.500001)
    result = analyser.analyse_audio(clip.audio, clip.sample_rate,
                                    salience=salience)

    assert result.beats_per_bar == 0
    assert not result.downbeats.size
    assert not result.downbeat_confident


def test_an_external_backend_can_supply_its_own_range_gate(
        analyser, clip, grid):
    salience = np.where(np.arange(len(grid.beats)) % 4 == 0,
                        0.500003, 0.500001)
    result = analyser.analyse_audio(
        clip.audio,
        clip.sample_rate,
        salience=salience,
        salience_min_range=1e-7,
    )

    assert result.beats_per_bar == 4
    assert result.downbeats[0] == pytest.approx(grid.beats[0])
    # Admitting this backend's scale does not make the ordinary cue backend's
    # much larger margin thresholds magically applicable to it.
    assert not result.downbeat_confident


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_an_external_range_gate_must_be_non_negative_and_finite(
        analyser, clip, grid, bad):
    with pytest.raises(
            RuntimeError,
            match=r"--salience-min-range must be a finite, non-negative number"):
        analyser.analyse_audio(
            clip.audio,
            clip.sample_rate,
            salience=np.ones(len(grid.beats)),
            salience_min_range=bad,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf],
                         ids=["nan", "positive infinity", "negative infinity"])
def test_non_finite_salience_is_rejected_with_its_position_named(
        analyser, clip, grid, bad):
    salience = np.ones(len(grid.beats))
    salience[1] = bad

    with pytest.raises(RuntimeError,
                       match=r"non-finite salience value 2 at byte"):
        analyser.analyse_audio(clip.audio, clip.sample_rate, salience=salience)


def test_a_count_mismatch_is_refused_with_both_counts_named(analyser, clip):
    with pytest.raises(RuntimeError, match=r"3 value\(s\).*beat\(s\)"):
        analyser.analyse_audio(clip.audio, clip.sample_rate,
                               salience=np.array([1.0, 0.0, 1.0]))
