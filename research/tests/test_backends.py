"""Backend plumbing: sampling a model's activation, and carrying its calibration.

No model runs here — this environment has neither torch nor the weights. What
runs is everything between a model and a verdict, exercised with a stand-in
scorer, so that plugging in the real one is the only untested step left.
"""

import numpy as np
import pytest

from eval.analysis import Analyser, DEFAULT_BINARY
from eval.backends import (
    BEAT_THIS_CHECKPOINT,
    Backend,
    Calibration,
    CUE_CALIBRATION,
    beat_this_backend,
    cue_backend,
    sample_at_beats,
)
from tiktak.synth import make_clip


# ------------------------------------------------------- sampling at beats --

def test_the_peak_in_the_window_is_taken_not_the_nearest_frame():
    # A spike one frame wide, sitting between two frame centres. The nearest
    # frame would miss it entirely and report the floor.
    frame_times = np.arange(0, 1.0, 0.02)
    activation = np.zeros_like(frame_times)
    activation[25] = 1.0                       # t = 0.50

    beats = np.array([0.49])
    assert sample_at_beats(activation, frame_times, beats)[0] == 1.0


def test_a_beat_and_a_models_idea_of_it_may_disagree_by_tens_of_milliseconds():
    frame_times = np.arange(0, 2.0, 0.02)
    activation = np.zeros_like(frame_times)
    activation[50] = 1.0                       # t = 1.00

    # Inside ±70 ms it is the same beat; well outside, it is not.
    assert sample_at_beats(activation, frame_times, np.array([1.06]))[0] == 1.0
    assert sample_at_beats(activation, frame_times, np.array([1.30]))[0] == 0.0


def test_widening_the_window_must_not_average_the_spike_away():
    # The failure a mean would produce: every beat converging on the piece's
    # average, which says nothing about where the bar line is.
    frame_times = np.arange(0, 4.0, 0.01)
    activation = np.zeros_like(frame_times)
    activation[::100] = 1.0                    # a spike every second

    narrow = sample_at_beats(activation, frame_times, np.array([2.0]), 0.05)
    wide = sample_at_beats(activation, frame_times, np.array([2.0]), 0.4)
    assert narrow[0] == wide[0] == 1.0


def test_every_beat_gets_a_value_even_where_the_model_has_no_frames():
    # The count must equal the beat count: the resolver refuses a mismatch,
    # and dropping a beat here would turn that into an alignment puzzle.
    frame_times = np.arange(0, 1.0, 0.02)
    activation = np.ones_like(frame_times)
    beats = np.array([0.5, 5.0, 9.0])

    out = sample_at_beats(activation, frame_times, beats)
    assert out.shape == beats.shape
    assert out[0] == 1.0 and out[1] == 0.0 and out[2] == 0.0


def test_a_mismatched_activation_is_refused_with_both_counts():
    with pytest.raises(ValueError, match="3.*5"):
        sample_at_beats(np.zeros(3), np.zeros(5), np.zeros(2))


def test_a_non_finite_activation_is_refused_rather_than_propagated():
    frames = np.arange(4) * 0.1
    with pytest.raises(ValueError, match="non-finite"):
        sample_at_beats(np.array([0.0, np.nan, 1.0, 0.0]), frames, np.array([0.1]))


def test_an_empty_activation_still_answers_once_per_beat():
    out = sample_at_beats(np.zeros(0), np.zeros(0), np.array([0.0, 0.5]))
    assert out.tolist() == [0.0, 0.0]


# -------------------------------------------------------------- backends ----

def test_the_cue_backend_is_the_one_that_needs_no_artifacts():
    backend = cue_backend()
    assert backend.is_builtin
    assert backend.calibration == CUE_CALIBRATION


def test_a_missing_checkpoint_refuses_rather_than_falling_back_to_the_cues():
    # Silently scoring the cues twice and labelling one column with a model's
    # name would look exactly like a result.
    if BEAT_THIS_CHECKPOINT.is_file():
        pytest.skip("the checkpoint is present in this checkout")
    with pytest.raises(FileNotFoundError, match="models/fetch.py pin"):
        beat_this_backend()


def test_a_backend_can_be_injected_so_the_rest_can_be_tested_without_weights():
    def loader(path):
        return lambda audio, rate, beats: np.ones(len(beats))

    backend = beat_this_backend(loader=loader)
    assert not backend.is_builtin
    assert backend.name == "beat_this_small"


def test_a_calibration_is_three_flags_and_never_a_subset():
    flags = Calibration(0.1, 0.2, 0.3).flags()
    assert flags.count("--salience-min-range") == 1
    assert flags.count("--salience-min-phase-margin") == 1
    assert flags.count("--salience-min-meter-margin") == 1


# --------------------------------------------- the calibration, end to end --

pytestmark_binary = pytest.mark.skipif(
    not Analyser(DEFAULT_BINARY).available,
    reason="dump_analysis is not built — see eval/analysis.py",
)


@pytestmark_binary
def test_a_backends_own_thresholds_decide_its_confidence():
    clip = make_clip(duration_sec=25, bpm=120, beats_per_bar=4, seed=1)
    analyser = Analyser(DEFAULT_BINARY)
    grid = analyser.analyse_audio(clip.audio, clip.sample_rate)

    # A clean four-four salience in a model's units: small numbers, real
    # structure. Under a calibration scaled to it, this is an answer.
    salience = np.where(np.arange(len(grid.beats)) % 4 == 0, 0.9, 0.1)
    generous = analyser.analyse_audio(
        clip.audio, clip.sample_rate, salience=salience,
        calibration=Calibration(0.05, 0.05, 0.05))
    assert generous.beats_per_bar == 4
    assert generous.downbeat_confident

    # The same salience under thresholds belonging to a different scorer is
    # withheld — which is the whole reason the three numbers travel together.
    strict = analyser.analyse_audio(
        clip.audio, clip.sample_rate, salience=salience,
        calibration=Calibration(0.05, 5.0, 5.0))
    assert strict.beats_per_bar == 4
    assert not strict.downbeat_confident


@pytestmark_binary
def test_half_a_calibration_is_refused_by_the_tool():
    # Enforced on the C++ side too, because that is where a hand-run command
    # would otherwise inherit the cue margins without saying so.
    clip = make_clip(duration_sec=10, bpm=120, beats_per_bar=4, seed=2)
    analyser = Analyser(DEFAULT_BINARY)
    grid = analyser.analyse_audio(clip.audio, clip.sample_rate)
    salience = np.where(np.arange(len(grid.beats)) % 4 == 0, 0.9, 0.1)

    import subprocess
    import tempfile
    import pathlib

    audio = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    audio.write(np.asarray(clip.audio, dtype=np.float32).tobytes())
    audio.close()
    values = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    values.write("\n".join(repr(float(v)) for v in salience))
    values.close()
    try:
        done = subprocess.run(
            [str(DEFAULT_BINARY), audio.name, "48000.0",
             "--salience", values.name, "--salience-min-range", "0.05"],
            capture_output=True, text=True)
        assert done.returncode == 2
        assert "all 3" in done.stderr
    finally:
        pathlib.Path(audio.name).unlink(missing_ok=True)
        pathlib.Path(values.name).unlink(missing_ok=True)


@pytestmark_binary
def test_the_cue_backend_needs_no_calibration_flags_at_all():
    clip = make_clip(duration_sec=25, bpm=120, beats_per_bar=4, seed=3)
    result = Analyser(DEFAULT_BINARY).analyse_audio(clip.audio, clip.sample_rate)
    assert result.salience_source == "cues"
    assert result.beats_per_bar == 4


def test_a_calibration_without_salience_is_a_mistake_not_a_default():
    with pytest.raises(ValueError, match="only applies with salience"):
        Analyser(DEFAULT_BINARY).analyse_audio(
            np.zeros(1000, dtype=np.float32), 48000.0,
            calibration=CUE_CALIBRATION)
