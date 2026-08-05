"""BeatNet's front end, held to a signal whose tempo is known.

Every constant here was transcribed from madmom and from BeatNet.py rather
than imported, because madmom does not build against current numpy in this
environment. A transcription error does not crash: it produces a slightly
different spectrogram, the network's activations get quietly worse, and the
model looks less useful than it is. That is the failure these prevent.

Two of them were caught this way while the module was being written, and both
are pinned below: the feature parameters live in BeatNet.py and differ from the
defaults of the class it calls in frame size, hop and bands; and the three
output classes are ordered beat, downbeat, null rather than the intuitive way
round.
"""

import numpy as np
import importlib.util

import pytest

from eval.beatnet_onnx import (
    FEATURES,
    FPS,
    HOP,
    SAMPLE_RATE,
    WEIGHTS_PATH,
    BeatNet,
    filterbank,
    log_filtered_spectrogram,
)


# ------------------------------------------------------ the front end alone --

def test_the_frame_rate_is_exactly_fifty():
    # 22050 / 441. A hop read off the feature class's own default gives 43.07
    # and every activation lands at the wrong time by a growing amount.
    assert FPS == 50.0
    assert HOP == 441


def test_the_filterbank_has_the_width_the_network_expects():
    # The check that catches the wrong parameters: the first layer's weights
    # are shaped for 272 features, and only one set of frame size, hop and
    # band count produces them. Read the defaults instead of BeatNet.py and
    # this is 168.
    bank = filterbank()
    assert bank.shape[1] * 2 == FEATURES


def test_every_filter_is_normalised_to_unit_area():
    # Unit area, not unit peak — madmom's norm_filters=True. Getting this wrong
    # scales each band by its own width, which grows with frequency, so the top
    # of the spectrum would arrive several times too loud.
    bank = filterbank()
    areas = bank.sum(axis=0)
    assert np.allclose(areas[areas > 0], 1.0)
    assert (areas > 0).all()


def test_a_pure_tone_lands_in_the_band_that_contains_it():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    for hz, lower, upper in ((110.0, 0.0, 0.35), (3000.0, 0.55, 0.95)):
        spec = log_filtered_spectrogram(np.sin(2 * np.pi * hz * t))
        bands = spec.shape[1] // 2
        loudest = spec[:, :bands].mean(axis=0).argmax() / bands
        assert lower <= loudest <= upper, (hz, loudest)


def test_the_difference_half_is_positive_and_starts_at_zero():
    rng = np.random.default_rng(0)
    spec = log_filtered_spectrogram(rng.normal(size=SAMPLE_RATE))
    bands = spec.shape[1] // 2
    difference = spec[:, bands:]
    assert (difference >= 0).all()          # positive_diffs
    assert np.all(difference[0] == 0.0)     # nothing to difference against yet


def test_silence_is_zero_rather_than_minus_infinity():
    # log10(x + 1), not log10(x): silence has to be zero, not -inf, or the
    # first layer sees a value it was never trained on.
    spec = log_filtered_spectrogram(np.zeros(SAMPLE_RATE))
    assert np.all(np.isfinite(spec))
    assert np.allclose(spec, 0.0)


def test_frames_advance_by_one_hop():
    seconds = 4
    spec = log_filtered_spectrogram(np.zeros(SAMPLE_RATE * seconds))
    assert abs(len(spec) / FPS - seconds) < 0.05


def test_too_little_audio_is_not_a_crash():
    assert log_filtered_spectrogram(np.zeros(10)).shape == (0, FEATURES)


# ------------------------------------------------------- the model, end to end --

# Two things have to be here, and they go missing independently: the weights,
# which are fetched separately and never in git, and torch, which the reference
# runs the network through. Naming both is the difference between a skip
# somebody can act on and one that sends them looking in the wrong place.
_HAS_TORCH = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.is_file() or not _HAS_TORCH,
    reason=("BeatNet weights are not present — see models/README.md"
            if not WEIGHTS_PATH.is_file()
            else "the reference runs the network on torch, which is not installed"))


@pytest.fixture(scope="module")
def click():
    import pathlib
    import soundfile as sf
    path = pathlib.Path(__file__).resolve().parents[2] / "core/tests/data/click_120.mp3"
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), float(rate)


def test_the_model_finds_the_tempo_it_is_given(click):
    # The whole point. If the filterbank, the window, the hop, the difference
    # or the class order were wrong, this comes back approximately right at
    # best and nonsense at worst.
    audio, rate = click
    activation = BeatNet().beat_activation(audio, rate)
    peaks = [i for i in range(1, len(activation) - 1)
             if activation[i] > 0.5
             and activation[i] >= activation[i - 1]
             and activation[i] > activation[i + 1]]
    times = np.array(peaks) / FPS
    assert len(times) >= 15
    assert 60.0 / np.median(np.diff(times)) == pytest.approx(120.0, abs=1.0)


def test_the_classes_are_beat_downbeat_null_in_that_order(click):
    # BeatNet.py keeps preds[:2] and discards the third, so the kept two are
    # the beat and the downbeat. Assume the intuitive order and the null class
    # ends up read as a downbeat — which, being high almost always, looks like
    # a model hearing a downbeat everywhere rather than like an index error.
    audio, rate = click
    probabilities = BeatNet().activations(audio, rate)
    assert probabilities.shape[1] == 3
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    means = probabilities.mean(axis=0)
    assert means[2] > 0.7, "the null class should hold most frames"
    assert means[0] > means[1], "beats are at least as common as downbeats"
    assert means[0] < 0.3


def test_the_activation_is_causal(click):
    # The property that makes this the online candidate: truncating the audio
    # must not change the activations that came before the cut.
    audio, rate = click
    net = BeatNet()
    whole = net.beat_activation(audio, rate)
    half = net.beat_activation(audio[: len(audio) // 2], rate)
    assert len(half) >= 10
    shared = min(len(half), len(whole)) - 2      # the last frame sees padding
    assert np.allclose(whole[:shared], half[:shared], atol=1e-9)
