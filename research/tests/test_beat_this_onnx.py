"""The Beat This! preprocessing, held to a signal whose answer is known.

Every constant in eval/beat_this_onnx.py was transcribed from a reference C++
port. A transcription error there does not crash and does not look wrong — it
produces a slightly blurred activation and a model that quietly underperforms
its published numbers, which is the failure mode most likely to be mistaken for
"the model is not that good on this material". These tests are what stands
between that mistake and the conclusions drawn from it.

Skipped without the model file, which is deliberately not in git.
"""

import numpy as np
import pytest

from eval.beat_this_onnx import (
    FPS,
    MODEL_PATH,
    N_MELS,
    SAMPLE_RATE,
    BeatThisOnnx,
    beats_and_downbeats,
    log_mel_spectrogram,
    mel_filterbank,
    pick_peaks,
    resample_to_model_rate,
)

CLICK = "core/tests/data/click_120.mp3"


# ------------------------------------------------------ preprocessing alone --

def test_the_frame_rate_is_exactly_fifty():
    # 22050 / 441 divides exactly. A hop that did not would put every frame time
    # slightly off and no test would notice until beats drifted late in a song.
    assert FPS == 50.0


def test_the_filterbank_is_triangles_that_do_not_get_area_normalised():
    bank = mel_filterbank()
    assert bank.shape == (513, N_MELS)
    peaks = bank.max(axis=0)
    assert bank.min() >= 0.0

    # Slaney *scale*, but no Slaney *normalisation*. The distinction that
    # matters: normalisation divides each triangle by its width, so wide
    # high-frequency filters would peak far below narrow low-frequency ones.
    # Here every triangle reaches 1.0 at its centre, and the only reason a
    # sampled peak falls short is that no FFT bin lands exactly on that centre
    # — worst at the bottom, where the bins are wide relative to the spacing.
    assert peaks.max() <= 1.0 + 1e-9
    assert peaks[N_MELS // 2:].min() > 0.8, "wide filters are not scaled down"
    assert peaks.min() > 0.5, peaks.min()


def test_a_pure_tone_lands_in_the_band_that_contains_it():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    spec = log_mel_spectrogram(np.sin(2 * np.pi * 440.0 * t))

    assert spec.shape[1] == N_MELS
    loudest = spec.mean(axis=0).argmax()
    # 440 Hz sits low in a 30..11000 Hz bank; anywhere in the top half would
    # mean the mel scale or the filterbank is wrong.
    assert loudest < N_MELS // 2


def test_silence_compresses_to_zero_rather_than_minus_infinity():
    # log1p of a clamped floor, not dB: dB would give -200 and the model has
    # never seen that.
    spec = log_mel_spectrogram(np.zeros(SAMPLE_RATE))
    assert np.all(np.isfinite(spec))
    assert np.allclose(spec, 0.0, atol=1e-6)


def test_frames_advance_by_one_hop():
    seconds = 4
    spec = log_mel_spectrogram(np.zeros(SAMPLE_RATE * seconds))
    assert abs(len(spec) / FPS - seconds) < 0.05


def test_resampling_keeps_the_length_in_seconds():
    audio = np.zeros(48000 * 3)
    out = resample_to_model_rate(audio, 48000)
    assert abs(len(out) / SAMPLE_RATE - 3.0) < 0.01
    # Already at the model rate, nothing should happen at all.
    same = resample_to_model_rate(audio, SAMPLE_RATE)
    assert len(same) == len(audio)


def test_peaks_are_local_maxima_above_a_half_and_never_adjacent():
    logits = np.array([-5.0, 1.0, 2.0, 1.0, -5.0, 0.5, 0.6, -5.0])
    peaks = pick_peaks(logits)

    assert 2 in peaks                       # the clear maximum
    assert 0 not in peaks and 4 not in peaks  # negative logits are below a half
    assert np.all(np.diff(peaks) > 1)


def test_no_frames_is_not_a_crash():
    assert len(pick_peaks(np.zeros(0))) == 0
    assert log_mel_spectrogram(np.zeros(10)).shape == (0, N_MELS)


# ------------------------------------------------- the model, end to end -----

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason="beat_this.onnx is not present — see models/README.md",
)


@pytest.fixture(scope="module")
def click():
    import pathlib
    import soundfile as sf

    path = pathlib.Path(__file__).resolve().parents[2] / CLICK
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), float(rate)


def test_the_model_finds_the_tempo_and_metre_it_is_given(click):
    # The whole point of this file. 120 BPM, four-four, kick on the one — if
    # the preprocessing were wrong in any of the ways it could be, this comes
    # back approximately right instead of exactly right.
    audio, rate = click
    beats, downbeats = beats_and_downbeats(BeatThisOnnx().activations(audio, rate))

    assert len(beats) >= 15
    assert np.median(np.diff(beats)) == pytest.approx(0.5, abs=0.01)
    assert len(downbeats) >= 4
    bar = np.median(np.diff(downbeats)) / np.median(np.diff(beats))
    assert bar == pytest.approx(4.0, abs=0.1)


def test_downbeats_are_a_subset_of_the_beats(click):
    audio, rate = click
    beats, downbeats = beats_and_downbeats(BeatThisOnnx().activations(audio, rate))
    # Snapped on purpose: the two heads are independent and a downbeat one frame
    # off its beat would otherwise score as a miss against our own grid.
    assert set(np.round(downbeats, 9)) <= set(np.round(beats, 9))


def test_activations_cover_every_frame_of_the_piece(click):
    # The chunking stitches 1500-frame windows with a 6-frame border. A gap
    # would leave the -1000 sentinel behind and read as certain silence.
    audio, rate = click
    act = BeatThisOnnx().activations(audio, rate)
    assert len(act.beat) == len(act.downbeat) > 0
    assert act.beat.min() > -999.0
    assert act.downbeat.min() > -999.0
