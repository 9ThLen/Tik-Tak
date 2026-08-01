import numpy as np
import pytest

from tiktak.odf import (
    OdfConfig,
    compute_odf,
    hann_periodic,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
)
from tiktak.synth import make_clip

SR = 48000


def test_mel_scale_round_trips():
    for hz in (0.0, 27.5, 440.0, 4000.0, 16000.0):
        assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, abs=1e-6)


def test_mel_scale_compresses_high_frequencies():
    # A fixed span in Hz covers fewer mel the higher it sits.
    low = hz_to_mel(200.0) - hz_to_mel(100.0)
    high = hz_to_mel(8100.0) - hz_to_mel(8000.0)
    assert low > high * 5.0


def test_hann_is_periodic_not_symmetric():
    w = hann_periodic(64)
    assert w[0] == pytest.approx(0.0)
    assert w[32] == pytest.approx(1.0)
    # Periodic: overlapping at half the window sums to a constant.
    assert np.allclose(w[:32] + w[32:], 1.0, atol=1e-9)


def test_no_mel_band_is_silent():
    filters = mel_filterbank(2048, SR, 81, 27.5, 16000.0)
    assert filters.shape == (81, 1025)
    assert np.all(filters.sum(axis=1) > 0.0), "every filter must touch an FFT bin"


def test_silence_produces_no_onsets():
    result = compute_odf(np.zeros(SR), OdfConfig(sample_rate=SR))
    assert len(result) > 0
    assert np.all(result.full == 0.0)
    assert np.all(result.low == 0.0)
    assert np.all(result.high == 0.0)


def test_first_frame_has_no_flux():
    # No previous spectrum to difference against, so it must be zero rather
    # than a phantom onset at t=0.
    clip = make_clip(bpm=120, duration_sec=5, seed=0)
    result = compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))
    assert result.full[0] == 0.0


def test_timestamps_are_window_centres():
    config = OdfConfig(sample_rate=SR, frame_size=1024, hop_size=256)
    result = compute_odf(np.zeros(8192), config)
    assert result.times[0] == pytest.approx(1024 * 0.5 / SR)
    assert result.times[1] - result.times[0] == pytest.approx(256 / SR)


def test_onsets_land_on_the_beat():
    clip = make_clip(bpm=120, duration_sec=20, seed=7)
    result = compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))

    on_beat = np.zeros(len(result), dtype=bool)
    for beat in clip.beats:
        nearest = int(np.argmin(np.abs(result.times - beat)))
        on_beat[max(0, nearest - 1) : nearest + 2] = True

    assert result.full[on_beat].mean() > result.full[~on_beat].mean() * 3.0


def test_low_band_takes_bass_and_high_band_takes_treble():
    def burst(freq):
        audio = np.zeros(24000)
        length = 4000
        envelope = 1.0 - np.arange(length) / length
        audio[8000 : 8000 + length] = envelope * np.sin(
            2 * np.pi * freq * np.arange(length) / SR
        )
        return compute_odf(audio, OdfConfig(sample_rate=SR))

    bass, treble = burst(60.0), burst(9000.0)
    assert bass.low.max() > bass.high.max() * 2.0
    assert treble.high.max() > treble.low.max() * 2.0


def test_whitening_strength_trades_band_balance_for_level_invariance():
    # The two properties come from the same normalisation and cannot both be
    # maximised — see the module docstring. Pin the shape, not the setting.
    def level_sensitivity(strength):
        clip = make_clip(bpm=120, duration_sec=10, seed=3)
        config = OdfConfig(sample_rate=clip.sample_rate, whitening_strength=strength)
        quiet = compute_odf(clip.audio * 0.02, config).full.max()
        loud = compute_odf(clip.audio, config).full.max()
        return loud / quiet

    def band_balance(strength):
        audio = np.zeros(24000)
        length = 4000
        envelope = 1.0 - np.arange(length) / length
        audio[8000 : 8000 + length] = envelope * np.sin(
            2 * np.pi * 60.0 * np.arange(length) / SR
        )
        result = compute_odf(audio, OdfConfig(sample_rate=SR, whitening_strength=strength))
        return result.low.max() / max(result.high.max(), 1e-9)

    assert level_sensitivity(0.0) > level_sensitivity(0.5) > level_sensitivity(1.0)
    assert level_sensitivity(1.0) == pytest.approx(1.0, abs=0.05)

    assert band_balance(0.0) > band_balance(0.5) > band_balance(1.0)
    assert band_balance(1.0) == pytest.approx(1.0, abs=0.05)
    assert band_balance(0.5) > 2.0


def test_rejects_invalid_config():
    for bad in (
        OdfConfig(sample_rate=0.0),
        OdfConfig(frame_size=1000),
        OdfConfig(hop_size=0),
        OdfConfig(hop_size=9999),
        OdfConfig(mel_bands=0),
        OdfConfig(low_band_hz=5000.0, high_band_hz=100.0),
        OdfConfig(whitening_strength=1.5),
        OdfConfig(whitening_floor_rel=2.0),
    ):
        with pytest.raises(ValueError):
            bad.validate()
