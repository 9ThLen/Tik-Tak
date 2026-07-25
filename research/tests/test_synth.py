import numpy as np
import pytest

from tiktak.synth import Clip, beat_times, make_clip, make_sparse_clip


def _beat_window_rms(clip: Clip, times: np.ndarray, half_win: float = 0.010) -> np.ndarray:
    sr = clip.sample_rate
    out = []
    for t in times:
        i0 = max(int((t - half_win) * sr), 0)
        i1 = min(int((t + half_win) * sr), len(clip.audio))
        seg = clip.audio[i0:i1].astype(np.float64)
        out.append(np.sqrt(np.mean(seg**2)) if len(seg) else 0.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# Ground-truth grid
# ---------------------------------------------------------------------------

class TestBeatTimes:
    def test_constant_tempo_exact(self):
        t = beat_times(120.0, 10.0)
        assert t[0] == 0.0
        np.testing.assert_allclose(np.diff(t), 0.5, atol=1e-12)
        assert t[-1] < 10.0
        assert len(t) == 20  # beats at 0.0 .. 9.5

    def test_matches_requested_bpm_and_duration(self):
        for bpm in (60.0, 93.0, 187.5):
            t = beat_times(bpm, 30.0)
            np.testing.assert_allclose(np.diff(t), 60.0 / bpm, atol=1e-9)
            assert abs(len(t) - 30.0 * bpm / 60.0) <= 1

    def test_tempo_drift_endpoints(self):
        # 120 -> 180 BPM over 60 s: local IBI at the start ~0.5 s, at the end ~1/3 s.
        t = beat_times(120.0, 60.0, tempo_drift=60.0)
        ibi = np.diff(t)
        assert abs(ibi[0] - 60.0 / 120.0) < 0.01
        assert abs(ibi[-1] - 60.0 / 180.0) < 0.01
        # strictly accelerating
        assert np.all(np.diff(ibi) < 0)

    def test_tempo_drift_phase_is_exact(self):
        # The n-th beat must satisfy the integrated phase equation exactly:
        # phi(t_n) = n, with phi(t) = (bpm*t + drift*t^2/(2*D)) / 60.
        bpm, drift, D = 100.0, -30.0, 45.0
        t = beat_times(bpm, D, tempo_drift=drift)
        n = np.arange(len(t))
        phi = (bpm * t + drift * t**2 / (2.0 * D)) / 60.0
        np.testing.assert_allclose(phi, n, atol=1e-9)

    def test_negative_drift_decelerates(self):
        t = beat_times(140.0, 30.0, tempo_drift=-40.0)
        assert np.all(np.diff(np.diff(t)) > 0)  # IBIs grow

    def test_offset_shifts_grid(self):
        t0 = beat_times(120.0, 10.0)
        t1 = beat_times(120.0, 10.0, offset_sec=2.0)
        assert t1[0] == 2.0
        np.testing.assert_allclose(np.diff(t1), 0.5, atol=1e-12)
        assert len(t1) < len(t0)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            beat_times(0.0, 10.0)
        with pytest.raises(ValueError):
            beat_times(100.0, 10.0, tempo_drift=-100.0)


# ---------------------------------------------------------------------------
# Rendered audio
# ---------------------------------------------------------------------------

class TestMakeClip:
    def test_basic_shape_and_range(self):
        c = make_clip(bpm=120, duration_sec=8.0, seed=1)
        assert c.audio.dtype == np.float32
        assert c.audio.ndim == 1
        assert len(c.audio) == 8 * 48000
        assert np.max(np.abs(c.audio)) <= 1.0
        assert c.beats.dtype == np.float64
        # downbeats are a strict subset of beats
        assert np.all(np.isin(c.downbeats, c.beats))
        np.testing.assert_allclose(c.downbeats, c.beats[:: c.beats_per_bar])

    def test_energy_at_beats_exceeds_energy_between(self):
        # subdivisions=1: nothing is rendered between beats, so the contrast
        # must be stark.
        c = make_clip(bpm=100, duration_sec=10.0, seed=2, subdivisions=1)
        on = _beat_window_rms(c, c.beats)
        mids = (c.beats[:-1] + c.beats[1:]) / 2.0
        off = _beat_window_rms(c, mids)
        assert np.median(on) > 3.0 * np.median(off)
        assert np.all(on > 0)

    def test_energy_at_beats_with_subdivisions(self):
        # With hi-hat subdivisions there IS energy between beats, but the
        # annotated beats still dominate on average.
        c = make_clip(bpm=110, duration_sec=12.0, seed=3, subdivisions=2)
        on = _beat_window_rms(c, c.beats)
        mids = (c.beats[:-1] + c.beats[1:]) / 2.0
        off = _beat_window_rms(c, mids)
        assert np.median(on) > np.median(off)

    def test_tempo_drift_ground_truth_follows_ramp(self):
        c = make_clip(bpm=90, duration_sec=30.0, tempo_drift=30.0, seed=4)
        ibi = np.diff(c.beats)
        assert abs(ibi[0] - 60.0 / 90.0) < 0.02
        assert abs(ibi[-1] - 60.0 / 120.0) < 0.02
        # and audio still peaks at those drifting beat times
        on = _beat_window_rms(c, c.beats)
        mids = (c.beats[:-1] + c.beats[1:]) / 2.0
        off = _beat_window_rms(c, mids)
        assert np.median(on) > np.median(off)

    def test_silence_lead(self):
        lead = 2.0
        c = make_clip(bpm=120, duration_sec=10.0, silence_lead=lead, seed=5,
                      noise_db=None)
        assert c.beats[0] == pytest.approx(lead)
        head = c.audio[: int((lead - 0.05) * c.sample_rate)]
        assert np.max(np.abs(head)) == 0.0
        assert np.max(np.abs(c.audio)) > 0.1

    def test_noise_db_sets_snr(self):
        c_clean = make_clip(bpm=120, duration_sec=8.0, seed=6)
        c_noisy = make_clip(bpm=120, duration_sec=8.0, seed=6, noise_db=10.0)
        # noisy version has strictly more broadband energy in gaps
        lead_free = slice(0, len(c_clean.audio))
        p_clean = np.mean(c_clean.audio[lead_free].astype(np.float64) ** 2)
        p_noisy = np.mean(c_noisy.audio[lead_free].astype(np.float64) ** 2)
        assert p_noisy > p_clean
        assert np.max(np.abs(c_noisy.audio)) <= 1.0

    def test_swing_moves_subdivisions_not_beats(self):
        c_straight = make_clip(bpm=120, duration_sec=10.0, seed=7, swing=0.0)
        c_swung = make_clip(bpm=120, duration_sec=10.0, seed=7, swing=0.5)
        # ground truth identical
        np.testing.assert_array_equal(c_straight.beats, c_swung.beats)
        # straight midpoints carry hats; swung midpoints do not
        mids = (c_swung.beats[:-1] + c_swung.beats[1:]) / 2.0
        off_straight = _beat_window_rms(c_straight, mids)
        off_swung = _beat_window_rms(c_swung, mids)
        assert np.median(off_swung) < 0.5 * np.median(off_straight)

    def test_seed_reproducible_and_varying(self):
        a = make_clip(seed=42, duration_sec=5.0)
        b = make_clip(seed=42, duration_sec=5.0)
        c = make_clip(seed=43, duration_sec=5.0)
        np.testing.assert_array_equal(a.audio, b.audio)
        assert not np.array_equal(a.audio, c.audio)

    def test_hits_are_not_identical_impulses(self):
        c = make_clip(bpm=60, duration_sec=8.0, seed=8, subdivisions=1)
        sr = c.sample_rate
        # compare the first two downbeat kick waveforms — velocity/timbre
        # jitter must make them differ
        w = int(0.05 * sr)
        i0 = int(c.downbeats[0] * sr)
        i1 = int(c.downbeats[1] * sr)
        assert not np.allclose(c.audio[i0 : i0 + w], c.audio[i1 : i1 + w])

    def test_waltz_meter(self):
        c = make_clip(bpm=150, beats_per_bar=3, duration_sec=10.0, seed=9)
        np.testing.assert_allclose(c.downbeats, c.beats[::3])


class TestSparseClip:
    def test_sparse_has_energy_and_exact_beats(self):
        c = make_sparse_clip(bpm=80, duration_sec=12.0, seed=10)
        assert c.meta["sparse"] is True
        np.testing.assert_allclose(np.diff(c.beats), 60.0 / 80.0, atol=1e-9)
        # sustained: energy between beats comparable to energy at beats
        # (that is the point — no percussive contrast)
        on = _beat_window_rms(c, c.beats[1:-1])
        mids = (c.beats[:-1] + c.beats[1:]) / 2.0
        off = _beat_window_rms(c, mids)
        assert np.median(off) > 0.05  # sound is sustained, not gated
        assert np.median(on) > 0.05
