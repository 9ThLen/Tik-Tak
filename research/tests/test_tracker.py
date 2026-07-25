import numpy as np
import pytest

from eval.harness import evaluate
from tiktak.odf import OdfConfig, compute_odf
from tiktak.synth import make_clip
from tiktak.tracker import TrackerConfig, track_beats


def track(clip, **kwargs):
    result = compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))
    return track_beats(result.full, result.times, result.fps, **kwargs), result


@pytest.mark.parametrize("bpm", [90, 100, 120, 140])
def test_tracks_steady_tempo_accurately(bpm):
    clip = make_clip(bpm=bpm, duration_sec=25, seed=bpm)
    beats, _ = track(clip)
    metrics = evaluate(clip.beats, beats.beats)

    assert metrics["f_measure"] > 0.95
    assert metrics["cmlt"] > 0.9


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("swing", dict(swing=0.3)),
        ("noise", dict(noise_db=6.0)),
        ("silence lead", dict(silence_lead=3.0)),
        ("three four", dict(beats_per_bar=3)),
        ("sparse", dict(sparse=True)),
    ],
)
def test_survives_realistic_material(name, kwargs):
    clip = make_clip(bpm=120, duration_sec=25, seed=23, **kwargs)
    beats, _ = track(clip)
    metrics = evaluate(clip.beats, beats.beats)

    assert metrics["f_measure"] > 0.9, f"{name}: F={metrics['f_measure']:.2f}"


def test_sustained_tones_are_tracked_without_percussion():
    # The a cappella-adjacent case: no percussive attacks at all, and energy
    # between beats comparable to energy on them. Only spectral flux sees these
    # onsets — an energy-based detector would find nothing.
    clip = make_clip(bpm=100, duration_sec=25, sparse=True, seed=29)
    beats, _ = track(clip)
    assert evaluate(clip.beats, beats.beats)["f_measure"] > 0.9


def test_follows_a_tempo_drift():
    # The DP runs at one fixed period, but the transition penalty is soft
    # (log-squared), so the grid can still stretch with a gradual ramp.
    clip = make_clip(bpm=120, duration_sec=25, tempo_drift=20.0, seed=31)
    beats, _ = track(clip)
    metrics = evaluate(clip.beats, beats.beats)

    assert metrics["f_measure"] > 0.9
    assert metrics["cmlt"] > 0.8


def test_fixing_the_tempo_skips_estimation():
    # This is the app's manual mode: the user sets BPM, the tracker only has to
    # find the phase.
    clip = make_clip(bpm=190, duration_sec=25, seed=37)
    beats, _ = track(clip, bpm=190.0)

    assert beats.bpm == pytest.approx(190.0)
    assert evaluate(clip.beats, beats.beats)["f_measure"] > 0.9


def test_recovers_from_a_slightly_wrong_manual_tempo():
    # Manual mode has to tolerate a user who taps in 97 when the track is at
    # 120. The DP searches gaps from half the period to twice it, so the true
    # interval is still reachable, and the onset term outweighs the soft
    # log-squared penalty for deviating. The grid snaps to the real beats.
    clip = make_clip(bpm=120, duration_sec=25, seed=41)
    beats, _ = track(clip, bpm=97.0)

    assert evaluate(clip.beats, beats.beats)["f_measure"] > 0.9
    assert np.median(beats.intervals) == pytest.approx(60.0 / 120.0, rel=0.1)


def test_a_hint_beyond_the_search_window_does_fail():
    # The tolerance above is not unlimited: at 40 BPM against 120 BPM material
    # the true interval falls outside [period/2, 2*period] and cannot be
    # reached at all. Worth pinning, so the recovery above is not mistaken for
    # the tracker ignoring its tempo input entirely.
    clip = make_clip(bpm=120, duration_sec=25, seed=41)
    beats, _ = track(clip, bpm=40.0)

    assert evaluate(clip.beats, beats.beats)["f_measure"] < 0.8
    assert np.median(beats.intervals) > 60.0 / 120.0 * 1.5


def test_beat_intervals_are_regular():
    clip = make_clip(bpm=120, duration_sec=25, seed=43)
    beats, _ = track(clip)

    intervals = beats.intervals
    assert len(intervals) > 20
    # Coefficient of variation: a DP grid on steady material should be tight.
    assert intervals.std() / intervals.mean() < 0.1


def test_silence_yields_no_beats():
    beats, _ = track(make_clip(bpm=120, duration_sec=5, seed=1))
    result = compute_odf(np.zeros(48000 * 5), OdfConfig(sample_rate=48000))
    empty = track_beats(result.full, result.times, result.fps, bpm=120.0)
    assert len(empty) == 0
    del beats


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        track_beats(np.zeros(10), np.zeros(5), 100.0, bpm=120.0)
    with pytest.raises(ValueError):
        track_beats(np.zeros(10), np.zeros(10), 0.0, bpm=120.0)
    with pytest.raises(ValueError):
        track_beats(np.ones(10), np.zeros(10), 100.0, bpm=-1.0)
    with pytest.raises(ValueError):
        TrackerConfig(tightness=0.0).validate()
