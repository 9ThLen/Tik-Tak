"""The slate resolves an offset that correlation cannot, and refuses when it can't.

The last test is the one that matters. It builds material with the property
that lost two real captures -- strongly metrical music, where the onset
envelope correlates with itself at every beat -- and asks both methods for the
offset. Correlation is entitled to be ambiguous there; the slate is not.
"""
import pathlib

import numpy as np
import pytest
from scipy.signal import fftconvolve

from eval.room_recording import align
from eval.slate import (LEAD_SECONDS, MIN_PEAK_TO_SIDELOBE_DB, RATE,
                        align_by_slate, build_take, find_slate, slate)

BPM = 125.0
BEAT_SEC = 60.0 / BPM


def click_track(seconds: float, rate: int = RATE, bpm: float = BPM,
                seed: int = 7) -> np.ndarray:
    """Metrical music, in the only respect this test needs: it repeats.

    A click every beat with a louder one every fourth, which is exactly the
    self-similarity that gives an envelope correlation rival peaks a beat
    apart.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(int(round(seconds * rate)), dtype=np.float64)
    period = int(round(60.0 / bpm * rate))
    decay = np.exp(-np.arange(int(0.05 * rate)) / (0.01 * rate))
    for index, start in enumerate(range(0, len(out) - len(decay), period)):
        height = 1.0 if index % 4 == 0 else 0.55
        burst = decay * rng.standard_normal(len(decay)) * height
        out[start:start + len(decay)] += burst
    return out


MEASURED_IR = pathlib.Path(
    "D:/Programs/D/Tik-tak/music/room-ir/measured/ir.wav")


def impulse(rate: int = RATE, rt60: float = 0.35, seed: int = 3) -> np.ndarray:
    """A tail that decays *under* a direct path, which is what a room does.

    An earlier version of this helper gave the reverberant tail an amplitude
    comparable to the direct sound. That is not a room, it is a reverberation
    chamber, and it drove the slate's peak-to-sidelobe margin to 2.8 dB -- the
    method was refused on material no phone would ever record. The direct path
    dominates here, and `rt60` matches the 0.33 to 0.37 s the real room
    measured.

    Not a model of a room in any load-bearing sense:
    `room_simulation_measured.json` established that convolving even a
    *measured* response does not reproduce a real capture. It is here only so
    the test does not run on arithmetically clean audio.
    """
    rng = np.random.default_rng(seed)
    length = int(rt60 * rate)
    tail = rng.standard_normal(length) * np.exp(-np.arange(length) / (rt60 * rate / 6))
    tail *= 0.25 / max(np.max(np.abs(tail)), 1e-12)
    tail[0] = 1.0
    return tail


def room(signal: np.ndarray, rate: int = RATE, snr_db: float = 20.0,
         seed: int = 3, response: np.ndarray | None = None) -> np.ndarray:
    """Convolve, normalise, add a noise floor. FFT because the tests are long."""
    rng = np.random.default_rng(seed)
    kernel = impulse(rate=rate, seed=seed) if response is None else response
    wet = fftconvolve(signal, kernel)[: len(signal)]
    wet /= max(np.max(np.abs(wet)), 1e-12)
    return wet + rng.standard_normal(len(wet)) * 10 ** (-snr_db / 20.0)


def capture_of(take: np.ndarray, offset_sec: float, **room_kwargs) -> np.ndarray:
    """What the recorder holds: silence, then the take, through a room."""
    lead = np.zeros(int(round(offset_sec * RATE)), dtype=np.float64)
    return room(np.concatenate([lead, take]), **room_kwargs)


def test_the_slate_deconvolves_to_one_spike():
    found = find_slate(slate())
    assert found["accepted"]
    assert found["peak_to_sidelobe_db"] >= MIN_PEAK_TO_SIDELOBE_DB
    assert found["offset_sec"] == pytest.approx(0.0, abs=0.01)


def test_the_peak_is_at_the_end_of_the_sweep_and_the_answer_is_not():
    """Pins Farina's convention, because getting it wrong is silent.

    The deconvolution spike lands at `round(seconds * rate) - 1` -- the *end*
    of the slate. Reporting that as the arrival time puts every capture late by
    the whole slate, 0.5 s, which is 1.04 beats at 115 BPM: the same size as
    the ambiguity this module exists to remove, and nothing downstream would
    look wrong. It would simply score a different bar.
    """
    marker = slate()
    found = find_slate(marker)
    assert found["peak_index_sec"] == pytest.approx(
        (len(marker) - 1) / RATE, abs=1e-6)
    assert found["slate_lead_sec"] == pytest.approx(
        (len(marker) - 1) / RATE, abs=1e-6)
    assert found["offset_sec"] == pytest.approx(0.0, abs=1e-6)


def test_the_slate_is_not_at_full_scale():
    """0 dBFS into a phone invites clipping on the way in, or an AGC duck."""
    peak = float(np.max(np.abs(slate())))
    # The fades at either end keep it just under the stated amplitude rather
    # than exactly on it, which is the correct direction to be wrong in.
    assert peak == pytest.approx(0.5, abs=1e-4)
    assert peak <= 0.5


def test_the_take_says_where_the_music_starts():
    music = click_track(4.0)
    take, layout = build_take(music)
    assert layout["music_start_sec"] == pytest.approx(0.5 + LEAD_SECONDS, abs=1e-6)
    assert layout["music_seconds"] == pytest.approx(4.0, abs=1e-6)
    expected = layout["music_start_sec"] + 4.0 + LEAD_SECONDS
    assert layout["tail_slate_start_sec"] == pytest.approx(expected, abs=1e-6)
    assert len(take) / RATE == pytest.approx(expected + 0.5, abs=1e-6)


@pytest.mark.parametrize("offset_sec", [0.0, 0.476, 0.910, 3.2])
def test_the_offset_is_recovered_through_a_room(offset_sec):
    take, layout = build_take(click_track(8.0))
    got = align_by_slate(capture_of(take, offset_sec), layout)
    assert got["accepted"], got
    assert got["head"]["offset_sec"] == pytest.approx(offset_sec, abs=0.01)
    assert got["music_offset_sec"] == pytest.approx(
        offset_sec + layout["music_start_sec"], abs=0.01)
    assert abs(got["drift_sec"]) < 0.01


def test_a_capture_without_a_slate_is_refused_not_guessed():
    """The failure that matters is a confident wrong answer, not a missing one."""
    got = find_slate(room(click_track(8.0)), search=(0.0, 4.0))
    assert not got["accepted"]
    assert got["peak_to_sidelobe_db"] < MIN_PEAK_TO_SIDELOBE_DB


def test_a_missing_tail_slate_fails_the_whole_alignment():
    take, layout = build_take(click_track(6.0))
    truncated = capture_of(take, 0.4)[: int(4.0 * RATE)]
    got = align_by_slate(truncated, layout)
    assert not got["accepted"]
    assert got["reason"] == "tail slate not found"


def test_the_slate_settles_what_correlation_leaves_ambiguous():
    """The case that lost two captures, put to both methods.

    `align` is given the source and the capture with no slate in either, which
    is how the real sessions were aligned. The music repeats every beat, so its
    rivals sit a beat apart and the winner is not reliably the truth. The slate
    reads the same capture and is not entitled to an opinion about metre.
    """
    music = click_track(20.0)
    truth = 0.910

    correlated = align(music, room(np.concatenate(
        [np.zeros(int(truth * RATE)), music])), RATE, RATE, beat_sec=BEAT_SEC)
    correlation_error = abs(correlated["offset_sec"] - truth)

    take, layout = build_take(music)
    slated = align_by_slate(capture_of(take, truth), layout)

    assert slated["accepted"]
    slate_error = abs(slated["head"]["offset_sec"] - truth)
    assert slate_error < 0.01

    # The claim is not that correlation always fails -- it is that when it does,
    # it fails by about a beat, and the slate does not fail at all. Whichever
    # way correlation lands here, the slate must be the more accurate of the
    # two by a wide margin or this method buys nothing.
    assert slate_error < max(correlation_error, 0.0) or correlation_error < 0.01
    assert slate_error <= correlation_error


@pytest.mark.skipif(not MEASURED_IR.is_file(),
                    reason="the measured room response is not on this machine")
def test_the_offset_survives_the_room_that_was_actually_measured():
    """The stronger version of the room test, on a real response.

    RT60 0.33 to 0.37 s, measured through a phone with three repeats, and the
    measurement itself reports `lti: false`. If the slate cannot be read
    through the one room this project has actually been in, the procedure is
    not ready to collect with.
    """
    import soundfile

    response, rate = soundfile.read(str(MEASURED_IR), dtype="float64",
                                    always_2d=True)
    assert int(rate) == RATE
    take, layout = build_take(click_track(8.0))
    truth = 0.476
    lead = np.zeros(int(round(truth * RATE)), dtype=np.float64)
    capture = room(np.concatenate([lead, take]),
                   response=response.mean(axis=1))

    got = align_by_slate(capture, layout)
    assert got["accepted"], got
    assert got["head"]["offset_sec"] == pytest.approx(truth, abs=0.01)


def test_two_identical_slates_need_a_bounded_search():
    """A take holds two of the same marker, so an unbounded search is 0 dB.

    `find_slate` over a whole take finds the head and then measures it against
    the tail, which is exactly as tall. The margin is not small, it is zero, and
    a self-test that used it would reject a take that is perfectly good.
    `align_by_slate` bounds each search, which is the whole reason it does.
    """
    take, layout = build_take(click_track(6.0))

    unbounded = find_slate(take)
    assert unbounded["peak_to_sidelobe_db"] == pytest.approx(0.0, abs=1e-9)
    assert not unbounded["accepted"]

    bounded = align_by_slate(take, layout)
    assert bounded["accepted"]
    assert bounded["head"]["offset_sec"] == pytest.approx(0.0, abs=1e-3)
