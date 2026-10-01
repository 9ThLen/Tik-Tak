import numpy as np

from eval.closed_loop import (CLICK_ENERGY, click_to_music_db, lag_between, paired, removed_db,
                              sustain, synthetic_path, take_windows)


def layout():
    take = lambda offset, track: {"track": track, "offset_sec": offset,
                                  "layout": {"music_start_sec": 2.0, "music_seconds": 30.0}}
    return {"seconds": 100.0, "takes": [take(0.0, "a"), take(42.0, "b")]}


def test_windows_run_from_the_music_to_the_next_take_s_music():
    first, second = take_windows(layout())
    assert (first["start"], first["end"]) == (2.0, 32.0)
    # The silence after a take ends where the next take's music begins, less a
    # margin on each side so that neither the music nor its slate counts.
    assert first["silence"] == (32.5, 43.5)
    assert second["silence"] == (74.5, 99.5)


def test_coasting_and_feeding_on_its_own_click_separate_on_confidence():
    times = np.arange(0.0, 11.0, 0.02)
    beats = np.arange(0.0, 11.0, 0.5)
    coasting = sustain(beats, times, np.linspace(0.5, 0.1, len(times)), (0.0, 11.0), 0.5)
    feeding = sustain(beats, times, np.full(len(times), 0.8), (0.0, 11.0), 0.5)
    # Both keep clicking, so the click count cannot tell them apart ...
    assert coasting["beats"] == feeding["beats"] == 22
    # ... but one decays out of lock and the other holds.
    assert coasting["confidence_trend"] < -0.2
    assert abs(feeding["confidence_trend"]) < 1e-9
    assert feeding["locked_share"] == 1.0 and coasting["locked_share"] < 0.7


def test_paired_differences_use_only_takes_both_passes_scored():
    a = {"records": [{"track": "x", "f_measure": 0.8, "sustain": {"locked_share": 0.9}},
                     {"track": "y", "f_measure": 0.6, "sustain": {"locked_share": 0.5}}]}
    b = {"records": [{"track": "x", "f_measure": 0.5, "sustain": {"locked_share": 0.4}}]}
    f = paired(a, b, "f_measure")
    assert f["n"] == 1 and abs(f["mean"] - 0.3) < 1e-12
    assert abs(paired(a, b, "locked_share")["mean"] - 0.5) < 1e-12


def test_the_click_ratio_moves_one_for_one_with_the_gain():
    rate = 1000.0
    programme = np.full(40_000, 0.1, dtype=np.float32)
    window = {"start": 2.0, "end": 32.0}
    at_zero = click_to_music_db(programme, rate, window, 0.5, 0.0)
    # float32 music: 0.1 is not exactly 0.1, so allow a hair.
    assert abs(at_zero - 10.0 * np.log10(CLICK_ENERGY / 0.5 / 0.01)) < 1e-5
    assert abs(click_to_music_db(programme, rate, window, 0.5, -12.0) - (at_zero - 12.0)) < 1e-9


def test_the_lag_is_found_to_the_sample():
    rng = np.random.default_rng(1)
    rate = 8000.0
    programme = rng.standard_normal(int(120 * rate)).astype(np.float32)
    delay = 333
    capture = np.concatenate([np.zeros(delay, np.float32), programme])[:len(programme)]
    lag, clarity = lag_between(programme, capture, rate, 10.0, span_sec=20.0)
    assert abs(lag * rate - delay) < 0.5
    assert clarity > 0.99


def test_removed_is_what_was_heard_against_what_was_left_at_the_clicks():
    rate = 8000.0
    raw = np.zeros(int(4 * rate), dtype=np.float32)
    clean = np.zeros_like(raw)
    beats = np.array([1.0, 2.0, 3.0])
    for beat in beats:
        a = int(beat * rate)
        raw[a:a + 200] = 0.1
        clean[a:a + 200] = 0.001   # 40 dB down
    assert abs(removed_db(raw, clean, rate, 0.0, beats, (0.5, 3.5)) - 40.0) < 1e-3
    # Only the clicks inside the window count, and none means no figure.
    assert removed_db(raw, clean, rate, 0.0, beats, (3.6, 3.9)) is None


def test_the_made_up_room_is_the_same_every_time_and_mostly_direct():
    first, second = synthetic_path(48000.0), synthetic_path(48000.0)
    assert np.array_equal(first, second)
    assert first[0] == 1.0
    # Its diffuse tail carries a tenth of the direct sound's energy by default.
    tail = float(np.sum(first[int(0.020 * 48000):] ** 2))
    assert abs(10.0 * np.log10(tail) + 10.0) < 0.2
