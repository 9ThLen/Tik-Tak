"""Splitting one programme capture back into its takes.

The operator's convenience is only worth having if the split is as trustworthy
as capturing take by take was. These tests build a programme, simulate a
recording of it, and check that every take is found where it actually is --
including under the drift that made a fixed search window unsafe.
"""
import numpy as np
import pytest

from eval.click_programme import build_programme, locate_takes
from eval.slate import RATE, build_take


def _take(seconds: float, seed: int):
    rng = np.random.default_rng(seed)
    music = rng.normal(0, 0.05, int(round(seconds * RATE)))
    return build_take(music, rate=RATE)


def _programme(count=3, seconds=4.0):
    takes = []
    for index in range(count):
        audio, layout = _take(seconds, seed=index)
        takes.append((f"track{index}", audio, layout))
    return build_programme(takes)


def _record(programme, offset_sec=0.0, drift_ppm=0.0):
    """A capture: silence, then the programme, optionally clock-stretched."""
    if drift_ppm:
        n = len(programme)
        stretched = np.interp(
            np.arange(0, n, 1.0 + drift_ppm * 1e-6), np.arange(n), programme)
        programme = stretched
    lead = np.zeros(int(round(offset_sec * RATE)))
    return np.concatenate([lead, programme, np.zeros(RATE)])


def test_every_take_is_found_at_its_true_position():
    programme, layout = _programme()
    capture = _record(programme, offset_sec=1.25)
    found = locate_takes(capture, layout)
    assert [row["track"] for row in found] == ["track0", "track1", "track2"]
    assert all(row["accepted"] for row in found), found
    for row, entry in zip(found, layout["takes"]):
        expected = 1.25 + entry["offset_sec"] + entry["layout"]["music_start_sec"]
        assert row["music_offset_sec"] == pytest.approx(expected, abs=0.01)


def test_a_late_record_button_is_tolerated_up_to_a_stated_bound():
    """The tolerance is the first take's own head-to-tail distance, not a constant.

    Every slate is the same signal, so a search window holding two of them makes
    them equal height and the finder refuses at 0 dB -- correctly, because it
    cannot tell which one it found. Long takes therefore buy the operator more
    slop than short ones, and the bound has to be derived from the take rather
    than assumed to fit it.
    """
    programme, layout = _programme(count=3, seconds=20.0)
    found = locate_takes(_record(programme, offset_sec=12.0), layout)
    assert all(row["accepted"] for row in found), found


def test_starting_too_early_refuses_and_says_by_how_much():
    """A refusal that does not name its bound is indistinguishable from a bad room."""
    programme, layout = _programme(count=2, seconds=4.0)
    found = locate_takes(_record(programme, offset_sec=12.0), layout)
    assert not any(row["accepted"] for row in found)
    assert "within the first" in found[0]["reason"]


def test_drift_does_not_walk_the_later_takes_out_of_their_windows():
    """Why the window is re-anchored per take rather than fixed.

    At 1000 ppm a fourteen-minute programme moves its last take by most of a
    second; a window sized to absorb that for take five would be loose enough
    for take one to lock onto a neighbour.
    """
    programme, layout = _programme(count=4)
    found = locate_takes(_record(programme, offset_sec=0.5, drift_ppm=800.0),
                         layout)
    assert all(row["accepted"] for row in found), found
    assert [row["track"] for row in found] == [
        f"track{i}" for i in range(4)]


def test_a_capture_that_stops_early_refuses_rather_than_guesses():
    programme, layout = _programme()
    capture = _record(programme)[: int(len(programme) * 0.55)]
    found = locate_takes(capture, layout)
    assert found[0]["accepted"]
    assert not found[-1]["accepted"]
    assert "ends before" in found[-1]["reason"] or "not found" in found[-1]["reason"]


def test_no_programme_head_means_every_take_is_refused_not_skipped():
    """Silence in place of a recording must not read as zero problems."""
    programme, layout = _programme()
    found = locate_takes(np.zeros(len(programme)), layout)
    assert len(found) == len(layout["takes"])
    assert not any(row["accepted"] for row in found)


def test_the_layout_records_where_each_take_starts():
    _, layout = _programme(count=3, seconds=4.0)
    offsets = [entry["offset_sec"] for entry in layout["takes"]]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0.0
    gaps = [b - a - layout["takes"][i]["seconds"]
            for i, (a, b) in enumerate(zip(offsets, offsets[1:]))]
    assert all(gap == pytest.approx(layout["gap_seconds"]) for gap in gaps)
