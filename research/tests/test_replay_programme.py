"""The P1-B0 replay programme: the subset rule, the split, and the session gate.

The subset has to be the one the rule names, the programme has to split back
under the conditions a real pass produces -- a late record button, two device
clocks, a quiet level over a noise floor -- and a session that did not record
its OS-side processing has to be refused rather than scored.
"""
import numpy as np
import pytest

from eval.click_programme import build_programme, locate_takes
from eval.replay_programme import (GAP_SECONDS, LEVELS_DB, SESSION_SCHEMA,
                                   check, paired_interval, resample,
                                   select_excerpts, validate_session)
from eval.slate import RATE, build_take

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal",
          "pop", "reggae", "rock"]


def _rows(per_genre=100, bad=()):
    return [{"dataset": "gtzan", "genre": genre,
             "track_id": f"{genre}.{index:05d}",
             "status": "exclude" if f"{genre}.{index:05d}" in bad else "ok"}
            for genre in GENRES for index in range(per_genre)]


def test_two_per_genre_from_the_middle_of_equal_slices():
    chosen = [row["track_id"] for row in select_excerpts(_rows())]
    assert len(chosen) == 20
    assert chosen[:2] == ["blues.00025", "blues.00075"]
    assert chosen[-2:] == ["rock.00025", "rock.00075"]


def test_only_ok_rows_are_eligible_and_the_rule_is_deterministic():
    rows = _rows(bad={"jazz.00054", "reggae.00075"})
    first = select_excerpts(rows)
    assert all(row["status"] == "ok" for row in first)
    assert "reggae.00075" not in [row["track_id"] for row in first]
    assert first == select_excerpts(list(reversed(rows)))


def test_other_corpora_never_enter():
    rows = _rows() + [{"dataset": "ballroom", "genre": "blues",
                       "track_id": "blues.99999", "status": "ok"}]
    assert "blues.99999" not in [r["track_id"] for r in select_excerpts(rows)]


def test_resampling_keeps_duration():
    audio = np.random.default_rng(0).normal(0, 0.1, 22050 * 3)
    assert len(resample(audio, 22050)) == RATE * 3


def _excerpt_programme(count, gain_db=0.0, seconds=30.0):
    takes = []
    for index in range(count):
        rng = np.random.default_rng(index)
        music = rng.normal(0, 0.1, int(round(seconds * RATE)))
        take, layout = build_take(music * 10 ** (gain_db / 20))
        takes.append((f"x{index}", take, layout))
    return build_programme(takes, gap_seconds=GAP_SECONDS)


def _record(programme, offset_sec, drift_ppm=0.0, noise_dbfs=None, seed=99):
    if drift_ppm:
        n = len(programme)
        programme = np.interp(np.arange(0, n, 1.0 + drift_ppm * 1e-6),
                              np.arange(n), programme)
    capture = np.concatenate([np.zeros(int(round(offset_sec * RATE))),
                              programme, np.zeros(RATE)])
    if noise_dbfs is not None:
        rng = np.random.default_rng(seed)
        capture = capture + rng.normal(0, 10 ** (noise_dbfs / 20), len(capture))
    return capture


@pytest.mark.parametrize("offset_sec,drift_ppm", [
    (0.5, 0.0), (20.0, 0.0), (3.0, 200.0), (3.0, -200.0)])
def test_thirty_second_takes_split_back_with_an_eight_second_gap(
        offset_sec, drift_ppm):
    """The gap is pinned here: a slow record button and two device clocks."""
    programme, layout = _excerpt_programme(4)
    found = check(_record(programme, offset_sec, drift_ppm), layout)
    assert found["aligned"] == 4, found["takes"]
    for row, entry in zip(found["takes"], layout["takes"]):
        expected = (offset_sec + entry["offset_sec"]
                    + entry["layout"]["music_start_sec"])
        assert row["music_offset_sec"] == pytest.approx(
            expected, abs=0.01 + abs(drift_ppm) * 1e-6 * expected)


def test_the_quiet_programme_aligns_over_a_room_noise_floor():
    programme, layout = _excerpt_programme(3, gain_db=LEVELS_DB["quiet"])
    found = check(_record(programme, 2.0, noise_dbfs=-50.0), layout)
    assert found["aligned"] == 3, found["takes"]


def test_a_capture_that_stops_early_is_refused_not_guessed():
    programme, layout = _excerpt_programme(3)
    cut = programme[: int(len(programme) * 0.6)]
    found = check(_record(cut, 1.0), layout)
    assert found["takes"][-1]["accepted"] is False
    assert not found["a1_fraction_met"]


def _session(**overrides):
    capture = {"file": "captures/a.m4a", "level": "normal", "distance": "near",
               "device": "phone", "device_model": "x", "channels": 2,
               "os_processing": "unknown"}
    capture.update(overrides)
    return {"schema": SESSION_SCHEMA, "captures": [capture]}


def test_unknown_processing_is_accepted_when_it_is_said():
    validate_session(_session())


@pytest.mark.parametrize("field", ["os_processing", "device_model", "file"])
def test_a_capture_missing_a_required_record_is_refused(field):
    with pytest.raises(ValueError, match=field):
        validate_session(_session(**{field: None}))


def test_an_undeclared_level_is_refused():
    with pytest.raises(ValueError, match="undeclared level"):
        validate_session(_session(level="loud"))


def test_the_interval_is_repeatable_and_brackets_the_mean():
    deltas = list(np.random.default_rng(1).normal(-0.2, 0.1, 20))
    first, second = paired_interval(deltas), paired_interval(deltas)
    assert first == second
    assert first[0] < np.mean(deltas) < first[1]
    assert paired_interval([0.1]) is None


def test_a2_brackets_its_own_estimate_and_grows_with_spread():
    from eval.replay_programme import captures_needed

    narrow = captures_needed(list(np.random.default_rng(2).normal(0, 0.05, 20)))
    wide = captures_needed(list(np.random.default_rng(2).normal(0, 0.15, 20)))
    assert narrow["ci95"][0] <= narrow["estimate"] <= narrow["ci95"][1]
    assert wide["estimate"] > narrow["estimate"]
    assert captures_needed([0.1]) is None
