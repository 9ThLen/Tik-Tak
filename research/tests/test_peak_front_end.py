"""Regression tests for the experiment driver, not only the peak map."""

import hashlib
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from eval.activation_recall import matched, top_n_times_and_chance
from eval.peak_front_end import (alignment_offsets, score, selection_record,
                                 shared_reference, write_features)


def test_top_n_reuses_the_signal_specific_shuffled_reference():
    reference = np.arange(1.0, 11.0)
    candidate_times = reference.copy()
    candidate_heights = np.linspace(10.0, 1.0, len(reference))

    selected, chance_reference = top_n_times_and_chance(
        reference, candidate_times, candidate_heights, "track")

    assert selected.tolist() == candidate_times.tolist()
    assert matched(reference, selected) == len(reference)
    assert matched(chance_reference, selected) < len(reference)


def test_top_n_ties_resolve_by_earliest_time():
    reference = np.array([1.0, 2.0])
    candidates = np.array([1.0, 2.0, 3.0])
    selected, _ = top_n_times_and_chance(
        reference, candidates, np.ones(3), "ties")
    assert selected.tolist() == [1.0, 2.0]


def test_shared_beats_give_both_conditions_the_same_population():
    times_short = np.arange(0.0, 10.0, 0.02)
    times_long = np.arange(0.0, 12.0, 0.02)
    raw = np.arange(5.0, 12.0, 0.5)
    shared, _, common_end = shared_reference(
        raw, [SimpleNamespace(times=times_short),
              SimpleNamespace(times=times_long)])

    short = score(np.sin(times_short * 4.0) + 2.0,
                  times_short, shared, "paired")
    long = score(np.sin(times_long * 4.0) + 2.0,
                 times_long, shared, "paired")
    assert short["beats"] == long["beats"] == len(shared)


def test_out_of_interval_candidates_cannot_consume_top_n_slots():
    times = np.arange(0.0, 20.0, 0.02)
    beats = np.arange(5.0, 10.0, 0.5)
    baseline = np.full(len(times), 0.1)
    for beat in beats:
        baseline[np.argmin(np.abs(times - beat))] = 2.0
    tailed = baseline.copy()
    for at in np.arange(12.0, 19.0, 0.5):
        tailed[np.argmin(np.abs(times - at))] = 20.0
    interval = {"start_sec": 5.0, "end_sec": 10.0}

    clean = score(baseline, times, beats, "paired", interval)
    room = score(tailed, times, beats, "paired", interval)
    assert clean["top_n"] == room["top_n"]
    assert clean["top_n_chance"] == room["top_n_chance"]


def test_feature_dump_is_regenerated_even_when_the_path_exists(
        tmp_path, monkeypatch):
    out = tmp_path / "cached.ttfd"
    out.write_bytes(b"stale")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        pathlib.Path(command[-1]).write_bytes(b"fresh")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("eval.peak_front_end.subprocess.run", fake_run)
    monkeypatch.setattr("eval.peak_front_end.read_features", lambda path: object())
    write_features(pathlib.Path("dump_analysis"), pathlib.Path("audio.wav"), out)

    assert len(calls) == 1
    assert out.read_bytes() == b"fresh"


def test_successful_noop_cannot_reuse_a_stale_feature_dump(
        tmp_path, monkeypatch):
    out = tmp_path / "cached.ttfd"
    out.write_bytes(b"stale")
    monkeypatch.setattr(
        "eval.peak_front_end.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))

    with pytest.raises(RuntimeError, match="invalid feature file"):
        write_features(pathlib.Path("dump_analysis"),
                       pathlib.Path("audio.wav"), out)
    assert out.read_bytes() == b"stale"


def test_selected_novelty_horizon_is_serialized_effectively():
    label = "b2_p6_f0_r2_n10_union"
    record = selection_record(label, "novelty25")
    assert record["selection_key"] == {
        "parameters": label, "readout": "novelty25"}
    assert record["parameters"]["novelty_frames"] == 25


def test_alignment_record_must_name_the_room_input(tmp_path):
    rows = []
    inputs = {}
    for track in ("0116_goodies", "0132_iceicebaby", "0466_onthedarkside",
                  "0707_halfwaygone", "0837_nottonight"):
        actual = (tmp_path / track / f"{track}.wav").resolve()
        actual.parent.mkdir(parents=True)
        content = track.encode("utf-8")
        actual.write_bytes(content)
        inputs[track] = actual
        rows.append({"name": track, "aligned_audio": str(actual),
                     "aligned_audio_sha256": hashlib.sha256(content).hexdigest(),
                     "alignment": {"offset_sec": 1.0, "skip_sec": 0.0}})
    artifact = tmp_path / "alignment.json"
    artifact.write_text(json.dumps({"records": rows}), "utf-8")

    assert alignment_offsets(artifact, inputs)["0116_goodies"]["offset_sec"] == 1.0
    relocated = tmp_path / "relocated.wav"
    relocated.write_bytes(b"0116_goodies")
    inputs["0116_goodies"] = relocated
    assert alignment_offsets(artifact, inputs)["0116_goodies"][
        "aligned_audio"] == str(relocated.resolve())

    relocated.write_bytes(b"different")
    with pytest.raises(ValueError, match="does not match"):
        alignment_offsets(artifact, inputs)
