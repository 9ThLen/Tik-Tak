"""Integration of the fixed veto schedule with the real C++ live tracker."""

from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pytest

from eval.octave_veto_experiment import VetoInterval, write_schedule

ROOT = pathlib.Path(__file__).resolve().parents[2]
BINARY = (ROOT / "tools" / "eval" / "build" / "RelWithDebInfo" /
          "dump_analysis.exe")


def activation(seconds: float = 12.0, fps: float = 50.0) -> np.ndarray:
    times = np.arange(0.0, seconds, 1.0 / fps)
    values = np.full(len(times), 0.02)
    for index, beat in enumerate(np.arange(0.0, seconds, 0.5)):
        nearest = int(round(beat * fps))
        if nearest < len(values):
            values[nearest] = 0.95 if index % 2 == 0 else 0.45
    return values


def invoke(audio, values, *extra):
    args = [str(BINARY), str(audio), "48000", "--live-activation", str(values),
            "--activation-fps", "50", "--live-sample-hz", "50", *extra]
    return subprocess.run(args, capture_output=True, text=True, check=False)


@pytest.mark.skipif(not BINARY.is_file(), reason="dump_analysis is not built")
def test_schedule_changes_the_anchor_but_not_the_measured_series(tmp_path) -> None:
    audio = tmp_path / "silence.f32"
    np.zeros(12 * 48_000, dtype=np.float32).tofile(audio)
    values = tmp_path / "activation.txt"
    np.savetxt(values, activation(), fmt="%.6g")
    schedule = tmp_path / "schedule.txt"
    write_schedule(schedule, (VetoInterval(7.0, 12.0, 60.0),))

    baseline = invoke(audio, values)
    veto = invoke(audio, values, "--live-anchor-veto", str(schedule))
    assert baseline.returncode == 0, baseline.stderr
    assert veto.returncode == 0, veto.stderr
    before, after = json.loads(baseline.stdout), json.loads(veto.stdout)

    assert after["live_anchor_veto_intervals"] == 1
    assert after["live_anchor_veto_frames"] > 0
    assert after["live_anchor_bpm"] == before["live_anchor_bpm"]
    assert after["live_bpms"] != before["live_bpms"]


@pytest.mark.skipif(not BINARY.is_file(), reason="dump_analysis is not built")
def test_overlapping_schedule_is_rejected_before_replay(tmp_path) -> None:
    audio = tmp_path / "silence.f32"
    np.zeros(48_000, dtype=np.float32).tofile(audio)
    values = tmp_path / "activation.txt"
    np.savetxt(values, activation(1.0), fmt="%.6g")
    schedule = tmp_path / "bad.txt"
    schedule.write_text("0 2 120\n1 3 120\n", encoding="utf-8")

    done = invoke(audio, values, "--live-anchor-veto", str(schedule))
    assert done.returncode != 0
    assert "sorted and non-overlapping" in done.stderr


@pytest.mark.skipif(not BINARY.is_file(), reason="dump_analysis is not built")
def test_model_timing_holds_back_frames_without_a_complete_window(tmp_path) -> None:
    audio = tmp_path / "silence.f32"
    np.zeros(12 * 48_000, dtype=np.float32).tofile(audio)
    values = tmp_path / "activation.txt"
    supplied = activation()
    np.savetxt(values, supplied, fmt="%.9g")

    done = invoke(audio, values, "--activation-model-timing")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert payload["live_frames"] < len(supplied)
    assert payload["live_frames"] == 597


@pytest.mark.skipif(not BINARY.is_file(), reason="dump_analysis is not built")
def test_schedule_uses_callback_time_not_the_old_model_frame_time(tmp_path) -> None:
    audio = tmp_path / "silence.f32"
    np.zeros(12 * 48_000, dtype=np.float32).tofile(audio)
    values = tmp_path / "activation.txt"
    np.savetxt(values, activation(), fmt="%.9g")
    schedule = tmp_path / "schedule.txt"
    # At seven seconds the available model frame is timestamped roughly 64 ms
    # earlier. A narrow callback-time interval therefore catches a frame only
    # when the replay uses the same clock as proposal extraction.
    write_schedule(schedule, (VetoInterval(7.0, 7.02, 60.0),))

    done = invoke(audio, values, "--activation-model-timing",
                  "--live-anchor-veto", str(schedule))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["live_anchor_veto_frames"] > 0
