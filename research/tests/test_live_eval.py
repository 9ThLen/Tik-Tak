"""The eval stand reports the causal tracker's tempo, not the offline tempo."""

import numpy as np
import pytest

from eval.analysis import Analyser, DEFAULT_BINARY, Estimate
from eval.live_corpus_benchmark import (
    load_reference_beats,
    local_reference_bpm,
    octave_statistics,
    tempo_state,
)
from tiktak.synth import make_clip


pytestmark = pytest.mark.skipif(
    not Analyser(DEFAULT_BINARY).available,
    reason="dump_analysis is not built — see eval/analysis.py",
)


def test_live_tempo_has_a_separate_final_value_and_time_series():
    clip = make_clip(duration_sec=12, bpm=120, seed=7)
    result = Analyser(DEFAULT_BINARY).analyse_live_audio(
        clip.audio, clip.sample_rate, manual_bpm=137.0)

    assert result.bpm == pytest.approx(120.0, abs=2.0)
    assert result.live_bpm == pytest.approx(137.0)
    assert result.live_tempo_spread_octaves == pytest.approx(0.0, abs=1e-6)

    columns = (
        result.live_times,
        result.live_bpms,
        result.live_confidences,
        result.live_tempo_spreads_octaves,
    )
    assert len(result.live_times) >= 10
    assert all(len(values) == len(result.live_times) for values in columns)
    assert np.all(np.diff(result.live_times) > 0.0)
    np.testing.assert_allclose(result.live_bpms, 137.0, atol=1e-6)
    assert np.all(np.isfinite(result.live_confidences))


def test_octave_statistics_separates_in_lock_and_reacquisition_switches():
    reference = np.arange(0.0, 20.0, 0.5)
    estimate = Estimate(
        beats=np.zeros(0),
        downbeats=np.zeros(0),
        beats_per_bar=0,
        downbeat_strength=0.0,
        downbeat_phase_margin=0.0,
        downbeat_meter_margin=0.0,
        live_bpm=60.0,
        live_times=np.arange(5.0, 10.0),
        live_bpms=np.array([120.0, 120.0, 240.0, 240.0, 60.0]),
        live_confidences=np.array([0.30, 0.20, 0.20, 0.01, 0.30]),
        live_tempo_spreads_octaves=np.zeros(5),
    )

    result = octave_statistics(estimate, reference)

    assert local_reference_bpm(reference, 8.0) == pytest.approx(120.0)
    assert tempo_state(60.0, 120.0) == "half"
    assert tempo_state(120.0, 120.0) == "same"
    assert tempo_state(240.0, 120.0) == "double"
    assert result["within_switches"] == 1
    assert result["reacquire_switches"] == 1
    assert result["switches"] == 2
    assert result["final_state"] == "half"


def test_normalized_manifest_annotation_header_is_accepted(tmp_path):
    annotation = tmp_path / "track.csv"
    annotation.write_text(
        "time_seconds,beat_position,is_downbeat\n"
        "0.25,1,1\n"
        "0.75,2,0\n",
        encoding="utf-8-sig",
    )

    np.testing.assert_allclose(load_reference_beats(annotation), [0.25, 0.75])
