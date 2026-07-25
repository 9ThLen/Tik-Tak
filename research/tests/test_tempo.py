import numpy as np
import pytest

from tiktak.odf import OdfConfig, compute_odf
from tiktak.synth import make_clip
from tiktak.tempo import (
    TempoConfig,
    autocorrelation,
    estimate_tempo,
    estimate_tempo_windowed,
    tempo_prior,
)


def odf_of(clip):
    return compute_odf(clip.audio, OdfConfig(sample_rate=clip.sample_rate))


def test_autocorrelation_peaks_at_the_true_period():
    fps = 100.0
    period = 25  # frames
    pulses = np.zeros(2000)
    pulses[::period] = 1.0

    acf = autocorrelation(pulses)
    # Lag 0 aside, the period and its multiples should dominate.
    assert acf[period] > acf[period // 2] * 3
    assert acf[period] > acf[period + period // 2] * 3


def test_autocorrelation_of_silence_is_finite():
    acf = autocorrelation(np.zeros(100))
    assert np.all(np.isfinite(acf))


def test_prior_peaks_at_its_centre():
    config = TempoConfig(prior_centre_bpm=120.0)
    grid = np.array([60.0, 120.0, 240.0])
    prior = tempo_prior(grid, config)

    assert prior[1] == pytest.approx(1.0)
    # Symmetric in octaves: half and double are equally penalised.
    assert prior[0] == pytest.approx(prior[2])
    assert prior[0] < prior[1]


@pytest.mark.parametrize("bpm", [90, 100, 120, 140])
def test_estimates_moderate_tempi_exactly(bpm):
    # Tempi near the prior centre must come out right; the octave errors this
    # estimator does make are at the extremes, and are tested separately.
    clip = make_clip(bpm=bpm, duration_sec=25, seed=bpm)
    result = odf_of(clip)
    estimate = estimate_tempo(result.full, result.fps)

    assert estimate.bpm == pytest.approx(bpm, rel=0.04)
    assert estimate.confidence > 0.5


@pytest.mark.parametrize("bpm", [72, 168, 190])
def test_extreme_tempi_are_at_worst_an_octave_out(bpm):
    # A documented, accepted limitation: the log-normal prior centred at 120 BPM
    # pulls the extremes towards it. The estimate must still be metrically
    # related to the truth — a non-metrical answer would be a real failure.
    clip = make_clip(bpm=bpm, duration_sec=25, seed=bpm)
    result = odf_of(clip)
    estimate = estimate_tempo(result.full, result.fps)

    ratio = estimate.bpm / bpm
    assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.06, (
        f"{bpm} BPM estimated as {estimate.bpm:.1f}, which is not a metrical relative"
    )


def test_comb_summation_removes_non_metrical_peaks():
    # Without the comb, a kick/snare pattern produces a strong autocorrelation
    # peak at two-thirds of the beat, where unlike events happen to align. With
    # it, candidates must be supported at every metrical level above them.
    clip = make_clip(bpm=120, duration_sec=25, seed=11)
    result = odf_of(clip)

    plain = estimate_tempo(result.full, result.fps, TempoConfig(comb_harmonics=1))
    combed = estimate_tempo(result.full, result.fps, TempoConfig(comb_harmonics=4))

    def metrical(bpm):
        ratio = bpm / 120.0
        return min(abs(ratio - r) for r in (0.25, 0.5, 1.0, 2.0, 4.0)) < 0.06

    assert metrical(combed.bpm)
    # The plain estimator is allowed to be wrong here; the point is that the
    # combed one is not. Assert only on what we depend on.
    del plain


def test_silence_reports_no_confidence():
    estimate = estimate_tempo(np.zeros(1000), 93.75)
    assert estimate.confidence == 0.0


def test_top_candidates_are_distinct():
    clip = make_clip(bpm=120, duration_sec=25, seed=13)
    result = odf_of(clip)
    estimate = estimate_tempo(result.full, result.fps)

    candidates = estimate.top_candidates(3)
    assert len(candidates) == 3
    assert candidates[0][1] == pytest.approx(1.0)

    bpms = [bpm for bpm, _ in candidates]
    for i, a in enumerate(bpms):
        for b in bpms[i + 1 :]:
            assert abs(np.log2(a / b)) >= 0.2, "candidates must be genuinely distinct"


def test_windowed_estimate_follows_a_drift():
    clip = make_clip(bpm=100, duration_sec=40, tempo_drift=40.0, seed=17)
    result = odf_of(clip)

    times, bpms, confidence = estimate_tempo_windowed(
        result.full, result.fps, window_sec=8.0, hop_sec=2.0
    )
    assert len(times) == len(bpms) == len(confidence)

    # Compare the first and last thirds rather than adjacent windows, which are
    # noisy. Allow an octave: what is under test is that the estimate *moves*.
    early = np.median(bpms[: len(bpms) // 3])
    late = np.median(bpms[-len(bpms) // 3 :])
    assert late > early, f"tempo ramped up but estimate went {early:.0f} -> {late:.0f}"


def test_rejects_invalid_config():
    for bad in (
        TempoConfig(min_bpm=0.0),
        TempoConfig(min_bpm=200.0, max_bpm=100.0),
        TempoConfig(prior_width_octaves=0.0),
        TempoConfig(grid_size=2),
        TempoConfig(comb_harmonics=0),
        TempoConfig(comb_weight_decay=-1.0),
    ):
        with pytest.raises(ValueError):
            bad.validate()

    with pytest.raises(ValueError):
        estimate_tempo(np.zeros(100), fps=0.0)
