"""The moving-phase decoder, held to signals whose bar lines are known.

The reason these matter: this decoder is what a measurement said was worth
0.18 F on a learned activation, and that measurement is the argument for a
delicate change to the C++ resolver. A decoder that quietly drifts, or that
does not actually reduce to the current resolver at infinite cost, would make
that argument out of nothing.
"""

import numpy as np
import pytest

from eval.moving_phase import SINGLE_PHASE, bar_positions, decode, salience_from_cues


def planted(meter, bars, offset=0, height=1.0, noise=0.0, seed=0):
    """A salience with a downbeat every `meter` beats, starting at `offset`."""
    n = meter * bars
    salience = np.zeros(n)
    salience[(np.arange(n) - offset) % meter == 0] = height
    if noise:
        salience += np.random.default_rng(seed).normal(0.0, noise, n)
    return salience


# --------------------------------------------------- reducing to the resolver --

@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_an_infinite_switch_cost_finds_the_one_planted_phase(offset):
    downbeats = decode(planted(4, 12, offset), 4, SINGLE_PHASE)
    assert list(downbeats) == list(range(offset, 48, 4))


def test_an_infinite_switch_cost_never_changes_phase():
    # However badly the evidence argues for moving, infinity cannot be repaid.
    salience = np.concatenate([planted(4, 8, 0), planted(4, 8, 2)])
    path = bar_positions(salience, 4, SINGLE_PHASE)
    assert len(set(path.tolist())) == 1


def test_it_agrees_with_scoring_every_phase_the_way_the_resolver_does():
    # The resolver picks the phase maximising mean(in) - mean(out). At an
    # infinite switch cost this decoder must reach the same answer, or the
    # baseline row of the comparison is not the resolver's row.
    rng = np.random.default_rng(7)
    for meter in (2, 3, 4, 6):
        salience = rng.normal(size=meter * 15)
        n = len(salience)
        contrasts = []
        for phase in range(meter):
            inside = (np.arange(n) % meter) == phase
            contrasts.append(salience[inside].mean() - salience[~inside].mean())
        expected = int(np.argmax(contrasts))
        assert bar_positions(salience, meter, SINGLE_PHASE)[0] == expected


# ------------------------------------------------------------ moving the phase --

def test_a_planted_phase_change_is_found_when_switching_is_cheap():
    # Eight bars on phase 0, then eight on phase 2 — the section-boundary bar
    # insertion that a single global phase provably cannot represent.
    salience = np.concatenate([planted(4, 8, 0), planted(4, 8, 2)])
    downbeats = set(decode(salience, 4, switch_cost=0.5).tolist())

    assert {0, 4, 8}.issubset(downbeats)          # the first section
    assert {34, 38, 42}.issubset(downbeats)       # the second, two beats over
    # A single global phase gets one section right and the other entirely wrong.
    single = set(decode(salience, 4, SINGLE_PHASE).tolist())
    assert not {34, 38, 42}.issubset(single)


def test_it_does_not_switch_on_material_with_nothing_to_switch_on():
    # The failure that would make the F gain meaningless: a decoder free to
    # move the phase drifting until it matches anything. On a regular signal
    # it has to stay put across the range of costs that get used.
    salience = planted(4, 30, offset=1, noise=0.15, seed=3)
    for cost in (4.0, 1.0, 0.25):
        path = bar_positions(salience, 4, cost)
        assert len(set(path.tolist())) == 1, f"drifted at switch_cost={cost}"


def test_a_vanishing_switch_cost_does_chase_noise():
    # Pinned rather than hidden, because it marks where the usable range ends.
    # With switching nearly free the decoder has no reason to prefer the phase
    # it is on, and noise is enough to move it. Anything reading the sweep
    # should know the bottom of it is not a better setting.
    salience = planted(4, 30, offset=1, noise=0.15, seed=3)
    assert len(set(bar_positions(salience, 4, 0.01).tolist())) > 1


def test_cheaper_switching_never_produces_fewer_switches():
    salience = np.concatenate([planted(4, 6, 0), planted(4, 6, 1), planted(4, 6, 3)])
    counts = [np.sum(np.diff(bar_positions(salience, 4, c)) != 0)
              for c in (8.0, 4.0, 2.0, 1.0, 0.5)]
    assert counts == sorted(counts)


# ------------------------------------------------------------------- the edges --

def test_degenerate_input_is_not_a_crash():
    assert len(decode([], 4)) == 0
    assert len(bar_positions([], 4)) == 0
    assert list(decode([1.0], 4)) == [0]
    assert list(decode(np.zeros(8), 4, 1.0)) == [0, 4]   # flat: earliest phase
    with pytest.raises(ValueError):
        decode(np.zeros(8), 1)


def test_the_cue_mix_matches_what_the_core_computes():
    # Standardised low band plus floored, fixed-scale harmony — the asymmetry
    # is deliberate in the core and copying it wrongly here would sweep weights
    # that mean something else. See DownbeatConfig.
    estimate = {"cue_low": [0.0, 1.0, 2.0, 3.0], "cue_harmony": [0.0, 0.05, 0.10, 0.25]}
    mixed = salience_from_cues(estimate)

    low = np.array([0.0, 1.0, 2.0, 3.0])
    low = (low - low.mean()) / low.std()
    harmony = np.array([0.0, 0.0, 0.05, 0.20]) * 12.0
    assert mixed == pytest.approx(low + harmony)

    # Harmony below the floor is noise and must not vote at all.
    quiet = salience_from_cues({"cue_low": [0.0, 1.0], "cue_harmony": [0.04, 0.04]})
    loud = salience_from_cues({"cue_low": [0.0, 1.0], "cue_harmony": [0.0, 0.0]})
    assert quiet == pytest.approx(loud)
