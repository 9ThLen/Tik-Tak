import numpy as np

from eval.learned_file_path import paired, rendered_downbeats


def _moved(intro_phase, intro_bars, body_phase, body_bars):
    beats = 0.5 * np.arange(4 * (intro_bars + body_bars + 1))
    downbeats = [beats[4 * bar + intro_phase] for bar in range(intro_bars)]
    downbeats += [beats[4 * bar + body_phase]
                  for bar in range(intro_bars, intro_bars + body_bars)]
    return beats, np.asarray(downbeats)


def test_a_player_is_handed_the_bodys_phase_not_the_intros():
    # Mirrors the core's Offline.APlayerIsHandedTheBodysBarPhaseNotTheIntros.
    beats, downbeats = _moved(1, 2, 2, 20)
    np.testing.assert_array_equal(rendered_downbeats(beats, downbeats, 4), beats[2::4])


def test_a_tie_keeps_the_first_bar_lines_phase():
    beats, downbeats = _moved(1, 4, 2, 4)
    np.testing.assert_array_equal(rendered_downbeats(beats, downbeats, 4), beats[1::4])


def test_nothing_to_render_without_a_metre_or_bar_lines():
    beats, downbeats = _moved(0, 0, 0, 8)
    assert len(rendered_downbeats(beats, downbeats, 0)) == 0
    assert len(rendered_downbeats(beats, np.zeros(0), 4)) == 0


def test_paired_skips_what_either_arm_could_not_score():
    records = [{"a": {"x": 1.0}, "b": {"x": 0.5}},
               {"a": {"x": float("nan")}, "b": {"x": 0.2}},
               {"a": {"x": 0.4}, "b": {}}]
    result = paired(records, "a", "b", "x")
    assert result["n"] == 1
    assert result["delta"] == 0.5
    assert result["better"] == 1 and result["worse"] == 0
