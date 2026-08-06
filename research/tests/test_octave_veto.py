"""I1-I7 from `eval/PREREGISTERED_octave_veto.md`, plus event extraction.

These run on constructed activation tracks and no audio. The pre-registration
requires all of them to pass before a corpus is touched, and names I1 and I4 as
the ones to write first: they are the specific defect the previous audit had,
and a decoder that passes the waltz test while failing them is that run again
wearing a new formula.

**I2 does not pass as registered.** That is recorded here and in the
pre-registration's deviations appendix rather than sanded off, and the test
below asserts the behaviour that was measured instead of the behaviour that was
predicted. See `TestI2BeatOnlyChannel` for the arithmetic and for why the
obvious repair is the octave freeze's mistake in new clothes.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.octave_veto import (Decision, WindowTrack, admissible_metres,
                              committed_grid, common_metres, doubled_grid,
                              extract_proposals, halved_grids, judge, octave_k,
                              window_beats, zscores)

FPS = 50.0
PERIOD = 0.5  # 120 BPM committed


def frames(n_beats: int, period: float = PERIOD) -> np.ndarray:
    return np.arange(0.0, (n_beats + 2) * period, 1.0 / FPS)


def beats(n: int, period: float = PERIOD, t0: float = 1.0) -> np.ndarray:
    return t0 + period * np.arange(n)


def pulse(t: np.ndarray, times, width: float = 0.06,
          high: float = 1.0, low: float = 0.02) -> np.ndarray:
    """A downbeat channel: gaussian bumps at `times`, near-silent elsewhere."""
    out = np.full(len(t), low)
    for x in times:
        out = np.maximum(out, high * np.exp(-0.5 * ((t - x) / width) ** 2))
    return out


# --- I1: iid noise decides nothing ------------------------------------------


class TestI1NoiseDecidesNothing:
    """The registered form: mean within 0.05 of zero, `P(D > 0)` 50% +- 3.

    Stated at `tau = 0`, because the invariants run before `tau` exists.
    """

    @pytest.mark.parametrize("k", [+1, -1])
    def test_neither_grid_wins_systematically(self, k: int) -> None:
        t, b = frames(20), beats(16)
        rng = np.random.default_rng(20260806)
        deltas = np.array([judge(WindowTrack(t, rng.random(len(t))), b, k).delta
                           for _ in range(1000)])
        assert abs(float(deltas.mean())) < 0.05
        assert abs(100.0 * float((deltas > 0).mean()) - 50.0) < 3.0

    def test_the_null_subtraction_is_what_removes_the_bias(self) -> None:
        """Without it the halving side wins on noise, which is the old defect.

        `k = -1` takes a maximum over two alignments of the proposed grid and
        the committed side takes none, so the raw difference is biased against
        the committed level before any evidence is read. The identical maximum
        inside the null is what cancels it.
        """
        t, b = frames(20), beats(16)
        rng = np.random.default_rng(7)
        raw, full = [], []
        for _ in range(400):
            d = judge(WindowTrack(t, rng.random(len(t))), b, -1)
            raw.append(d.delta_raw)
            full.append(d.delta)
        assert float(np.mean(raw)) < -0.10
        assert abs(float(np.mean(full))) < 0.05


# --- I4: length alone does not flip the sign --------------------------------


class TestI4LengthDoesNotDecide:
    @pytest.mark.parametrize("n", [16, 24, 32])
    def test_a_clean_signal_keeps_its_sign_at_every_length(self, n: int) -> None:
        t, b = frames(n + 4), beats(n)
        track = WindowTrack(t, pulse(t, b[0::4]))
        assert judge(track, b, +1, window_beats=n).delta > 0.0
        assert judge(track, b, -1, window_beats=n).delta > 0.0

    @pytest.mark.parametrize("n", [16, 24, 32])
    def test_noise_stays_centred_at_every_length(self, n: int) -> None:
        t, b = frames(n + 4), beats(n)
        rng = np.random.default_rng(100 + n)
        deltas = np.array([judge(WindowTrack(t, rng.random(len(t))), b, -1,
                                 window_beats=n).delta for _ in range(400)])
        assert abs(float(deltas.mean())) < 0.06


# --- I2: what the registered invariant asked, and what is true --------------


class TestI2BeatOnlyChannel:
    """The registered I2 asked for `|D| < 0.05` on a beat-only channel. False.

    Measured: `D = -2.02` at `k = +1`. The mechanism, from the arithmetic:

    * a channel high at every committed beat is **constant** when sampled on the
      committed grid, so `sd = 0`, every `z` is 0, and `score_committed = 0`;
    * the same channel on the doubled grid alternates high, low, high, low,
      which is a perfect metre-2 contrast at `z = 5.57`;
    * so the decoder prefers the doubled grid and allows the doubling.

    Metre 2 on a grid of half-beats implies a **bar every committed beat** —
    0.50 s, 120 bars a minute. Not a bar in any music this product is for. The
    decoder maximises over (metre, phase) with no admissibility constraint, and
    §3 of the pre-registration defines it that way, so I2 contradicted §3 and I2
    is the half that was wrong.

    **The obvious repair is refused.** Constraining the implied bar period to
    something musical would decide the octave by arithmetic on the committed
    beat period — the very quantity in dispute — and
    `PREREGISTERED_downbeat_channel.md` already refused to break its ties with
    the tempo prior for that reason: it turns independent evidence into a
    re-reading of the belief that sank the octave freeze.

    So this is carried as a **named limitation with a direction**: the failure
    produces false *allows*, never false vetoes, so A3 cannot see it and A1 will
    pay for it.
    """

    def test_a_flat_channel_creates_no_preference_at_all(self) -> None:
        """This is what I2 was for, and it holds exactly.

        Geometry alone manufactures nothing: with no content on either grid the
        statistic is 0.0, not merely small.
        """
        t, b = frames(20), beats(16)
        track = WindowTrack(t, np.full(len(t), 0.3))
        for k in (+1, -1):
            decision = judge(track, b, k)
            assert decision.delta == 0.0
            assert decision.delta_raw == 0.0

    def test_a_beat_only_channel_allows_the_doubling(self) -> None:
        t, b = frames(20), beats(16)
        decision = judge(WindowTrack(t, pulse(t, b)), b, +1)
        assert decision.delta < -1.0
        assert decision.score_committed == 0.0

    def test_the_committed_grid_is_constant_and_scores_zero(self) -> None:
        """`sd < EPS` returns 0 rather than dividing by zero."""
        t, b = frames(20), beats(16)
        sampled = WindowTrack(t, pulse(t, b)).sample(b)
        assert float(sampled.std()) == pytest.approx(0.0, abs=1e-12)
        assert set(zscores(sampled, (2, 3, 4, 6)).values()) == {0.0}

    def test_the_winning_hypothesis_is_not_a_musically_possible_bar(self) -> None:
        """The reason the invariant fails, as a number rather than a story."""
        t, b = frames(20), beats(16)
        grid = doubled_grid(b)
        scores = zscores(WindowTrack(t, pulse(t, b)).sample(grid), (2, 3, 4, 6))
        (metre, _), _ = max(scores.items(), key=lambda kv: kv[1])
        assert metre == 2
        implied_bar_sec = metre * (PERIOD / 2.0)
        assert implied_bar_sec == pytest.approx(0.5)
        assert 60.0 / implied_bar_sec == pytest.approx(120.0)


# --- I3, I5, I6, I7 ---------------------------------------------------------


class TestI3CleanFourFour:
    @pytest.mark.parametrize("k", [+1, -1])
    def test_a_correct_committed_level_is_defended(self, k: int) -> None:
        t, b = frames(20), beats(16)
        decision = judge(WindowTrack(t, pulse(t, b[0::4])), b, k)
        assert decision.delta > 0.0
        assert decision.metre == 4


class TestI5Waltz:
    @pytest.mark.parametrize("k", [+1, -1])
    def test_three_to_the_bar_is_not_pulled_towards_four(self, k: int) -> None:
        t, b = frames(20), beats(16)
        decision = judge(WindowTrack(t, pulse(t, b[0::3])), b, k)
        assert decision.metre == 3
        assert decision.delta > 0.0


class TestI6TheDecoderCanSayAllow:
    def test_a_genuinely_doubled_committed_state_is_released(self) -> None:
        """Committed at half the true period; halving the grid is correct.

        The magnitude matters as much as the sign and is recorded here: the
        allow signal came out at -0.18 against +1.97 for the veto case in I3.
        An order of magnitude weaker, which bounds where `tau` can sit and is
        why P6 predicts the gain, if any, comes from allows.
        """
        t = frames(20)
        committed = beats(16, 0.5)
        true_beats = beats(8, 1.0)
        decision = judge(WindowTrack(t, pulse(t, true_beats[0::4])), committed, -1)
        assert decision.delta < 0.0


class TestI7SixIsDecidedWhenReachable:
    def test_six_eight_is_read_when_both_grids_admit_six(self) -> None:
        t, b = frames(28), beats(24)
        track = WindowTrack(t, pulse(t, b[0::6]))
        for k in (+1, -1):
            decision = judge(track, b, k, window_beats=24)
            assert decision.metre == 6
            assert 6 in decision.metres_scored
            assert decision.delta > 0.0

    def test_six_leaves_both_sides_on_a_sixteen_beat_halving(self) -> None:
        """Symmetric exclusion: lost sensitivity, not a tilt."""
        t, b = frames(20), beats(16)
        decision = judge(WindowTrack(t, pulse(t, b[0::4])), b, -1)
        assert decision.metres_scored == (2, 3, 4)
        assert common_metres(16, 8) == (2, 3, 4)
        assert admissible_metres(31) == (2, 3, 4, 6)


# --- The grids and the shared null ------------------------------------------


class TestGrids:
    def test_doubling_adds_the_midpoints(self) -> None:
        got = doubled_grid(np.array([0.0, 1.0, 2.0]))
        assert list(got) == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_halving_offers_both_alignments(self) -> None:
        a, b = halved_grids(np.arange(6, dtype=float))
        assert list(a) == [0.0, 2.0, 4.0]
        assert list(b) == [1.0, 3.0, 5.0]

    def test_the_null_shifts_the_track_not_the_grid(self) -> None:
        """Both grids read the same shifted track, so their nesting survives.

        The previous audit permuted each grid independently, which gives each
        one a null of its own and cannot adjudicate a comparison between them.
        """
        t, b = frames(20), beats(16)
        track = WindowTrack(t, pulse(t, b[0::4]))
        shift = track.shifts()[0]
        committed = track.sample(b, shift)
        doubled = track.sample(doubled_grid(b), shift)
        # Every committed beat still appears in the doubled grid, at the same
        # value, because one track was shifted and then both were read off it.
        assert np.allclose(committed, doubled[0::2])

    def test_four_deterministic_shifts_and_no_rng(self) -> None:
        t, b = frames(20), beats(16)
        track = WindowTrack(t, pulse(t, b[0::4]))
        assert len(track.shifts()) == 4
        assert track.shifts() == WindowTrack(t, track.downbeat.copy()).shifts()


# --- Events -----------------------------------------------------------------


def trace(ks: list[int], dt: float = 0.02, committed: float = 120.0,
          locked_from: int = 0):
    """A replayed trace with a prescribed sequence of octave disagreements."""
    n = len(ks)
    times = np.arange(n) * dt
    committed_bpm = np.full(n, committed)
    measured_bpm = np.array([committed * (2.0 ** k) for k in ks])
    answered = np.ones(n, dtype=bool)
    locked = np.arange(n) >= locked_from
    return times, committed_bpm, measured_bpm, answered, locked


class TestEventExtraction:
    def test_one_sustained_disagreement_is_one_event(self) -> None:
        """Not one per frame. At 50 fps the naive reading gives 250 of them."""
        events = extract_proposals(*trace([0] * 50 + [1] * 250 + [0] * 100))
        assert len(events) == 1
        assert events[0].k == 1
        assert events[0].onset_sec == pytest.approx(1.0)

    def test_a_brief_return_to_agreement_does_not_close_the_event(self) -> None:
        """The estimator drops in and out far faster than a conflict resolves."""
        events = extract_proposals(*trace([1] * 100 + [0] * 20 + [1] * 100))
        assert len(events) == 1

    def test_a_gap_longer_than_the_close_window_ends_it(self) -> None:
        ks = [1] * 100 + [0] * 200 + [1] * 100
        events = extract_proposals(*trace(ks))
        assert len(events) == 2
        assert [e.onset_sec for e in events] == pytest.approx([0.0, 6.0])

    def test_close_is_when_the_one_second_agreement_is_confirmed(self) -> None:
        """The veto stays latched through the close debounce, not just to its start."""
        events = extract_proposals(*trace([1] * 100 + [0] * 100))
        assert len(events) == 1
        assert events[0].close_sec == pytest.approx(3.0)

    def test_a_sign_flip_splits_the_event(self) -> None:
        events = extract_proposals(*trace([1] * 200 + [-1] * 200))
        assert [e.k for e in events] == [1, -1]

    def test_onsets_closer_than_the_minimum_separation_merge(self) -> None:
        # Two conflicts 1.0 s apart: 50 frames of k=1, 50 of k=0, 50 of k=1.
        # The gap exceeds the close window at 0.02 s a frame, so two events are
        # found and the second is merged for being 2.0 s too close.
        ks = [1] * 10 + [0] * 60 + [1] * 60
        events = extract_proposals(*trace(ks))
        assert len(events) == 1
        assert events[0].close_sec == pytest.approx((len(ks) - 1) * 0.02)

    def test_proposals_before_the_first_lock_are_excluded(self) -> None:
        """The exclusion is on the onset, not on any frame of the event.

        Two conflicts, onsetting at 0.0 s and 4.0 s. Lock lands at 3.0 s, so
        the first is dropped and the second kept.
        """
        ks = [1] * 100 + [0] * 100 + [1] * 100
        events = extract_proposals(*trace(ks, locked_from=150))
        assert len(events) == 1
        assert events[0].onset_sec == pytest.approx(4.0)

    def test_a_disagreement_still_live_at_the_lock_opens_there(self) -> None:
        """The committed level becomes a claim at the lock, so the event does too.

        Pre-lock frames are not examined at all under the held-committed
        definition, so a conflict that was already running when the tracker
        locked is timestamped at the lock rather than discarded. What the
        product is publishing at that moment is on screen, and an estimator an
        octave away from it is about to change what is on screen.
        """
        ks = [1] * 100 + [0] * 100 + [1] * 100
        events = extract_proposals(*trace(ks, locked_from=250))
        assert len(events) == 1
        assert events[0].onset_sec == pytest.approx(5.0)


class TestWhatCountsAsAnOctave:
    """Both defects the smoke run on GTZAN exposed, pinned.

    Neither was visible in the pre-registration, and neither is visible in a
    synthetic trace built to have octave jumps in it. They needed real replayed
    traces, which is why the order of work put parity before any corpus.
    """

    def test_a_three_to_two_tempo_relation_is_not_an_octave_proposal(self) -> None:
        """`round(log2(r))` alone maps every ratio in (1.41, 2.83) to +1.

        Four of the eight proposals the first GTZAN smoke found were 3:2 —
        122->176, 87->136, 119->176, 106->176 — and the proposed grid is built
        as exactly doubled or halved, so those would have been scored against a
        grid nobody proposed.
        """
        assert octave_k(176.0, 122.0) == 0
        assert octave_k(136.0, 87.0) == 0
        assert octave_k(176.0, 119.0) == 0
        # The genuine ones from the same run survive.
        assert octave_k(83.0, 168.0) == -1
        assert octave_k(130.0, 67.0) == 1
        assert octave_k(176.0, 91.0) == 1
        assert octave_k(97.0, 188.0) == -1

    def test_the_tolerance_is_the_eight_percent_the_labels_use(self) -> None:
        assert octave_k(120.0 * 2.0 * 1.07, 120.0) == 1
        assert octave_k(120.0 * 2.0 * 1.09, 120.0) == 0

    def test_the_committed_level_follows_drift_inside_its_octave(self) -> None:
        """A band moving 128 -> 132 is never a proposal, at any length."""
        n = 400
        times = np.arange(n) * 0.02
        published = np.linspace(128.0, 132.0, n)
        answered = np.ones(n, dtype=bool)
        locked = np.ones(n, dtype=bool)
        assert extract_proposals(times, published, published.copy(),
                                 answered, locked) == []

    def test_the_committed_level_does_not_chase_the_published_bpm(self) -> None:
        """The registered instantaneous definition finds nothing on real audio.

        Measured over 7821 locked-and-answered GTZAN frames, `k != 0` on two of
        them, because the anchor pins the filter to the estimator. Here the
        published BPM follows the estimator into the new octave and the event
        still opens, because `committed` froze at the old one.
        """
        n = 400
        times = np.arange(n) * 0.02
        measured = np.where(np.arange(n) < 200, 120.0, 240.0)
        published = np.where(np.arange(n) < 202, 120.0, 240.0)  # follows in 40 ms
        events = extract_proposals(times, published, measured,
                                   np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        assert len(events) == 1
        assert events[0].k == 1
        assert events[0].committed_bpm == pytest.approx(120.0)

    def test_an_unanswered_estimator_proposes_nothing(self) -> None:
        times, committed, measured, answered, locked = trace([1] * 200)
        answered[:] = False
        assert extract_proposals(times, committed, measured, answered, locked) == []

    def test_the_octave_index_rounds_in_log_space(self) -> None:
        assert octave_k(240.0, 120.0) == 1
        assert octave_k(60.0, 120.0) == -1
        assert octave_k(132.0, 128.0) == 0
        assert octave_k(0.0, 120.0) == 0


class TestCommittedGrid:
    """A committed beat is a grid position, not a beat the shell played.

    Reading the published list instead left 10 of 22 GTZAN proposals without a
    window, at onsets as late as 19.6 s with 2 to 11 played beats behind them —
    because the list is gated by the lock hysteresis, which thins out exactly
    when confidence is low, which is exactly when an octave conflict happens.
    Reconstructing raised coverage from 45% to 86%.
    """

    @staticmethod
    def _series(n: int = 2000, bpm_value: float = 120.0):
        times = np.arange(n) * 0.02
        return times, np.full(n, bpm_value)

    def test_it_rebuilds_sixteen_beats_from_one_played_beat(self) -> None:
        times, bpm = self._series()
        played = np.array([30.0])
        grid = committed_grid(played, times, bpm, onset_sec=31.0)
        assert len(grid) == 16
        assert grid[-1] == pytest.approx(30.0)
        assert np.allclose(np.diff(grid), 0.5)

    def test_it_walks_back_at_the_bpm_of_each_step(self) -> None:
        """A drifting tempo must not be smeared by one period for the window."""
        n = 3000
        times = np.arange(n) * 0.02
        bpm = np.where(times < 20.0, 100.0, 140.0)
        grid = committed_grid(np.array([25.0]), times, bpm, onset_sec=25.5)
        steps = np.diff(grid)
        assert steps[-1] == pytest.approx(60.0 / 140.0)
        assert steps[0] == pytest.approx(60.0 / 100.0)

    def test_nothing_after_the_proposal_is_read(self) -> None:
        times, bpm = self._series()
        played = np.array([10.0, 20.0, 30.0])
        grid = committed_grid(played, times, bpm, onset_sec=20.0)
        assert grid[-1] == pytest.approx(20.0)
        assert float(grid.max()) <= 20.0

    def test_too_little_history_yields_nothing(self) -> None:
        times, bpm = self._series(n=200)  # 4 s, and 16 beats need 7.5
        assert len(committed_grid(np.array([3.9]), times, bpm, 3.95)) == 0

    def test_no_played_beat_at_all_yields_nothing(self) -> None:
        times, bpm = self._series()
        assert len(committed_grid(np.zeros(0), times, bpm, 20.0)) == 0


class TestWindowIsCausal:
    def test_no_beat_after_the_proposal_is_visible(self) -> None:
        all_beats = beats(40)
        got = window_beats(all_beats, onset_sec=all_beats[20] + 0.01)
        assert len(got) == 16
        assert got[-1] <= all_beats[20]
        assert got[-1] == pytest.approx(all_beats[20])

    def test_too_few_beats_yields_nothing_and_the_policy_falls_to_baseline(self) -> None:
        got = window_beats(beats(10), onset_sec=100.0)
        assert len(got) == 0
        t = frames(20)
        assert judge(WindowTrack(t, np.zeros(len(t))), got, +1).answered is False


class TestUnansweredAllows:
    @pytest.mark.parametrize("tau", [-5.0, 0.0, 5.0])
    def test_an_unanswered_decision_never_vetoes(self, tau: float) -> None:
        assert Decision(False).veto(tau) is False

    def test_k_zero_is_not_an_octave_proposal(self) -> None:
        t, b = frames(20), beats(16)
        decision = judge(WindowTrack(t, pulse(t, b[0::4])), b, 0)
        assert decision.answered is False
        assert decision.reason == "not an octave proposal"
