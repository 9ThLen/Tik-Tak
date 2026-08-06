"""Stateful replay, event labels and A2/A3 accounting for octave veto."""

from __future__ import annotations

import concurrent.futures
import pathlib

import numpy as np
import pytest

from eval.octave_veto import Decision, Proposal
from eval.octave_veto_experiment import (
    AMBIGUOUS, CORRECT_TO_WRONG, WRONG_TO_CORRECT, EventResult, TrackEvents,
    VetoInterval, CorpusArm, balanced_accuracy, cluster_bootstrap_difference,
    converge_replay, debounce_schedule, decoder_schedule, event_counts,
    event_label, evaluate_events, false_veto_rate, rate_limit_schedule,
    schedule_signature, select_matched_policy, select_tau, total_ban_schedule,
    transfer_verdict, verify_cached_parity, write_schedule, run_policy_grid,
    _require_frozen_selection, ambiguity_diagnostic, d1_zero_committed_score,
    event_coverage_by_meter, raw_sign_agreement, apply_protocol_diagnostics,
    _bounded_results,
)
from eval.octave_veto_replay import from_payload

FPS = 50.0


def reference_beats(seconds: float = 30.0, bpm: float = 120.0) -> np.ndarray:
    return np.arange(0.0, seconds, 60.0 / bpm)


def payload_with_proposal() -> dict:
    times = np.arange(0.0, 20.0, 1.0 / FPS)
    measured = np.full(len(times), 120.0)
    measured[(times >= 10.0) & (times < 12.0)] = 240.0
    downbeat = np.full(len(times), 0.02)
    for beat in reference_beats(20.0)[0::4]:
        downbeat = np.maximum(
            downbeat, np.exp(-0.5 * ((times - beat) / 0.04) ** 2))
    return {
        "beats": reference_beats(20.0).tolist(),
        "live_times": times.tolist(),
        "live_bpms": np.full(len(times), 120.0).tolist(),
        "live_confidences": np.full(len(times), 0.5).tolist(),
        "live_anchor_bpm": measured.tolist(),
        "live_anchor_margin": np.ones(len(times)).tolist(),
        "activation_times": times.tolist(),
        "activation_downbeat": downbeat.tolist(),
    }


def proposal(committed: float, measured: float, onset: float = 10.0,
             close: float = 12.0) -> Proposal:
    return Proposal(onset, close, 1 if measured > committed else -1,
                    committed, measured)


def event(label: str, delta: float, null_delta: float = 0.0,
          onset: float = 10.0) -> EventResult:
    p = proposal(120.0, 240.0, onset, onset + 2.0)
    return EventResult(
        p,
        Decision(True, delta=delta, delta_raw=delta,
                 null_committed=null_delta,
                 null_proposed=0.0),
        label,
    )


class TestLabelsAndWindows:
    def test_labels_both_directions_and_ambiguous(self) -> None:
        beats = reference_beats()
        assert event_label(proposal(120.0, 240.0), beats) == CORRECT_TO_WRONG
        assert event_label(proposal(240.0, 120.0), beats) == WRONG_TO_CORRECT
        assert event_label(proposal(120.0, 124.0), beats) == AMBIGUOUS

    def test_real_event_is_judged_on_the_causal_window(self) -> None:
        replay = from_payload(payload_with_proposal())
        got = evaluate_events("synthetic", replay, reference_beats())
        assert len(got.events) == 1
        assert got.events[0].label == CORRECT_TO_WRONG
        assert got.events[0].decision.answered
        assert got.events[0].decision.delta > 0.0

    def test_future_activation_cannot_change_the_onset_decision(self) -> None:
        payload = payload_with_proposal()
        before = evaluate_events("a", from_payload(payload), reference_beats())
        changed = dict(payload)
        downbeat = np.asarray(payload["activation_downbeat"], dtype=float)
        times = np.asarray(payload["activation_times"], dtype=float)
        downbeat[times > 10.0] = np.linspace(0.0, 100.0, np.sum(times > 10.0))
        changed["activation_downbeat"] = downbeat.tolist()
        after = evaluate_events("a", from_payload(changed), reference_beats())
        assert after.events[0].decision == before.events[0].decision


class TestSchedules:
    def test_decoder_vetoes_only_answered_delta_above_tau(self) -> None:
        rows = TrackEvents("x", (
            event(CORRECT_TO_WRONG, 1.0, onset=10.0),
            event(WRONG_TO_CORRECT, 0.25, onset=20.0),
            EventResult(proposal(120.0, 240.0, 30.0, 32.0), Decision(False),
                        CORRECT_TO_WRONG),
        ))
        assert [row.onset_sec for row in decoder_schedule(rows, 0.5)] == [10.0]

    def test_shift_control_reads_the_null_not_delta(self) -> None:
        rows = TrackEvents("x", (event(CORRECT_TO_WRONG, -2.0, 1.0),))
        assert decoder_schedule(rows, 0.5) == ()
        assert len(decoder_schedule(rows, 0.5, control=True)) == 1

    def test_debounce_vetoes_only_the_opening_part(self) -> None:
        rows = TrackEvents("x", (event(CORRECT_TO_WRONG, 1.0),))
        assert debounce_schedule(rows, 0.5) == (VetoInterval(10.0, 10.5, 120.0),)

    def test_rate_limit_allows_first_and_vetoes_too_soon(self) -> None:
        rows = TrackEvents("x", (
            event(CORRECT_TO_WRONG, 1.0, onset=10.0),
            event(CORRECT_TO_WRONG, 1.0, onset=14.0),
            event(CORRECT_TO_WRONG, 1.0, onset=21.0),
        ))
        assert [row.onset_sec for row in rate_limit_schedule(rows, 10.0)] == [14.0]
        assert len(total_ban_schedule(rows)) == 3

    def test_schedule_round_trip_format_and_signature(self, tmp_path) -> None:
        path = tmp_path / "schedule.txt"
        rows = (VetoInterval(1.23456789, 3.0, 119.9999999996),)
        write_schedule(path, rows)
        assert path.read_text().splitlines()[1].split() == [
            "1.234567890", "3.000000000", "120"]
        assert schedule_signature(rows) == ((1.234568, 3.0, 120.0),)


class TestFixedPoint:
    def test_nine_digits_round_trip_a_float_but_not_the_activation(self) -> None:
        """The true half of the claim that cost twenty of twenty on parity.

        Nine significant digits do round trip a float exactly, and the tool's
        note said so. What it missed is that the activation is not a float:
        `beatnet.hpp` emits `(double)p[0] + (double)p[1]` scaled, and the sum of
        two floats in double precision generally is not one.
        """
        values = np.random.default_rng(9).random(1000, dtype=np.float32)
        restored = np.asarray([float(f"{value:.9g}") for value in values],
                              dtype=np.float32)
        assert np.array_equal(restored, values)

        rng = np.random.default_rng(11)
        sums = [float(a) + float(b) for a, b in
                zip(rng.random(1000, dtype=np.float32),
                    rng.random(1000, dtype=np.float32))]
        assert any(float(f"{value:.9g}") != value for value in sums)
        assert all(float(f"{value:.17g}") == value for value in sums)

    def test_empty_policy_is_baseline_without_an_extra_run(self) -> None:
        called = []
        payload = payload_with_proposal()
        got = converge_replay(payload, "x", reference_beats(),
                              lambda rows: called.append(rows) or payload,
                              lambda _: ())
        assert got.passes == 0
        assert called == []

    def test_cached_parity_includes_the_proposal_event_list(self) -> None:
        payload = payload_with_proposal()
        verify_cached_parity("x", payload, dict(payload), reference_beats())
        changed = dict(payload)
        changed["live_anchor_bpm"] = np.full(
            len(payload["live_anchor_bpm"]), 120.0).tolist()
        with pytest.raises(RuntimeError, match="diverged"):
            verify_cached_parity("x", payload, changed, reference_beats())

    def test_nonempty_policy_is_replayed_until_the_schedule_agrees(self) -> None:
        called = []
        payload = payload_with_proposal()

        def rerun(rows):
            called.append(rows)
            return payload

        got = converge_replay(
            payload, "x", reference_beats(), rerun,
            lambda rows: decoder_schedule(rows, tau=-100.0),
        )
        assert got.passes == 1
        assert len(called) == 1
        assert got.schedule == called[0]

    def test_a_genuine_cycle_resolves_toward_the_fewest_vetoes(self) -> None:
        """Two schedules that each imply the other have no fixed point.

        Iterating only alternates, so the harness stops and takes the most
        baseline-like member. Where the seam cannot decide it does *less*, which
        is the direction that cannot flatter the policy being measured; dropping
        the recording instead would exclude exactly those where vetoing changes
        the trajectory most.
        """
        payload = payload_with_proposal()
        flip = False

        def build(_):
            nonlocal flip
            flip = not flip
            return (VetoInterval(10.0, 12.0, 120.0),) if flip else ()

        got = converge_replay(payload, "x", reference_beats(),
                              lambda _: payload, build, max_passes=8)
        assert got.cycled is True
        assert got.schedule == ()

    def test_slow_convergence_is_not_mistaken_for_a_cycle(self) -> None:
        """A repeat of the *previous* pass is the fixed point, not an orbit.

        Measured on RWC, 39 of 48 track-policy pairs settle on the first pass
        and the tail runs to 15. The old limit of 8 called those failures.
        """
        payload = payload_with_proposal()
        schedules = [
            (VetoInterval(10.0, 12.0, 120.0),) * n for n in (3, 2, 1, 1, 1)
        ]
        step = iter(schedules)

        got = converge_replay(payload, "x", reference_beats(),
                              lambda _: payload, lambda _: next(step),
                              max_passes=8)
        assert got.cycled is False
        assert len(got.schedule) == 1

    def test_running_out_of_passes_without_repeating_still_raises(self) -> None:
        count = 0

        def build(_):
            nonlocal count
            count += 1
            return (VetoInterval(float(count), float(count) + 1.0, 120.0),) * count

        with pytest.raises(RuntimeError, match="did not converge"):
            converge_replay(payload_with_proposal(), "x", reference_beats(),
                            lambda _: payload_with_proposal(), build,
                            max_passes=3)

    def test_empty_grid_reaches_the_corpus_import_seam(self) -> None:
        """The corpus runner imports annotation loading from its real module."""
        assert run_policy_grid([], pathlib.Path("binary"), pathlib.Path("weights"),
                               (), corpus="rwc", workers=1) == {}

    def test_corpus_submission_window_does_not_retain_every_future(self) -> None:
        class CountingItems:
            seen = 0

            def __iter__(self):
                for value in range(20):
                    self.seen += 1
                    yield value

        items = CountingItems()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = _bounded_results(pool, lambda value: value, items, 3)
            first = next(results)
            assert first in range(20)
            # Three initial submissions plus the replacement for the yielded
            # result. An eager corpus submission would have consumed all 20.
            assert items.seen == 4
            assert sorted((first, *results)) == list(range(20))


class TestFrozenSelection:
    def test_selection_commit_contains_only_the_frozen_result(
            self, tmp_path, monkeypatch) -> None:
        selection = tmp_path / "research" / "results" / "selection.json"
        selection.parent.mkdir(parents=True)
        selection.write_text("{}", encoding="utf-8")

        def fake_git(_repository, *args):
            if args == ("status", "--porcelain"):
                return ""
            if args[:2] == ("ls-files", "--error-unmatch"):
                return str(args[2])
            if args[:4] == ("log", "-1", "--format=%H", "--"):
                return "selection-commit"
            if args == ("rev-parse", "HEAD"):
                return "selection-commit"
            if args == ("rev-parse", "HEAD^"):
                return "implementation-commit"
            if args == ("diff", "--name-only", "implementation-commit",
                        "selection-commit"):
                return "research/results/selection.json"
            raise AssertionError(args)

        monkeypatch.setattr("eval.octave_veto_experiment._git", fake_git)
        assert _require_frozen_selection(
            tmp_path, selection, "implementation-commit") == "selection-commit"

    def test_selection_commit_rejects_code_changes(
            self, tmp_path, monkeypatch) -> None:
        selection = tmp_path / "research" / "results" / "selection.json"
        selection.parent.mkdir(parents=True)
        selection.write_text("{}", encoding="utf-8")

        def fake_git(_repository, *args):
            if args == ("status", "--porcelain"):
                return ""
            if args[:2] == ("ls-files", "--error-unmatch"):
                return str(args[2])
            if args[:4] == ("log", "-1", "--format=%H", "--"):
                return "selection-commit"
            if args == ("rev-parse", "HEAD"):
                return "selection-commit"
            if args == ("rev-parse", "HEAD^"):
                return "implementation-commit"
            if args == ("diff", "--name-only", "implementation-commit",
                        "selection-commit"):
                return "research/results/selection.json\ncore/src/tracking/live.cpp"
            raise AssertionError(args)

        monkeypatch.setattr("eval.octave_veto_experiment._git", fake_git)
        with pytest.raises(RuntimeError, match="only the RWC selection"):
            _require_frozen_selection(tmp_path, selection,
                                      "implementation-commit")


class TestA2A3Accounting:
    def tracks(self):
        return [
            TrackEvents("a", (event(CORRECT_TO_WRONG, 1.0),
                              event(WRONG_TO_CORRECT, -1.0))),
            TrackEvents("b", (event(CORRECT_TO_WRONG, -1.0),
                              event(WRONG_TO_CORRECT, 1.0))),
        ]

    def test_balanced_accuracy_and_false_veto_are_recording_balanced(self) -> None:
        tracks = self.tracks()
        assert balanced_accuracy(tracks, 0.0) == pytest.approx(0.5)
        assert false_veto_rate(tracks, 0.0) == pytest.approx(0.5)

    def test_unanswered_and_ambiguous_are_counted_but_not_scored(self) -> None:
        rows = TrackEvents("a", (
            event(CORRECT_TO_WRONG, 1.0),
            EventResult(proposal(120.0, 240.0), Decision(False), WRONG_TO_CORRECT),
            event(AMBIGUOUS, 1.0),
        ))
        assert event_counts([rows]) == {
            "events": 3, "answered": 2, CORRECT_TO_WRONG: 1,
            WRONG_TO_CORRECT: 1, AMBIGUOUS: 1,
        }
        assert np.isnan(balanced_accuracy([rows], 0.0))

    def test_cluster_bootstrap_resamples_recordings_as_pairs(self) -> None:
        decoder = {
            f"t{i}": TrackEvents(f"t{i}", (
                event(CORRECT_TO_WRONG, 1.0),
                event(WRONG_TO_CORRECT, -1.0),
            )) for i in range(6)
        }
        control = {
            f"t{i}": TrackEvents(f"t{i}", (
                event(CORRECT_TO_WRONG, 1.0, null_delta=-1.0),
                event(WRONG_TO_CORRECT, -1.0, null_delta=1.0),
            )) for i in range(6)
        }
        got = cluster_bootstrap_difference(decoder, control, 0.0,
                                           replicates=200, seed=7)
        assert got["difference"] == pytest.approx(1.0)
        assert got["ci_low"] == pytest.approx(1.0)
        assert got["ci_high"] == pytest.approx(1.0)

    def test_registered_diagnostics_keep_their_denominators_visible(self) -> None:
        answered = Decision(True, delta=1.0, delta_raw=1.0,
                            score_committed=0.0)
        rows = TrackEvents("a", (
            EventResult(proposal(120.0, 240.0), answered, CORRECT_TO_WRONG),
            event(AMBIGUOUS, -1.0),
        ), meter="4")
        assert raw_sign_agreement([rows]) == {
            "events": 2, "agreements": 2, "fraction": 1.0}
        assert ambiguity_diagnostic([rows]) == {
            "ambiguous": 1, "labelled": 1, "dominates": False}
        assert d1_zero_committed_score([rows]) == {
            "events": 1, "zero_score": 1, "fraction": 1.0}
        assert event_coverage_by_meter([rows])["4"] == {
            "events": 2, "answered": 2, "coverage": 1.0}


def fake_arm(name, parameter, metrics, tracks=(), scores=(), corpus="rwc"):
    return CorpusArm(
        name=name, parameter=parameter, corpus=corpus,
        scores=list(scores), tracks={track.name: track for track in tracks},
        summary={"by_corpus": {corpus: metrics}}, max_passes=1,
    )


def cost_metrics(episode=0.5, correct=0.8, strict=0.3, switches=4.0,
                 settle=30.0, f_measure=0.8):
    return {
        "no_wrong_level_episode_fraction": episode,
        "mean_correct_share_of_eligible": correct,
        "usable_rate_strict": strict,
        "switches_per_five_minutes": switches,
        "p90_settle_sec": settle,
        "f_measure": f_measure,
    }


class TestRegisteredSelectionAndVerdict:
    def test_tau_selection_applies_a3_costs_objective_and_ties(self) -> None:
        safe = TrackEvents("safe", (event(WRONG_TO_CORRECT, -1.0),))
        harmful = TrackEvents("harm", (event(WRONG_TO_CORRECT, 1.0),))
        baseline = fake_arm("baseline", None, cost_metrics(episode=0.4), [safe])
        low = fake_arm("tau_0", 0.0, cost_metrics(episode=0.5), [safe])
        winner = fake_arm("tau_025", 0.25, cost_metrics(episode=0.6), [safe])
        bad_a3 = fake_arm("tau_05", 0.5, cost_metrics(episode=0.9), [harmful])
        bad_cost = fake_arm("tau_075", 0.75,
                            cost_metrics(episode=0.95, switches=4.1), [safe])
        assert select_tau([low, winner, bad_a3, bad_cost], baseline) is winner

    def test_matched_policy_must_be_within_half_a_point(self) -> None:
        decoder = fake_arm("decoder", 0.5, cost_metrics(episode=0.6, correct=0.80))
        near = fake_arm("debounce", 1.0, cost_metrics(episode=0.55, correct=0.804))
        farther_better = fake_arm("ban", None, cost_metrics(episode=0.9, correct=0.81))
        assert select_matched_policy([near, farther_better], decoder) is near

    def test_all_four_transfer_conditions_can_pass_together(self) -> None:
        decoder_tracks = []
        control_tracks = []
        decoder_scores, baseline_scores, matched_scores = [], [], []
        for index in range(20):
            name = f"t{index}"
            decoder_tracks.append(TrackEvents(name, (
                event(CORRECT_TO_WRONG, 1.0),
                event(WRONG_TO_CORRECT, -1.0),
            )))
            control_tracks.append(TrackEvents(name, (
                event(CORRECT_TO_WRONG, 1.0, null_delta=-1.0),
                event(WRONG_TO_CORRECT, -1.0, null_delta=1.0),
            )))
            decoder_scores.append({"name": name, "ok": True, "annotated": True,
                                   "worst_wrong_octave_sec": 0.0})
            baseline_scores.append({"name": name, "ok": True, "annotated": True,
                                    "worst_wrong_octave_sec": 0.0 if index < 10 else 5.0})
            matched_scores.append({"name": name, "ok": True, "annotated": True,
                                   "worst_wrong_octave_sec": 0.0 if index < 10 else 5.0})

        metrics = cost_metrics(episode=1.0, correct=0.80, strict=0.31,
                               switches=4.0, settle=30.0, f_measure=0.80)
        decoder = fake_arm("decoder", 0.5, metrics, decoder_tracks,
                           decoder_scores, corpus="harmonix")
        baseline = fake_arm("baseline", None, cost_metrics(episode=0.5), (),
                            baseline_scores, corpus="harmonix")
        matched = fake_arm("debounce", 1.0,
                           cost_metrics(episode=0.5, correct=0.804), (),
                           matched_scores, corpus="harmonix")
        control = fake_arm("control", 0.5, metrics, control_tracks, (),
                           corpus="harmonix")

        got = transfer_verdict(baseline, decoder, matched, control, tau=0.5)
        assert [got[key] for key in ("A1", "A2", "A3", "A4", "accepted")] == [
            True, True, True, True, True]

    def test_a_winning_decoder_with_failed_p2_is_not_accepted(self) -> None:
        disagree = TrackEvents("x", (
            EventResult(proposal(120.0, 240.0),
                        Decision(True, delta=1.0, delta_raw=-1.0),
                        CORRECT_TO_WRONG),
        ))
        verdict = {"A1": True, "accepted": True}
        apply_protocol_diagnostics(
            verdict, [disagree], {"ambiguous": 0, "labelled": 1,
                                  "dominates": False})
        assert not verdict["accepted"]
        assert verdict["protocol_diagnostics"]["P2_sink_triggered"]

    def test_ambiguity_must_dominate_both_corpora_to_sink_transfer(self) -> None:
        ambiguous = TrackEvents("x", (event(AMBIGUOUS, 1.0),))
        verdict = {"A1": False, "accepted": True}
        apply_protocol_diagnostics(
            verdict, [ambiguous], {"ambiguous": 2, "labelled": 1,
                                   "dominates": True})
        assert not verdict["accepted"]
        assert verdict["protocol_diagnostics"]["ambiguity_sink_triggered"]


class TestActivationReplayFidelity:
    """The three reasons cached-activation replay was not the same run.

    Each was found by the registered parity gate and each on its own failed it
    on every full-length recording tried: twenty of twenty RWC, with beat counts
    as far apart as 116 against 74. The gate did its job; what it caught is
    pinned here so a future edit cannot quietly undo one of them.
    """

    def test_nine_digits_cannot_round_trip_the_activation(self) -> None:
        """`beatnet.hpp` emits a double, not the float nine digits assume.

        The value is `(double)p[0] + (double)p[1]` scaled by `1 / models`. A sum
        of two floats in double precision is generally not a float, so the note
        that "nine significant digits round trip every float" was true and its
        conclusion was not.
        """
        value = float(np.float32(0.41579765)) + float(np.float32(0.13931973))
        assert float(f"{value:.9g}") != value
        assert float(f"{value:.17g}") == value

    def test_the_reconstructed_frame_time_is_a_different_double(self) -> None:
        """`(n * 441.0) / 22050.0` against `n * (1.0 / 50.0)`.

        The first is an exact product and one correctly rounded division; the
        second multiplies an already rounded 0.02. The filter integrates over
        the gaps between observations, so the last bits are not cosmetic.
        """
        differing = [n for n in range(1, 5000)
                     if (n * 441.0) / 22050.0 != n * (1.0 / 50.0)]
        assert differing, "the two expressions agree; the fix would be moot"

    def test_a_recorded_release_time_can_round_up_past_its_block(self) -> None:
        """Why the release schedule is an integer block index and not a clock.

        The seventh block boundary is 0.081269841269..., which prints at nine
        significant digits as a fractionally *larger* number. Compared against
        the clock it came from, `emit <= now` then fails and the frame arrives a
        block late.
        """
        boundary = sum(512.0 / 44100.0 for _ in range(7))
        assert float(f"{boundary:.9g}") > boundary

    def test_the_replay_is_handed_all_three(self) -> None:
        import subprocess
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args
            return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

        original = subprocess.run
        subprocess.run = fake_run
        try:
            from eval.octave_veto_replay import run_activation
            run_activation(pathlib.Path("bin"), pathlib.Path("a.wav"),
                           pathlib.Path("act.txt"),
                           emit_path=pathlib.Path("emit.txt"),
                           times_path=pathlib.Path("times.txt"))
        finally:
            subprocess.run = original
        args = seen["args"]
        assert "--activation-emit" in args
        assert "--activation-times" in args
        assert "--activation-model-timing" in args


class TestBothReplayPathsAreHandedTheCache:
    """There were two copies of the activation setup and one was fixed.

    The parity gate caught the other on the next run, 26 minutes in. Both paths
    now call one function; this asserts that neither can drift again.
    """

    def test_the_cache_writer_emits_all_three_files(self, tmp_path) -> None:
        from eval.octave_veto_experiment import write_activation_cache

        initial = {"activation_beat": [0.5, 0.25], "activation_emit": [1, 2],
                   "activation_times": [0.0, 0.02]}
        act = tmp_path / "a.txt"
        emit = tmp_path / "e.txt"
        times = tmp_path / "t.txt"
        write_activation_cache(initial, act, emit, times)
        assert act.exists() and emit.exists() and times.exists()
        assert emit.read_text().split() == ["1", "2"]

    def test_the_values_survive_the_round_trip_exactly(self, tmp_path) -> None:
        from eval.octave_veto_experiment import write_activation_cache

        value = float(np.float32(0.41579765)) + float(np.float32(0.13931973))
        initial = {"activation_beat": [value], "activation_emit": [1],
                   "activation_times": [(1 * 441.0) / 22050.0]}
        act = tmp_path / "a.txt"
        times = tmp_path / "t.txt"
        write_activation_cache(initial, act, tmp_path / "e.txt", times)
        assert float(act.read_text().strip()) == value
        assert float(times.read_text().strip()) == (1 * 441.0) / 22050.0

    def test_neither_replay_path_calls_the_core_without_the_cache(self) -> None:
        """Both call sites pass emit_path and times_path, or the gate fails."""
        import inspect

        from eval import octave_veto_experiment as module

        source = inspect.getsource(module)
        calls = source.count("run_activation(")
        assert calls == source.count("emit_path=") == source.count("times_path=")
        assert calls >= 4


class TestCycleDetectionGranularity:
    """Convergence is exact; cycle detection is not, and they differ on purpose.

    On RWC_C003 every decoder arm settles into a period-2 orbit whose members
    are bitwise identical — drift 0.000e+00 — so the exact test finds them. The
    comparison policies do not: debounce builds an interval from every event
    rather than a thresholded subset, so its schedule moves with the whole event
    list and wandered forty passes without landing on a float it had seen.
    """

    def test_a_millisecond_apart_is_the_same_decision(self) -> None:
        from eval.octave_veto_experiment import decision_signature

        a = (VetoInterval(10.0000001, 12.0000002, 120.0000001),)
        b = (VetoInterval(10.0, 12.0, 120.0),)
        assert schedule_signature(a) != schedule_signature(b)
        assert decision_signature(a) == decision_signature(b)

    def test_two_real_proposals_do_not_merge(self) -> None:
        """The poll the onsets come from is ~23 ms; the resolution is 1 ms."""
        from eval.octave_veto_experiment import decision_signature

        a = (VetoInterval(10.000, 12.0, 120.0),)
        b = (VetoInterval(10.023, 12.0, 120.0),)
        assert decision_signature(a) != decision_signature(b)

    def test_a_drifting_orbit_is_caught_as_a_cycle(self) -> None:
        """Alternating decisions whose floats never repeat exactly."""
        payload = payload_with_proposal()
        step = 0

        def build(_):
            nonlocal step
            step += 1
            jitter = step * 1e-9
            if step % 2:
                return (VetoInterval(10.0 + jitter, 12.0, 120.0),)
            return ()

        got = converge_replay(payload, "x", reference_beats(),
                              lambda _: payload, build, max_passes=12)
        assert got.cycled is True
        assert got.schedule == ()

    def test_missing_memory_reads_as_absent_not_as_zero(self) -> None:
        from eval.octave_veto_experiment import _resident_mb

        value = _resident_mb()
        assert np.isnan(value) or value > 0.0


class TestDegenerateIntervals:
    """A proposal opening on the last frame has no extent to act on.

    The core requires close > onset and rejects the whole schedule otherwise,
    which stopped a twelve-hour run seven recordings in. Every schedule builder
    must therefore drop the empty ones — dropping is behaviour-preserving,
    because the core skips such an interval before it can apply.
    """

    def test_a_zero_length_veto_is_not_scheduled(self) -> None:
        rows = TrackEvents("x", (
            EventResult(Proposal(10.0, 10.0, 1, 120.0, 240.0),
                        Decision(True, delta=5.0, delta_raw=5.0), CORRECT_TO_WRONG),
            EventResult(Proposal(20.0, 22.0, 1, 120.0, 240.0),
                        Decision(True, delta=5.0, delta_raw=5.0), CORRECT_TO_WRONG),
        ))
        got = decoder_schedule(rows, 0.5)
        assert [row.onset_sec for row in got] == [20.0]

    def test_every_builder_agrees_on_that(self) -> None:
        rows = TrackEvents("x", (
            EventResult(Proposal(10.0, 10.0, 1, 120.0, 240.0),
                        Decision(True, delta=5.0, delta_raw=5.0), CORRECT_TO_WRONG),
        ))
        assert decoder_schedule(rows, 0.5) == ()
        assert debounce_schedule(rows, 1.0) == ()
        for row in (*rate_limit_schedule(rows, 5.0), *total_ban_schedule(rows)):
            assert row.close_sec > row.onset_sec
