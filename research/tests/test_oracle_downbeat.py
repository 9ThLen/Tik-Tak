"""The parts of the oracle decomposition that can be checked without a model.

The table itself needs weights and a corpus. What is checked here is the
arithmetic that gives the table its meaning: that a fold map is read the way the
split file writes it, that a metrical level is named from the beat interval, and
above all that grid errors and label errors are counted separately — the whole
file exists to tell those two apart, so a bug there would not be a wrong number,
it would be a wrong conclusion.
"""

import numpy as np
import pytest

from eval.oracle_downbeat import (auprc, fold_map, matched_beat_stats,
                                  octave_label, resolve_checkpoint)


def grid(bpm: float, count: int, start: float = 0.0) -> np.ndarray:
    return start + np.arange(count) * (60.0 / bpm)


class TestFoldMap:
    def test_it_reads_both_spellings_of_a_name(self, tmp_path):
        path = tmp_path / "8-folds.split"
        path.write_text("ballroom_Media-105418\t6\nballroom_Albums-Foo-01\t0\n",
                        encoding="utf-8")
        folds = fold_map(path)
        # The split file carries the corpus prefix and the audio does not, so
        # both have to resolve or every recording looks unsplit.
        assert folds["ballroom_Media-105418"] == 6
        assert folds["Media-105418"] == 6
        assert folds["Albums-Foo-01"] == 0

    def test_a_malformed_line_is_skipped_rather_than_crashing_the_run(self, tmp_path):
        path = tmp_path / "s.split"
        path.write_text("good\t1\n\nnonsense\nalso good\t2\n", encoding="utf-8")
        folds = fold_map(path)
        assert folds["good"] == 1 and folds["also good"] == 2
        assert "nonsense" not in folds


class TestOctaveLabel:
    @pytest.mark.parametrize("bpm,expected", [
        (120.0, "1x"), (60.0, "1/2"), (240.0, "2x"), (180.0, "3/2"), (80.0, "2/3"),
    ])
    def test_it_names_the_level_from_the_beat_interval(self, bpm, expected):
        assert octave_label(grid(bpm, 40), grid(120.0, 40)) == expected

    def test_a_tempo_that_is_not_a_simple_ratio_is_not_called_an_octave_error(self):
        assert octave_label(grid(137.0, 40), grid(120.0, 40)) == "other"

    def test_too_few_beats_to_have_an_interval_is_not_a_level(self):
        assert octave_label(np.zeros(0), grid(120.0, 40)) == "none"
        assert octave_label(grid(120.0, 1), grid(120.0, 40)) == "none"


class TestMatchedBeatStats:
    def test_a_perfect_grid_has_nothing_missing_and_nothing_extra(self):
        beats = grid(120.0, 40)
        downbeats = beats[::4]
        stats = matched_beat_stats(beats, beats, downbeats, downbeats)
        assert stats["missing"] == pytest.approx(0.0)
        assert stats["extra"] == pytest.approx(0.0)
        assert stats["downbeat_accuracy"] == pytest.approx(1.0)

    def test_dropping_half_the_beats_is_counted_as_missing_not_as_extra(self):
        reference = grid(120.0, 40)
        stats = matched_beat_stats(reference, reference[::2],
                                   reference[::4], reference[::4])
        assert stats["missing"] == pytest.approx(0.5, abs=0.05)
        assert stats["extra"] == pytest.approx(0.0, abs=0.05)

    def test_beats_that_are_not_in_the_annotation_are_counted_as_extra(self):
        reference = grid(120.0, 20)
        doubled = grid(240.0, 40)
        stats = matched_beat_stats(reference, doubled,
                                   reference[::4], doubled[::8])
        assert stats["extra"] > 0.4
        assert stats["missing"] == pytest.approx(0.0, abs=0.05)

    def test_the_right_grid_with_the_wrong_phase_scores_zero_on_labels_not_on_the_grid(self):
        # This is the separation the whole module is for: the grid is perfect
        # and every beat matches, but every bar line is on the wrong beat.
        beats = grid(120.0, 40)
        stats = matched_beat_stats(beats, beats, beats[::4], beats[1::4])
        assert stats["missing"] == pytest.approx(0.0)
        assert stats["extra"] == pytest.approx(0.0)
        assert stats["downbeat_accuracy"] < 0.6

    def test_an_empty_estimate_yields_no_opinion_rather_than_a_zero(self):
        stats = matched_beat_stats(grid(120.0, 40), np.zeros(0),
                                   grid(120.0, 10), np.zeros(0))
        assert np.isnan(stats["downbeat_accuracy"])
        assert stats["matched"] == 0


class TestAuprc:
    def test_a_perfect_ranking_scores_one(self):
        assert auprc(np.array([0.9, 0.8, 0.2, 0.1]),
                     np.array([1, 1, 0, 0], dtype=bool)) == pytest.approx(1.0)

    def test_an_uninformative_score_lands_near_the_prevalence(self):
        rng = np.random.default_rng(0)
        labels = rng.random(4000) < 0.25
        assert auprc(rng.random(4000), labels) == pytest.approx(0.25, abs=0.05)

    def test_no_positives_is_undefined_rather_than_zero(self):
        assert np.isnan(auprc(np.array([0.5, 0.4]), np.array([0, 0], dtype=bool)))


class TestCheckpointResolution:
    def test_a_local_file_is_preferred_over_a_download(self, tmp_path):
        (tmp_path / "beat_this_fold3.ckpt").write_bytes(b"not really a checkpoint")
        assert resolve_checkpoint("fold3", tmp_path).endswith("beat_this_fold3.ckpt")

    def test_a_name_with_no_local_file_is_left_for_the_library_to_fetch(self, tmp_path):
        assert resolve_checkpoint("final0", tmp_path) == "final0"
        assert resolve_checkpoint("final0", None) == "final0"
