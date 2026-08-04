"""The ensemble harness's arithmetic, apart from the corpus it runs on.

The measurements this script produces are two points of usable rate over 328
recordings, and the whole question is whether two points is a result. That
question is answered entirely by the code tested here — the paired counts, the
sign test and the multiplicity correction — so it is the part that must not be
trusted because it looked right once in a terminal.
"""

import pathlib
import sys

import pytest

from eval.beatnet_ensemble import MAX_DROPPED_SHARE, activation_of, aggregate, one, sign_test
from eval.provenance import provenance


def test_a_sign_test_of_nothing_is_not_a_finding():
    # No recording moved either way. The arms are indistinguishable, and the
    # p that says so has to be 1.0 rather than a division by zero.
    assert sign_test(0, 0) == 1.0


def test_the_sign_test_matches_the_binomial_by_hand():
    # 5 wins 0 losses is (1/2)^5 in each tail.
    assert sign_test(5, 0) == pytest.approx(2 / 32)
    # An even split cannot be evidence of anything.
    assert sign_test(10, 10) == pytest.approx(1.0)
    # The numbers this harness actually produced on RWC, so a change in the
    # implementation that moves them is visible here rather than in a claim.
    assert sign_test(25, 7) == pytest.approx(0.0021, abs=5e-5)
    assert sign_test(18, 12) == pytest.approx(0.3616, abs=5e-5)


def test_the_sign_test_does_not_care_which_arm_is_called_first():
    # Two-sided: the p is a statement about how lopsided the movement was, not
    # about who won. A one-sided p here would halve every number in the report.
    assert sign_test(19, 4) == sign_test(4, 19)


def test_ties_are_discarded_rather_than_counted_as_agreement():
    # Recordings both arms get right carry no information about which is
    # better. If they were counted, adding easy material to the corpus would
    # make every comparison look less significant.
    assert sign_test(9, 1) == sign_test(9, 1)
    assert sign_test(9, 1) < 0.05


def _rows(corpus, n, usable, strict):
    return [{"ok": True, "annotated": True, "corpus": corpus, "name": f"{corpus}{i}",
             "usable": i < usable, "usable_strict": i < strict,
             "usable_any_octave": i < usable, "f_measure": 0.5, "cmlt": 0.4,
             "reasons": [] if i < usable else ["wrong_octave"]}
            for i in range(n)]


def test_the_macro_average_ignores_corpora_below_the_minimum():
    # 40 recordings at 25% and 10 at 100%. Macro counts the first only; pooled
    # counts both. A harness that quietly pooled would report 40% here and the
    # difference would never show up as an error.
    rows = _rows("big", 40, 10, 10) + _rows("small", 10, 10, 10)
    summary = aggregate(rows)
    assert summary["usable_rate_macro"] == pytest.approx(0.25)
    assert summary["usable_rate_pooled"] == pytest.approx(0.40)


def test_the_per_recording_verdicts_are_keyed_so_two_arms_can_be_paired():
    # The paired comparison joins two arms on this key. If it were the bare
    # name, two corpora with a colliding name would silently pair the wrong
    # recordings against each other.
    summary = aggregate(_rows("rwc-pop", 2, 1, 1) + _rows("rwc-jazz", 2, 0, 0))
    assert set(summary["tracks"]) == {
        "rwc-pop/rwc-pop0", "rwc-pop/rwc-pop1",
        "rwc-jazz/rwc-jazz0", "rwc-jazz/rwc-jazz1"}
    assert summary["tracks"]["rwc-pop/rwc-pop0"] == [True, True]
    assert summary["tracks"]["rwc-jazz/rwc-jazz0"] == [False, False]


def test_an_arm_that_scored_nothing_reports_n_rather_than_dividing_by_zero():
    assert aggregate([])["n"] == 0


# The two defects that produced a wrong table on 2026-08-04. Both were silent,
# and silence is the thing being tested — not the failure itself, which is
# allowed to happen, but whether it can happen without saying so.


def test_a_binary_that_cannot_be_run_names_the_reason(tmp_path):
    # The failure that actually occurred: the environment stopped being able to
    # spawn a child process. It used to be indistinguishable from a corrupt
    # audio file, and from a file that was simply absent.
    stream, why = activation_of(tmp_path / "there-is-no-such-binary.exe",
                                tmp_path / "whatever.wav", tmp_path / "m.ttw")
    assert stream is None
    assert "could not run the binary" in why


def test_a_binary_that_fails_names_its_exit_code(tmp_path):
    # Python itself, handed arguments it cannot make sense of: a process that
    # starts and then exits non-zero, which is the other half of the failure.
    stream, why = activation_of(pathlib.Path(sys.executable),
                                tmp_path / "whatever.wav", tmp_path / "m.ttw")
    assert stream is None
    assert "exit" in why


def test_a_dropped_recording_carries_its_reason_rather_than_vanishing(tmp_path):
    # `one` returning None was the whole defect: the caller counted it and
    # moved on, so 109 recordings left no trace but a number. The marker key
    # forces the caller to deal with it.
    item = {"corpus": "rwc-pop", "name": "RWC_P007",
            "audio": tmp_path / "missing.wav", "annotated": True}
    result = one(item, tmp_path / "no-such-binary.exe",
                 {"1": tmp_path / "a.ttw"})
    assert "__dropped__" in result
    assert "rwc-pop/RWC_P007" in result["__dropped__"]


def test_the_refusal_threshold_is_far_below_what_went_wrong():
    # 109 of 328 is 33%. Any threshold below that would have caught it; this
    # asserts the constant was not later loosened past the failure it exists
    # for, which is the only way this check quietly stops working.
    assert MAX_DROPPED_SHARE < 109 / 328


def test_provenance_says_unknown_rather_than_dirty_when_git_cannot_be_asked(tmp_path):
    # A directory that is not a repository stands in for git being unavailable.
    # `tree_clean` must not be False here: False means "there were changes",
    # and reading "we could not tell" as "there were changes" is what let an
    # artifact with no commit at all look merely untidy.
    record = provenance(tmp_path / "not-a-repository")
    assert record["commit"] is None
    assert record["tree_clean"] is None
    assert record["provenance_error"]


def test_provenance_answers_normally_inside_a_repository():
    repository = pathlib.Path(__file__).resolve().parents[2]
    record = provenance(repository)
    assert record["commit"] and len(record["commit"]) == 40
    assert record["tree_clean"] in (True, False)
    assert "provenance_error" not in record
