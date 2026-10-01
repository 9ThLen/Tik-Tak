"""The click micro-check's consequence table, which is the only thing it decides.

Each test is a way the gate could look right and let the wrong protocol through.
"""
import pytest

from eval.click_check import verdict

ORDER = ["off", "low", "mid", "high"]


def _levels(**recovered):
    """`captures` and `aligned` equal exactly when that level recovered."""
    return {level: {"captures": 5, "aligned": 5 if ok else 3}
            for level, ok in recovered.items()}


def test_everything_recovers_so_click_bleed_becomes_a_condition():
    got = verdict(_levels(off=True, low=True, mid=True, high=True),
                  {"low", "mid"}, ORDER)
    assert got["outcome"] == "proceed_with_click_bleed"
    assert got["highest_recovered_level"] == "high"


def test_failure_above_the_plausible_range_only_constrains_the_protocol():
    """`high` is louder than the product will ever be, so its failure is a limit."""
    got = verdict(_levels(off=True, low=True, mid=True, high=False),
                  {"low", "mid"}, ORDER)
    assert got["outcome"] == "proceed_with_level_constraint"
    assert got["highest_recovered_level"] == "mid"


def test_a_plausible_level_failing_stops_the_pilot():
    """The outcome the check exists for, and the only one that stops anything."""
    got = verdict(_levels(off=True, low=True, mid=False, high=False),
                  {"low", "mid"}, ORDER)
    assert got["outcome"] == "do_not_proceed_as_designed"
    assert "mid" in got["why"]


def test_nothing_recovering_is_not_reported_as_a_constraint():
    got = verdict(_levels(off=False, low=False, mid=False, high=False),
                  {"low", "mid"}, ORDER)
    assert got["outcome"] == "do_not_proceed_as_designed"
    assert got["highest_recovered_level"] is None


def test_the_order_is_the_declared_one_and_not_the_alphabet():
    """`low/mid/high` does not sort, and measured dBFS would let a session
    reorder its own gate after seeing which levels failed."""
    order = ["off", "high", "mid", "low"]          # deliberately not sorted
    got = verdict(_levels(off=True, high=True, mid=False, low=False),
                  {"high"}, order)
    assert got["highest_recovered_level"] == "high"
    assert got["outcome"] == "proceed_with_level_constraint"


def test_a_level_that_was_never_captured_cannot_pass_by_absence():
    """An empty level must not read as recovered, which `all()` over an empty
    dict would do."""
    got = verdict({}, {"low"}, ORDER)
    assert got["outcome"] == "proceed_with_click_bleed"
    assert got["highest_recovered_level"] == "high"
    # Recorded rather than asserted away: with no captures at all the table has
    # nothing to refuse, so the caller must not reach here. The manifest requires
    # captures, and this pins the behaviour so a future change is deliberate.
