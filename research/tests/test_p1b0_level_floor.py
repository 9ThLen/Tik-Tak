import pathlib

from eval.p1b0_level_floor import SEEDS, excerpt_deltas, parse_take


def test_a_take_name_splits_into_excerpt_and_cell():
    assert parse_take(pathlib.Path("blues.00025__quiet__far__phone.wav")) == \
        ("blues.00025", "quiet__far__phone")
    assert parse_take(pathlib.Path("jazz.00001__normal__loopback.wav")) == \
        ("jazz.00001", "normal__loopback")


def rows(track, cell, off, floor):
    out = []
    for seed, (a, b) in zip(SEEDS, zip(off, floor)):
        out.append({"ok": True, "track": track, "cell": cell, "arm": "off", "seed": seed,
                    "f_measure": a})
        out.append({"ok": True, "track": track, "cell": cell, "arm": "floor", "seed": seed,
                    "f_measure": b})
    return out


def test_deltas_pair_by_seed_and_average_cells_within_an_excerpt():
    records = (rows("a", "quiet__near__phone", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
               + rows("a", "quiet__far__phone", [0.5, 0.5, 0.5], [0.4, 0.4, 0.4])
               + rows("b", "quiet__near__phone", [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]))
    deltas = excerpt_deltas(records, ("quiet__near__phone", "quiet__far__phone"))
    # Excerpt a: +0.3 near and -0.1 far, one value between them, not two.
    assert set(deltas) == {"a", "b"}
    assert abs(deltas["a"] - 0.1) < 1e-12
    assert abs(deltas["b"] - 0.2) < 1e-12


def test_a_take_missing_a_seed_is_dropped_rather_than_unbalanced():
    records = rows("a", "quiet__near__phone", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    records = [r for r in records if not (r["arm"] == "floor" and r["seed"] == "2")]
    assert excerpt_deltas(records, ("quiet__near__phone",)) == {}
