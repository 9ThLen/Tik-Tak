import csv
import pathlib

from eval.prepare_m0b import (
    _rows_from_complete_bars,
    _rows_from_position_annotation,
    meter_contract,
)


def test_meter_contract_separates_compound_tactus_from_notated_numerator():
    assert meter_contract("6/8") == {
        "grouping": 2,
        "subdivisions_per_tactus": 3,
        "meter_family": "compound_duple",
        "canonical_time_signature": "6/8",
        "notation_basis": "annotated",
        "supported": True,
    }
    assert meter_contract("6/4")["grouping"] == 6
    assert meter_contract("6/4")["subdivisions_per_tactus"] == 2


def test_measured_performance_downsamples_subdivision_grain_to_tactus():
    beats = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    rows, rejected = _rows_from_complete_bars(
        beats, [0.0, 0.6, 1.2], ["6/8", "6/8"])
    assert rejected == {}
    assert [(row["time_seconds"], row["position"], row["grouping"])
            for row in rows] == [
                (0.0, 1, 2), (0.3, 2, 2),
                (0.6, 1, 2), (0.9, 2, 2),
            ]


def test_position_annotation_keeps_changes_and_marks_unsupported(tmp_path):
    path = pathlib.Path(tmp_path) / "beats.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(("time_seconds", "beat_position"))
        for time, position in enumerate((1, 2, 3, 1, 2, 1, 2, 3, 4, 5, 1)):
            writer.writerow((time * 0.5, position))
    rows, rejected = _rows_from_position_annotation(path)
    assert rejected == {}
    assert [row["grouping"] for row in rows] == [3, 3, 3, 2, 2, 5, 5, 5, 5, 5]
    assert all(row["supported"] for row in rows[:5])
    assert not any(row["supported"] for row in rows[5:])
    assert [row["segment_id"] for row in rows] == [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
