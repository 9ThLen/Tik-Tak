import csv

import pytest

from eval.prepare_rwc2 import COLLECTIONS, prepare


COUNTS = {"C": 61, "G": 102, "J": 50, "P": 100, "R": 15}


@pytest.fixture
def rwc(tmp_path):
    root = tmp_path / "rwc2"
    annotations = root / "annotations"
    beats_root = annotations / "01_annotations_preprocessed" / "beats"
    audio_root = root / "audio"

    metadata_rows = []
    for coll_id, (folder, _) in COLLECTIONS.items():
        for number in range(1, COUNTS[coll_id] + 1):
            stem = f"RWC_{coll_id}{number:03d}"
            beat_file = beats_root / folder / f"{stem}.csv"
            beat_file.parent.mkdir(parents=True, exist_ok=True)
            beat_file.write_text(
                "t;beat\n0.100;1\n0.600;2\n1.100;3\n1.600;4\n"
                "2.100;1\n",
                encoding="utf-8",
            )
            wav = audio_root / folder / f"{stem}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()
            metadata_rows.append({
                "RWCID": stem, "CollID": coll_id, "Tempo": "120.0",
                "GenreMain": "Test", "GenreSub": folder,
            })

    with (annotations / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("RWCID", "CollID", "Tempo", "GenreMain", "GenreSub"),
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(metadata_rows)
    return root


def test_prepares_all_collections_without_copying_audio(rwc):
    rows = prepare(rwc)

    assert len(rows) == 328
    assert {row["dataset"] for row in rows} == {
        "rwc-classical", "rwc-genre", "rwc-jazz", "rwc-pop",
        "rwc-royalty-free",
    }
    assert (rwc / "manifest.csv").is_file()
    assert len(list((rwc / "normalized").rglob("*.csv"))) == 328
    assert len(list((rwc / "audio").rglob("*.wav"))) == 328

    normalized = rwc / rows[0]["annotation_relpath"]
    assert normalized.read_text(encoding="utf-8").splitlines()[0] == (
        "time_seconds,beat_position"
    )


def test_refuses_an_incomplete_download(rwc):
    next((rwc / "audio").rglob("*.wav")).unlink()

    with pytest.raises(ValueError, match="annotation has no WAV"):
        prepare(rwc)
