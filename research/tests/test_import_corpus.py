"""Importing a corpus: pairing, refusing, and counting groups rather than clips.

Built on synthetic trees rather than a real corpus, because the shapes that
matter are the ones that go wrong — a stem that appears twice, an annotation
with no audio, a file that does not parse — and a healthy corpus exercises none
of them.
"""

import json

import pytest
import soundfile as sf
import numpy as np

from eval.import_corpus import find_by_stem, group_of, main


def write_audio(path, seconds=1.0, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(seconds * rate), dtype="float32"), rate)


def write_beats(path, bars=4, per_bar=4, gap=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(bars * per_bar):
        lines.append(f"{i * gap:.3f}\t{i % per_bar + 1}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def corpus(tmp_path):
    """Two tracks, annotations and audio in differently shaped trees."""
    ann = tmp_path / "ann"
    aud = tmp_path / "aud"
    for name in ("track-one", "track-two"):
        write_beats(ann / "beats" / f"{name}.beats")
        write_audio(aud / "wav" / name[:5] / f"{name}.wav")
    return tmp_path


def run(corpus, *extra):
    return main(["--annotations", str(corpus / "ann"),
                 "--audio", str(corpus / "aud"),
                 "--out", str(corpus / "out"),
                 "--dataset", "testset", *extra])


def test_a_matching_pair_is_imported_whatever_shape_the_trees_have(corpus):
    # The annotations are flat, the audio is nested one level deeper. No corpus
    # agrees with another about this, which is why the stem is what pairs them.
    assert run(corpus) == 0

    out = corpus / "out" / "testset"
    assert (out / "track-one.wav").is_file()
    assert (out / "track-one.beats").is_file()
    assert (out / "track-two.beats").is_file()


def test_the_group_manifest_is_written_and_counts_recordings(corpus):
    assert run(corpus) == 0
    groups = json.loads((corpus / "out" / "groups.json").read_text())

    assert groups == {
        "testset/track-one": "testset/track-one",
        "testset/track-two": "testset/track-two",
    }


def test_two_corpora_can_share_an_out_folder_without_colliding(corpus):
    assert run(corpus) == 0
    assert main(["--annotations", str(corpus / "ann"),
                 "--audio", str(corpus / "aud"),
                 "--out", str(corpus / "out"),
                 "--dataset", "otherset"]) == 0

    groups = json.loads((corpus / "out" / "groups.json").read_text())
    assert len(groups) == 4
    assert groups["testset/track-one"] != groups["otherset/track-one"]


def test_takes_of_one_session_can_be_folded_into_one_group(tmp_path):
    ann, aud = tmp_path / "ann", tmp_path / "aud"
    for name in ("song-a_take-1", "song-a_take-2", "song-b_take-1"):
        write_beats(ann / f"{name}.beats")
        write_audio(aud / f"{name}.wav")

    assert main(["--annotations", str(ann), "--audio", str(aud),
                 "--out", str(tmp_path / "out"), "--dataset", "session",
                 "--group-by", "prefix"]) == 0

    groups = json.loads((tmp_path / "out" / "groups.json").read_text())
    assert groups["session/song-a_take-1"] == groups["session/song-a_take-2"]
    assert groups["session/song-b_take-1"] != groups["session/song-a_take-1"]


def test_an_unpaired_annotation_stops_the_import(corpus, capsys):
    write_beats(corpus / "ann" / "orphan.beats")

    assert run(corpus) == 2
    assert "orphan: annotated but no audio" in capsys.readouterr().out
    assert not (corpus / "out").exists()


def test_unpaired_files_can_be_accepted_deliberately(corpus):
    write_beats(corpus / "ann" / "orphan.beats")

    assert run(corpus, "--allow-partial") == 0
    assert (corpus / "out" / "testset" / "track-one.wav").is_file()
    assert not (corpus / "out" / "testset" / "orphan.beats").exists()


def test_a_repeated_stem_is_a_collision_and_not_a_coin_toss(corpus, capsys):
    # Picking one silently would pair an annotation with a different take.
    write_audio(corpus / "aud" / "elsewhere" / "track-one.wav")

    assert run(corpus) == 2
    assert "appears more than once" in capsys.readouterr().out


def test_an_annotation_that_does_not_parse_is_caught_at_import(corpus, capsys):
    (corpus / "ann" / "beats" / "track-one.beats").write_text("not a time at all\n")

    assert run(corpus) == 2
    out = capsys.readouterr().out
    assert "track-one" in out and "is not a time" in out


def test_beats_without_bar_lines_are_kept_and_announced(tmp_path, capsys):
    ann, aud = tmp_path / "ann", tmp_path / "aud"
    (ann).mkdir()
    (ann / "tapped.beats").write_text("\n".join(f"{i * 0.5:.3f}" for i in range(16)))
    write_audio(aud / "tapped.wav")

    assert main(["--annotations", str(ann), "--audio", str(aud),
                 "--out", str(tmp_path / "out"), "--dataset", "tap"]) == 0
    assert "no bar lines" in capsys.readouterr().out


def test_a_dry_run_writes_nothing(corpus, capsys):
    assert run(corpus, "--dry-run") == 0
    assert not (corpus / "out").exists()
    assert "nothing written" in capsys.readouterr().out


def test_an_existing_clip_is_not_overwritten_by_accident(corpus):
    assert run(corpus) == 0
    assert run(corpus) == 2
    assert run(corpus, "--force") == 0


def test_the_group_count_is_reported_rather_than_the_clip_count(tmp_path, capsys):
    # The number the Wilson bound is computed over. Reporting clips here would
    # be the same overcounting the group split exists to prevent.
    ann, aud = tmp_path / "ann", tmp_path / "aud"
    for i in range(6):
        write_beats(ann / f"song-a_take-{i}.beats")
        write_audio(aud / f"song-a_take-{i}.wav")

    main(["--annotations", str(ann), "--audio", str(aud),
          "--out", str(tmp_path / "out"), "--dataset", "s",
          "--group-by", "prefix", "--dry-run"])
    assert "1 independent group(s)" in capsys.readouterr().out


def test_an_empty_side_is_named_rather_than_producing_an_empty_dataset(tmp_path, capsys):
    (tmp_path / "ann").mkdir()
    (tmp_path / "aud").mkdir()

    assert main(["--annotations", str(tmp_path / "ann"),
                 "--audio", str(tmp_path / "aud"),
                 "--out", str(tmp_path / "out"), "--dataset", "empty"]) == 2
    assert "no annotation files" in capsys.readouterr().out


def test_stems_pair_across_arbitrary_nesting():
    # find_by_stem is what makes one script work for sixteen corpus layouts.
    assert group_of("x", "file", "d") == "d/x"
    assert group_of("a_b", "prefix", "d") == "d/a"
    assert group_of("nounderscore", "prefix", "d") == "d/nounderscore"
