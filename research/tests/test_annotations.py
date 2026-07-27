import numpy as np
import pytest

from eval.annotations import find_pairs, load_annotation, parse_annotation


def test_reads_times_and_bar_positions():
    beats, downbeats, bpb, lengths = parse_annotation(
        "0.5 1\n1.0 2\n1.5 3\n2.0 4\n2.5 1\n3.0 2\n3.5 3\n4.0 4\n4.5 1\n"
    )
    assert np.allclose(beats, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    assert np.allclose(downbeats, [0.5, 2.5, 4.5])
    assert bpb == 4
    assert lengths == (4, 4)


def test_accepts_the_separators_the_tools_actually_emit():
    tab = parse_annotation("0.5\t1\n1.0\t2\n1.5\t1\n")
    comma = parse_annotation("0.5,1\n1.0,2\n1.5,1\n")
    spaces = parse_annotation("0.5   1\n1.0   2\n1.5   1\n")
    for parsed in (comma, spaces):
        assert np.allclose(parsed[0], tab[0])
        assert parsed[2] == tab[2]


def test_bare_times_are_beats_without_bar_lines():
    # Tapping beats is quick, marking bar lines is not, so a half-annotated set
    # has to be usable for the half it has rather than rejected.
    beats, downbeats, bpb, lengths = parse_annotation("0.5\n1.0\n1.5\n")
    assert len(beats) == 3
    assert len(downbeats) == 0
    assert bpb == 0
    assert lengths == ()


def test_comments_and_blank_lines_are_ignored():
    beats, _, bpb, _ = parse_annotation(
        "# exported from Sonic Visualiser\n\n0.5 1\n\n1.0 2  # snare\n1.5 1\n"
    )
    assert np.allclose(beats, [0.5, 1.0, 1.5])
    assert bpb == 2


def test_out_of_order_lines_are_sorted_with_their_positions():
    beats, downbeats, bpb, _ = parse_annotation("2.0 4\n0.5 1\n2.5 1\n1.5 3\n1.0 2\n")
    assert np.allclose(beats, [0.5, 1.0, 1.5, 2.0, 2.5])
    assert np.allclose(downbeats, [0.5, 2.5])
    assert bpb == 4


def test_the_metre_is_the_common_bar_and_not_the_first_one():
    # A pickup bar is the normal case, not an error: the first stretch is
    # incomplete and must not be allowed to name the metre.
    beats, _, bpb, lengths = parse_annotation(
        "0.0 3\n0.5 4\n1.0 1\n1.5 2\n2.0 3\n2.5 4\n3.0 1\n3.5 2\n4.0 3\n4.5 4\n5.0 1\n"
    )
    assert bpb == 4
    assert lengths == (4, 4)


def test_a_recording_that_changes_metre_says_so():
    from eval.annotations import Reference

    steady = Reference(beats=np.zeros(0), downbeats=np.zeros(0), bar_lengths=(4, 4, 4))
    changing = Reference(beats=np.zeros(0), downbeats=np.zeros(0), bar_lengths=(4, 4, 3))
    assert steady.meter_is_stable
    assert not changing.meter_is_stable


@pytest.mark.parametrize(
    "text, message",
    [
        ("nonsense 1\n", "not a time"),
        ("0.5 first\n", "not a beat number"),
        ("nan 1\n", "usable beat time"),
        ("0.5 0\n", "count from 1"),
    ],
)
def test_a_bad_line_is_named_rather_than_guessed_at(text, message):
    with pytest.raises(ValueError, match=message):
        parse_annotation(text)


def test_empty_annotation_is_empty_rather_than_an_error():
    beats, downbeats, bpb, lengths = parse_annotation("# nothing yet\n")
    assert len(beats) == 0 and len(downbeats) == 0 and bpb == 0 and lengths == ()


def _write_clip(folder, stem, suffix=".wav", annotation="0.5 1\n1.0 2\n1.5 1\n2.0 2\n"):
    (folder / f"{stem}{suffix}").write_bytes(b"not really audio")
    if annotation is not None:
        (folder / f"{stem}.beats").write_text(annotation)


def test_pairs_audio_with_the_annotation_beside_it(tmp_path):
    _write_clip(tmp_path, "take-one")
    _write_clip(tmp_path, "take-two", suffix=".mp3")

    pairs, problems = find_pairs(tmp_path)
    assert problems == []
    assert [p.name for p in pairs] == ["take-one", "take-two"]
    assert pairs[0].audio_path.name == "take-one.wav"
    assert pairs[0].beats_per_bar == 2


def test_a_missing_annotation_is_reported_not_skipped(tmp_path):
    # Silently scoring 12 of the 40 files someone recorded is the worst
    # outcome here: the mean looks fine and most of the set never ran.
    _write_clip(tmp_path, "annotated")
    _write_clip(tmp_path, "forgotten", annotation=None)

    pairs, problems = find_pairs(tmp_path)
    assert [p.name for p in pairs] == ["annotated"]
    assert any("forgotten" in p and "no annotation" in p for p in problems)


def test_an_unparsable_annotation_does_not_stop_the_rest(tmp_path):
    _write_clip(tmp_path, "good")
    _write_clip(tmp_path, "broken", annotation="this is not an annotation\n")

    pairs, problems = find_pairs(tmp_path)
    assert [p.name for p in pairs] == ["good"]
    assert any("broken" in p for p in problems)


def test_two_annotations_for_one_file_is_refused_rather_than_chosen_between(tmp_path):
    _write_clip(tmp_path, "take")
    (tmp_path / "take.txt").write_text("0.0 1\n1.0 1\n")

    pairs, problems = find_pairs(tmp_path)
    assert pairs == []
    assert any("remove all but one" in p for p in problems)


def test_an_empty_folder_says_so(tmp_path):
    pairs, problems = find_pairs(tmp_path)
    assert pairs == []
    assert any("no " in p for p in problems)


def test_subfolders_are_searched_and_keep_their_path_in_the_name(tmp_path):
    (tmp_path / "waltzes").mkdir()
    _write_clip(tmp_path / "waltzes", "one")

    pairs, _ = find_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].name.replace("\\", "/") == "waltzes/one"


def test_loads_from_a_path(tmp_path):
    path = tmp_path / "clip.beats"
    path.write_text("0.0 1\n0.5 2\n1.0 3\n1.5 1\n")
    reference = load_annotation(path)
    assert reference.name == "clip"
    assert reference.beats_per_bar == 3
    assert reference.has_downbeats


def test_a_beat_annotated_just_before_zero_is_dropped_not_refused():
    # Seven of the 911 Harmonix annotations open like this: the annotator's
    # grid extrapolated back past the start of the file. Refusing the file
    # would have thrown away seven real recordings over a beat no estimate
    # could have matched anyway, because the audio starts at zero.
    beats, downbeats, bpb, _ = parse_annotation(
        "-0.019183673\t4\n0.5\t1\n1.0\t2\n1.5\t3\n2.0\t4\n2.5\t1\n")

    assert beats[0] == pytest.approx(0.5)
    assert len(beats) == 5
    assert downbeats[0] == pytest.approx(0.5)
    assert bpb == 4


def test_an_annotation_entirely_before_zero_yields_nothing_rather_than_lying():
    beats, downbeats, bpb, lengths = parse_annotation("-2.0\t1\n-1.0\t2\n")

    assert len(beats) == 0 and len(downbeats) == 0
    assert bpb == 0 and lengths == ()


def test_a_time_that_is_not_a_number_at_all_is_still_refused():
    with pytest.raises(ValueError, match="not a usable beat time"):
        parse_annotation("nan\t1\n")
