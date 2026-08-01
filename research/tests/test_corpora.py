"""The benchmark's composition, checked rather than assumed.

Most of these run on synthetic trees, because the annotations are not in git —
they are fetched separately and a fresh checkout has none. The last test runs
against the real set when it happens to be present, which is where the numbers
in eval/corpora.py were verified in the first place.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from eval import corpora


def write(folder: pathlib.Path, name: str, lines: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.beats").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: pathlib.Path, name: str, count: int, *, downbeats: bool = True,
          beat_only: tuple[str, ...] = ()) -> None:
    """`count` annotations, of which the ones named in `beat_only` carry no bars.

    The named files are written under exactly those names — the real corpora do
    not number their files the way a loop would, and a helper that invented its
    own stems would build a tree the profile can never match.
    """
    folder = root / name / "annotations" / "beats"
    stems = list(beat_only)
    stems += [f"{name}_{i:05d}" for i in range(count - len(stems))]
    assert len(stems) == count

    for stem in stems:
        bare = not downbeats or stem in beat_only
        lines = [f"{0.5 * j:.4f}" + ("" if bare else f"\t{j % 4 + 1}") for j in range(8)]
        write(folder, stem, lines)


def test_the_profile_adds_up():
    assert corpora.total_tracks() == 2812
    assert {c.name for c in corpora.PROFILE} == {"harmonix", "ballroom", "gtzan", "smc"}
    assert dict((c.name, c.tracks) for c in corpora.PROFILE) == {
        "harmonix": 911, "ballroom": 685, "gtzan": 999, "smc": 217}


def test_a_matching_tree_produces_no_complaints(tmp_path):
    build(tmp_path, "harmonix", 911)
    build(tmp_path, "ballroom", 685)
    build(tmp_path, "gtzan", 999, beat_only=corpora.BEAT_ONLY["gtzan"])
    build(tmp_path, "smc", 217, downbeats=False)
    assert corpora.verify(tmp_path) == []


def test_a_missing_corpus_is_named(tmp_path):
    build(tmp_path, "ballroom", 685)
    complaints = corpora.verify(tmp_path)
    assert any("harmonix: not present" in c for c in complaints)
    assert any("smc: not present" in c for c in complaints)


def test_a_short_corpus_is_caught(tmp_path):
    # The failure this exists for: a corpus that paired 670 of its 685 files and
    # reported a healthy-looking average over the ones that worked.
    build(tmp_path, "harmonix", 911)
    build(tmp_path, "ballroom", 670)
    build(tmp_path, "gtzan", 999, beat_only=corpora.BEAT_ONLY["gtzan"])
    build(tmp_path, "smc", 217, downbeats=False)

    complaints = corpora.verify(tmp_path)
    assert len(complaints) == 1
    assert "ballroom: 670 annotations, profile says 685" in complaints[0]


def test_an_annotation_that_lost_its_bar_lines_is_caught(tmp_path):
    # Scoring a file with no bar positions as though it had them turns absent
    # ground truth into a run of downbeat failures, which looks exactly like a
    # tracker that got worse.
    build(tmp_path, "harmonix", 911, beat_only=("harmonix_untapped",))
    build(tmp_path, "ballroom", 685)
    build(tmp_path, "gtzan", 999, beat_only=corpora.BEAT_ONLY["gtzan"])
    build(tmp_path, "smc", 217, downbeats=False)

    complaints = corpora.verify(tmp_path)
    assert len(complaints) == 1
    assert "harmonix_untapped" in complaints[0]


def test_a_recovered_annotation_asks_to_be_recorded(tmp_path):
    # The opposite drift: upstream fills in the bar lines, and the exclusion
    # list quietly keeps excluding something that no longer needs it.
    build(tmp_path, "harmonix", 911)
    build(tmp_path, "ballroom", 685)
    build(tmp_path, "gtzan", 999, beat_only=corpora.BEAT_ONLY["gtzan"][1:])
    build(tmp_path, "smc", 217, downbeats=False)

    complaints = corpora.verify(tmp_path)
    assert len(complaints) == 1
    assert "gtzan_jazz_00003" in complaints[0]
    assert "update BEAT_ONLY" in complaints[0]


def test_a_beat_only_corpus_that_grew_bar_lines_is_caught(tmp_path):
    build(tmp_path, "harmonix", 911)
    build(tmp_path, "ballroom", 685)
    build(tmp_path, "gtzan", 999, beat_only=corpora.BEAT_ONLY["gtzan"])
    build(tmp_path, "smc", 217)   # profile says none of these have bar positions

    complaints = corpora.verify(tmp_path)
    assert len(complaints) == 1
    assert "the profile says the whole corpus does" in complaints[0]


def test_comments_and_blank_lines_do_not_make_a_file_look_beat_only(tmp_path):
    folder = tmp_path / "ballroom" / "annotations" / "beats"
    write(folder, "ballroom_00000", ["# tapped by hand", "", "0.5\t1", "1.0\t2"])
    found = corpora.census(tmp_path)
    assert found.beat_only["ballroom"] == []


def test_the_real_annotations_match_the_profile_if_they_are_here():
    """Point TIKTAK_ANNOTATIONS at the annotation tree to run this for real.

    Skipped without it, because the annotations are not in git. That makes this
    a test that mostly does not run, which is usually a smell — here it is the
    only honest arrangement: the alternative is asserting the profile against
    nothing and calling it verified.
    """
    where = os.environ.get("TIKTAK_ANNOTATIONS")
    if not where:
        pytest.skip("set TIKTAK_ANNOTATIONS to the annotation tree")
    root = pathlib.Path(where)
    if not root.is_dir():
        pytest.skip(f"TIKTAK_ANNOTATIONS points at {root}, which is not a directory")
    assert corpora.verify(root) == []
