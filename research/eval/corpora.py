"""Which recordings the public benchmark is, and what is missing from it.

A benchmark drifts by losing files, not by announcing that it has. Ten tracks
that failed to pair, a corpus that arrived at 685 where 698 were expected, a set
of annotations with no bar lines quietly scored as bar-line failures — each of
these moves a number without moving anything visible. So the composition is
written down here and checked, rather than being whatever happened to be in a
folder on the day.

The profile is Beat This!'s, which is what makes our numbers comparable with
theirs. Four corpora, 2812 recordings::

    harmonix   911    modern pop and rock, the closest public stand-in for
                      the material this product is actually used on
    ballroom   685    steady programmed dance music — the easy end, and so
                      the place a high score has to appear first
    gtzan      999    thirty-second excerpts across ten genres
    smc        217    deliberately hard: rubato, classical, solo playing

**Two kinds of fact live in this file and they are not equally certain.**
Counts, exclusions-by-absence and the beat-only census are verified here
against the annotations themselves, by test_corpora.py. The *reasons* for the
exclusions are reported from the upstream projects and are not checkable from
the annotations alone — a file that is absent looks the same whatever the
reason. Each reason below says which it is.

**Audio is not here and mostly cannot be.** Only Ballroom and SMC distribute
it; GTZAN's is no longer published and carries no stated licence, and Harmonix
never distributed audio at all. Whatever is obtained stays out of git and out
of the product, exactly as the model weights do — it is material for measuring
quality, not material to ship.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

__all__ = ["Corpus", "PROFILE", "BEAT_ONLY", "census", "verify"]


@dataclass(frozen=True)
class Corpus:
    name: str
    tracks: int
    has_downbeats: bool
    audio_available: bool
    note: str


# Counts verified against the annotations; notes reported from upstream.
PROFILE: tuple[Corpus, ...] = (
    Corpus("harmonix", 911, True, False,
           "912 upstream; 0120_hallowedbethyname is absent here. Audio was "
           "never distributed — the project publishes mel spectrograms."),
    Corpus("ballroom", 685, True, True,
           "698 upstream; 13 absent here, reported as 4 exact duplicates and "
           "9 re-recordings of the same performance."),
    Corpus("gtzan", 999, True, False,
           "1000 upstream; reggae_00086 is absent here, reported as an "
           "annotation that stops at 6.84 s. Audio is no longer published and "
           "carries no stated licence."),
    Corpus("smc", 217, False, True,
           "Beat times only, with no bar positions anywhere in the corpus."),
)

# Annotations carrying beat times but no bar positions, so they can score beat
# metrics and must not be scored for downbeats. Counted, not guessed: scoring a
# file that has no bar lines as though it had them turns absent ground truth
# into a run of failures, which is indistinguishable from a tracker that got
# worse.
BEAT_ONLY: dict[str, tuple[str, ...]] = {
    "gtzan": (
        "gtzan_jazz_00003",
        "gtzan_jazz_00009",
        "gtzan_jazz_00010",
        "gtzan_jazz_00014",
        "gtzan_jazz_00018",
        "gtzan_jazz_00020",
    ),
    # Every file in SMC. Listed by rule rather than by name because 217 names
    # would be a wall that nobody reads and that a single new file invalidates.
    "smc": (),
}


@dataclass
class Census:
    """What is actually on disk, as opposed to what was expected."""

    tracks: dict[str, int] = field(default_factory=dict)
    beat_only: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _is_beat_only(path: pathlib.Path) -> bool:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if len(line.replace(",", " ").split()) >= 2:
            return False
    return True


def census(root: pathlib.Path) -> Census:
    """Count what is under `root`, one directory per corpus."""
    out = Census()
    for corpus in PROFILE:
        folder = root / corpus.name
        if not folder.is_dir():
            out.missing.append(corpus.name)
            continue
        files = sorted(folder.rglob("*.beats"))
        out.tracks[corpus.name] = len(files)
        out.beat_only[corpus.name] = sorted(f.stem for f in files if _is_beat_only(f))
    return out


def verify(root: pathlib.Path) -> list[str]:
    """Complaints, one per discrepancy. Empty means the set is the profile.

    Returns them rather than raising, because a caller assembling a corpus
    wants every problem at once, not the first one.
    """
    found = census(root)
    complaints: list[str] = []

    for corpus in PROFILE:
        if corpus.name in found.missing:
            complaints.append(f"{corpus.name}: not present under {root}")
            continue

        actual = found.tracks[corpus.name]
        if actual != corpus.tracks:
            complaints.append(
                f"{corpus.name}: {actual} annotations, profile says {corpus.tracks}")

        without = found.beat_only[corpus.name]
        if not corpus.has_downbeats:
            if len(without) != actual:
                complaints.append(
                    f"{corpus.name}: {len(without)} of {actual} lack bar positions, "
                    f"and the profile says the whole corpus does")
            continue

        expected = set(BEAT_ONLY.get(corpus.name, ()))
        surprise = sorted(set(without) - expected)
        recovered = sorted(expected - set(without))
        if surprise:
            complaints.append(
                f"{corpus.name}: {len(surprise)} annotation(s) lost their bar "
                f"positions: {', '.join(surprise[:5])}"
                + (" …" if len(surprise) > 5 else ""))
        if recovered:
            complaints.append(
                f"{corpus.name}: {', '.join(recovered)} now carry bar positions; "
                f"update BEAT_ONLY")
    return complaints


def total_tracks() -> int:
    return sum(c.tracks for c in PROFILE)
