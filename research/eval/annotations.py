"""Reading beat annotations, and pairing them with the audio they describe.

The format is the one the annotated corpora already use and the one Sonic
Visualiser exports, so a recording can be annotated in an ordinary tool and
dropped into a folder here with nothing in between::

    0.487   1
    0.975   2
    1.486   3
    1.974   4
    2.484   1

One line per beat: the time in seconds, then the beat's position in its bar,
counting from 1. Separator is any whitespace or a comma. Blank lines and lines
starting with ``#`` are ignored.

The beat number is optional — a file of bare times is a valid beat annotation
with no downbeats in it, and scores on beat metrics only. That matters because
tapping beats is quick and marking bar lines is not, so a set will realistically
grow in two passes.

**The metre is read, not declared.** It is the most common number of beats
between one bar line and the next, which is what makes a pickup bar or a single
inserted 2/4 harmless instead of a reason to reject the file. A recording whose
metre genuinely changes partway has no single right answer here; see
``Reference.meter_is_stable``.
"""

from __future__ import annotations

import pathlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Reference", "AUDIO_SUFFIXES", "ANNOTATION_SUFFIXES",
           "parse_annotation", "load_annotation", "find_pairs"]

# What the core's decoder actually accepts. Anything else in the folder is
# reported as unusable rather than skipped silently — a set that quietly scored
# 12 of the 40 files someone recorded is worse than one that refuses to start.
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")

# ``.beats`` is this project's name for it; the other two are what the public
# corpora ship, and accepting them costs a tuple entry.
ANNOTATION_SUFFIXES = (".beats", ".txt", ".csv")


@dataclass
class Reference:
    """Ground truth for one recording."""

    beats: np.ndarray                 # every beat, seconds
    downbeats: np.ndarray             # the subset that starts a bar
    beats_per_bar: int = 0            # 0 = not annotated
    name: str = ""
    # Independent statistical unit: every excerpt from the same song, session,
    # or backing track shares this id. None means this recording is its own
    # group, which is the right fallback for synthetic and exploratory runs.
    group_id: str | None = None
    audio_path: pathlib.Path | None = None
    annotation_path: pathlib.Path | None = None
    bar_lengths: tuple[int, ...] = field(default_factory=tuple)

    @property
    def has_downbeats(self) -> bool:
        return len(self.downbeats) > 0 and self.beats_per_bar > 0

    @property
    def meter_is_stable(self) -> bool:
        """Whether every complete bar has the same length.

        False means the recording changes metre, and any single answer the
        analysis gives is wrong for part of it — see the note in
        core/src/analysis/downbeat.hpp about what it would take to represent
        that. Such clips are reported separately rather than counted as
        failures.
        """
        return len(set(self.bar_lengths)) <= 1

    @property
    def duration_sec(self) -> float:
        return float(self.beats[-1]) if len(self.beats) else 0.0


def parse_annotation(text: str) -> tuple[np.ndarray, np.ndarray, int, tuple[int, ...]]:
    """Parses annotation text into (beats, downbeats, beats_per_bar, bar_lengths)."""
    times: list[float] = []
    positions: list[int | None] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        try:
            t = float(fields[0])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {fields[0]!r} is not a time") from exc
        if not np.isfinite(t):
            raise ValueError(f"line {lineno}: {t} is not a usable beat time")
        if t < 0.0:
            # Dropped, not refused. Seven of the 911 Harmonix annotations open
            # with a beat a few milliseconds before zero — the annotator's grid
            # extrapolated back past the start of the file, which is an ordinary
            # convention and not a broken file. Such a beat cannot be matched by
            # any estimate anyway, since the audio begins at zero, so dropping
            # it costs nothing and refusing it would have cost seven recordings.
            # The leading partial bar it leaves behind is the same shape as a
            # pickup, which the metre vote already ignores.
            continue

        position: int | None = None
        if len(fields) > 1:
            try:
                # Written as a float in some corpora ("1.000"), and as a bar.beat
                # pair in none of the ones we accept.
                position = int(round(float(fields[1])))
            except ValueError as exc:
                raise ValueError(f"line {lineno}: {fields[1]!r} is not a beat number") from exc
            if position < 1:
                raise ValueError(f"line {lineno}: beat numbers count from 1, got {position}")

        times.append(t)
        positions.append(position)

    if not times:
        return np.zeros(0), np.zeros(0), 0, ()

    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    beats = np.asarray(times, dtype=np.float64)[order]
    ordered_positions = [positions[i] for i in order]

    downbeat_index = [i for i, p in enumerate(ordered_positions) if p == 1]
    downbeats = beats[downbeat_index]

    # Bar lengths in beats, from one annotated bar line to the next. The stretch
    # before the first and after the last are incomplete by construction and are
    # not evidence about the metre.
    bar_lengths = tuple(
        int(b - a) for a, b in zip(downbeat_index, downbeat_index[1:])
    )
    beats_per_bar = Counter(bar_lengths).most_common(1)[0][0] if bar_lengths else 0

    return beats, downbeats, beats_per_bar, bar_lengths


def load_annotation(path: pathlib.Path) -> Reference:
    path = pathlib.Path(path)
    beats, downbeats, bpb, lengths = parse_annotation(path.read_text(encoding="utf-8"))
    return Reference(
        beats=beats,
        downbeats=downbeats,
        beats_per_bar=bpb,
        name=path.stem,
        annotation_path=path,
        bar_lengths=lengths,
    )


def find_pairs(folder: pathlib.Path) -> tuple[list[Reference], list[str]]:
    """Finds audio files with annotations beside them, by matching stem.

    Returns the pairs and a list of complaints about everything in the folder
    that looks like it was meant to be part of the set but is not usable. The
    complaints are returned rather than raised so one mistyped filename does not
    hide the other thirty-nine files.
    """
    folder = pathlib.Path(folder)
    pairs: list[Reference] = []
    problems: list[str] = []

    audio: dict[str, pathlib.Path] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in AUDIO_SUFFIXES:
            key = str(path.with_suffix("").relative_to(folder))
            if key in audio:
                problems.append(
                    f"{path.name}: two audio files share the stem {key!r}"
                    f" — cannot tell which the annotation belongs to"
                )
                continue
            audio[key] = path

    if not audio:
        problems.append(
            f"{folder}: no {'/'.join(AUDIO_SUFFIXES)} files found"
        )

    for key, path in sorted(audio.items()):
        found = [
            candidate
            for suffix in ANNOTATION_SUFFIXES
            if (candidate := path.with_suffix(suffix)).is_file()
        ]
        if not found:
            problems.append(f"{path.name}: no annotation beside it"
                            f" (expected {path.stem}{ANNOTATION_SUFFIXES[0]})")
            continue
        if len(found) > 1:
            problems.append(
                f"{path.name}: {len(found)} annotations beside it"
                f" ({', '.join(p.name for p in found)}) — remove all but one"
            )
            continue
        try:
            reference = load_annotation(found[0])
        except (ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{found[0].name}: {exc}")
            continue
        if len(reference.beats) < 2:
            problems.append(f"{found[0].name}: fewer than two beats annotated")
            continue
        reference.audio_path = path
        reference.name = key
        pairs.append(reference)

    return pairs, problems
