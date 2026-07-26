#!/usr/bin/env python3
"""Assemble an eval dataset out of a corpus's annotations and its audio.

    python -m eval.import_corpus --annotations ~/beat_this_annotations/ballroom \
                                 --audio ~/audio/ballroom \
                                 --out ~/clips --dataset ballroom

The annotations for sixteen public corpora live in one MIT-licensed repository
(CPJKU/beat_this_annotations); the audio does not, and has to be obtained
separately per corpus. This script is the join: it pairs the two by filename,
checks that every annotation actually parses, writes the result in the layout
research/eval expects, and builds the groups.json that keeps excerpts of one
recording on one side of the validation/test split.

**It discovers the layout rather than assuming one.** Each corpus nests its
files differently and no two agree, so both trees are walked and paired on the
filename stem. That also means this same script imports a folder of your own
recordings, which is the case it is most likely to be used for first.

**Nothing is skipped quietly.** An annotation with no audio, audio with no
annotation, a stem that appears twice, a file that does not parse — all are
reported, and by default the import refuses to write a partial dataset. A set
that silently covered 200 of someone's 700 recordings would report healthy
numbers over a set that mostly never ran, and the thresholds calibrated on it
would belong to a different dataset than the one they thought they built.

**Licences are the caller's problem and are not checked here.** Several of
these corpora ship audio under no stated licence at all (see docs/ml-models.md):
usable for measuring quality, which is what this is for, and not for training a
model you sell. RWC is CC BY-NC. This script copies whatever it is pointed at.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from collections import defaultdict

from eval.annotations import (
    ANNOTATION_SUFFIXES,
    AUDIO_SUFFIXES,
    parse_annotation,
)

GROUPS_FILENAME = "groups.json"


def find_by_stem(folder: pathlib.Path, suffixes: tuple[str, ...],
                 ) -> tuple[dict[str, pathlib.Path], list[str]]:
    """Every file under `folder` with one of `suffixes`, keyed by filename stem.

    A stem that appears twice is a collision the caller has to resolve: picking
    one silently would pair an annotation with audio from a different take, and
    the resulting score would be a mystery to debug.
    """
    found: dict[str, pathlib.Path] = {}
    collisions: dict[str, list[pathlib.Path]] = defaultdict(list)
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stem = path.stem
        if stem in found:
            collisions[stem].append(path)
            continue
        found[stem] = path

    problems = []
    for stem, extra in collisions.items():
        others = ", ".join(str(p) for p in [found[stem], *extra])
        problems.append(f"the name {stem!r} appears more than once: {others}")
    return found, problems


def group_of(stem: str, mode: str, dataset: str) -> str:
    """Which independent group a clip belongs to.

    Default is one group per recording, which is right for corpora of distinct
    tracks. `--group-by prefix` folds `song-01_take-2` and `song-01_take-3`
    together on everything before the last underscore, which is the shape most
    multi-take recording sessions come out as. Anything more particular is
    better done by editing groups.json than by growing a rule here.
    """
    if mode == "prefix" and "_" in stem:
        return f"{dataset}/{stem.rsplit('_', 1)[0]}"
    return f"{dataset}/{stem}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--audio", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", required=True,
                        help="name this corpus is filed under; also the prefix "
                             "of every group id, so two corpora can share an "
                             "--out folder without their names colliding")
    parser.add_argument("--group-by", choices=("file", "prefix"), default="file",
                        help="'file' is one group per recording (default); "
                             "'prefix' folds name_take-1 and name_take-2 into "
                             "one group")
    parser.add_argument("--allow-partial", action="store_true",
                        help="import the pairs that are complete even when "
                             "others could not be paired or parsed")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be imported and write nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite clips already present in --out")
    args = parser.parse_args(argv)

    for folder, what in ((args.annotations, "--annotations"), (args.audio, "--audio")):
        if not folder.is_dir():
            print(f"{what}: {folder} is not a directory")
            return 2

    annotations, problems = find_by_stem(args.annotations, ANNOTATION_SUFFIXES)
    audio, audio_problems = find_by_stem(args.audio, AUDIO_SUFFIXES)
    problems.extend(audio_problems)

    if not annotations:
        print(f"no annotation files under {args.annotations} "
              f"(looked for {', '.join(ANNOTATION_SUFFIXES)})")
        return 2
    if not audio:
        print(f"no audio under {args.audio} "
              f"(looked for {', '.join(AUDIO_SUFFIXES)})")
        return 2

    paired = sorted(set(annotations) & set(audio))
    for stem in sorted(set(annotations) - set(audio)):
        problems.append(f"{stem}: annotated but no audio found")
    for stem in sorted(set(audio) - set(annotations)):
        problems.append(f"{stem}: audio but no annotation found")

    # Parsed now rather than at scoring time. A corpus with a handful of broken
    # files is normal; finding out during a calibration run is not.
    usable: list[str] = []
    no_downbeats: list[str] = []
    for stem in paired:
        try:
            beats, downbeats, beats_per_bar, _ = parse_annotation(
                annotations[stem].read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            problems.append(f"{stem}: {error}")
            continue
        if len(beats) == 0:
            problems.append(f"{stem}: the annotation has no beats in it")
            continue
        if len(downbeats) == 0 or beats_per_bar == 0:
            # Kept: beats alone still score the tracker, and the eval already
            # separates clips it cannot ask the bar-line question of.
            no_downbeats.append(stem)
        usable.append(stem)

    print(f"{len(annotations)} annotation(s), {len(audio)} audio file(s), "
          f"{len(usable)} usable pair(s)")
    if no_downbeats:
        print(f"{len(no_downbeats)} pair(s) have beats but no bar lines — they "
              f"will be scored on beats only")
    for complaint in problems:
        print(f"  ! {complaint}")

    if problems and not args.allow_partial:
        print(f"\n{len(problems)} problem(s) above. Fix them, or pass "
              f"--allow-partial to import the {len(usable)} pair(s) that are "
              f"complete.")
        return 2
    if not usable:
        print("nothing to import.")
        return 2

    groups = {stem: group_of(stem, args.group_by, args.dataset) for stem in usable}
    distinct = len(set(groups.values()))
    print(f"{distinct} independent group(s) — this is the number the confidence "
          f"bounds are computed over, not {len(usable)}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    destination = args.out / args.dataset
    destination.mkdir(parents=True, exist_ok=True)

    written = 0
    for stem in usable:
        audio_target = destination / (stem + audio[stem].suffix.lower())
        beats_target = destination / (stem + ".beats")
        if not args.force and (audio_target.exists() or beats_target.exists()):
            print(f"  ! {stem}: already in {destination} — pass --force to replace")
            return 2
        shutil.copyfile(audio[stem], audio_target)
        # Copied verbatim rather than rewritten: the parser already accepts the
        # corpus convention, and a rewrite would put this script's rounding
        # between the corpus and every future score.
        shutil.copyfile(annotations[stem], beats_target)
        written += 1

    # Merged, not replaced: two corpora may share an --out, and the group ids
    # are prefixed by dataset so they cannot collide.
    groups_path = args.out / GROUPS_FILENAME
    existing = {}
    if groups_path.is_file():
        existing = json.loads(groups_path.read_text(encoding="utf-8"))
    for stem, group in groups.items():
        existing[f"{args.dataset}/{stem}"] = group
    groups_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"\nwrote {written} clip(s) to {destination}")
    print(f"{groups_path} now covers {len(existing)} clip(s)")
    print(f"\nscore them with:\n"
          f"  python -m eval.downbeat_benchmark --dataset {args.out} "
          f"--groups {groups_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
