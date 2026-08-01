#!/usr/bin/env python3
"""Prepare the locally downloaded RWC 2.0 corpus for Tik-Tak evaluation.

The audio stays where the Zenodo archives were extracted.  Only the small beat
annotations are normalized and a manifest is written, so preparing the corpus
does not duplicate roughly thirty gigabytes of WAV files.

    python -m eval.prepare_rwc2 --root ../music/rwc2

RWC 2.0 audio and annotations are CC BY-NC 4.0.  This helper prepares an
evaluation corpus; it does not make the data suitable for commercial training
or redistribution.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import Counter


COLLECTIONS = {
    "C": ("RWC-C", "rwc-classical"),
    "G": ("RWC-G", "rwc-genre"),
    "J": ("RWC-J", "rwc-jazz"),
    "P": ("RWC-P", "rwc-pop"),
    "R": ("RWC-R", "rwc-royalty-free"),
}

ARCHIVE_MD5 = {
    "RWC-C.zip": "2ac9139c4f03a65885ae0d0d299f67f8",
    "RWC-G.zip": "e78cddfb6fa639bcb6a61ad873f3cceb",
    "RWC-J.zip": "c5d7d989e1afb8257ec50a3696d90c37",
    "RWC-P.zip": "960a11a2d7fb603ad0dae8428f53d4f0",
    "RWC-R.zip": "63e3b6263656a42c592ce1e90a88caa3",
}

MANIFEST_FIELDS = (
    "dataset", "track_id", "genre", "tempo_bpm", "meter", "beat_count",
    "downbeat_count", "annotation_relpath", "audio_relpath",
    "feature_relpath", "segment_relpath", "jams_relpath", "status", "notes",
)


def _read_metadata(path: pathlib.Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["RWCID"]: row for row in csv.DictReader(handle, delimiter=";")}


def _read_beats(path: pathlib.Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if not rows or not {"t", "beat"}.issubset(rows[0]):
        raise ValueError(f"{path}: expected semicolon-separated columns t;beat")

    beats: list[tuple[str, str]] = []
    previous = -math.inf
    for number, row in enumerate(rows, start=2):
        try:
            time_sec = float(row["t"])
            position = int(round(float(row["beat"])))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{number}: invalid beat row") from error
        if not math.isfinite(time_sec) or time_sec < 0 or time_sec <= previous:
            raise ValueError(f"{path}:{number}: beat times must increase")
        if position < 1:
            raise ValueError(f"{path}:{number}: beat position must be positive")
        previous = time_sec
        beats.append((row["t"], str(position)))
    return beats


def _meter(beats: list[tuple[str, str]]) -> int:
    downbeats = [index for index, (_, position) in enumerate(beats)
                 if position == "1"]
    lengths = [right - left for left, right in zip(downbeats, downbeats[1:])]
    return Counter(lengths).most_common(1)[0][0] if lengths else 0


def prepare(root: pathlib.Path) -> list[dict[str, str]]:
    root = root.resolve()
    annotations = root / "annotations" / "01_annotations_preprocessed" / "beats"
    audio = root / "audio"
    metadata_path = root / "annotations" / "metadata.csv"
    for folder in (annotations, audio):
        if not folder.is_dir():
            raise ValueError(f"missing directory: {folder}")
    if not metadata_path.is_file():
        raise ValueError(f"missing metadata: {metadata_path}")

    metadata = _read_metadata(metadata_path)
    normalized = root / "normalized"
    manifest: list[dict[str, str]] = []

    all_audio = {path.stem: path for path in audio.rglob("*.wav")}
    all_annotations: set[str] = set()

    for coll_id, (source_name, dataset) in COLLECTIONS.items():
        source = annotations / source_name
        for annotation in sorted(source.glob("*.csv")):
            stem = annotation.stem
            all_annotations.add(stem)
            wav = all_audio.get(stem)
            if wav is None:
                raise ValueError(f"{stem}: annotation has no WAV")
            beats = _read_beats(annotation)
            info = metadata.get(stem, {})

            target = normalized / source_name / annotation.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("time_seconds", "beat_position"))
                writer.writerows(beats)

            downbeats = sum(position == "1" for _, position in beats)
            manifest.append({
                "dataset": dataset,
                "track_id": stem,
                "genre": info.get("GenreSub") or info.get("GenreMain", ""),
                "tempo_bpm": info.get("Tempo", ""),
                "meter": str(_meter(beats)),
                "beat_count": str(len(beats)),
                "downbeat_count": str(downbeats),
                "annotation_relpath": target.relative_to(root).as_posix(),
                "audio_relpath": wav.relative_to(root).as_posix(),
                "feature_relpath": "",
                "segment_relpath": "",
                "jams_relpath": "",
                "status": "audio-aligned",
                "notes": "RWC 2.0; CC BY-NC 4.0; evaluation only",
            })

    unannotated = sorted(set(all_audio) - all_annotations)
    if unannotated:
        raise ValueError(f"{len(unannotated)} WAV file(s) have no annotation: "
                         + ", ".join(unannotated[:5]))
    if len(manifest) != 328:
        raise ValueError(f"expected 328 annotated tracks, found {len(manifest)}")

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)

    (root / "SOURCE.md").write_text(
        "# RWC 2.0 local evaluation corpus\n\n"
        "Audio: https://doi.org/10.5281/zenodo.18656623\n\n"
        "Annotations: https://github.com/rwc-music/rwc-annotations\n\n"
        "License: CC BY-NC 4.0. Evaluation/research only; do not bundle with "
        "the product or use for commercial model training without permission.\n\n"
        "Official archive MD5 values:\n\n"
        + "".join(f"- `{name}`: `{digest}`\n"
                  for name, digest in ARCHIVE_MD5.items()),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path,
                        default=repository / "music" / "rwc2")
    args = parser.parse_args(argv)
    try:
        rows = prepare(args.root)
    except (OSError, ValueError) as error:
        print(f"RWC 2.0 preparation failed: {error}", file=sys.stderr)
        return 2
    counts = Counter(row["dataset"] for row in rows)
    print(f"prepared {len(rows)} RWC 2.0 tracks at {args.root.resolve()}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print("benchmark with:")
    print("  python -m eval.live_corpus_benchmark "
          f"--manifest {args.root.resolve() / 'manifest.csv'} --mode model "
          "--model ../models/beatnet_model_1.ttw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
