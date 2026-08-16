#!/usr/bin/env python3
"""Build the canonical, work-grouped meter-diverse manifest used by M0b.

The source audio is never copied.  Small canonical tactus annotations and one
manifest are written below ``--output``.  Paths in the manifest are relative
to ``--music-root`` or to the manifest itself, so a report never publishes a
local user path.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import pathlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


SCHEMA = "tiktak.m0b_manifest/v1"
ANNOTATION_FIELDS = (
    "time_seconds", "position", "grouping", "subdivisions_per_tactus",
    "meter_family", "canonical_time_signature", "notation_basis",
    "supported", "segment_id",
)
SUPPORTED_GROUPINGS = frozenset({2, 3, 4, 6})
ALIGN_TOLERANCE_SEC = 0.070


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def meter_contract(signature: str) -> dict:
    """Translate notation into the acoustic contract without claiming it was inferred."""
    try:
        numerator_text, denominator_text = signature.strip().split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid time signature: {signature!r}") from error
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"invalid time signature: {signature!r}")

    compound = numerator > 3 and numerator % 3 == 0 and denominator in {8, 16}
    grouping = numerator // 3 if compound else numerator
    subdivisions = 3 if compound else 2
    names = {2: "duple", 3: "triple", 4: "quadruple", 6: "sextuple"}
    family = f"{'compound' if compound else 'simple'}_{names.get(grouping, 'other')}"
    return {
        "grouping": grouping,
        "subdivisions_per_tactus": subdivisions,
        "meter_family": family,
        "canonical_time_signature": f"{numerator}/{denominator}",
        "notation_basis": "annotated",
        "supported": grouping in SUPPORTED_GROUPINGS,
    }


def _float_rows(path: pathlib.Path, delimiter: str, time_key: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    out = []
    previous = -math.inf
    for number, row in enumerate(rows, start=2):
        try:
            value = float(row[time_key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}:{number}: invalid {time_key}") from error
        if not math.isfinite(value) or value < 0 or value <= previous:
            raise ValueError(f"{path.name}:{number}: times must increase")
        out.append({**row, "_time": value})
        previous = value
    return out


@functools.lru_cache(maxsize=None)
def score_signatures(path: pathlib.Path) -> list[str]:
    """Inherited time signature for every measure in the first MusicXML part."""
    root = ET.parse(path).getroot()
    part = root.find("part")
    if part is None:
        raise ValueError(f"{path.name}: MusicXML has no part")
    current = ""
    signatures = []
    for measure in part.findall("measure"):
        time = measure.find("./attributes/time")
        if time is not None:
            beats = time.findtext("beats")
            beat_type = time.findtext("beat-type")
            if beats and beat_type:
                current = f"{beats}/{beat_type}"
        if not current:
            raise ValueError(f"{path.name}: first measure has no time signature")
        signatures.append(current)
    return signatures


def _rows_from_complete_bars(
    beat_times: list[float], measure_times: list[float], signatures: list[str]
) -> tuple[list[dict], dict]:
    if len(measure_times) < 3:
        raise ValueError("fewer than two complete annotated measures")
    rows: list[dict] = []
    rejected = Counter()
    segment = -1
    previous_contract: tuple | None = None

    for index, (start, end) in enumerate(zip(measure_times, measure_times[1:])):
        if index >= len(signatures):
            rejected["missing_score_measure"] += 1
            continue
        contract = meter_contract(signatures[index])
        grouping = contract["grouping"]
        candidates = [candidate for candidate, time in enumerate(beat_times)
                      if abs(time - start) <= ALIGN_TOLERANCE_SEC]
        if not candidates:
            rejected["missing_measure_downbeat"] += 1
            continue
        first = min(candidates, key=lambda candidate: (
            abs(beat_times[candidate] - start), candidate))
        raw = [time for time in beat_times[first:] if time < end - 1e-9]
        if len(raw) < grouping or len(raw) % grouping:
            rejected["beat_grain_not_divisible_by_grouping"] += 1
            continue
        stride = len(raw) // grouping
        tactus = raw[::stride]
        if len(tactus) != grouping:
            rejected["wrong_tactus_count"] += 1
            continue

        key = (contract["grouping"], contract["subdivisions_per_tactus"],
               contract["canonical_time_signature"])
        if key != previous_contract:
            segment += 1
            previous_contract = key
        for position, time_sec in enumerate(tactus, start=1):
            rows.append({
                "time_seconds": time_sec,
                "position": position,
                **contract,
                "segment_id": segment,
            })
    if not rows:
        raise ValueError("no complete measures survived canonicalisation")
    return rows, dict(rejected)


def _rows_from_position_annotation(path: pathlib.Path) -> tuple[list[dict], dict]:
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    delimiter = ";" if ";" in header else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle, delimiter=delimiter))
    values = [(float(row["time_seconds"]), int(row["beat_position"]))
              for row in source]
    starts = [index for index, (_, position) in enumerate(values) if position == 1]
    rows: list[dict] = []
    rejected = Counter()
    segment = -1
    previous_grouping = None
    for left, right in zip(starts, starts[1:]):
        bar = values[left:right]
        grouping = len(bar)
        if [position for _, position in bar] != list(range(1, grouping + 1)):
            rejected["nonsequential_positions"] += 1
            continue
        if grouping != previous_grouping:
            segment += 1
            previous_grouping = grouping
        names = {2: "duple", 3: "triple", 4: "quadruple", 6: "sextuple"}
        for time_sec, position in bar:
            rows.append({
                "time_seconds": time_sec,
                "position": position,
                "grouping": grouping,
                "subdivisions_per_tactus": "unknown",
                "meter_family": f"unknown_{names.get(grouping, 'other')}",
                "canonical_time_signature": "unknown",
                "notation_basis": "unknown",
                "supported": grouping in SUPPORTED_GROUPINGS,
                "segment_id": segment,
            })
    if not rows:
        raise ValueError("no complete bars in beat-position annotation")
    return rows, dict(rejected)


def _write_annotation(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ANNOTATION_FIELDS})


def _record(
    *, name: str, corpus: str, work_id: str, audio: pathlib.Path,
    source_annotation: pathlib.Path, rows: list[dict], rejected: dict,
    music_root: pathlib.Path, output: pathlib.Path, hash_audio: bool,
    source_groupings: set[int] | None = None,
) -> dict:
    annotation = output / "annotations" / corpus / f"{name}.csv"
    _write_annotation(annotation, rows)
    groupings = sorted(source_groupings or {
        int(row["grouping"]) for row in rows})
    families = sorted({str(row["meter_family"]) for row in rows})
    supported = all(bool(row["supported"]) for row in rows)
    result = {
        "name": name,
        "corpus": corpus,
        "work_id": work_id,
        "audio_relpath": _relative(audio, music_root),
        "annotation_relpath": annotation.relative_to(output).as_posix(),
        "source_annotation_relpath": _relative(source_annotation, music_root),
        "audio_bytes": audio.stat().st_size,
        "annotation_bytes": annotation.stat().st_size,
        "annotation_sha256": sha256(annotation),
        "source_annotation_sha256": sha256(source_annotation),
        "groupings": groupings,
        "meter_families": families,
        "meter_changes": max(0, len({row["segment_id"] for row in rows}) - 1),
        "primary_eligible": supported,
        "canonical_rows": len(rows),
        "rejected_measures": rejected,
    }
    result["audio_sha256"] = sha256(audio) if hash_audio else None
    return result


def _matching_score(stem: str, scores: dict[str, pathlib.Path]) -> pathlib.Path:
    matches = [path for name, path in scores.items()
               if stem == name or stem.startswith(name + "_")]
    if not matches:
        raise ValueError(f"{stem}: no matching score")
    return max(matches, key=lambda path: len(path.stem))


def add_measured_dataset(
    *, corpus: str, audio_dir: pathlib.Path, beat_dir: pathlib.Path,
    measure_dir: pathlib.Path, score_dir: pathlib.Path, music_root: pathlib.Path,
    output: pathlib.Path, records: list[dict], exclusions: list[dict],
    hash_audio: bool,
) -> None:
    scores = {path.stem: path for path in score_dir.glob("*.xml")}
    annotations = {path.stem: path for path in beat_dir.glob("*.csv")}
    measures = {path.stem: path for path in measure_dir.glob("*.csv")}
    for audio in sorted(path for path in audio_dir.glob("*.wav") if path.is_file()):
        name = audio.stem
        try:
            beat_path = annotations[name]
            measure_path = measures[name]
            score = _matching_score(name, scores)
            beats = [row["_time"] for row in _float_rows(beat_path, ";", "time")]
            measure_times = [row["_time"] for row in
                             _float_rows(measure_path, ";", "time")]
            signatures = score_signatures(score)
            rows, rejected = _rows_from_complete_bars(
                beats, measure_times, signatures)
            records.append(_record(
                name=name, corpus=corpus, work_id=score.stem, audio=audio,
                source_annotation=beat_path, rows=rows, rejected=rejected,
                music_root=music_root, output=output, hash_audio=hash_audio,
                source_groupings={meter_contract(value)["grouping"]
                                  for value in signatures}))
        except Exception as error:
            exclusions.append({"name": name, "corpus": corpus,
                               "reason": str(error)})


def add_rwc2(
    root: pathlib.Path, music_root: pathlib.Path, output: pathlib.Path,
    records: list[dict], exclusions: list[dict], hash_audio: bool,
) -> None:
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    for item in source:
        name = item["track_id"]
        corpus = "rwc2"
        annotation = root / item["annotation_relpath"]
        audio = root / item["audio_relpath"]
        try:
            rows, rejected = _rows_from_position_annotation(annotation)
            records.append(_record(
                name=name, corpus=corpus, work_id=name, audio=audio,
                source_annotation=annotation, rows=rows, rejected=rejected,
                music_root=music_root, output=output, hash_audio=hash_audio))
        except Exception as error:
            exclusions.append({"name": name, "corpus": corpus,
                               "reason": str(error)})


def add_kraisler(
    root: pathlib.Path, music_root: pathlib.Path, output: pathlib.Path,
    records: list[dict], exclusions: list[dict], hash_audio: bool,
) -> None:
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    for item in metadata:
        name = item["track"]
        annotation = root / item["beats_csv"]
        audio = root / item["audio_mix_dry"]
        try:
            source = _float_rows(annotation, ",", "beat_time")
            current = ""
            starts = []
            for index, row in enumerate(source):
                if row.get("time_signature", "").strip():
                    current = row["time_signature"].strip()
                row["_signature"] = current
                if row.get("beat_type", "").strip().lower() == "db":
                    starts.append(index)
            rows = []
            rejected = Counter()
            segment = -1
            previous = None
            for left, right in zip(starts, starts[1:]):
                bar = source[left:right]
                contract = meter_contract(bar[0]["_signature"])
                grouping = contract["grouping"]
                if len(bar) < grouping or len(bar) % grouping:
                    rejected["beat_grain_not_divisible_by_grouping"] += 1
                    continue
                stride = len(bar) // grouping
                tactus = bar[::stride]
                key = (grouping, contract["subdivisions_per_tactus"],
                       contract["canonical_time_signature"])
                if key != previous:
                    segment += 1
                    previous = key
                for position, beat in enumerate(tactus, start=1):
                    rows.append({"time_seconds": beat["_time"], "position": position,
                                 **contract, "segment_id": segment})
            if not rows:
                raise ValueError("no complete bars survived canonicalisation")
            records.append(_record(
                name=f"kraisler_{name}", corpus="kraisler", work_id=name,
                audio=audio, source_annotation=annotation, rows=rows,
                rejected=dict(rejected), music_root=music_root, output=output,
                hash_audio=hash_audio))
        except Exception as error:
            exclusions.append({"name": name, "corpus": "kraisler",
                               "reason": str(error)})


def add_candombe(
    annotation_root: pathlib.Path, audio_root: pathlib.Path,
    music_root: pathlib.Path, output: pathlib.Path, records: list[dict],
    exclusions: list[dict], hash_audio: bool,
) -> None:
    for annotation in sorted(annotation_root.glob("*.csv")):
        name = annotation.stem
        audio = audio_root / f"{name}.flac"
        try:
            # Candombe has no header; parse it explicitly after the monotonic check.
            values = []
            previous = -math.inf
            with annotation.open(encoding="utf-8-sig", newline="") as handle:
                for number, raw in enumerate(csv.reader(handle), start=1):
                    time_sec = float(raw[0])
                    if time_sec <= previous:
                        raise ValueError(f"row {number}: times must increase")
                    bar_text, position_text = raw[1].split(".", 1)
                    values.append((time_sec, int(bar_text), int(position_text)))
                    previous = time_sec
            by_bar: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for time_sec, bar, position in values:
                by_bar[bar].append((time_sec, position))
            rows = []
            segment = 0
            rejected = Counter()
            contract = meter_contract("4/4")
            for bar in sorted(by_bar):
                events = by_bar[bar]
                if [position for _, position in events] != [1, 2, 3, 4]:
                    rejected["incomplete_bar"] += 1
                    continue
                for time_sec, position in events:
                    rows.append({"time_seconds": time_sec, "position": position,
                                 **contract, "segment_id": segment})
            if not rows:
                raise ValueError("no complete bars survived canonicalisation")
            records.append(_record(
                name=name, corpus="candombe", work_id=name, audio=audio,
                source_annotation=annotation, rows=rows,
                rejected=dict(rejected), music_root=music_root, output=output,
                hash_audio=hash_audio))
        except Exception as error:
            exclusions.append({"name": name, "corpus": "candombe",
                               "reason": str(error)})


def profile(records: list[dict]) -> dict:
    grouping_works: dict[int, set[str]] = defaultdict(set)
    corpus_counts = Counter()
    primary = 0
    changes = 0
    for record in records:
        corpus_counts[record["corpus"]] += 1
        primary += bool(record["primary_eligible"])
        changes += record["meter_changes"]
        if not record["primary_eligible"]:
            continue
        for grouping in record["groupings"]:
            if grouping in SUPPORTED_GROUPINGS:
                grouping_works[grouping].add(record["work_id"])
    return {
        "records": len(records),
        "primary_eligible_records": primary,
        "corpora": dict(sorted(corpus_counts.items())),
        "independent_works_by_grouping": {
            str(grouping): len(grouping_works[grouping])
            for grouping in sorted(SUPPORTED_GROUPINGS)},
        "meter_change_boundaries": changes,
    }


def build(music_root: pathlib.Path, output: pathlib.Path,
          hash_audio: bool = True) -> dict:
    music_root = music_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    exclusions: list[dict] = []

    add_rwc2(music_root / "rwc2", music_root, output,
             records, exclusions, hash_audio)
    selected = music_root / "selected-datasets"
    bpsd = selected / "bpsd" / "Beethoven_Piano_Sonata_Dataset_v2"
    add_measured_dataset(
        corpus="bpsd", audio_dir=bpsd / "1_Audio",
        beat_dir=bpsd / "2_Annotations" / "ann_audio_beat",
        measure_dir=bpsd / "2_Annotations" / "ann_audio_measure",
        score_dir=bpsd / "0_RawData" / "score_xml_unfolded",
        music_root=music_root, output=output, records=records,
        exclusions=exclusions, hash_audio=hash_audio)
    rubato = selected / "rubato" / "rubato"
    add_measured_dataset(
        corpus="rubato", audio_dir=rubato / "01_RawData" / "wav_22050_mono",
        beat_dir=rubato / "02_Annotations" / "ann_audio_beat",
        measure_dir=rubato / "02_Annotations" / "ann_audio_measure",
        score_dir=rubato / "01_RawData" / "score_musicxml",
        music_root=music_root, output=output, records=records,
        exclusions=exclusions, hash_audio=hash_audio)
    add_kraisler(selected / "kraisler" / "KRAISLER", music_root, output,
                 records, exclusions, hash_audio)
    add_candombe(
        selected / "candombe" / "candombe_annotations" / "with_bar_number",
        selected / "candombe-audio" / "candombe_audio", music_root, output,
        records, exclusions, hash_audio)

    records.sort(key=lambda row: (row["corpus"], row["work_id"], row["name"]))
    manifest = {
        "schema": SCHEMA,
        "supported_groupings": sorted(SUPPORTED_GROUPINGS),
        "audio_hashes_complete": hash_audio,
        "records": records,
        "technical_exclusions": exclusions,
        "profile": profile(records),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music-root", type=pathlib.Path,
                        default=repository / "music")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--skip-audio-hashes", action="store_true",
                        help="diagnostic only; a binding M0b run refuses it")
    args = parser.parse_args(argv)
    manifest = build(args.music_root, args.output,
                     hash_audio=not args.skip_audio_hashes)
    print(json.dumps(manifest["profile"], sort_keys=True))
    print(f"M0B_MANIFEST={args.output.resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
