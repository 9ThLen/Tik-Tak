#!/usr/bin/env python3
"""The P1-B0 replay session: one file to play, and what to do with the captures.

`PREREGISTERED_P1B0.md` turned the pilot into a replay: GTZAN excerpts played
through a loudspeaker into microphones in a room, annotation inherited through
the slates rather than produced. This is the tooling for that session, in the
order the session needs it.

    build   choose the excerpts by a rule fixed before any capture, wrap each
            in slates, and write one programme per playback level
    check   align a capture of a programme, take by take -- run after every
            pass, while a failed pass still costs minutes rather than a session
    score   after the session: the same tracker on the clean excerpt and on
            every aligned take, per condition cell

**The excerpts are chosen by position, not by content.** Per genre, sorted by
track id, evenly spaced through the list. A property of the music -- tempo,
say -- would be the second way this repository has picked a subset and got a
different one than it meant (`docs/ROOM_PROTOCOL.md`, step 1).

**The level is in the file, not on the dial.** Every programme after the first
is the same samples attenuated by a stated number of decibels, so the speaker's
volume is set once and never touched. A dial moved between passes is a level
nobody can state afterwards; a gain in the file is exact and repeatable.

**The clean arm is the programme itself, not the excerpt.** The same file the
speaker was sent, read, split and cut by the same code as a capture, at the
same level. That is stricter than it sounds necessary, and it is necessary: the
live tracker is deterministic but not smooth, and on a hard excerpt the same
music re-quantised at 1e-7 moves F by 0.4, while 12 dB of pure digital gain
moves the mean. A clean arm that differs from the capture path by anything but
the room writes both into the room's column. They are reported instead, as two
controls with no room in either.

    cd research
    .venv/Scripts/python -m eval.replay_programme build \
        --manifest ../music/ground-truth/manifest.csv --music ../music \
        --output ../music/replay-p1b0
    .venv/Scripts/python -m eval.replay_programme check \
        --layout ../music/replay-p1b0/programme_normal.layout.json \
        --capture "../music/replay-p1b0/captures/near_normal_phone.m4a"
    .venv/Scripts/python -m eval.replay_programme score \
        --session ../music/replay-p1b0/session.json \
        --manifest ../music/ground-truth/manifest.csv --music ../music \
        --binary <dump_analysis> --model <beatnet.ttw> \
        --output ../research/results/p1b0_replay.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from math import gcd

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.click_programme import build_programme, locate_takes  # noqa: E402
from eval.slate import RATE, align_by_slate, build_take  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "tiktak.replay_programme/v1"
SESSION_SCHEMA = "tiktak.replay_session/v1"
RESULT_SCHEMA = "tiktak.replay_result/v1"

# Two per genre is twenty excerpts: at 42 s a take, a 14-minute pass, and
# four passes inside the two-hour ceiling with the setup and the checks.
PER_GENRE = 2
# The programme gap. `click_programme` needed thirty seconds for takes over
# two minutes long, because the operator's slop is derived from the first
# take's own head-to-tail distance. Thirty-second excerpts need far less, and
# a thirty-second gap would make half of every pass silence. On thirty-second
# takes a 1 s gap loses three takes of four and 2 s splits back cleanly even
# at 1000 ppm; eight is four times that, and lets a room's tail die before the
# next slate. The round trip in the tests pins it.
GAP_SECONDS = 8.0
# The registered levels. `normal` is whatever the speaker dial gives once it
# is set; `quiet` is the same programme 12 dB down, with the dial untouched.
LEVELS_DB = {"normal": 0.0, "quiet": -12.0}
# Headroom for the loudest sample of the whole programme, applied once to
# every level alike, so resampling overshoot cannot clip one excerpt and not
# another.
PEAK_CEILING = 10 ** (-1.0 / 20)
A1_ALIGNED_FRACTION = 0.90
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_DRAWS = 10_000
# A2: the full collection is sized to this half-width on a cell's mean paired
# difference, and reported for the worst cell.
A2_HALF_WIDTH = 0.03


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_excerpts(rows: list[dict], per_genre: int = PER_GENRE) -> list[dict]:
    """Evenly spaced through each genre's sorted list; nothing about the music.

    Position `round((j + 0.5) * n / per_genre)` for `j` in range, so the picks
    sit in the middle of equal slices rather than at the ends of the list,
    where a corpus's first and last files are the likeliest to be special.
    Only rows the manifest marks `ok` are eligible: GTZAN ships a file that
    will not open and one whose annotation is under review.
    """
    by_genre: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row["dataset"] == "gtzan" and row["status"] == "ok":
            by_genre[row["genre"]].append(row)
    chosen = []
    for genre in sorted(by_genre):
        ordered = sorted(by_genre[genre], key=lambda row: row["track_id"])
        count = len(ordered)
        for j in range(per_genre):
            chosen.append(ordered[min(count - 1,
                                      int((j + 0.5) * count / per_genre))])
    return chosen


def resample(audio: np.ndarray, rate: int, target: int = RATE) -> np.ndarray:
    from scipy.signal import resample_poly

    if int(rate) == target:
        return np.asarray(audio, dtype=np.float64)
    divisor = gcd(int(target), int(rate))
    return resample_poly(np.asarray(audio, dtype=np.float64),
                         target // divisor, int(rate) // divisor)


def build(manifest: pathlib.Path, music: pathlib.Path, output: pathlib.Path,
          per_genre: int = PER_GENRE, gap_seconds: float = GAP_SECONDS) -> dict:
    import csv

    import soundfile

    from eval.live_corpus_benchmark import _audio_path

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen = select_excerpts(rows, per_genre)

    clean_dir = output / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    excerpts = []
    for row in chosen:
        source = _audio_path(manifest.parent, row)
        audio, rate = soundfile.read(str(source), dtype="float64",
                                     always_2d=True)
        excerpts.append((row, source, resample(audio.mean(axis=1), rate)))

    # One factor for the whole programme, measured on the takes themselves.
    peak = max(float(np.max(np.abs(build_take(x)[0]))) for _, _, x in excerpts)
    headroom = min(1.0, PEAK_CEILING / peak)

    selection = []
    for row, source, audio in excerpts:
        clean = clean_dir / f"{row['track_id']}.wav"
        # The clean arm is written at the programme's headroom too, so the
        # only difference between the arms is the room.
        soundfile.write(str(clean), audio * headroom, RATE, subtype="FLOAT")
        selection.append({"track": row["track_id"], "genre": row["genre"],
                          "source": source.relative_to(music).as_posix(),
                          "source_sha256": sha256(source),
                          "clean": clean.relative_to(output).as_posix(),
                          "clean_sha256": sha256(clean)})

    written = {}
    for level, gain_db in LEVELS_DB.items():
        gain = headroom * 10 ** (gain_db / 20)
        takes = []
        for (row, _, audio) in excerpts:
            take, layout = build_take(audio * gain)
            takes.append((row["track_id"], take, layout))
        programme, layout = build_programme(takes, gap_seconds=gap_seconds)
        # The programme must split back on itself before anyone plays it.
        refused = [r["track"] for r in locate_takes(programme, layout)
                   if not r["accepted"]]
        if refused:
            raise RuntimeError(f"{level}: programme does not split back "
                               f"on itself: {refused}")
        path = output / f"programme_{level}.wav"
        soundfile.write(str(path), programme, RATE, subtype="PCM_24")
        layout.update({"schema_replay": SCHEMA, "level": level,
                       "gain_db": gain_db, "headroom": headroom,
                       "programme_sha256": sha256(path)})
        path.with_suffix(".layout.json").write_text(
            json.dumps(layout, indent=1) + "\n", encoding="utf-8")
        written[level] = {"file": path.name, "minutes": layout["seconds"] / 60,
                          "sha256": layout["programme_sha256"]}

    summary = {"schema": SCHEMA, "rule": {
        "corpus": "gtzan", "status": "ok", "per_genre": per_genre,
        "position": "round((j + 0.5) * n / per_genre) in sorted track_id"},
        "gap_seconds": gap_seconds, "levels_db": LEVELS_DB,
        "headroom": headroom, "programmes": written, "excerpts": selection}
    (output / "selection.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def read_capture(path: pathlib.Path) -> tuple[np.ndarray, float]:
    """Mono at the slate rate, and the rate the device actually wrote."""
    from eval.room_recording import read_audio

    mono, rate = read_audio(path)
    return resample(mono, int(rate)), float(rate)


def check(capture: np.ndarray, layout: dict) -> dict:
    rows = locate_takes(capture, layout)
    aligned = sum(row["accepted"] for row in rows)
    return {"takes": rows, "aligned": aligned, "total": len(rows),
            "a1_fraction_met": aligned >= A1_ALIGNED_FRACTION * len(rows)}


def paired_interval(deltas: list[float], seed: int = BOOTSTRAP_SEED,
                    draws: int = BOOTSTRAP_DRAWS) -> list[float] | None:
    """Percentile bootstrap of the mean paired difference, resampling excerpts."""
    if len(deltas) < 2:
        return None
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def captures_needed(deltas: list[float],
                    half_width: float = A2_HALF_WIDTH) -> dict | None:
    """`n = (1.96 SD / h)^2`, with the interval the SD's own uncertainty gives.

    A point estimate of n from twenty excerpts would read as more certain than
    it is; the chi-square interval on the SD carries straight through to n.
    """
    from scipy.stats import chi2

    if len(deltas) < 2:
        return None
    k = len(deltas) - 1
    sd = float(np.std(deltas, ddof=1))
    variance_ci = (k * sd ** 2 / chi2.ppf(0.975, k),
                   k * sd ** 2 / chi2.ppf(0.025, k))
    need = lambda variance: float(np.ceil(1.96 ** 2 * variance / half_width ** 2))
    return {"half_width": half_width, "estimate": need(sd ** 2),
            "ci95": [need(variance_ci[0]), need(variance_ci[1])]}


REQUIRED_CAPTURE_FIELDS = ("file", "level", "distance", "device",
                           "device_model", "channels", "os_processing")


def validate_session(session: dict) -> None:
    """Refuse a session that did not record what the preregistration requires.

    `os_processing` may say "unknown"; it may not be missing. Capturing with
    unknown processing is permitted and not recording that it was unknown
    voids the capture (`PREREGISTERED_P1B0.md`).
    """
    if session.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"session schema must be {SESSION_SCHEMA}")
    for index, entry in enumerate(session["captures"]):
        missing = [key for key in REQUIRED_CAPTURE_FIELDS
                   if entry.get(key) in (None, "")]
        if missing:
            raise ValueError(f"capture {index} does not record {missing}")
        if entry["level"] not in LEVELS_DB:
            raise ValueError(f"capture {index}: undeclared level "
                             f"{entry['level']!r}")


def score(session_path: pathlib.Path, manifest: pathlib.Path,
          music: pathlib.Path, binary: pathlib.Path, model: pathlib.Path,
          output: pathlib.Path) -> dict:
    import soundfile

    from eval.live_corpus_benchmark import _score_one, load_corpus
    from eval.provenance import experiment_provenance

    session = json.loads(session_path.read_text(encoding="utf-8"))
    validate_session(session)
    base = session_path.parent
    selection = json.loads((base / "selection.json").read_text(encoding="utf-8"))
    layouts = {level: json.loads((base / f"programme_{level}.layout.json")
                                 .read_text(encoding="utf-8"))
               for level in LEVELS_DB}

    files = {"binary": binary, "model": model, "manifest": manifest,
             "session": session_path, "selection": base / "selection.json"}
    for level in LEVELS_DB:
        files[f"programme_{level}"] = base / f"programme_{level}.wav"
    for index, entry in enumerate(session["captures"]):
        files[f"capture_{index}"] = base / entry["file"]
    run_provenance = experiment_provenance(REPOSITORY, files,
                                           experiment="p1b0_replay",
                                           alignment="slate", rate=RATE)

    items = {item["name"]: item for item in load_corpus(
        manifest, music, False, frozenset({"gtzan"}))}
    aligned_dir = base / "aligned"
    aligned_dir.mkdir(exist_ok=True)

    def takes_of(capture: np.ndarray, level: str, tag: str):
        """Every take of one capture, aligned, cut and scored -- or refused."""
        layout = layouts[level]
        found = check(capture, layout)
        for row, take in zip(found["takes"], layout["takes"]):
            if not row["accepted"]:
                yield row, None
                continue
            start = int(round(row["music_offset_sec"] * RATE))
            stop = start + int(round(take["layout"]["music_seconds"] * RATE))
            if stop > len(capture):
                yield dict(row, accepted=False,
                           reason="capture ends before the music does"), None
                continue
            path = aligned_dir / f"{row['track']}__{tag}.wav"
            soundfile.write(str(path), capture[start:stop], RATE,
                            subtype="FLOAT")
            yield row, _score_one(dict(items[row["track"]], audio=path),
                                  "model", binary, model)

    # The clean arm is the programme itself, read and cut exactly as a capture
    # is: same file, same 24-bit samples, same level. The live tracker is
    # deterministic but not smooth -- on a hard excerpt, re-quantising the same
    # music at 1e-7 moves F by 0.4 -- and it is not gain-invariant either. A
    # clean arm that differs from the capture path by anything but the room
    # puts both effects into the room's column.
    clean: dict[str, dict[str, dict]] = {}
    for level in LEVELS_DB:
        loopback, _ = read_capture(base / f"programme_{level}.wav")
        clean[level] = {}
        for row, scored in takes_of(loopback, level, f"{level}__loopback"):
            if scored is None:
                raise RuntimeError(f"{level}: the programme does not split "
                                   f"back on itself at {row['track']}")
            clean[level][row["track"]] = scored
    # Not an arm: the float excerpt the programme was built from, scored only
    # to report how far the digital path alone moves the tracker.
    reference = {excerpt["track"]: _score_one(
        dict(items[excerpt["track"]], audio=base / excerpt["clean"]),
        "model", binary, model) for excerpt in selection["excerpts"]}

    records = []
    for entry in session["captures"]:
        capture, device_rate = read_capture(base / entry["file"])
        cell = {key: entry[key] for key in ("level", "distance", "device")}
        tag = "__".join(map(str, cell.values()))
        for row, room in takes_of(capture, entry["level"], tag):
            record = dict(cell, track=row["track"], file=entry["file"],
                          device_rate=device_rate, aligned=row["accepted"])
            if room is None:
                record["reason"] = row.get("reason")
                records.append(record)
                continue
            base_row = clean[entry["level"]][row["track"]]
            record.update({
                "drift_sec": row.get("drift_sec"),
                "head_margin_db": row.get("head_margin_db"),
                "f_clean": base_row.get("f_measure"),
                "f_room": room.get("f_measure"),
                "usable_clean": base_row.get("usable"),
                "usable_room": room.get("usable"),
            })
            records.append(record)

    cells = collections.defaultdict(list)
    for record in records:
        cells[(record["level"], record["distance"], record["device"])].append(
            record)
    summary = {}
    for key, rows in sorted(cells.items()):
        scored = [r for r in rows if r["aligned"] and r.get("f_room") is not None
                  and r.get("f_clean") is not None]
        deltas = [r["f_room"] - r["f_clean"] for r in scored]
        summary["/".join(key)] = {
            "takes": len(rows), "aligned": sum(r["aligned"] for r in rows),
            "mean_f_clean": float(np.mean([r["f_clean"] for r in scored]))
            if scored else None,
            "mean_f_room": float(np.mean([r["f_room"] for r in scored]))
            if scored else None,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1
            else None,
            "delta_ci95": paired_interval(deltas),
            "a2_captures_needed": captures_needed(deltas),
            "usable_room": float(np.mean([bool(r["usable_room"])
                                          for r in scored])) if scored else None,
        }
    def paired(a: dict, b: dict) -> dict:
        deltas = [a[t]["f_measure"] - b[t]["f_measure"] for t in a
                  if a[t].get("f_measure") is not None
                  and b.get(t, {}).get("f_measure") is not None]
        return {"n": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "changed": sum(abs(d) > 1e-9 for d in deltas),
                "delta_ci95": paired_interval(deltas)}

    # What the numbers above are to be read against: the digital path alone,
    # and the level alone, with no room anywhere in either.
    controls = {
        "digital_path": paired(clean["normal"], reference),
        "tracker_level": paired(clean["quiet"], clean["normal"]),
    }
    aligned = sum(r["aligned"] for r in records)
    sized = [(cell, row["a2_captures_needed"]) for cell, row in summary.items()
             if row["a2_captures_needed"] is not None]
    worst = max(sized, key=lambda pair: pair[1]["estimate"], default=None)
    result = {
        "schema": RESULT_SCHEMA, "research_only": True,
        "provenance": run_provenance, "session": session,
        "a1": {"aligned": aligned, "takes": len(records),
               "fraction": aligned / len(records) if records else None,
               "gate": A1_ALIGNED_FRACTION,
               "met": bool(records) and
               aligned >= A1_ALIGNED_FRACTION * len(records)},
        "controls": controls,
        "clean": {level: {track: {"f": row.get("f_measure"),
                                  "usable": row.get("usable")}
                          for track, row in rows.items()}
                  for level, rows in clean.items()},
        "a2": {"worst_cell": worst[0], **worst[1]} if worst else None,
        "cells": summary, "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    made = commands.add_parser("build")
    made.add_argument("--manifest", type=pathlib.Path, required=True)
    made.add_argument("--music", type=pathlib.Path, required=True)
    made.add_argument("--output", type=pathlib.Path, required=True)

    checked = commands.add_parser("check")
    checked.add_argument("--layout", type=pathlib.Path, required=True)
    checked.add_argument("--capture", type=pathlib.Path, required=True)

    scored = commands.add_parser("score")
    scored.add_argument("--session", type=pathlib.Path, required=True)
    scored.add_argument("--manifest", type=pathlib.Path, required=True)
    scored.add_argument("--music", type=pathlib.Path, required=True)
    scored.add_argument("--binary", type=pathlib.Path, required=True)
    scored.add_argument("--model", type=pathlib.Path, required=True)
    scored.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "build":
        summary = build(args.manifest, args.music, args.output)
        for level, row in summary["programmes"].items():
            print(f"{level:>7}: {row['file']}  {row['minutes']:.1f} min")
        print(f"{len(summary['excerpts'])} excerpts, headroom "
              f"{20 * np.log10(summary['headroom']):.2f} dB")
        return 0

    if args.command == "check":
        layout = json.loads(args.layout.read_text(encoding="utf-8"))
        capture, rate = read_capture(args.capture)
        found = check(capture, layout)
        print(f"{args.capture.name}: {rate:.0f} Hz, "
              f"{len(capture) / RATE / 60:.1f} min")
        for row in found["takes"]:
            if row["accepted"]:
                drift, margin = row.get("drift_sec"), row.get("head_margin_db")
                print(f"  ok      {row['track']:<18} drift "
                      f"{'n/a' if drift is None else f'{drift * 1000:+.1f} ms'}"
                      f"  margin "
                      f"{'n/a' if margin is None else f'{margin:.1f} dB'}")
            else:
                print(f"  REFUSED {row['track']:<18} {row.get('reason')}")
        print(f"aligned {found['aligned']}/{found['total']} -- "
              + ("pass" if found["a1_fraction_met"] else
                 f"below the {A1_ALIGNED_FRACTION:.0%} gate: redo this pass"))
        return 0 if found["a1_fraction_met"] else 1

    result = score(args.session, args.manifest, args.music, args.binary,
                   args.model, args.output)
    print(json.dumps(result["a1"]))
    print(json.dumps(result["a2"]))
    for name, row in result["controls"].items():
        ci = row["delta_ci95"]
        print(f"  control {name:<14} dF {row['mean_delta']:+.3f}"
              f"{'' if ci is None else f' [{ci[0]:+.3f}, {ci[1]:+.3f}]'}"
              f"  changed {row['changed']}/{row['n']}")
    for cell, row in result["cells"].items():
        delta, ci = row["mean_delta"], row["delta_ci95"]
        print(f"  {cell:<24} aligned {row['aligned']}/{row['takes']}  dF "
              f"{'n/a' if delta is None else f'{delta:+.3f}'}"
              f"{'' if ci is None else f' [{ci[0]:+.3f}, {ci[1]:+.3f}]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
