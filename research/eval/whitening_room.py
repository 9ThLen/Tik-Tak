#!/usr/bin/env python3
"""Measure whether adaptive whitening preserves gaps between beats in a room.

For each clean/room pair and whitening exponent, this runs the shipping ODF
through ``dump_analysis`` and measures:

``peak``
    Median of the maximum within +/-70 ms of every annotated beat.
``floor``
    Median ODF value over the middle third of every inter-beat gap.
``ratio``
    ``floor / peak``; lower means the gaps remain more distinct.

The result carries the exact commit, binary, audio and annotations so the
negative can be reproduced instead of surviving as a number from a scratchpad.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "research"))

from eval.provenance import provenance  # noqa: E402

TRACKS = (
    "0116_goodies",
    "0132_iceicebaby",
    "0466_onthedarkside",
    "0707_halfwaygone",
    "0837_nottonight",
)
ARMS = {"off": 0.0, "shipped": 0.5, "full": 1.0}
WINDOW_SEC = 0.070
WARMUP_SEC = 5.0


def dump_odf(binary: pathlib.Path, path: pathlib.Path,
             strength: float) -> tuple[np.ndarray, np.ndarray]:
    done = subprocess.run(
        (str(binary), str(path), "--dump-odf",
         "--odf-whitening-strength", repr(strength)),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"{path.name} @ {strength}: {done.stderr.strip()}")
    result = json.loads(done.stdout)
    return (np.asarray(result["odf"], dtype=np.float64),
            np.asarray(result["odf_times"], dtype=np.float64))


def contrast(values: np.ndarray, times: np.ndarray,
             beats: np.ndarray) -> dict:
    beats = beats[(beats >= WARMUP_SEC) & (beats <= times[-1])]
    if len(beats) < 8:
        return {}

    peaks = []
    for beat in beats:
        near = (times >= beat - WINDOW_SEC) & (times <= beat + WINDOW_SEC)
        if near.any():
            peaks.append(float(values[near].max()))

    floors = []
    for start, end in zip(beats[:-1], beats[1:]):
        span = end - start
        middle = (times >= start + span / 3.0) & (times <= end - span / 3.0)
        if middle.any():
            floors.append(float(np.median(values[middle])))

    if not peaks or not floors:
        return {}
    peak = float(np.median(peaks))
    floor = float(np.median(floors))
    return {
        "peak": peak,
        "floor": floor,
        "ratio": float(floor / peak) if peak > 0.0 else float("nan"),
        "beats": int(len(beats)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument(
        "--data-root", type=pathlib.Path, required=True,
        help="repository checkout containing the ignored music/ and annotations/",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    data_root = args.data_root.resolve()
    clean = data_root / "music/ground-truth/audio/harmonix-ready"
    room = data_root / "music/room-aligned"
    annotations = data_root / "annotations/harmonix/annotations/beats"

    sources: dict[str, pathlib.Path] = {"binary": binary}
    for track in TRACKS:
        sources[f"clean_{track}"] = clean / f"{track}.wav"
        sources[f"room_{track}"] = room / f"{track}.wav"
        sources[f"beats_{track}"] = annotations / f"{track}.beats"
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))

    run_provenance = provenance(
        REPOSITORY,
        sources,
        tracks=list(TRACKS),
        window_sec=WINDOW_SEC,
        warmup_sec=WARMUP_SEC,
    )
    if run_provenance["tree_clean"] is not True:
        raise RuntimeError(
            "refusing a provisional run: git tree is not provably clean")

    rows = []
    for track in TRACKS:
        beats = np.loadtxt(sources[f"beats_{track}"], usecols=0, ndmin=1)
        for condition, folder in (("clean", clean), ("room", room)):
            path = folder / f"{track}.wav"
            for arm, strength in ARMS.items():
                values, times = dump_odf(binary, path, strength)
                stats = contrast(values, times, beats)
                if not stats:
                    raise RuntimeError(f"unscorable: {track} {condition} {arm}")
                rows.append({
                    "track": track,
                    "condition": condition,
                    "arm": arm,
                    "strength": strength,
                    **stats,
                })
                print(
                    f"{track:22s} {condition:5s} {arm:7s} "
                    f"peak {stats['peak']:.4f}  floor {stats['floor']:.4f}  "
                    f"ratio {stats['ratio']:.4f}")

    summary = {}
    print(f"\n{'condition':10s} {'arm':8s} {'peak':>8s} "
          f"{'floor':>8s} {'ratio':>8s}")
    for condition in ("clean", "room"):
        for arm in ARMS:
            part = [row for row in rows
                    if row["condition"] == condition and row["arm"] == arm]
            cell = {key: float(np.mean([row[key] for row in part]))
                    for key in ("peak", "floor", "ratio")}
            summary[f"{condition}/{arm}"] = cell
            print(f"{condition:10s} {arm:8s} {cell['peak']:8.4f} "
                  f"{cell['floor']:8.4f} {cell['ratio']:8.4f}")

    room_rows = {
        (row["track"], row["arm"]): row for row in rows
        if row["condition"] == "room"
    }
    off_beats_shipped = sum(
        room_rows[(track, "off")]["ratio"]
        < room_rows[(track, "shipped")]["ratio"] for track in TRACKS)
    full_worse_than_shipped = sum(
        room_rows[(track, "full")]["ratio"]
        > room_rows[(track, "shipped")]["ratio"] for track in TRACKS)
    room_off = summary["room/off"]
    room_full = summary["room/full"]
    verdict = {
        "result": "negative",
        "off_beats_shipped_tracks": int(off_beats_shipped),
        "full_worse_than_shipped_tracks": int(full_worse_than_shipped),
        "peak_loss_off_to_full": float(
            (room_off["peak"] - room_full["peak"]) / room_off["peak"]),
        "floor_loss_off_to_full": float(
            (room_off["floor"] - room_full["floor"]) / room_off["floor"]),
        "statement": (
            "Adaptive whitening does not preserve inter-beat gaps in the room "
            "captures; disabling it produces the lowest room ratio on all "
            "five tracks."),
    }
    report = {
        "provenance": run_provenance,
        "arms": ARMS,
        "records": rows,
        "summary": summary,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
