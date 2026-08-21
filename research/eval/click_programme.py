#!/usr/bin/env python3
"""One file to play and one recording to make, per click level.

The click micro-check is five takes at each of several levels. Captured take by
take that is five starts and five stops per level -- twenty file names to get
right by hand, in a session whose whole justification is that it costs an hour.
A programme is the five takes concatenated into one playable file, so the
operator presses record, presses play, and stops: three actions per level.

Splitting it back apart needs no new alignment idea. Every take already carries
its own head and tail slate, and `find_slate` already takes a search window, so
each take is located inside the programme and then handed to `align_by_slate`
unchanged -- the checked primitive keeps doing the checking.

Windows are predicted **sequentially**: the measured position of take k sets the
window for take k+1. Drift across a fourteen-minute programme is small (0.84 s
at 1000 ppm) but it accumulates, and a fixed window sized for the last take
would be loose enough for the first to catch the wrong slate.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from eval.slate import RATE, align_by_slate, find_slate

SCHEMA = "tiktak.click_programme/v1"
# Long enough that a window wide enough for a slow record button still
# contains exactly one slate. Every slate in a programme is the same signal,
# so two of them inside one search window are the same height: `find_slate`
# measures a 0 dB margin and refuses, correctly, because it genuinely cannot
# tell which one it found. A three-second gap put the tail of one take and
# the head of the next inside any usable window.
GAP_SECONDS = 30.0
ANCHOR_SEARCH_SEC = 25.0
# The slice handed to `align_by_slate` starts this far before the take's head
# slate. It has to be under `music_start_sec + 2.0` -- the head search window
# that function uses -- or the slate falls outside its own search.
LEAD_MARGIN_SEC = 2.0
SEARCH_HALF_WIDTH_SEC = 2.0


def build_programme(takes: list[tuple[str, np.ndarray, dict]],
                    gap_seconds: float = GAP_SECONDS,
                    rate: int = RATE) -> tuple[np.ndarray, dict]:
    """Concatenate takes with silence between them, and say where each starts."""
    gap = np.zeros(int(round(gap_seconds * rate)))
    pieces, entries, cursor = [], [], 0.0
    for name, audio, layout in takes:
        entries.append({"track": name, "offset_sec": cursor, "layout": layout,
                        "seconds": len(audio) / rate})
        pieces.append(audio)
        pieces.append(gap)
        cursor += (len(audio) + len(gap)) / rate
    programme = np.concatenate(pieces) if pieces else np.zeros(0)
    return programme, {"schema": SCHEMA, "rate": rate,
                       "gap_seconds": gap_seconds,
                       "seconds": len(programme) / rate, "takes": entries}


def locate_takes(capture: np.ndarray, programme: dict,
                 rate: int = RATE) -> list[dict]:
    """Find each take inside one capture and align it on its own two slates."""
    entries = programme["takes"]
    if not entries:
        return []

    # The programme's own start: the first take's head slate, searched over a
    # window wide enough for a slow hand on the record button.
    first = entries[0]
    # How long the operator may take between record and play. Bounded by the
    # nearest *other* slate, which is the first take's own tail: every slate
    # is the same signal, so a window holding two of them makes them equal
    # height and `find_slate` refuses at 0 dB. Derived rather than assumed,
    # because a constant that happens to fit the real takes silently breaks
    # on any shorter one.
    slop = min(ANCHOR_SEARCH_SEC,
               0.8 * float(first["layout"]["tail_slate_start_sec"]))
    anchor = find_slate(capture, rate=rate,
                        seconds=first["layout"]["slate_seconds"],
                        search=(0.0, min(slop, len(capture) / rate)))
    if not anchor.get("accepted"):
        return [{"track": entry["track"], "accepted": False,
                 "reason": ("programme head slate not found within the "
                            f"first {slop:.1f} s")}
                for entry in entries]
    programme_offset = anchor["offset_sec"] - first["offset_sec"]

    out = []
    for entry in entries:
        layout = entry["layout"]
        expected = programme_offset + entry["offset_sec"]
        # Only the lead margin: `align_by_slate` looks for the head inside
        # (0, music_start + 2) of whatever slice it is given, so starting
        # further back would push the slate out of its own search window.
        start = expected - LEAD_MARGIN_SEC
        stop = expected + entry["seconds"] + SEARCH_HALF_WIDTH_SEC
        lo = max(0, int(round(start * rate)))
        hi = min(len(capture), int(round(stop * rate)))
        if hi - lo < int(round(entry["seconds"] * rate)):
            out.append({"track": entry["track"], "accepted": False,
                        "reason": "capture ends before this take does"})
            continue

        found = align_by_slate(capture[lo:hi], layout, rate=rate)
        row = {"track": entry["track"], "accepted": bool(found.get("accepted")),
               "slice_start_sec": lo / rate}
        if not found.get("accepted"):
            row["reason"] = found.get("reason", "alignment refused")
            out.append(row)
            continue
        # Back into capture time, so a caller can cut the music out of the
        # original recording rather than out of a slice it has to remember.
        row.update({
            "music_offset_sec": lo / rate + found["music_offset_sec"],
            "drift_sec": found.get("drift_sec"),
            "head_margin_db": found["head"].get("peak_to_sidelobe_db"),
            "tail_margin_db": (found.get("tail") or {}).get("peak_to_sidelobe_db"),
        })
        # Each accepted take re-anchors the prediction, so drift does not
        # accumulate into the window of the take after it.
        programme_offset = row["music_offset_sec"] - layout["music_start_sec"] \
            - entry["offset_sec"]
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    import soundfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takes", type=pathlib.Path, required=True)
    parser.add_argument("--track", action="append", required=True,
                        help="give once per take, in playing order")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    loaded = []
    for name in args.track:
        audio, rate = soundfile.read(str(args.takes / f"{name}.wav"),
                                     dtype="float64", always_2d=True)
        if int(rate) != RATE:
            raise SystemExit(f"{name}: {rate} Hz, expected {RATE}")
        layout = json.loads(
            (args.takes / f"{name}.layout.json").read_text(encoding="utf-8"))
        loaded.append((name, audio.mean(axis=1), layout))

    programme, layout = build_programme(loaded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(args.output), programme, RATE)
    args.output.with_suffix(".layout.json").write_text(
        json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({layout['seconds'] / 60:.1f} min, "
          f"{len(loaded)} takes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
