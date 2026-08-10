#!/usr/bin/env python3
"""A slate that makes a capture's offset measurable instead of inferred.

Two of eight room captures have been lost to alignment. `room_recording.py`
finds the offset by cross-correlating onset envelopes, and its own comments say
why that can fail: music repeats, so a correlation peak can be a later chorus
rather than a late start. On `0707_halfwaygone` it produced two candidates
0.476 s and 0.910 s apart -- about one beat at 125 BPM -- scoring room F 0.340
and 0.204. Four windows of seven agreed on one, the coherent sum peaked on the
other, and nothing in the recording could settle it.

No better correlation fixes that. The ambiguity is a property of the signal:
the autocorrelation of metrical music has peaks at metrical periods, and they
are real. The fix is to put something in the recording that has no metre.

A short Farina sweep has exactly one autocorrelation peak, which is what
`make_sweep.py` already exploits to measure a room. Played immediately before
the music, in the same recording, it turns the offset from something inferred
into something read.

    slate -> silence -> music        one file, one recorder start

The deconvolution peak is the moment the slate arrived. The music began a known
interval later, because the file that plays them was built here.

**A slate is refused rather than guessed.** `find_slate` reports how far its
peak stands above the rest of the deconvolution, and returns `accepted=False`
below a threshold instead of returning its best guess. A capture whose slate
cannot be trusted is a capture to redo, and knowing that during the session
costs minutes where discovering it afterwards costs the recording.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.make_sweep import deconvolve, sweep  # noqa: E402

RATE = 48000

# Short enough not to be a nuisance between takes, long enough that the sweep
# still resolves: the self-test below is what decides whether a given pair of
# numbers actually inverts, and it refuses to write a file that does not.
SLATE_SECONDS = 0.5
SLATE_F_LOW = 200.0
SLATE_F_HIGH = 8000.0

# Peak level, matching `make_sweep.AMPLITUDE`. `sweep()` returns unit amplitude
# and `make_sweep` scales it only when writing a file, so taking it raw would
# emit a slate at 0 dBFS -- into a phone, an invitation to clip on the way in
# or to make its automatic gain duck the music that follows.
#
# It is a trade rather than a free choice: a quieter slate is a slate that
# stands less far above a room's noise floor, which is exactly what
# `MIN_PEAK_TO_SIDELOBE_DB` measures. Both numbers are provisional until a real
# capture has been through them.
SLATE_AMPLITUDE = 0.5

# The silence between the slate and the first sample of music. Long enough that
# the room's own tail from the slate -- measured at RT60 0.33 to 0.37 s -- has
# died before the music starts, so the two do not overlap in the capture.
LEAD_SECONDS = 1.5

# Below this the peak is not distinguishable enough from the rest to be called
# a slate. `make_sweep.self_test` uses 40 dB for a ten-second sweep in a quiet
# room; a half-second slate through a phone in a live one is a harder case, and
# this is the bar for *accepting an alignment*, not for the arithmetic.
MIN_PEAK_TO_SIDELOBE_DB = 12.0

# How far either side of the peak is excluded when measuring what it stands
# above. The room's response is part of the peak, not a rival to it.
GUARD_SECONDS = 0.05


def slate(rate: int = RATE, seconds: float = SLATE_SECONDS,
          f_low: float = SLATE_F_LOW, f_high: float = SLATE_F_HIGH
          ) -> np.ndarray:
    """The marker itself: a short exponential sweep, at a stated level."""
    return SLATE_AMPLITUDE * sweep(rate=rate, seconds=seconds, f_low=f_low,
                                   f_high=f_high)


def build_take(music: np.ndarray, rate: int = RATE,
               lead_seconds: float = LEAD_SECONDS,
               tail_slate: bool = True) -> tuple[np.ndarray, dict]:
    """One playable file: slate, silence, music, silence, slate.

    The trailing slate is not decoration. `room_recording.py` fits the offset
    twice and calls a capture drifting when the halves disagree; two slates
    measure that drift directly, over the whole take rather than over a third
    of it, and cost a second.
    """
    marker = slate(rate=rate)
    gap = np.zeros(int(round(lead_seconds * rate)), dtype=np.float64)
    parts = [marker, gap, music]
    layout = {
        "rate": rate,
        "slate_seconds": len(marker) / rate,
        "lead_seconds": lead_seconds,
        "music_start_sec": (len(marker) + len(gap)) / rate,
        "music_seconds": len(music) / rate,
    }
    if tail_slate:
        parts += [gap, marker]
        layout["tail_slate_start_sec"] = (
            len(marker) + len(gap) + len(music) + len(gap)) / rate
    return np.concatenate(parts), layout


def find_slate(capture: np.ndarray, rate: int = RATE,
               seconds: float = SLATE_SECONDS, f_low: float = SLATE_F_LOW,
               f_high: float = SLATE_F_HIGH,
               search: tuple[float, float] | None = None) -> dict:
    """When the slate arrived, and whether the answer can be trusted.

    `search` bounds the window in seconds. The head slate is looked for near
    the start and the tail slate near the end, so that one cannot be mistaken
    for the other -- they are the same signal by construction.
    """
    response = deconvolve(capture, rate=rate, seconds=seconds, f_low=f_low,
                          f_high=f_high)
    magnitude = np.abs(response)

    lo, hi = 0, len(magnitude)
    if search is not None:
        lo = max(0, int(round(search[0] * rate)))
        hi = min(len(magnitude), int(round(search[1] * rate)))
        if hi <= lo:
            return {"accepted": False, "reason": "empty search window"}

    window = magnitude[lo:hi]
    if not len(window) or not np.any(window > 0):
        return {"accepted": False, "reason": "no signal in search window"}

    peak = int(np.argmax(window))
    height = float(window[peak])

    # Farina's deconvolution puts the spike at the *end* of the sweep, not its
    # start: `make_sweep.self_test` names the convention as
    # `expected_index = round(seconds * rate) - 1`, and it is 23999 for a
    # half-second slate at 48 kHz.
    #
    # Reading the peak as the arrival time therefore reports every capture late
    # by the whole length of the slate -- 0.5 s here, which is 1.04 beats at
    # 115 BPM. That is the same magnitude as the ambiguity this file exists to
    # remove, it is constant, and nothing downstream would look wrong: the
    # alignment would simply score a different bar. `test_slate.py` pins the
    # convention for that reason.
    lead = (int(round(seconds * rate)) - 1) / rate

    guard = int(round(GUARD_SECONDS * rate))
    rest = np.concatenate([window[: max(0, peak - guard)],
                           window[peak + guard:]])
    sidelobe = float(np.max(rest)) if len(rest) else 0.0
    ratio_db = float(20.0 * np.log10(height / max(sidelobe, 1e-30)))

    return {
        "offset_sec": (lo + peak) / rate - lead,
        "peak_index_sec": (lo + peak) / rate,
        "slate_lead_sec": lead,
        "peak": height,
        "sidelobe": sidelobe,
        "peak_to_sidelobe_db": ratio_db,
        "accepted": bool(ratio_db >= MIN_PEAK_TO_SIDELOBE_DB),
        "threshold_db": MIN_PEAK_TO_SIDELOBE_DB,
    }


def align_by_slate(capture: np.ndarray, layout: dict,
                   rate: int = RATE) -> dict:
    """The offset of the music inside a capture, and the drift across it.

    Returns `accepted=False` when either slate cannot be trusted. Refusing is
    the point: an alignment that is wrong by a beat scores a different piece of
    music, and it does not look wrong afterwards.
    """
    slate_len = layout["slate_seconds"]
    music_start = layout["music_start_sec"]

    head_span = music_start + 2.0
    head = find_slate(capture, rate=rate, seconds=slate_len,
                      search=(0.0, min(head_span, len(capture) / rate)))
    if not head.get("accepted"):
        return {"accepted": False, "reason": "head slate not found",
                "head": head}

    out = {"head": head, "capture_seconds": len(capture) / rate,
           "music_offset_sec": head["offset_sec"] + music_start,
           "accepted": True}

    expected_tail = layout.get("tail_slate_start_sec")
    if expected_tail is None:
        return out

    centre = head["offset_sec"] + expected_tail
    tail = find_slate(capture, rate=rate, seconds=slate_len,
                      search=(max(0.0, centre - 2.0),
                              min(centre + 2.0, len(capture) / rate)))
    out["tail"] = tail
    if not tail.get("accepted"):
        out["accepted"] = False
        out["reason"] = "tail slate not found"
        return out

    measured = tail["offset_sec"] - head["offset_sec"]
    out["drift_sec"] = float(measured - expected_tail)
    return out


def main(argv: list[str] | None = None) -> int:
    import soundfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music", type=pathlib.Path, required=True,
                        help="the track to wrap in slates")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--no-tail-slate", action="store_true")
    args = parser.parse_args(argv)

    audio, rate = soundfile.read(str(args.music), dtype="float64",
                                 always_2d=True)
    mono = audio.mean(axis=1)
    if int(rate) != RATE:
        print(f"expected {RATE} Hz, got {rate}; resample first",
              file=sys.stderr)
        return 1

    take, layout = build_take(mono, rate=RATE,
                              tail_slate=not args.no_tail_slate)

    # The same control `make_sweep` applies before writing anything: if the
    # marker does not deconvolve to a spike, nothing downstream can align to it
    # and a file would only invite a session that cannot be used.
    check = find_slate(take, rate=RATE)
    if not check["accepted"]:
        print(f"slate does not resolve on the take itself: "
              f"{check.get('peak_to_sidelobe_db', float('nan')):.1f} dB",
              file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(args.output), take, RATE)
    layout["self_test"] = check
    args.output.with_suffix(".layout.json").write_text(
        json.dumps(layout, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(take) / RATE:.1f} s); "
          f"music starts at {layout['music_start_sec']:.3f} s; "
          f"slate resolves {check['peak_to_sidelobe_db']:.1f} dB above the rest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
