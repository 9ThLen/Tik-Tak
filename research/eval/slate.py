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
#
# 0.05 was a guess, and it was too narrow. Measured on session 3, five of the
# six limiting sidelobes sat at +53 to +61 ms from their peak -- the slate's own
# response, just outside the old window, counted as a rival to itself. Widening
# past that ridge is worth 8 to 14 dB on those five.
#
# It cannot be widened freely: the ambiguity this file exists to resolve is one
# beat, so a rival a beat away has to stay outside the guard. 0.125 is at least
# twice the observed ridge and under half a beat even at 200 BPM, which is
# faster than anything these takes contain. The ridge is a property of the room
# that was measured and should be rechecked in a new one; the beat bound is not.
GUARD_SECONDS = 0.125

# How far either side of the *expected* tail position the trailing slate is
# looked for. The head cannot be bounded this tightly -- the capture's start
# offset is exactly what is unknown -- but once the head is found the tail's
# position is known to within clock drift, which measured 1.5 to 2.6 ms across a
# whole take.
#
# This was 2.0, and that was the real cause of session 3's narrowest margin.
# `build_take` leaves `LEAD_SECONDS` of silence between the music and the tail
# slate, so a 2.0 s window reaches 0.5 s back into the music, and on
# `0116_goodies` the loudest thing in the window was the end of the song, at
# -1999.3 ms from the peak. No guard width reaches that far; only the window
# does. Bounded below by drift and above by `LEAD_SECONDS`, 1.0 s sits inside a
# wide interval rather than at a fitted optimum.
#
# It is a ceiling rather than the width, because the width is not a constant at
# all: `lead_seconds` is written into every layout, and the bound that matters
# is a fraction of *that*. A take built with a shorter pause -- which is exactly
# what lengthening the tail gap for the phone's AGC would produce on the other
# side -- would silently put a hard-coded 1.0 back inside the music, and the
# symptom would again look like a weak slate.
TAIL_SEARCH_MAX_HALF_WIDTH_SEC = 1.0
TAIL_SEARCH_LEAD_FRACTION = 2.0 / 3.0

# Head and tail must agree about where the take starts, after the interval
# between them is removed. This is the check the peak-to-sidelobe margin was
# only ever a proxy for: the margin asks whether a peak looks convincing, this
# asks whether the two independent readings of the same take land in the same
# place. A head slate read one beat early does not produce a slightly worse
# margin -- it produces a drift of about half a second.
#
# No free parameter, and the bar does not need calibrating: measured drift is
# 1.5 to 2.6 ms, one beat is around 480 ms, and 50 ms sits twenty times above
# the first and ten times below the second. Nothing in between needs a decision.
MAX_DRIFT_SEC = 0.05


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
        # A window can be too small to support the measurement at all. Once the
        # peak and its guard are removed, what is left has to be at least a
        # slate long, or the sidelobe is estimated from less material than the
        # signal it is compared against. Swept on the session 3 captures, a
        # +/-0.25 s tail window returned 0.180 dB -- peak and rival were the same
        # ridge -- which reads as a terrible capture rather than as a
        # misconfigured search. Refusing says which one it is.
        #
        # Only for a bounded search: an unbounded call measures whatever buffer
        # it was handed, and that is the caller's business.
        minimum = seconds + 2.0 * GUARD_SECONDS
        if (hi - lo) / rate < minimum:
            return {"accepted": False,
                    "reason": (f"search window {(hi - lo) / rate:.3f} s is "
                               f"below the {minimum:.3f} s this measurement "
                               f"needs")}

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
        # Reported because they decide the number above. The first session 3
        # artifact recorded `threshold_db` -- the one of the three that turned
        # out not to matter -- and neither of these, so reading it afterwards
        # could not reveal which guard produced a 12.4 dB margin. A constant
        # that is not in the artifact is no better than a variable one.
        "guard_sec": GUARD_SECONDS,
        "search_seconds": (hi - lo) / rate,
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

    # A fraction of the take's own pause, not a constant: the window must not
    # reach back into the music, and how far away the music is, is a property of
    # the layout that built the take.
    lead = float(layout.get("lead_seconds", TAIL_SEARCH_MAX_HALF_WIDTH_SEC))
    half = min(TAIL_SEARCH_MAX_HALF_WIDTH_SEC, lead * TAIL_SEARCH_LEAD_FRACTION)

    out = {"head": head, "capture_seconds": len(capture) / rate,
           "music_offset_sec": head["offset_sec"] + music_start,
           "tail_search_half_width_sec": half,
           "max_drift_sec": MAX_DRIFT_SEC,
           "accepted": True}

    expected_tail = layout.get("tail_slate_start_sec")
    if expected_tail is None:
        return out

    centre = head["offset_sec"] + expected_tail
    tail = find_slate(capture, rate=rate, seconds=slate_len,
                      search=(max(0.0, centre - half),
                              min(centre + half, len(capture) / rate)))
    out["tail"] = tail
    if not tail.get("accepted"):
        out["accepted"] = False
        out["reason"] = "tail slate not found"
        return out

    measured = tail["offset_sec"] - head["offset_sec"]
    out["drift_sec"] = float(measured - expected_tail)

    # The primary check, and the only one here without a free parameter. Two
    # independent readings of one take have to land in the same place; a head
    # read a beat early shows up as half a second of "drift", not as a slightly
    # worse margin. The margin stays as the secondary signal because it is what
    # notices a capture degrading before it fails.
    if abs(out["drift_sec"]) > MAX_DRIFT_SEC:
        out["accepted"] = False
        out["reason"] = (
            f"head and tail disagree by {out['drift_sec'] * 1000:.1f} ms, "
            f"above the {MAX_DRIFT_SEC * 1000:.0f} ms two-slate bar; one of "
            f"them is not the slate it was taken for")
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
    #
    # Through `align_by_slate`, not a bare `find_slate`, because a take holds
    # *two* identical slates. Searching the whole file finds the head and then
    # measures it against the tail, which is exactly as tall: peak equals
    # sidelobe, 0.0 dB, and the check fails on a take that is perfectly good.
    # The head is looked for near the start and the tail near the end for that
    # reason, and only a bounded search can tell them apart.
    checked = align_by_slate(take, layout, rate=RATE)
    check = checked.get("head", {})
    if not checked.get("accepted"):
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
