"""The closed loop in a real room: run the passes, then score them.

Registered in `PREREGISTERED_closed_loop_room.md`. `tiktak loop` plays a
programme through the speaker with the metronome's own click on top while the
microphone feeds the tracker, so each arm hears its own clicks through a real
speaker, room and microphone. Each pass writes what the microphone heard
(`<arm>.wav`) and every beat and the 50 Hz estimate (`<arm>.json`), all on the
tracker's clock. This module:

* ``levels`` — the click gains that put the click at a given ratio to this
  programme's music, so that "-8 dB" means -8 dB against the music;
* ``path``   — a made-up room for the dry run: an impulse response the
  simulated click returns through, longer than anything that models it;
* ``run``    — every registered arm through `tiktak loop`, one after another;
* ``score``  — aligns each pass to the programme by cross-correlation, cuts it
  per take, scores each take exactly as the bench does, and measures whether
  the metronome keeps itself going in the silence after the music stops.

    cd research
    .venv/Scripts/python -m eval.closed_loop levels --programme <programme.wav> --layout <layout.json> ...
    .venv/Scripts/python -m eval.closed_loop run --tiktak <tiktak.exe> --model <beatnet.ttw> ...
    .venv/Scripts/python -m eval.closed_loop score --session <dir> --layout <layout.json> ...
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.analysis import Estimate  # noqa: E402
from eval.live_corpus_benchmark import (_score_one, load_corpus,  # noqa: E402
                                        load_reference_beats)

# ClickConfig::beat at 0 dB: a sine falling 60 dB over its length.
CLICK_GAIN, CLICK_LENGTH = 0.75, 0.060
CLICK_TAU = CLICK_LENGTH / math.log(1000.0)
CLICK_ENERGY = CLICK_GAIN ** 2 * CLICK_TAU / 4.0 * (1.0 - math.exp(-2.0 * CLICK_LENGTH / CLICK_TAU))

# The registered arms. `ratio` is the click's beat-averaged power against the
# programme's music, median over takes; None is the silent click.
ARMS = {
    "l0_silent": {"ratio": None, "gated": False},
    "l1_loud_gated": {"ratio": -8.0, "gated": True},
    "l2_loud_ungated": {"ratio": -8.0, "gated": False},
    "l3_quiet_gated": {"ratio": -20.0, "gated": True},
    "l4_quiet_ungated": {"ratio": -20.0, "gated": False},
    # Amendment 1 to the registration: the click taken out of what the tracker
    # hears instead of gated, without and with the gate kept for an empty room.
    "l5_loud_subtracted": {"ratio": -8.0, "gated": False, "subtract": "only"},
    "l6_loud_guarded": {"ratio": -8.0, "gated": False, "subtract": "guarded"},
}
CLICK_WINDOW_SEC = (-0.002, 0.100)
LOCK_CONFIDENCE = 0.25
SILENCE_MARGIN_SEC = 0.5
TREND_SEC = 3.0
TAIL_SEC = 60.0
# Where the rules read the silence after the programme, in seconds after the
# music: the confidence as first registered, and whether the metronome is still
# clicking at the end, which Amendment 2 added.
CONFIDENCE_AT = (25.0, 30.0)
CLICKING_AT = (55.0, 60.0)
DRAWS = 10_000
BOOTSTRAP_SEED = 20260930


def read_mono(path: pathlib.Path) -> tuple[np.ndarray, float]:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), float(rate)


def take_windows(layout: dict) -> list[dict]:
    """Each take's music, and the silence after it, in programme seconds."""
    takes = layout["takes"]
    windows = []
    for index, take in enumerate(takes):
        start = take["offset_sec"] + take["layout"]["music_start_sec"]
        end = start + take["layout"]["music_seconds"]
        if index + 1 < len(takes):
            following = takes[index + 1]
            silence_end = following["offset_sec"] + following["layout"]["music_start_sec"]
        else:
            silence_end = layout["seconds"]
        windows.append({"track": take["track"], "start": start, "end": end,
                        "silence": (end + SILENCE_MARGIN_SEC, silence_end - SILENCE_MARGIN_SEC)})
    return windows


def click_to_music_db(programme: np.ndarray, rate: float, window: dict, period: float,
                      click_db: float) -> float:
    """The click's power over one beat period against the take's music power."""
    a, b = int(window["start"] * rate), int(window["end"] * rate)
    music = float(np.mean(np.square(programme[a:b], dtype=np.float64)))
    click = CLICK_ENERGY * 10.0 ** (click_db / 10.0) / period
    return 10.0 * math.log10(click / max(music, 1e-20))


def periods(windows: list[dict], items: dict[str, dict]) -> dict[str, float]:
    out = {}
    for window in windows:
        beats = load_reference_beats(pathlib.Path(items[window["track"]]["annotation"]))
        out[window["track"]] = float(np.median(np.diff(beats)))
    return out


def gain_for_ratio(programme: np.ndarray, rate: float, windows: list[dict],
                   period_of: dict[str, float], ratio_db: float) -> float:
    """The click gain whose median click-to-music ratio over takes is `ratio_db`."""
    at_zero = [click_to_music_db(programme, rate, w, period_of[w["track"]], 0.0) for w in windows]
    return float(ratio_db - np.median(at_zero))


def synthetic_path(rate: float, direct_to_reverb_db: float = 10.0, rt60_sec: float = 0.4,
                   length_sec: float = 0.35, seed: int = 5) -> np.ndarray:
    """A click's way back through a made-up room.

    The direct sound, five early reflections inside 20 ms, and from there a
    diffuse tail decaying 60 dB in `rt60_sec`, whose whole energy is
    `direct_to_reverb_db` under the direct sound's. It is not a room: a
    measured response got levels wrong here before
    (`tiktak-room-cannot-be-convolved`). It is a path longer than any model of
    it, which is the one thing the straight-back dry run cannot be.
    """
    count = int(length_sec * rate)
    path = np.zeros(count)
    path[0] = 1.0
    for ms, gain in ((1.3, 0.45), (3.7, -0.30), (7.1, 0.22), (12.9, 0.18), (19.0, -0.12)):
        path[int(round(ms / 1000.0 * rate))] += gain
    time = np.arange(count) / rate
    tail = np.random.default_rng(seed).standard_normal(count) * np.exp(-6.908 * time / rt60_sec)
    tail[:int(0.020 * rate)] = 0.0
    tail *= math.sqrt(10.0 ** (-direct_to_reverb_db / 10.0) / float(np.sum(tail ** 2)))
    return (path + tail).astype(np.float32)


def lag_between(programme: np.ndarray, capture: np.ndarray, rate: float, at_sec: float,
                span_sec: float = 60.0, search_sec: float = 1.0) -> tuple[float, float]:
    """Where the programme around `at_sec` reappears in the capture, and how clearly.

    Coarse on 4 kHz envelopes over ±`search_sec`, then refined on the raw
    signal to one sample. Returns the lag in seconds and the normalised peak.
    """
    from scipy.signal import correlate, resample_poly

    a = int(at_sec * rate)
    b = min(len(programme), a + int(span_sec * rate))
    search = int(search_sec * rate)
    reference = programme[a:b].astype(np.float64)
    heard = capture[a:min(len(capture), b + search)].astype(np.float64)
    if len(heard) < len(reference):
        return float("nan"), 0.0
    down = int(round(rate / 4000.0))
    r = resample_poly(np.abs(reference), 1, down)
    h = resample_poly(np.abs(heard), 1, down)
    r -= r.mean()
    h -= h.mean()
    score = correlate(h, r, mode="valid")
    coarse = int(np.argmax(score)) * down
    lo, hi = max(0, coarse - 2 * down), coarse + 2 * down
    best, best_value = coarse, -np.inf
    for shift in range(lo, hi + 1):
        piece = heard[shift:shift + len(reference)]
        if len(piece) < len(reference):
            break
        value = float(np.dot(piece, reference))
        if value > best_value:
            best, best_value = shift, value
    piece = heard[best:best + len(reference)]
    norm = float(np.linalg.norm(piece) * np.linalg.norm(reference)) or 1.0
    return best / rate, best_value / norm


def sustain(beats: np.ndarray, times: np.ndarray, confidences: np.ndarray,
            window: tuple[float, float], period: float) -> dict:
    """How far into the silence the metronome kept going on its own.

    Clicks alone cannot tell: with nothing of its own to hear, the tracker
    still coasts through ten seconds of silence on its last prediction. What
    separates coasting from feeding on its own click is the confidence. Coasting
    decays, and a metronome that hears itself holds or climbs. So the share of
    the silence spent locked, and the change in confidence from its first three
    seconds to its last three, are the measures; the click count is reported
    beside them.
    """
    start, end = window
    length = max(end - start, 1e-9)
    inside = beats[(beats >= start) & (beats < end)]
    expected = length / period
    span = (times >= start) & (times < end)
    locked = confidences[span]
    head = confidences[(times >= start) & (times < min(end, start + TREND_SEC))]
    tail = confidences[(times >= max(start, end - TREND_SEC)) & (times < end)]
    return {"beats": int(len(inside)),
            "ratio": float(min(len(inside) / expected, 2.0)) if expected > 0 else 0.0,
            "last_beat_after_sec": float(inside[-1] - start) if len(inside) else 0.0,
            "locked_share": float(np.mean(locked >= LOCK_CONFIDENCE)) if len(locked) else 0.0,
            "confidence_trend": (float(tail.mean() - head.mean())
                                 if len(head) and len(tail) else 0.0)}


def tail_measures(beats: np.ndarray, times: np.ndarray, confidences: np.ndarray, after: float,
                  tail_end: float) -> dict:
    """The silence after the programme: does the metronome stop?

    The confidence is read where it was first registered, 25 to 30 s after the
    music, however long the pass went on listening. `still_clicking` is the
    measure Amendment 2 added, and a pass that did not listen for a minute
    cannot answer it.
    """
    inside = beats[(beats >= after) & (beats < tail_end)] - after
    elapsed = times - after
    half_minute = (elapsed >= 0.0) & (elapsed < min(CONFIDENCE_AT[1], tail_end - after))
    read = (elapsed >= CONFIDENCE_AT[0]) & (elapsed < CONFIDENCE_AT[1])
    listened = tail_end - after >= CLICKING_AT[1] - 0.5
    return {"seconds": tail_end - after,
            "beats": int(len(inside)),
            "last_beat_after_sec": float(inside[-1]) if len(inside) else 0.0,
            "final_confidence": float(confidences[read].mean()) if read.any() else None,
            "locked_share": (float(np.mean(confidences[half_minute] >= LOCK_CONFIDENCE))
                             if half_minute.any() else None),
            "still_clicking": (bool(np.any((inside >= CLICKING_AT[0]) & (inside < CLICKING_AT[1])))
                               if listened else None)}


def alone_clicks(beats: np.ndarray, flags, stretches: dict[str, list[tuple[float, float]]]) -> dict:
    """Where the clicks gated for having the room to themselves fell.

    In the gaps and the tail that is the empty-room rule working. Under a take
    it is the rule taking music for an empty room, and the tracker losing half
    of every such beat for nothing.
    """
    alone = (np.asarray(flags, dtype=np.float64) > 0.5 if flags is not None and len(flags)
             else np.zeros(len(beats), dtype=bool))
    out = {}
    for name, spans in stretches.items():
        inside = np.zeros(len(beats), dtype=bool)
        for start, end in spans:
            inside |= (beats >= start) & (beats < end)
        out[name] = {"alone": int(np.sum(alone & inside)), "clicks": int(np.sum(inside))}
    return out


def removed_db(raw: np.ndarray, clean: np.ndarray, rate: float, t0: float, beats: np.ndarray,
               window: tuple[float, float]) -> float | None:
    """What was heard against what subtraction left, at the clicks in `window`.

    Only meaningful where the click is all there is to hear: in a silence it is
    how far the click went down, and with a room's own noise under it, a lower
    bound on that.
    """
    ratios = []
    for beat in beats[(beats >= window[0]) & (beats < window[1])]:
        a = int((beat + CLICK_WINDOW_SEC[0] - t0) * rate)
        b = int((beat + CLICK_WINDOW_SEC[1] - t0) * rate)
        if a < 0 or b > len(raw) or b > len(clean):
            continue
        heard = float(np.sum(np.square(raw[a:b], dtype=np.float64)))
        left = float(np.sum(np.square(clean[a:b], dtype=np.float64)))
        if heard > 0.0 and left > 0.0:
            ratios.append(10.0 * math.log10(heard / left))
    return float(np.median(ratios)) if ratios else None


def score_pass(pass_json: pathlib.Path, programme: np.ndarray, rate: float, windows: list[dict],
               items: dict[str, dict], period_of: dict[str, float]) -> dict:
    log = json.loads(pass_json.read_text(encoding="utf-8"))
    capture, capture_rate = read_mono(pass_json.with_suffix(".wav"))
    if abs(capture_rate - rate) > 0.5:
        raise ValueError(f"{pass_json.name}: captured at {capture_rate} Hz, programme {rate} Hz")
    clean = None
    if log.get("subtracted"):
        clean, _ = read_mono(pass_json.with_name(pass_json.stem + ".clean.wav"))
    t0 = float(log["first_stream_sec"])
    beats = np.asarray(log["beats"], dtype=np.float64)
    times = np.asarray(log["live_times"], dtype=np.float64)
    bpms = np.asarray(log["live_bpms"], dtype=np.float64)
    confidences = np.asarray(log["live_confidences"], dtype=np.float64)
    spreads = np.asarray(log["live_tempo_spreads_octaves"], dtype=np.float64)

    # The programme's acoustic return, at the first take and the last, so that
    # a drift between the device's two clocks shows instead of hiding.
    first, last = windows[0]["start"], windows[-1]["start"]
    lag_first, clarity_first = lag_between(programme, capture, rate, first)
    lag_last, clarity_last = lag_between(programme, capture, rate, last)
    slope = (lag_last - lag_first) / (last - first) if last > first else 0.0

    def to_stream(programme_sec: float) -> float:
        return t0 + programme_sec + lag_first + slope * (programme_sec - first)

    records = []
    music, gaps = [], []
    for window in windows:
        track = window["track"]
        start, end = to_stream(window["start"]), to_stream(window["end"])
        keep = (times >= start) & (times < end)
        music.append((start, end))
        payload = {
            "beats": (beats[(beats >= start) & (beats < end)] - start).tolist(),
            "duration_sec": end - start, "sample_rate": rate,
            "live_times": (times[keep] - start).tolist(),
            "live_bpms": bpms[keep].tolist(),
            "live_confidences": confidences[keep].tolist(),
            "live_tempo_spreads_octaves": spreads[keep].tolist(),
            "live_bpm": float(bpms[keep][-1]) if keep.any() else 0.0,
            "live_confidence": float(confidences[keep][-1]) if keep.any() else 0.0,
        }
        scored = _score_one(dict(items[track]), "model", pathlib.Path("unused"), None,
                            estimate=Estimate.from_json(payload))
        quiet = (to_stream(window["silence"][0]), to_stream(window["silence"][1]))
        gaps.append(quiet)
        ratio = (None if log.get("click_silent") else
                 click_to_music_db(programme, rate, window, period_of[track], log["click_db"]))
        records.append({"track": track, "f_measure": scored.get("f_measure"),
                        "usable": scored.get("usable"),
                        "episode_free": (scored.get("worst_wrong_octave_sec") or 0.0) <= 4.0,
                        "correct_share_of_eligible": scored.get("correct_share_of_eligible"),
                        "click_to_music_db": ratio,
                        "removed_db": (removed_db(capture, clean, rate, t0, beats, quiet)
                                       if clean is not None else None),
                        "sustain": sustain(beats, times, confidences, quiet, period_of[track])})
    # The long silence after the programme: the direct look at whether the
    # metronome stops once nothing is playing.
    after = to_stream(float(len(programme)) / rate)
    tail_end = after + float(log.get("tail_sec") or 0.0)
    tail = tail_measures(beats, times, confidences, after, tail_end)
    tail["removed_db"] = (removed_db(capture, clean, rate, t0, beats,
                                     (after + 5.0, min(tail_end, after + CONFIDENCE_AT[1])))
                          if clean is not None else None)
    alone_in = alone_clicks(beats, log.get("beats_alone"),
                            {"music": music, "gaps": gaps, "tail": [(after, tail_end)]})
    return {"pass": pass_json.stem, "tail": tail, "alone": alone_in,
            "settings": {key: log.get(key) for key in (
                "click_db", "click_silent", "gated", "subtracted", "gate_when_alone",
                "alone_gate_sec", "alone_listen_sec",
                "subtraction", "round_trip_sec", "round_trip_measured", "probe", "front_end",
                "model",
                "simulated_ms", "simulated_path_taps", "device", "stats")},
            "lag_sec": {"first": lag_first, "last": lag_last, "drift_per_hour": slope * 3600.0},
            "clarity": {"first": clarity_first, "last": clarity_last},
            "gate_misalignment_sec": (lag_first - float(log.get("round_trip_sec") or 0.0))
            if log.get("gated") else None,
            "records": records}


def interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = data[rng.integers(0, len(data), (DRAWS, len(data)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired(a: dict, b: dict, key: str) -> dict:
    """`a - b` per take, over the takes both scored."""
    by_b = {r["track"]: r for r in b["records"]}
    deltas = []
    for r in a["records"]:
        other = by_b.get(r["track"])
        left = r["sustain"][key] if key in r["sustain"] else r.get(key)
        right = (other["sustain"][key] if key in other["sustain"] else other.get(key)) \
            if other else None
        if left is not None and right is not None:
            deltas.append(float(left) - float(right))
    return {"n": len(deltas), "mean": float(np.mean(deltas)) if deltas else None,
            "ci95": interval(deltas)}


def summarise(passes: dict[str, dict]) -> dict:
    summary = {}
    for name, scored in passes.items():
        rows = scored["records"]
        summary[name] = {
            "f": float(np.mean([r["f_measure"] for r in rows if r["f_measure"] is not None])),
            "usable": float(np.mean([bool(r["usable"]) for r in rows])),
            "episode_free": float(np.mean([bool(r["episode_free"]) for r in rows])),
            "gap_locked_share": float(np.mean([r["sustain"]["locked_share"] for r in rows])),
            "gap_confidence_trend": float(np.mean([r["sustain"]["confidence_trend"] for r in rows])),
            "gap_click_ratio": float(np.mean([r["sustain"]["ratio"] for r in rows])),
            "gap_removed_db_median": (float(np.median(
                [r["removed_db"] for r in rows if r.get("removed_db") is not None]))
                if any(r.get("removed_db") is not None for r in rows) else None),
            "tail": scored["tail"],
            "alone": scored.get("alone"),
            "click_to_music_db_median": (float(np.median([r["click_to_music_db"] for r in rows]))
                                         if rows[0]["click_to_music_db"] is not None else None),
            "lag_sec": scored["lag_sec"], "gate_misalignment_sec": scored["gate_misalignment_sec"],
        }
    comparisons = {}
    for ungated, gated in (("l2_loud_ungated", "l1_loud_gated"),
                           ("l4_quiet_ungated", "l3_quiet_gated")):
        if ungated in passes and gated in passes:
            comparisons[f"{ungated}_minus_{gated}"] = {
                key: paired(passes[ungated], passes[gated], key)
                for key in ("f_measure", "usable", "locked_share", "confidence_trend")}
    for taken in ("l5_loud_subtracted", "l6_loud_guarded"):
        for against in ("l2_loud_ungated", "l1_loud_gated"):
            if taken in passes and against in passes:
                comparisons[f"{taken}_minus_{against}"] = {
                    key: paired(passes[taken], passes[against], key)
                    for key in ("f_measure", "usable", "locked_share", "confidence_trend")}
    for arm in ("l1_loud_gated", "l2_loud_ungated", "l3_quiet_gated", "l4_quiet_ungated",
                "l5_loud_subtracted", "l6_loud_guarded"):
        if arm in passes and "l0_silent" in passes:
            comparisons[f"{arm}_minus_l0_silent"] = {
                key: paired(passes[arm], passes["l0_silent"], key)
                for key in ("f_measure", "locked_share", "confidence_trend")}
    return {"arms": summary, "comparisons": comparisons}


def command_levels(args) -> int:
    programme, rate = read_mono(args.programme)
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    windows = take_windows(layout)
    items = {i["name"]: i for i in load_corpus(args.manifest, args.music, False, frozenset({"gtzan"}))}
    period_of = periods(windows, items)
    out = {name: (None if arm["ratio"] is None else
                  round(gain_for_ratio(programme, rate, windows, period_of, arm["ratio"]), 1))
           for name, arm in ARMS.items()}
    print(json.dumps(out, indent=1))
    return 0


def command_path(args) -> int:
    import soundfile

    path = synthetic_path(args.rate, args.direct_to_reverb_db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(args.output), path, int(args.rate), subtype="FLOAT")
    late = float(np.sum(path[int(0.150 * args.rate):] ** 2))
    print(json.dumps({"taps": len(path), "energy": float(np.sum(path ** 2)),
                      "beyond_150ms_db": 10.0 * math.log10(late / float(np.sum(path ** 2)))},
                     indent=1))
    return 0


def command_run(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    gains = json.loads(args.gains.read_text(encoding="utf-8"))
    names = [name for name in ARMS if not args.only or name in args.only]
    for index, name in enumerate(names):
        arm = ARMS[name]
        flags = ["loop", str(args.programme), "--model", str(args.model),
                 "--tail", repr(float(args.tail)), "-o", str(args.out / name)]
        if args.latency_ms is not None:
            flags += ["--latency-ms", repr(float(args.latency_ms))]
        if arm["ratio"] is None:
            flags.append("--no-click")
        else:
            flags += ["--click-db", repr(float(gains[name]))]
            if arm.get("subtract"):
                flags.append("--subtract")
                if arm["subtract"] == "only":
                    flags.append("--no-alone-gate")
                if args.subtract_update is not None:
                    flags += ["--subtract-update", repr(float(args.subtract_update))]
                if args.subtract_span_ms is not None:
                    flags += ["--subtract-span-ms", repr(float(args.subtract_span_ms))]
            elif not arm["gated"]:
                flags.append("--no-gate")
        if args.device:
            flags += ["--device", args.device]
        if args.simulate_ms is not None:
            flags += ["--simulate-ms", repr(float(args.simulate_ms))]
            if args.simulate_ir is not None:
                flags += ["--simulate-ir", str(args.simulate_ir)]
        print(f"[{index + 1}/{len(names)}] {name}: {' '.join(flags[2:])}", flush=True)
        done = subprocess.run([str(args.tiktak), *flags], check=False)
        if done.returncode not in (0, 1):
            print(f"{name} failed ({done.returncode}); stopping", flush=True)
            return done.returncode
        if index + 1 < len(names) and args.simulate_ms is None:
            time.sleep(args.pause)
    return 0


def command_score(args) -> int:
    programme, rate = read_mono(args.programme)
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    windows = take_windows(layout)
    items = {i["name"]: i for i in load_corpus(args.manifest, args.music, False, frozenset({"gtzan"}))}
    period_of = periods(windows, items)
    passes = {}
    for name in ARMS:
        for folder in (args.session, *args.also):
            path = folder / f"{name}.json"
            if path.is_file():
                passes[name] = score_pass(path, programme, rate, windows, items, period_of)
                break
    result: dict[str, Any] = {"schema": "tiktak.closed_loop_result/v1",
                              "session": str(args.session.name),
                              "passes": passes, **summarise(passes)}
    if args.provenance:
        from eval.provenance import experiment_provenance

        repository = pathlib.Path(__file__).resolve().parents[2]
        files = {"programme": args.programme, "layout": args.layout, "manifest": args.manifest}
        for name in passes:
            for folder in (args.session, *args.also):
                if (folder / f"{name}.json").is_file():
                    files[f"{name}.json"] = folder / f"{name}.json"
                    files[f"{name}.wav"] = folder / f"{name}.wav"
                    break
        result["provenance"] = experiment_provenance(repository, files, experiment="closed_loop")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"arms": result["arms"], "comparisons": result["comparisons"]}, indent=1,
                     default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("levels", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--programme", type=pathlib.Path, required=True)
        sub.add_argument("--layout", type=pathlib.Path, required=True)
        sub.add_argument("--manifest", type=pathlib.Path, required=True)
        sub.add_argument("--music", type=pathlib.Path, required=True)
        if name == "score":
            sub.add_argument("--session", type=pathlib.Path, required=True)
            sub.add_argument("--output", type=pathlib.Path, required=True)
            sub.add_argument("--also", type=pathlib.Path, nargs="*", default=[],
                             help="further folders to take arms from that --session lacks")
            sub.add_argument("--provenance", action="store_true",
                             help="stamp the result as an experiment; needs a clean tree")
    made = commands.add_parser("path")
    made.add_argument("--output", type=pathlib.Path, required=True)
    made.add_argument("--rate", type=float, default=48000.0)
    made.add_argument("--direct-to-reverb-db", type=float, default=10.0)
    run = commands.add_parser("run")
    run.add_argument("--tiktak", type=pathlib.Path, required=True)
    run.add_argument("--model", type=pathlib.Path, required=True)
    run.add_argument("--programme", type=pathlib.Path, required=True)
    run.add_argument("--gains", type=pathlib.Path, required=True,
                     help="the JSON `levels` printed, saved to a file")
    run.add_argument("--latency-ms", type=float, default=None,
                     help="the round trip; left out, every pass measures its own")
    run.add_argument("--out", type=pathlib.Path, required=True)
    run.add_argument("--tail", type=float, default=TAIL_SEC)
    run.add_argument("--pause", type=float, default=5.0)
    run.add_argument("--device", default="")
    run.add_argument("--only", nargs="*", default=[])
    run.add_argument("--simulate-ms", type=float, default=None,
                     help="no device: the digital loop with this delay, for a dry run")
    run.add_argument("--simulate-ir", type=pathlib.Path, default=None,
                     help="with --simulate-ms: the click's way back, as an impulse response")
    run.add_argument("--subtract-update", type=float, default=None,
                     help="development only: the canceller's update, in the subtracting arms")
    run.add_argument("--subtract-span-ms", type=float, default=None,
                     help="development only: how much of the path the canceller models")
    args = parser.parse_args(argv)
    return {"levels": command_levels, "path": command_path, "run": command_run,
            "score": command_score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
