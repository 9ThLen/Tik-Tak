#!/usr/bin/env python3
"""Session 3: room captures aligned by a slate rather than by correlation.

Sessions 1 and 2 aligned by cross-correlating onset envelopes, and lost two of
eight captures doing it. `0837_nottonight` needed a hand-written `skip_sec` with
a paragraph explaining it; `0707_halfwaygone` was voided outright, because four
of seven windows agreed on 0.476 s while the coherent sum peaked at 0.910 s --
0.9 beats apart at 125 BPM, with nothing in the recording able to settle it.

This session played takes built by `eval/slate.py`: a short Farina sweep, a
gap, the music, a gap, the same sweep again. The offset is read from the
deconvolution peak instead of inferred from the music, and the second slate
measures clock drift across the whole take rather than over a third of it.

    cd research
    .venv/Scripts/python -m eval.room_session3 \
        --takes ../music/room-session3-takes \
        --captures "../music/записи з телефону" \
        --manifest ../music/ground-truth/manifest.csv --music ../music \
        --binary <dump_analysis> --model <beatnet.ttw> \
        --aligned ../music/room-session3-aligned \
        --output ../research/results/room_session3.json

**Items come from `load_corpus`, never assembled by hand.** The manifest points
Harmonix annotations at `normalized/harmonix/*.csv`, which carry the corrected
offsets; the raw `.beats` files do not. A hand-built pairing using the latter
scored this session's `0116_goodies` at F 0.003 while two other tracks still
read 0.95, so the mistake was invisible on two recordings of three and would
have travelled into a result. The loader cannot make it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import soundfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]

from eval.live_corpus_benchmark import _score_one, load_corpus  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402
from eval.room_recording import read_audio  # noqa: E402
from eval.slate import RATE, align_by_slate  # noqa: E402

# Capture file per track. The names are what the phone wrote, including a tag
# it prefixed to one of them, and they are listed rather than globbed so a
# stray file in the folder cannot silently join the session.
CAPTURES = {
    "0116_goodies": "0116_goodies3.m4a",
    "0707_halfwaygone": "Miła0707_halfwaygone3.m4a",
    "0837_nottonight": "0837_nottonight3.m4a",
}

# What sessions 1 and 2 did with the same tracks, quoted so the comparison in
# the artifact does not depend on a reader fetching two other files.
EARLIER = {
    "0116_goodies": {"session1_room_f": 0.984, "session2_room_f": 0.938},
    "0707_halfwaygone": {"session1_room_f": 0.151,
                         "session2": "voided: alignment ambiguous between "
                                     "0.476 s and 0.910 s"},
    "0837_nottonight": {"session1_room_f": 0.555,
                        "session1_note": "aligned only with a hand-written "
                                         "skip_sec of 14.0 s"},
}


def align_one(capture: pathlib.Path, layout: dict) -> tuple[np.ndarray, dict]:
    mono, rate = read_audio(capture)
    if int(rate) != RATE:
        raise RuntimeError(f"{capture.name}: {rate} Hz, expected {RATE}")
    found = align_by_slate(mono, layout, rate=RATE)
    if not found.get("accepted"):
        raise RuntimeError(
            f"{capture.name}: slate not accepted -- {found.get('reason')}")
    return mono, found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takes", type=pathlib.Path, required=True)
    parser.add_argument("--captures", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--aligned", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    sources: dict[str, pathlib.Path] = {
        "binary": args.binary, "model": args.model, "manifest": args.manifest}
    for track, filename in CAPTURES.items():
        sources[f"capture_{track}"] = args.captures / filename
        sources[f"take_{track}"] = args.takes / f"{track}.wav"
        sources[f"layout_{track}"] = args.takes / f"{track}.layout.json"
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))

    # Fail closed: a dirty or unknown tree raises rather than being recorded
    # beside numbers that are already written.
    run_provenance = provenance(REPOSITORY, sources, session=3,
                                alignment="slate", rate=RATE)

    items = {item["name"]: item for item in load_corpus(
        args.manifest, args.music, False, frozenset({"harmonix"}))}
    args.aligned.mkdir(parents=True, exist_ok=True)

    records = []
    for track, filename in CAPTURES.items():
        layout = json.loads(
            sources[f"layout_{track}"].read_text(encoding="utf-8"))
        mono, found = align_one(sources[f"capture_{track}"], layout)

        start = int(round(found["music_offset_sec"] * RATE))
        stop = start + int(round(layout["music_seconds"] * RATE))
        if stop > len(mono):
            raise RuntimeError(f"{track}: capture ends before the music does")
        aligned = args.aligned / f"{track}.wav"
        soundfile.write(str(aligned), mono[start:stop], RATE)

        base = items[track]
        scored = {}
        for condition in ("clean", "room"):
            item = dict(base)
            if condition == "room":
                item["audio"] = aligned
            result = _score_one(item, "model", args.binary, args.model)
            scored[condition] = {
                "f_measure": result.get("f_measure"),
                "p70": result.get("p70"), "r70": result.get("r70"),
                "usable": bool(result.get("usable", False)),
                "reasons": list(result.get("reasons", [])),
            }

        records.append({
            "name": track,
            "capture": filename,
            "capture_seconds": found["capture_seconds"],
            "alignment": {
                "music_offset_sec": found["music_offset_sec"],
                "head_offset_sec": found["head"]["offset_sec"],
                "head_peak_to_sidelobe_db": found["head"]["peak_to_sidelobe_db"],
                "tail_peak_to_sidelobe_db": found["tail"]["peak_to_sidelobe_db"],
                "drift_sec": found["drift_sec"],
                "threshold_db": found["head"]["threshold_db"],
                # The first version of this block recorded the threshold and
                # nothing else, and the threshold turned out to be the least
                # important of the three: swept afterwards, the guard and the
                # tail window moved the worst margin by 15 dB while the bar
                # stayed at 12. A reader of that artifact could not have found
                # this out from it. These are the numbers that decide the ones
                # above, so they travel with them.
                "guard_sec": found["head"]["guard_sec"],
                "head_search_seconds": found["head"]["search_seconds"],
                "tail_search_seconds": found["tail"]["search_seconds"],
                "tail_search_half_width_sec":
                    found["tail_search_half_width_sec"],
                "max_drift_sec": found["max_drift_sec"],
            },
            "clean": scored["clean"], "room": scored["room"],
            "delta_f": (scored["room"]["f_measure"]
                        - scored["clean"]["f_measure"]),
            "earlier_sessions": EARLIER[track],
        })

    def mean(condition: str, field: str) -> float:
        return float(np.mean([r[condition][field] for r in records]))

    margins = [r["alignment"][k] for r in records
               for k in ("head_peak_to_sidelobe_db",
                         "tail_peak_to_sidelobe_db")]
    summary = {
        "n": len(records),
        "aligned_without_ambiguity": len(records),
        "clean_mean_f": mean("clean", "f_measure"),
        "room_mean_f": mean("room", "f_measure"),
        "mean_delta_f": float(np.mean([r["delta_f"] for r in records])),
        "clean_usable_rate": mean("clean", "usable"),
        "room_usable_rate": mean("room", "usable"),
        # The threshold is provisional and this is the evidence about it: six
        # margins from three captures, and the smallest is what says whether
        # 12 dB was chosen with room to spare or only just.
        "slate_margin_db": {"min": float(min(margins)),
                            "max": float(max(margins)),
                            "threshold": records[0]["alignment"]["threshold_db"],
                            "guard_sec": records[0]["alignment"]["guard_sec"]},
        # The primary alignment check, and the only one without a tunable in it:
        # two independent readings of one take have to agree. A head slate taken
        # a beat early does not read as a poor margin, it reads as half a second
        # of drift.
        "drift_sec": {
            "max_abs": float(max(abs(r["alignment"]["drift_sec"])
                                 for r in records)),
            "bar": records[0]["alignment"]["max_drift_sec"]},
        "max_abs_drift_sec": float(max(abs(r["alignment"]["drift_sec"])
                                       for r in records)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "provenance": run_provenance,
        "procedure": "eval/slate.py takes; offset read from the deconvolution "
                     "peak; drift from the tail slate",
        "summary": summary,
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{'track':20s} {'clean F':>8s} {'room F':>8s} {'delta':>7s} "
          f"{'head dB':>8s} {'tail dB':>8s} {'drift ms':>9s}")
    for r in records:
        a = r["alignment"]
        print(f"{r['name']:20s} {r['clean']['f_measure']:8.3f} "
              f"{r['room']['f_measure']:8.3f} {r['delta_f']:+7.3f} "
              f"{a['head_peak_to_sidelobe_db']:8.1f} "
              f"{a['tail_peak_to_sidelobe_db']:8.1f} "
              f"{a['drift_sec'] * 1000:9.1f}")
    print(f"mean clean {summary['clean_mean_f']:.3f} -> "
          f"room {summary['room_mean_f']:.3f} "
          f"({summary['mean_delta_f']:+.3f}); "
          f"smallest slate margin {summary['slate_margin_db']['min']:.1f} dB "
          f"against a {summary['slate_margin_db']['threshold']:.0f} dB bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
