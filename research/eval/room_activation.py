#!/usr/bin/env python3
"""What does a room do to the activation, on the five captures that aligned?

Four of five phone captures lost between 0.26 and 0.80 of beat F, which is the
largest single cost measured anywhere here, and nothing explains it. This dumps
BeatNet's beat channel for the clean file and for the aligned capture and asks
the four questions registered in `PREREGISTERED_room_diagnosis.md`, whose
thresholds are fixed there and are not repeated as tunables:

1. **Discriminability.** AUC of beat salience against the floor. Scale-free on
   purpose -- a capture that is merely quieter must not read as one whose beats
   are invisible, which a raw peak height would.
2. **Smear and delay.** Cross-correlation of the two arms' beat channels, and
   the width at half height of the activation around annotated beats.
3. **Doubling.** How high the activation gets *between* beats, relative to the
   beats flanking it. A tail read as an onset raises this; a blurrier or
   quieter activation does not.
4. **The phone rather than the room.** The slope of room block level against
   clean block level in dB. Gain control compresses and pushes it below one
   while leaving the acoustics alone.

The falsifier is the point of the exercise and is applied in `verdicts()`:
`0116_goodies` went **up** in the room, so a statistic that condemns all five
equally is describing the room and not the failure, and is reported as not
explaining it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.live_corpus_benchmark import load_corpus, load_reference_beats  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402

# The window `sample_at_beats` means by "the activation at a beat", and the
# tolerance the scorer uses. One constant, because a diagnosis that measured
# salience over a different window than the score does would not be about the
# score.
BEAT_WINDOW_SEC = 0.070
# Level blocks for the gain-control question. One second is long against a beat
# and short against a chorus, so it follows an AGC without following the music.
LEVEL_BLOCK_SEC = 1.0
# How far the cross-correlation looks. Alignment has already removed the gross
# offset; anything beyond half a second here would be a different beat.
MAX_LAG_SEC = 0.5


def dump_activation(binary: pathlib.Path, audio: pathlib.Path,
                    model: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    done = subprocess.run(
        [str(binary), str(audio), "--dump-activation", "--live-model", str(model)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{audio.name}: {done.stderr.strip()[:300]}")
    payload = json.loads(done.stdout)
    return (np.asarray(payload["activation_times"], dtype=np.float64),
            np.asarray(payload["activation_beat"], dtype=np.float64))


def salience_and_floor(times: np.ndarray, activation: np.ndarray,
                       beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak in +-70 ms of each beat, and every frame not near any beat."""
    near = np.zeros(len(times), dtype=bool)
    salience = []
    for beat in beats:
        window = (times >= beat - BEAT_WINDOW_SEC) & (times <= beat + BEAT_WINDOW_SEC)
        near |= window
        if window.any():
            salience.append(float(activation[window].max()))
    return np.asarray(salience), activation[~near]


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(a random positive exceeds a random negative), ties counted as half.

    Computed by ranks rather than by the O(nm) pair count: the floor is tens of
    thousands of frames and the beats are hundreds, and the exact statistic is
    the Mann-Whitney U over the same numbers.
    """
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    joined = np.concatenate([positive, negative])
    order = joined.argsort(kind="mergesort")
    ranks = np.empty(len(joined), dtype=np.float64)
    ranks[order] = np.arange(1, len(joined) + 1, dtype=np.float64)
    # Average the ranks inside each tie group, which is what makes a tie count
    # as half a win rather than as a whole one.
    values = joined[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum = float(ranks[:len(positive)].sum())
    return ((rank_sum - len(positive) * (len(positive) + 1) / 2.0)
            / (len(positive) * len(negative)))


def half_height_width(times: np.ndarray, activation: np.ndarray,
                      beats: np.ndarray) -> float:
    """Mean seconds the activation spends above half its peak around a beat.

    Measured inside +-0.25 s so a neighbouring beat at 200 BPM cannot be
    counted as this beat's skirt.
    """
    widths = []
    step = float(np.median(np.diff(times))) if len(times) > 2 else 0.02
    for beat in beats:
        window = (times >= beat - 0.25) & (times <= beat + 0.25)
        if not window.any():
            continue
        local = activation[window]
        peak = float(local.max())
        if peak <= 0.0:
            continue
        widths.append(float((local >= peak / 2.0).sum()) * step)
    return float(np.mean(widths)) if widths else float("nan")


def between_beat_ratio(times: np.ndarray, activation: np.ndarray,
                       beats: np.ndarray) -> float:
    """Max activation in the middle half of each gap, over the flanking beats.

    The middle half and not the whole gap: the skirts of the two real beats
    reach into it, and including them would measure smear a second time instead
    of measuring a new peak.
    """
    ratios = []
    for first, second in zip(beats[:-1], beats[1:]):
        span = second - first
        if span <= 0:
            continue
        middle = (times >= first + 0.25 * span) & (times <= second - 0.25 * span)
        flank = [(times >= b - BEAT_WINDOW_SEC) & (times <= b + BEAT_WINDOW_SEC)
                 for b in (first, second)]
        if not middle.any() or not all(f.any() for f in flank):
            continue
        height = float(np.mean([activation[f].max() for f in flank]))
        if height > 0.0:
            ratios.append(float(activation[middle].max()) / height)
    return float(np.mean(ratios)) if ratios else float("nan")


def level_slope(clean: np.ndarray, room: np.ndarray, rate_a: float,
                rate_b: float) -> dict:
    """Room block level against clean block level, in dB, and its slope.

    A slope below one is compression: loud passages brought down or quiet ones
    brought up, which is what automatic gain control does and what a room does
    not. The two arms are already aligned at sample zero, so block *n* is the
    same music in both.
    """
    def blocks(mono: np.ndarray, rate: float) -> np.ndarray:
        size = max(1, int(LEVEL_BLOCK_SEC * rate))
        usable = len(mono) // size * size
        if usable == 0:
            return np.zeros(0)
        power = np.mean(mono[:usable].reshape(-1, size) ** 2, axis=1)
        return 10.0 * np.log10(np.maximum(power, 1e-12))

    a, b = blocks(clean, rate_a), blocks(room, rate_b)
    count = min(len(a), len(b))
    a, b = a[:count], b[:count]
    # Silence at either end is not a level, and a pair of them would anchor the
    # fit on the one thing neither arm is trying to reproduce.
    keep = (a > a.max() - 40.0) & (b > b.max() - 40.0)
    a, b = a[keep], b[keep]
    if len(a) < 8:
        return {"slope": float("nan"), "blocks": int(len(a))}
    slope, intercept = np.polyfit(a, b, 1)
    return {
        "slope": float(slope),
        "intercept_db": float(intercept),
        "blocks": int(len(a)),
        "clean_range_db": float(a.max() - a.min()),
        "room_range_db": float(b.max() - b.min()),
    }


def measure_one(item: dict, aligned: pathlib.Path, binary: pathlib.Path,
                model: pathlib.Path) -> dict:
    import soundfile

    from eval.room_recording import read_audio

    beats = load_reference_beats(item["annotation"])
    out: dict = {"name": item["name"], "beats": int(len(beats))}

    arms = {"clean": pathlib.Path(item["audio"]), "room": aligned}
    channels = {}
    for arm, path in arms.items():
        times, activation = dump_activation(binary, path, model)
        channels[arm] = (times, activation)
        inside = beats[beats <= times[-1]] if len(times) else beats
        salience, floor = salience_and_floor(times, activation, inside)
        out[arm] = {
            "frames": int(len(times)),
            "auc": auc(salience, floor),
            "mean_salience": float(np.mean(salience)) if len(salience) else float("nan"),
            "median_floor": float(np.median(floor)) if len(floor) else float("nan"),
            "half_height_width_sec": half_height_width(times, activation, inside),
            "between_beat_ratio": between_beat_ratio(times, activation, inside),
        }

    # One grid for the two arms: the frame times are identical by construction
    # only if the two files are the same length, and they are not.
    (ta, aa), (tb, ab) = channels["clean"], channels["room"]
    count = min(len(aa), len(ab))
    step = float(np.median(np.diff(ta))) if len(ta) > 2 else 0.02
    x = aa[:count] - aa[:count].mean()
    y = ab[:count] - ab[:count].mean()
    lags = int(MAX_LAG_SEC / step)
    correlation = np.correlate(y, x, mode="full")
    centre = len(x) - 1
    window = correlation[centre - lags: centre + lags + 1]
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    window = window / norm if norm > 0 else window
    peak = int(np.argmax(window))
    out["cross"] = {
        "lag_sec": float((peak - lags) * step),
        "peak": float(window[peak]),
        "at_zero_lag": float(window[lags]),
    }

    clean_audio, rate_a = read_audio(pathlib.Path(item["audio"]))
    room_audio, rate_b = read_audio(aligned)
    out["level"] = level_slope(clean_audio, room_audio, rate_a, rate_b)
    return out


def verdicts(records: list[dict], survivor: str) -> dict:
    """The registered readings, each with its falsifier applied.

    A candidate explanation fires on a recording if its threshold is crossed
    there. It *explains* the collapse only if it does not also fire on the
    recording that survived the room -- otherwise it is a description of the
    room, and the registration says to report it as not explaining anything.
    """
    def fired(record: dict) -> dict:
        clean, room = record["clean"], record["room"]
        drop = clean["auc"] - room["auc"]
        return {
            "front_end": bool(drop >= 0.10 or room["auc"] < 0.75),
            "decoder": bool(drop < 0.05 and room["auc"] >= 0.85),
            "doubling": bool(room["between_beat_ratio"]
                             - clean["between_beat_ratio"] >= 0.15),
            "gain_control": bool(record["level"]["slope"] <= 0.7),
        }

    per_track = {r["name"]: fired(r) for r in records}
    control = per_track.get(survivor, {})
    out = {}
    for candidate in ("front_end", "decoder", "doubling", "gain_control"):
        collapsed = [n for n, f in per_track.items()
                     if n != survivor and f[candidate]]
        on_control = bool(control.get(candidate))
        out[candidate] = {
            "fires_on_collapsed": collapsed,
            "fires_on_survivor": on_control,
            "explains": bool(collapsed and not on_control),
            "note": ("also fires on the recording that survived, so it "
                     "describes the room and not the failure")
            if on_control and collapsed else "",
        }
    out["per_track"] = per_track
    return out


def orderings(records: list[dict], drops: dict) -> dict:
    """Does each statistic order the recordings the way the F loss does?

    Reported with the count of statistics examined, because that count is the
    whole caveat. On five points the only Spearman that reaches two-sided 0.05
    is +-1.00, at p = 2/120 = 0.017, and eight statistics at 0.017 is 0.13 --
    so a perfect ordering here is suggestive and is not established. None of
    these was registered as the primary; the registered primary is the AUC
    falsifier in `verdicts()`.
    """
    def rank(values: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(values)).astype(np.float64)

    names = [r["name"] for r in records]
    loss = np.asarray([drops[n] for n in names])
    candidates = {
        "auc_drop": [r["clean"]["auc"] - r["room"]["auc"] for r in records],
        "room_auc": [r["room"]["auc"] for r in records],
        "room_salience": [r["room"]["mean_salience"] for r in records],
        "room_floor": [r["room"]["median_floor"] for r in records],
        "room_between_beat": [r["room"]["between_beat_ratio"] for r in records],
        "room_width": [r["room"]["half_height_width_sec"] for r in records],
        "cross_peak": [r["cross"]["peak"] for r in records],
        "level_slope": [r["level"]["slope"] for r in records],
    }
    out = {"n": len(names), "statistics_examined": len(candidates),
           "spearman": {}}
    for key, values in candidates.items():
        out["spearman"][key] = float(
            np.corrcoef(rank(np.asarray(values)), rank(loss))[0, 1])
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", default=["harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--aligned", type=pathlib.Path, required=True,
                        help="the --write-aligned directory from room_recording")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--survivor", default="0116_goodies",
                        help="the recording that did not lose F in the room; "
                             "the falsifier for every candidate")
    parser.add_argument("--scores", type=pathlib.Path,
                        help="room_recording_phone.json, to order the "
                             "statistics against the F actually lost")
    args = parser.parse_args(argv)

    run_provenance = provenance(
        repository,
        {"manifest": args.manifest, "binary": args.binary,
         "model": args.model, "scores": args.scores},
    )

    items = {i["name"]: i for i in
             load_corpus(args.manifest, args.music, False, frozenset(args.corpora))}
    seen = sorted({i["corpus"] for i in items.values()})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1

    records, failures = [], []
    for path in sorted(args.aligned.glob("*.wav")):
        item = items.get(path.stem)
        if item is None:
            failures.append({"name": path.stem, "error": "no corpus entry"})
            continue
        print(f"  {path.stem}", file=sys.stderr, flush=True)
        try:
            records.append(measure_one(item, path, args.binary, args.model))
        except Exception as error:  # noqa: BLE001
            failures.append({"name": path.stem, "error": str(error)[:300]})

    payload = {"provenance": run_provenance, "failures": failures,
               "survivor": args.survivor,
               "verdicts": verdicts(records, args.survivor) if records else {},
               "records": records}
    if args.scores and records:
        scored = json.loads(args.scores.read_text(encoding="utf-8"))
        drops = {r["name"]: r["clean"]["f_measure"] - r["room"]["f_measure"]
                 for r in scored["records"] if "room" in r}
        if all(r["name"] in drops for r in records):
            payload["orderings"] = orderings(records, drops)
        else:
            payload["orderings"] = {"error": "a recording here was not scored there"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for record in records:
        print(f"\n{record['name']}  ({record['beats']} beats)")
        for arm in ("clean", "room"):
            a = record[arm]
            print(f"  {arm:5s} AUC {a['auc']:.3f}  salience {a['mean_salience']:.3f}"
                  f"  floor {a['median_floor']:.4f}"
                  f"  width {a['half_height_width_sec']:.3f}s"
                  f"  between {a['between_beat_ratio']:.3f}")
        print(f"  cross lag {record['cross']['lag_sec']:+.3f}s peak "
              f"{record['cross']['peak']:.3f}   level slope "
              f"{record['level']['slope']:.2f}")
    print("\n" + json.dumps(payload["verdicts"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
