"""Do sparse time-frequency peaks hold the gaps open in a room?

Runs the experiment registered in ``eval/PEAK_FRONT_END_PLAN.md``. Nothing here
chooses a threshold: the four success conditions were written down before any
number existed and are applied as written.

    cd research
    .venv/Scripts/python -m eval.peak_front_end \
        --binary ../tools/eval/build/RelWithDebInfo/dump_analysis.exe \
        --data-root .. --features-dir <scratch> --output ../research/results/peak_front_end.json

The comparison is three-armed because two arms would attribute nothing. Both
the feature set and the picking change between `odf` and `peaks`, so `dense` —
the same BeatNet features with no picking — is the control that separates "the
peaks helped" from "the features changed".

Parameters are chosen leave-one-track-out: each recording's number comes from
parameters fitted without it. Picking the best of a sweep on the same five room
captures and calling the winner a result is fitting, not measuring, and it is
the most likely way this experiment produces a false positive.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.activation_recall import matched  # noqa: E402
from eval.features import read as read_features  # noqa: E402
from eval.peaks import (PeakParams, collapse, dense_signals,  # noqa: E402
                        peak_map)
from eval.provenance import provenance  # noqa: E402
from eval.whitening_room import WARMUP_SEC, contrast  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]

TRACKS = ("0116_goodies", "0132_iceicebaby", "0466_onthedarkside",
          "0707_halfwaygone", "0837_nottonight")
CONDITIONS = ("clean", "room")

# The sweep. Deliberately coarse: a finer grid searched on five recordings buys
# a better fit and no more evidence.
BAND_RADII = (1, 2, 4)
PAST_FRAMES = (3, 6, 12)
REFRACTORY = (0, 2, 5)
MERGES = ("difference", "union", "sum")
NOVELTY_HORIZONS = (5, 10, 25)

# The registered bars, quoted from the plan so a reader need not cross-check.
MIN_TRACKS_IMPROVED = 4
MAX_DEGRADATION_FRACTION = 2.0 / 3.0


def feature_path(folder: pathlib.Path, track: str, condition: str) -> pathlib.Path:
    return folder / f"{track}_{condition}.ttfd"


def audio_path(root: pathlib.Path, track: str, condition: str) -> pathlib.Path:
    if condition == "clean":
        return root / "music/ground-truth/audio/harmonix-ready" / f"{track}.wav"
    return root / "music/room-aligned" / f"{track}.wav"


def ensure_features(binary: pathlib.Path, audio: pathlib.Path,
                    out: pathlib.Path) -> None:
    if out.is_file():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([str(binary), str(audio), "--dump-features", str(out)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if done.returncode != 0:
        raise RuntimeError(f"dump_analysis failed on {audio}: {done.stderr.strip()}")


def signal_maxima(values: np.ndarray) -> np.ndarray:
    """Indices of the one-dimensional local maxima of a collapsed signal.

    A frame qualifies when it is at least its neighbours and strictly above the
    one before, which makes a flat ridge report its first frame — the same rule
    the two-dimensional map uses, so the two cannot disagree about plateaus.
    """
    if len(values) < 3:
        return np.flatnonzero(values > 0.0)
    higher_than_before = np.empty(len(values), dtype=bool)
    higher_than_before[0] = True
    higher_than_before[1:] = values[1:] > values[:-1]
    at_least_after = np.empty(len(values), dtype=bool)
    at_least_after[-1] = True
    at_least_after[:-1] = values[:-1] >= values[1:]
    return np.flatnonzero(higher_than_before & at_least_after & (values > 0.0))


def top_n(values: np.ndarray, times: np.ndarray, beats: np.ndarray,
          track: str) -> dict:
    """The N strongest peaks against N reference beats, matched one-to-one.

    N is the count of scorable beats, so a denser signal cannot score by being
    dense. The chance baseline is seeded from the track's own name by SHA-256,
    for the reason `activation_recall.py` records: Python salts `hash()` per
    process, and an unseeded baseline moved in the third digit between two runs
    of the same script.
    """
    beats = beats[(beats >= WARMUP_SEC) & (beats <= times[-1])]
    if len(beats) < 8:
        return {}
    candidates = signal_maxima(values)
    if len(candidates) == 0:
        return {"top_n": 0.0, "top_n_chance": 0.0, "candidates_per_sec": 0.0}

    heights = values[candidates]
    # A stable sort so ties in an integer-valued signal — which `count`
    # produces by the thousand — resolve by time and not by sort internals.
    order = np.argsort(-heights, kind="stable")[: len(beats)]
    chosen = np.sort(times[candidates[order]])

    digest = hashlib.sha256(track.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    chance = np.sort(rng.uniform(beats[0], beats[-1], len(beats)))

    span = float(times[-1] - times[0])
    return {"top_n": matched(beats, chosen) / len(beats),
            "top_n_chance": matched(beats, chance) / len(beats),
            "candidates_per_sec": len(candidates) / span if span > 0 else 0.0}


def score(values: np.ndarray, times: np.ndarray, beats: np.ndarray,
          track: str) -> dict:
    stats = contrast(values, times, beats)
    if not stats:
        return {}
    stats.update(top_n(values, times, beats, track))
    return stats


def grid(causal: bool) -> list[PeakParams]:
    out = []
    for radius, past, refractory, merge in itertools.product(
            BAND_RADII, PAST_FRAMES, REFRACTORY, MERGES):
        out.append(PeakParams(band_radius=radius, past_frames=past,
                              future_frames=0 if causal else past,
                              refractory_frames=refractory, merge=merge))
    return out


def readouts(mask, heights, params: PeakParams) -> dict[str, np.ndarray]:
    """Every collapse rule for one map, since the map is what costs."""
    out = {"count": collapse(mask, heights, "count", params),
           "weighted": collapse(mask, heights, "weighted", params)}
    for horizon in NOVELTY_HORIZONS:
        tuned = dataclasses.replace(params, novelty_frames=horizon)
        out[f"novelty{horizon}"] = collapse(mask, heights, "novelty", tuned)
    return out


def run(features: dict, beats: dict, arm: str) -> dict:
    """Every (parameters, readout) point, scored on every recording."""
    table: dict = {}
    for params in grid(causal=(arm == "causal")):
        for track in TRACKS:
            for condition in CONDITIONS:
                data = features[(track, condition)]
                mask, heights = peak_map(data.filterbank, data.difference,
                                         params)
                for rule, values in readouts(mask, heights, params).items():
                    stats = score(values, data.times, beats[track], track)
                    if not stats:
                        raise RuntimeError(
                            f"unscorable: {track} {condition} {params.label()} {rule}")
                    table[(params.label(), rule, track, condition)] = stats
    return table


def degradation(table: dict, key, track: str) -> float:
    return (table[(*key, track, "room")]["ratio"]
            - table[(*key, track, "clean")]["ratio"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--features-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="for development only; the artifact records it")
    args = parser.parse_args()

    root = args.data_root.resolve()
    sources: dict[str, pathlib.Path] = {}
    for track in TRACKS:
        sources[f"beats_{track}"] = (
            root / "annotations/harmonix/annotations/beats" / f"{track}.beats")
        for condition in CONDITIONS:
            sources[f"{condition}_{track}"] = audio_path(root, track, condition)
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))

    run_provenance = provenance(
        REPOSITORY, sources, tracks=list(TRACKS), warmup_sec=WARMUP_SEC,
        band_radii=list(BAND_RADII), past_frames=list(PAST_FRAMES),
        refractory=list(REFRACTORY), merges=list(MERGES),
        novelty_horizons=list(NOVELTY_HORIZONS))
    if run_provenance["tree_clean"] is not True and not args.allow_dirty:
        raise RuntimeError(
            "refusing a provisional run: git tree is not provably clean")

    features, beats = {}, {}
    for track in TRACKS:
        beats[track] = np.loadtxt(sources[f"beats_{track}"], usecols=0, ndmin=1)
        for condition in CONDITIONS:
            path = feature_path(args.features_dir, track, condition)
            ensure_features(args.binary, sources[f"{condition}_{track}"], path)
            features[(track, condition)] = read_features(path)

    # The control, which does not depend on any sweep parameter.
    dense: dict = {}
    for track in TRACKS:
        for condition in CONDITIONS:
            data = features[(track, condition)]
            for name, values in dense_signals(data.filterbank,
                                              data.difference).items():
                stats = score(values, data.times, beats[track], track)
                if not stats:
                    raise RuntimeError(f"unscorable dense: {track} {condition}")
                dense[(name, track, condition)] = stats

    dense_degradation = {
        track: (dense[("dense_difference", track, "room")]["ratio"]
                - dense[("dense_difference", track, "clean")]["ratio"])
        for track in TRACKS}

    arms = {arm: run(features, beats, arm) for arm in ("causal", "symmetric")}

    def loto(table: dict, keys: list) -> list[dict]:
        rows = []
        for held_out in TRACKS:
            others = [t for t in TRACKS if t != held_out]
            best = min(keys, key=lambda k: (
                float(np.mean([degradation(table, k, t) for t in others])), k))
            label, rule = best
            rows.append({
                "held_out": held_out,
                "selected_on": others,
                "parameters": next(p.as_dict() for p in grid("_f0_" in label)
                                   if p.label() == label) | {"readout": rule},
                "peaks": {"clean": table[(label, rule, held_out, "clean")],
                          "room": table[(label, rule, held_out, "room")],
                          "degradation": degradation(table, best, held_out)},
                "dense": {"clean": dense[("dense_difference", held_out, "clean")],
                          "room": dense[("dense_difference", held_out, "room")],
                          "degradation": dense_degradation[held_out]},
                "selection_objective_on_others": float(np.mean(
                    [degradation(table, best, t) for t in others])),
                # A ratio that did not move at all between conditions is worth
                # flagging rather than celebrating: `count` and `novelty` are
                # small integers, and a median of small integers can land on the
                # same value in a clean room and a live one. Zero degradation
                # then means the metric could not resolve the difference, not
                # that the signal survived it.
                "ratio_identical_across_conditions": bool(
                    table[(label, rule, held_out, "room")]["ratio"]
                    == table[(label, rule, held_out, "clean")]["ratio"]),
            })
        return rows

    folds = {}
    for arm, table in arms.items():
        folds[arm] = loto(table, sorted({(label, rule)
                                         for label, rule, _, _ in table}))

    # The registered null holds only "under every one of the three collapse
    # rules", so each family is selected and judged on its own. Not a retry:
    # the condition demands it, and a family passing in isolation would break
    # the null.
    families = {}
    for arm, table in arms.items():
        keys = sorted({(label, rule) for label, rule, _, _ in table})
        families[arm] = {
            family: loto(table, [k for k in keys if k[1].startswith(family)])
            for family in ("count", "weighted", "novelty")}

    def judge(arm: str, rows: list[dict]) -> dict:
        improved = sum(1 for r in rows
                       if r["peaks"]["degradation"] < r["dense"]["degradation"])
        peaks_mean = float(np.mean([r["peaks"]["degradation"] for r in rows]))
        dense_mean = float(np.mean([r["dense"]["degradation"] for r in rows]))
        strict = all(r["peaks"]["room"]["top_n"] >= r["dense"]["room"]["top_n"]
                     for r in rows)
        above_chance = all(r["peaks"]["room"]["top_n"]
                           > r["peaks"]["room"]["top_n_chance"] for r in rows)
        conditions = {
            "improved_on_at_least_four": improved >= MIN_TRACKS_IMPROVED,
            "degradation_at_most_two_thirds":
                peaks_mean <= MAX_DEGRADATION_FRACTION * dense_mean,
            "top_n_not_worse_and_above_chance": strict and above_chance,
            "parameters_selected_without_the_scored_track": True,
        }
        return {
            "tracks_improved": improved,
            "peaks_mean_degradation": peaks_mean,
            "dense_mean_degradation": dense_mean,
            "top_n": {
                "peaks_mean": float(np.mean([r["peaks"]["room"]["top_n"]
                                             for r in rows])),
                "dense_mean": float(np.mean([r["dense"]["room"]["top_n"]
                                             for r in rows])),
                "strict_per_track_not_worse": strict,
                "every_track_above_chance": above_chance,
                "binding_reading": "strict_per_track"},
            "folds_where_the_ratio_did_not_move": sum(
                1 for r in rows if r["ratio_identical_across_conditions"]),
            "conditions": conditions,
            "passes": arm == "causal" and all(conditions.values()),
        }

    by_family = {arm: {family: judge(arm, rows)
                       for family, rows in per_family.items()}
                 for arm, per_family in families.items()}

    verdict = {}
    for arm, rows in folds.items():
        improved = sum(1 for r in rows
                       if r["peaks"]["degradation"] < r["dense"]["degradation"])
        peaks_mean = float(np.mean([r["peaks"]["degradation"] for r in rows]))
        dense_mean = float(np.mean([r["dense"]["degradation"] for r in rows]))
        # The plan spelled out conditions 1 and 2 as per-track and mean
        # respectively, and left condition 3 as "top-N is not worse than
        # `dense`" with no qualifier. The strict per-track reading is the
        # binding one, because an ambiguous criterion should be the
        # conservative one; the mean reading is reported beside it rather than
        # substituted for it.
        #
        # Recorded plainly because of when it was noticed: a one-point smoke
        # run had already shown the two readings can disagree. Choosing between
        # them silently after that would be choosing a verdict.
        above_chance = all(r["peaks"]["room"]["top_n"]
                           > r["peaks"]["room"]["top_n_chance"] for r in rows)
        strict = all(r["peaks"]["room"]["top_n"] >= r["dense"]["room"]["top_n"]
                     for r in rows)
        peaks_top = float(np.mean([r["peaks"]["room"]["top_n"] for r in rows]))
        dense_top = float(np.mean([r["dense"]["room"]["top_n"] for r in rows]))
        conditions = {
            "improved_on_at_least_four": improved >= MIN_TRACKS_IMPROVED,
            "degradation_at_most_two_thirds": peaks_mean <= MAX_DEGRADATION_FRACTION * dense_mean,
            "top_n_not_worse_and_above_chance": strict and above_chance,
            "parameters_selected_without_the_scored_track": True,
        }
        verdict[arm] = {
            "tracks_improved": improved,
            "peaks_mean_degradation": peaks_mean,
            "dense_mean_degradation": dense_mean,
            "top_n": {"peaks_mean": peaks_top, "dense_mean": dense_top,
                      "strict_per_track_not_worse": strict,
                      "mean_not_worse": peaks_top >= dense_top,
                      "every_track_above_chance": above_chance,
                      "binding_reading": "strict_per_track"},
            "conditions": conditions,
            # Only the causal arm may pass, however large the symmetric one is.
            "passes": arm == "causal" and all(conditions.values()),
        }

    def flatten(table: dict) -> list[dict]:
        return [{"parameters": label, "readout": rule, "track": track,
                 "condition": condition, **stats}
                for (label, rule, track, condition), stats in sorted(table.items())]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "provenance": run_provenance,
        "registered_in": "research/eval/PEAK_FRONT_END_PLAN.md",
        "folds": folds,
        "verdict": verdict,
        # The null the plan registered holds only if every collapse family
        # fails on its own, so each is selected and judged separately here.
        "by_collapse_family": by_family,
        "folds_by_collapse_family": families,
        "dense": [{"signal": name, "track": track, "condition": condition, **stats}
                  for (name, track, condition), stats in sorted(dense.items())],
        "sweep": {arm: flatten(table) for arm, table in arms.items()},
    }, indent=2), "utf-8")

    for arm, result in verdict.items():
        print(f"{arm:10s} improved {result['tracks_improved']}/5  "
              f"peaks {result['peaks_mean_degradation']:.4f}  "
              f"dense {result['dense_mean_degradation']:.4f}  "
              f"passes {result['passes']}")
    for family, result in by_family["causal"].items():
        print(f"  causal/{family:9s} improved {result['tracks_improved']}/5  "
              f"peaks {result['peaks_mean_degradation']:.4f}  "
              f"topN {result['top_n']['peaks_mean']:.3f} vs "
              f"{result['top_n']['dense_mean']:.3f}  "
              f"passes {result['passes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
