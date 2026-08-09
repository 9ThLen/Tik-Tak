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
import itertools
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.activation_recall import matched, top_n_times_and_chance  # noqa: E402
from eval.features import read as read_features  # noqa: E402
from eval.peaks import (PeakParams, collapse, dense_signals,  # noqa: E402
                        peak_map)
from eval.provenance import (digest,  # noqa: E402
                             experiment_provenance as provenance)
from eval.whitening_room import WARMUP_SEC, contrast  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
ALIGNMENT_ARTIFACT = REPOSITORY / "research/results/room_recording_phone.json"

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


def write_features(binary: pathlib.Path, audio: pathlib.Path,
                   out: pathlib.Path) -> None:
    """Regenerate a dump so its producer is the binary recorded by this run."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=f".{out.name}.", suffix=".tmp", dir=out.parent,
            delete=False) as staging:
        staged = pathlib.Path(staging.name)
    try:
        done = subprocess.run(
            [str(binary), str(audio), "--dump-features", str(staged)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        if done.returncode != 0:
            raise RuntimeError(
                f"dump_analysis failed on {audio}: {done.stderr.strip()}")
        try:
            read_features(staged)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"dump_analysis produced an invalid feature file for {audio}: "
                f"{error}") from error
        staged.replace(out)
    finally:
        staged.unlink(missing_ok=True)


def alignment_offsets(
        path: pathlib.Path,
        room_inputs: dict[str, pathlib.Path]) -> dict[str, dict[str, float | str]]:
    """The measured offsets already applied to ``music/room-aligned``."""
    payload = json.loads(path.read_text("utf-8"))
    by_track = {row["name"]: row for row in payload.get("records", [])
                if row.get("aligned_audio")}
    missing = [track for track in TRACKS if track not in by_track]
    if missing:
        raise ValueError(f"alignment artifact lacks: {', '.join(missing)}")
    result: dict[str, dict[str, float | str]] = {}
    for track in TRACKS:
        actual = room_inputs[track].resolve()
        expected_sha256 = by_track[track].get("aligned_audio_sha256")
        actual_digest = digest(actual)
        if not expected_sha256:
            raise ValueError(f"alignment artifact lacks a digest for {track}")
        if actual_digest is None:
            raise ValueError(f"cannot hash aligned room input {actual}")
        if actual_digest["sha256"] != expected_sha256:
            raise ValueError(
                f"aligned room input for {track} does not match the artifact")
        result[track] = {
            "offset_sec": float(by_track[track]["alignment"]["offset_sec"]),
            "skip_sec": float(by_track[track]["alignment"].get("skip_sec", 0.0)),
            "aligned_audio": str(actual),
            "aligned_audio_sha256": actual_digest["sha256"],
        }
    return result


def shared_reference(raw_beats: np.ndarray,
                     recordings: list) -> tuple[np.ndarray, float, float]:
    """Use one scorable beat population for every condition in a pair."""
    start = max(WARMUP_SEC,
                *(float(recording.times[0]) for recording in recordings))
    end = min(float(recording.times[-1]) for recording in recordings)
    shared = raw_beats[(raw_beats >= start) & (raw_beats <= end)]
    return shared, start, end


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

    chosen, chance_reference = top_n_times_and_chance(
        beats, times[candidates], values[candidates], track)

    span = float(times[-1] - times[0])
    return {"top_n": matched(beats, chosen) / len(beats),
            "top_n_chance": matched(chance_reference, chosen) / len(beats),
            "candidates_per_sec": len(candidates) / span if span > 0 else 0.0}


def score(values: np.ndarray, times: np.ndarray, beats: np.ndarray,
          track: str, interval: dict | None = None) -> dict:
    if interval is not None:
        inside = ((times >= interval["start_sec"])
                  & (times <= interval["end_sec"]))
        values = values[inside]
        times = times[inside]
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


def run(features: dict, beats: dict, arm: str,
        scoring_intervals: dict) -> dict:
    """Every (parameters, readout) point, scored on every recording."""
    table: dict = {}
    for params in grid(causal=(arm == "causal")):
        for track in TRACKS:
            for condition in CONDITIONS:
                data = features[(track, condition)]
                mask, heights = peak_map(data.filterbank, data.difference,
                                         params)
                for rule, values in readouts(mask, heights, params).items():
                    stats = score(values, data.times, beats[track], track,
                                  scoring_intervals[track])
                    if not stats:
                        raise RuntimeError(
                            f"unscorable: {track} {condition} {params.label()} {rule}")
                    table[(params.label(), rule, track, condition)] = stats
    return table


def degradation(table: dict, key, track: str) -> float:
    return (table[(*key, track, "room")]["ratio"]
            - table[(*key, track, "clean")]["ratio"])


def selection_record(label: str, rule: str) -> dict:
    """Serialize both the exact sweep key and effective readout parameters."""
    selected = next(p for p in grid("_f0_" in label) if p.label() == label)
    if rule.startswith("novelty"):
        selected = dataclasses.replace(
            selected, novelty_frames=int(rule.removeprefix("novelty")))
    return {
        "selection_key": {"parameters": label, "readout": rule},
        "parameters": selected.as_dict() | {"readout": rule},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--features-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    for label, path in (("--features-dir", args.features_dir),
                        ("--output", args.output)):
        try:
            path.resolve().relative_to(REPOSITORY.resolve())
        except ValueError:
            pass
        else:
            parser.error(f"{label} must be outside the evaluation worktree")

    root = args.data_root.resolve()
    sources: dict[str, pathlib.Path] = {
        "binary": args.binary.resolve(),
        "room_alignment_artifact": ALIGNMENT_ARTIFACT,
    }
    for track in TRACKS:
        sources[f"beats_{track}"] = (
            root / "annotations/harmonix/annotations/beats" / f"{track}.beats")
        for condition in CONDITIONS:
            sources[f"{condition}_{track}"] = audio_path(root, track, condition)
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))

    room_inputs = {track: sources[f"room_{track}"] for track in TRACKS}
    alignment = alignment_offsets(ALIGNMENT_ARTIFACT, room_inputs)
    provenance_extra = dict(
        tracks=list(TRACKS), warmup_sec=WARMUP_SEC,
        band_radii=list(BAND_RADII), past_frames=list(PAST_FRAMES),
        refractory=list(REFRACTORY), merges=list(MERGES),
        novelty_horizons=list(NOVELTY_HORIZONS),
        room_alignment=alignment)
    provenance(REPOSITORY, sources, **provenance_extra)

    features, raw_beats = {}, {}
    feature_sources: dict[str, pathlib.Path] = {}
    for track in TRACKS:
        raw_beats[track] = np.loadtxt(
            sources[f"beats_{track}"], usecols=0, ndmin=1)
        for condition in CONDITIONS:
            path = feature_path(args.features_dir, track, condition)
            write_features(args.binary, sources[f"{condition}_{track}"], path)
            features[(track, condition)] = read_features(path)
            feature_sources[f"features_{condition}_{track}"] = path

    # Paired degradation must use the same annotated events in both conditions.
    # Independent truncation gave 0116_goodies 213 clean beats and 217 room
    # beats, so that subtraction was not paired.
    beats, scoring_intervals = {}, {}
    for track in TRACKS:
        shared, start, end = shared_reference(
            raw_beats[track], [features[(track, c)] for c in CONDITIONS])
        if len(shared) < 8:
            raise RuntimeError(f"unscorable common interval: {track}")
        beats[track] = shared
        scoring_intervals[track] = {
            "start_sec": start, "end_sec": end, "beats": int(len(shared))}

    run_provenance = provenance(
        REPOSITORY, sources | feature_sources, **provenance_extra,
        scoring_intervals=scoring_intervals)

    # The control, which does not depend on any sweep parameter.
    dense: dict = {}
    for track in TRACKS:
        for condition in CONDITIONS:
            data = features[(track, condition)]
            for name, values in dense_signals(data.filterbank,
                                              data.difference).items():
                stats = score(values, data.times, beats[track], track,
                              scoring_intervals[track])
                if not stats:
                    raise RuntimeError(f"unscorable dense: {track} {condition}")
                dense[(name, track, condition)] = stats

    dense_degradation = {
        track: (dense[("dense_difference", track, "room")]["ratio"]
                - dense[("dense_difference", track, "clean")]["ratio"])
        for track in TRACKS}

    arms = {arm: run(features, beats, arm, scoring_intervals)
            for arm in ("causal", "symmetric")}

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
                **selection_record(label, rule),
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
        peaks_top = float(np.mean([r["peaks"]["room"]["top_n"] for r in rows]))
        dense_top = float(np.mean([r["dense"]["room"]["top_n"] for r in rows]))
        return {
            "tracks_improved": improved,
            "peaks_mean_degradation": peaks_mean,
            "dense_mean_degradation": dense_mean,
            # Both readings of condition 3. The strict per-track one binds; the
            # mean is reported beside it and never substituted for it, which is
            # the whole point of writing the disambiguation down.
            "top_n": {
                "peaks_mean": peaks_top,
                "dense_mean": dense_top,
                "strict_per_track_not_worse": strict,
                "mean_not_worse": peaks_top >= dense_top,
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

    # One judge for both, so the pooled verdict and the per-family ones
    # cannot drift apart. They did on the first pass: loto() was shared and
    # the scoring was not, so the pooled verdict silently lacked a field the
    # families carried.
    verdict = {arm: judge(arm, rows) for arm, rows in folds.items()}

    def flatten(table: dict) -> list[dict]:
        return [{**selection_record(label, rule), "track": track,
                 "condition": condition, **stats}
                for (label, rule, track, condition), stats in sorted(table.items())]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
    }
    staged = args.output.with_name(f".{args.output.name}.tmp")
    staged.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    staged.replace(args.output)

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
