"""Run the pre-registered M0a causal BarTracker oracle ladder.

All arms use ``tracking::BarTracker`` through dump_analysis's causal replay.
The old ``--beats --salience`` batch resolver is intentionally not used.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import wave
from collections import defaultdict

import numpy as np

from eval.causal_metre import annotated_metre, score_phase
from eval.harness import evaluate_downbeats
from eval.live_corpus_benchmark import (
    _without_local_paths,
    load_corpus,
    load_reference_beats,
    load_reference_downbeats,
)
from eval.octave_veto_replay import run, run_activation
from eval.provenance import digest, experiment_provenance
from eval.s0_reset import InvariantError, paired_bootstrap


ARMS = ("A1", "A2", "A3", "A4")
REQUIRED_CORPORA = frozenset({"gtzan", "harmonix"})
LOOKAHEAD_SEC = 0.05
BLOCK_SAMPLES = 512
SYNTHETIC_METERS = (3, 4, 6)


def oracle_channel(frame_times: np.ndarray, events: np.ndarray,
                   amplitude: float = 1.0) -> np.ndarray:
    out = np.zeros(len(frame_times), dtype=np.float64)
    if len(frame_times) == 0:
        return out
    for event in events:
        right = int(np.searchsorted(frame_times, event, side="left"))
        if right == 0:
            index = 0
        elif right == len(frame_times):
            index = len(frame_times) - 1
        else:
            # <= makes an exact tie select the earlier frame.
            index = right - 1 if event - frame_times[right - 1] <= frame_times[right] - event else right
        out[index] = max(out[index], amplitude)
    return out


def reference_emit(beats: np.ndarray, sample_rate: float) -> np.ndarray:
    """First device block whose ending clock puts a beat in 50 ms lookahead."""
    blocks = np.ceil((beats - LOOKAHEAD_SEC) * sample_rate / BLOCK_SAMPLES)
    blocks = np.maximum(blocks, 1.0)
    return np.maximum.accumulate(blocks).astype(np.float64)


def best_phase_events(beats: np.ndarray, meter: int,
                      reference_downbeats: np.ndarray) -> np.ndarray:
    if meter <= 0 or len(beats) == 0:
        return np.zeros(0, dtype=np.float64)
    scores = []
    for phase in range(meter):
        claimed = beats[np.arange(len(beats)) % meter == phase]
        score = evaluate_downbeats(reference_downbeats, claimed, trim=False)
        scores.append(score["downbeat_f_measure"])
    # np.argmax returns the earliest phase on a tie, fixing the oracle.
    phase = int(np.nanargmax(np.asarray(scores, dtype=np.float64)))
    return beats[np.arange(len(beats)) % meter == phase]


def visible_indices(all_beats: np.ndarray, visible: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    start = 0
    for beat in visible:
        found = np.flatnonzero(all_beats[start:] == beat)
        if len(found) == 0:
            raise InvariantError(
                "playable beat is absent from complete bar grid")
        index = start + int(found[0])
        indices.append(index)
        start = index + 1
    return np.asarray(indices, dtype=np.int64)


def fixed_meter_label(item: dict) -> str:
    label = str(item.get("meter") or "").strip()
    if not label:
        raise RuntimeError(
            f"{item['name']}: manifest does not identify a fixed meter")
    return label


def _write(path: pathlib.Path, values: np.ndarray, fmt: str = "%.17g") -> None:
    np.savetxt(path, np.asarray(values, dtype=np.float64), fmt=fmt)


def replay_bar(binary: pathlib.Path, audio: pathlib.Path, beat_activation: np.ndarray,
               downbeat_activation: np.ndarray, frame_emit: np.ndarray,
               frame_times: np.ndarray, beats: np.ndarray,
               beat_emit: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        paths = {name: root / f"{name}.txt" for name in
                 ("beat", "downbeat", "frame_emit", "frame_times",
                  "beats", "beat_emit")}
        _write(paths["beat"], beat_activation)
        _write(paths["downbeat"], downbeat_activation)
        _write(paths["frame_emit"], frame_emit, "%.0f")
        _write(paths["frame_times"], frame_times)
        _write(paths["beats"], beats)
        _write(paths["beat_emit"], beat_emit, "%.0f")
        return run_activation(
            binary, audio, paths["beat"],
            extra=["--live-bars", "--live-downbeat", str(paths["downbeat"]),
                   "--bar-replay-beats", str(paths["beats"]),
                   "--bar-replay-emit", str(paths["beat_emit"])],
            emit_path=paths["frame_emit"], times_path=paths["frame_times"])


def synthetic_preflight(binary: pathlib.Path) -> dict:
    """Prove the causal C++ seam recovers planted fixed meter and phase."""
    sample_rate = 48000.0
    duration_sec = 32
    frame_times = np.arange(0.0, duration_sec, 0.02, dtype=np.float64)
    frame_emit = np.ceil(
        (frame_times + 0.064) * sample_rate / BLOCK_SAMPLES)
    frame_emit = np.maximum(frame_emit, 1.0)
    beat_activation = np.zeros(len(frame_times), dtype=np.float64)
    beats = np.arange(1.0, duration_sec - 1.0, 0.5, dtype=np.float64)
    beat_emit = reference_emit(beats, sample_rate)

    with tempfile.TemporaryDirectory() as directory:
        audio = pathlib.Path(directory) / "planted_phase.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            silence = b"\0\0" * int(sample_rate)
            for _ in range(duration_sec):
                handle.writeframesraw(silence)

        fixtures = {}
        for meter in SYNTHETIC_METERS:
            downbeats = beats[1::meter]
            payload = replay_bar(
                binary, audio, beat_activation,
                oracle_channel(frame_times, downbeats), frame_emit,
                frame_times, beats, beat_emit)
            positions = np.asarray(payload["bar_replay_positions"], dtype=np.float64)
            meters = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
            phase = score_phase(beats, positions, meters, downbeats)
            f1 = phase["f1"]
            final_meter = int(meters[meters > 0][-1]) if np.any(meters > 0) else 0
            if f1 is None or f1 < 0.90 or final_meter != meter:
                raise InvariantError(
                    f"synthetic {meter}-beat phase failed: "
                    f"final_meter={final_meter}, f1={f1}")
            fixtures[str(meter)] = {"final_meter": final_meter,
                                    "phase_f1": float(f1)}
    return {"passed": True, "fixtures": fixtures}


def _phase_with_null(beats: np.ndarray, positions: np.ndarray,
                     meters: np.ndarray, reference_downbeats: np.ndarray,
                     start_sec: float) -> dict:
    actual = score_phase(beats, positions, meters, reference_downbeats,
                         start_sec=start_sec)
    answered = actual["f1"] is not None
    if not answered:
        actual = {**actual, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                  "phase_correct_share": 0.0}
    decided = meters[meters > 0]
    final = int(decided[-1]) if len(decided) else 0
    rotations = [score_phase(beats, positions, meters, reference_downbeats,
                             shift=shift, start_sec=start_sec)
                 for shift in range(final)]
    null_values = [r["f1"] for r in rotations if r["f1"] is not None]
    eligible = (positions >= 0) & (meters > 0)
    before = beats < start_sec
    after = ~before
    return {"actual": actual, "answered": answered,
            "first_decision_sec": (float(beats[np.flatnonzero(meters > 0)[0]])
                                   if np.any(meters > 0) else None),
            "coverage": {
                "decided_before_common": int(np.sum(eligible & before)),
                "beats_before_common": int(np.sum(before)),
                "decided_after_common": int(np.sum(eligible & after)),
                "beats_after_common": int(np.sum(after)),
            },
            "random_phase_f1": (float(np.mean(null_values))
                                 if null_values else 0.0)}


def measure_one(item: dict, binary: pathlib.Path, model: pathlib.Path) -> dict:
    meter_label = fixed_meter_label(item)
    reference = load_reference_beats(item["annotation"])
    reference_downbeats = load_reference_downbeats(item["annotation"])
    meter = annotated_metre(reference, reference_downbeats)
    if meter <= 0:
        raise RuntimeError(f"{item['name']}: no fixed annotated meter")

    initial = run(binary, item["audio"], model, extra=["--live-bars"])
    frame_times = np.asarray(initial["activation_times"], dtype=np.float64)
    frame_emit = np.asarray(initial["activation_emit"], dtype=np.float64)
    beat_activation = np.asarray(initial["activation_beat"], dtype=np.float64)
    predicted_downbeat = np.asarray(initial["activation_downbeat"], dtype=np.float64)
    predicted_all = np.asarray(initial["live_bar_beats_all"], dtype=np.float64)
    predicted_emit = np.asarray(initial["live_bar_emit_all"], dtype=np.float64)
    predicted_visible = np.asarray(initial["beats"], dtype=np.float64)
    predicted_visible_index = visible_indices(predicted_all, predicted_visible)
    ref_emit = reference_emit(reference, float(initial["sample_rate"]))

    oracle_reference = oracle_channel(frame_times, reference_downbeats)
    a3_events = best_phase_events(predicted_all, meter, reference_downbeats)
    oracle_predicted = oracle_channel(frame_times, a3_events)
    definitions = {
        "A1": (reference, ref_emit, oracle_reference, None),
        "A2": (reference, ref_emit, predicted_downbeat, None),
        "A3": (predicted_all, predicted_emit, oracle_predicted,
               predicted_visible_index),
        "A4": (predicted_all, predicted_emit, predicted_downbeat,
               predicted_visible_index),
    }

    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for arm, (grid, emit, channel, select) in definitions.items():
        payload = replay_bar(binary, item["audio"], beat_activation, channel,
                             frame_emit, frame_times, grid, emit)
        positions = np.asarray(payload["bar_replay_positions"], dtype=np.float64)
        meters = np.asarray(payload["bar_replay_meters"], dtype=np.float64)
        if len(positions) != len(grid) or len(meters) != len(grid):
            raise InvariantError(
                f"{item['name']} {arm}: incomplete bar replay")
        if select is not None:
            raw[arm] = (grid[select], positions[select], meters[select])
        else:
            raw[arm] = (grid, positions, meters)
        if arm == "A4":
            if not (np.array_equal(positions,
                                   np.asarray(initial["live_bar_positions_all"]))
                    and np.array_equal(meters,
                                       np.asarray(initial["live_bar_meters_all"]))):
                raise InvariantError(
                    f"{item['name']}: A4 causal parity failed")

    first_decisions = [beats[np.flatnonzero(meters > 0)[0]]
                       for beats, _, meters in raw.values()
                       if np.any(meters > 0)]
    common_start = float(max(first_decisions)) if first_decisions else 0.0
    arms = {arm: _phase_with_null(*raw[arm], reference_downbeats, common_start)
            for arm in ARMS}

    # Registered A1 format sensitivities use the same reference grid and clock.
    jittered = reference_downbeats + np.where(
        np.arange(len(reference_downbeats)) % 2 == 0, -0.02, 0.02)
    sensitivity = {}
    for name, channel in (
        ("amplitude_0_5", oracle_channel(frame_times, reference_downbeats, 0.5)),
        ("alternating_jitter_20ms", oracle_channel(frame_times, jittered)),
    ):
        payload = replay_bar(binary, item["audio"], beat_activation, channel,
                             frame_emit, frame_times, reference, ref_emit)
        sensitivity[name] = _phase_with_null(
            reference,
            np.asarray(payload["bar_replay_positions"], dtype=np.float64),
            np.asarray(payload["bar_replay_meters"], dtype=np.float64),
            reference_downbeats, common_start)["actual"]["f1"]

    return {
        "name": item["name"], "corpus": item["corpus"],
        "annotation": digest(item["annotation"]),
        "annotation_meter_label": meter_label, "annotated_meter": meter,
        "common_start_sec": common_start, "arms": arms,
        "A1_sensitivity": sensitivity,
        "invariants": {"A4_causal_parity": True,
                       "A1_A2_grid_identical": True,
                       "A3_A4_grid_identical": True},
    }


def summarise(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["corpus"]].append(record)
    out = {}
    for corpus, rows in sorted(grouped.items()):
        means = {arm: float(np.mean([
            r["arms"][arm]["actual"]["f1"] for r in rows])) for arm in ARMS}
        delta = np.asarray([
            r["arms"]["A1"]["actual"]["f1"]
            - r["arms"]["A4"]["actual"]["f1"] for r in rows])
        sensitivity = max(
            abs(float(np.mean([r["A1_sensitivity"][kind] for r in rows]))
                - means["A1"])
            for kind in ("amplitude_0_5", "alternating_jitter_20ms"))
        complete = [r for r in rows
                    if all(r["arms"][arm]["answered"] for arm in ARMS)]
        complete_delta = np.asarray([
            r["arms"]["A1"]["actual"]["f1"]
            - r["arms"]["A4"]["actual"]["f1"] for r in complete])
        out[corpus] = {
            "records": len(rows), "phase_f1": means,
            "A1-A4": {"mean": float(np.mean(delta)),
                       "ci": paired_bootstrap(delta)},
            "A1_random_phase_f1": float(np.mean([
                r["arms"]["A1"]["random_phase_f1"] for r in rows])),
            "A1_max_format_sensitivity": sensitivity,
            "format_sensitivity_withholds_verdict": sensitivity > 0.05,
            "complete_case_sensitivity": {
                "records": len(complete),
                "phase_f1": ({arm: float(np.mean([
                    r["arms"][arm]["actual"]["f1"] for r in complete]))
                              for arm in ARMS} if complete else None),
                "A1-A4": ({"mean": float(np.mean(complete_delta)),
                            "ci": paired_bootstrap(complete_delta)}
                           if complete else None),
            },
        }
    if out:
        present = frozenset(out)
        required_present = REQUIRED_CORPORA.issubset(present)
        blocks = [out[name] for name in sorted(REQUIRED_CORPORA & present)]
        withheld = any(b["format_sensitivity_withholds_verdict"] for b in blocks)
        null_pass = all(
            b["phase_f1"]["A1"] - b["A1_random_phase_f1"] >= 0.20
            for b in blocks)
        high_a4_guard = any(b["phase_f1"]["A4"] > 0.80 for b in blocks)
        hard_negative = (required_present
                         and any(b["A1-A4"]["ci"][1] < 0.20 for b in blocks)
                         and null_pass and not high_a4_guard and not withheld)
        band3 = (required_present
                 and all(b["A1-A4"]["ci"][0] >= 0.20 for b in blocks)
                 and all(b["phase_f1"]["A1"] >= 0.90 for b in blocks)
                 and null_pass and not withheld)
        out["decision"] = {
            "band": ("band1_hard_negative" if hard_negative else
                     "band3_decoder_not_falsified" if band3 else
                     "band2_inconclusive"),
            "format_sensitivity_withheld": withheld,
            "A1_null_precondition": null_pass,
            "high_A4_guard": high_a4_guard,
            "required_corpora_present": required_present,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=repository / "music" / "ground-truth" / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path, default=repository / "music")
    parser.add_argument("--corpora", nargs="+", default=["gtzan", "harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    items = load_corpus(args.manifest, args.music, False,
                        frozenset(args.corpora))
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("no recordings")
    provenance = experiment_provenance(
        repository, files={"binary": args.binary, "model": args.model,
                           "manifest": args.manifest},
        experiment="M0a", arms=list(ARMS), bootstrap_draws=2000)
    synthetic = synthetic_preflight(args.binary)
    records = []
    exclusions = []
    for item in items:
        try:
            records.append(measure_one(item, args.binary, args.model))
        except InvariantError:
            raise
        except Exception as error:
            exclusions.append({"name": item["name"], "corpus": item["corpus"],
                               "error_type": type(error).__name__,
                               "reason": _without_local_paths(str(error)),
                               "annotation": digest(item.get("annotation"))})
    artifact = {"provenance": provenance,
                "synthetic_preflight": synthetic,
                "selected": len(items), "scored": len(records),
                "technical_exclusions": exclusions, "records": records,
                "summary": summarise(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
