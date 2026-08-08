#!/usr/bin/env python3
"""Can the room damage be taken back out, and from where?

Registered in `PREREGISTERED_room_repair.md`. Six arms on the five aligned
captures: three that transform the audio before BeatNet hears it, and three
that transform the activation after BeatNet has produced it and before
`LiveTracker` sees it. The split is what makes the run answer *where* a repair
belongs rather than only whether one exists.

Every constant is fixed in the registration and none is swept. They come from
the mechanism the diagnosis measured -- a floor risen fifty to three hundred
fold and a tail that fills the gaps -- rather than from trying values on these
five recordings, which are also the recordings the answer is read on.

The parity precondition is not ceremony. An activation replayed with values,
release order or timestamps merely close reproduces nothing: 0 of 20 recordings
on an earlier experiment. If the untouched replay does not reproduce the
baseline, every activation-side arm is measuring the replay.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.analysis import Estimate  # noqa: E402
from eval.live_corpus_benchmark import _score_one, load_corpus  # noqa: E402
from eval.octave_veto_experiment import write_activation_cache  # noqa: E402
from eval.octave_veto_replay import run_activation  # noqa: E402

SAMPLE_HZ = 50.0
ACTIVATION_HZ = 50.0

# STFT for the audio arms. 46 ms at 44.1 kHz with 75% overlap: long enough to
# resolve a reverb tail's decay per band, short enough not to smear the onset
# the whole exercise is trying to protect.
FFT = 2048
HOP = 512

# Registered constants, none swept.
GATE_PERCENTILE = 10.0      # the band's own quiet level
GATE_OVERSUBTRACT = 1.5
SPECTRAL_FLOOR = 0.05       # never take a band below this share of itself
DEREVERB_RT60 = 0.5         # seconds
DEREVERB_STRENGTH = 1.0
ACT_WINDOW_SEC = 2.0        # running statistic for the activation arms
ACT_PERCENTILE = 95.0
ACT_SHARPEN_TAU = 0.3
ACT_SHARPEN_STRENGTH = 1.0

# The control that already works in a room, and the floor it may not go under.
SURVIVOR = "0116_goodies"
SURVIVOR_FLOOR = 0.95


# --------------------------------------------------------------------------
# audio-side transforms


def stft(mono: np.ndarray) -> np.ndarray:
    from scipy.signal import stft as _stft
    _, _, spectrum = _stft(mono, nperseg=FFT, noverlap=FFT - HOP,
                           boundary="zeros", padded=True)
    return spectrum


def istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    from scipy.signal import istft as _istft
    _, out = _istft(spectrum, nperseg=FFT, noverlap=FFT - HOP,
                    boundary=True)
    # Exactly the original length: the annotations were not moved, and a file
    # a few samples longer or shorter is a file whose beats are at different
    # times than the ones being scored against.
    if len(out) < length:
        out = np.pad(out, (0, length - len(out)))
    return out[:length]


def spectral_gate(spectrum: np.ndarray) -> np.ndarray:
    """Subtract each band's own quiet level. Aimed at the risen floor."""
    magnitude = np.abs(spectrum)
    floor = np.percentile(magnitude, GATE_PERCENTILE, axis=1, keepdims=True)
    kept = np.maximum(magnitude - GATE_OVERSUBTRACT * floor,
                      SPECTRAL_FLOOR * magnitude)
    # Phase untouched: the ear is not listening and the model reads magnitude
    # features, so reconstructing with the original phase is both correct here
    # and the only thing that keeps the transform invertible without artefacts
    # of its own.
    return kept * np.exp(1j * np.angle(spectrum))


def dereverb(spectrum: np.ndarray) -> np.ndarray:
    """Subtract a decaying trace of each band's past. Aimed at the tail.

    `trace[t] = max(trace[t-1] * decay, magnitude[t-1])` is a running envelope
    that falls at the room's rate, so what it subtracts at time t is what the
    previous onsets would still be contributing if they decayed at RT60.
    """
    magnitude = np.abs(spectrum)
    frame_sec = HOP / 44100.0
    decay = float(np.power(10.0, -3.0 * frame_sec / DEREVERB_RT60))
    trace = np.zeros_like(magnitude)
    for frame in range(1, magnitude.shape[1]):
        trace[:, frame] = np.maximum(trace[:, frame - 1] * decay,
                                     magnitude[:, frame - 1] * decay)
    kept = np.maximum(magnitude - DEREVERB_STRENGTH * trace,
                      SPECTRAL_FLOOR * magnitude)
    return kept * np.exp(1j * np.angle(spectrum))


AUDIO_ARMS = {
    "audio_gate": (spectral_gate,),
    "audio_dereverb": (dereverb,),
    "audio_both": (spectral_gate, dereverb),
}


# --------------------------------------------------------------------------
# activation-side transforms


def running_statistic(values: np.ndarray, window: int, percentile: float
                      ) -> np.ndarray:
    """A percentile over a centred sliding window, without an O(n*w) loop."""
    if window >= len(values):
        return np.full(len(values), float(np.percentile(values, percentile)))
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.percentile(view, percentile, axis=1)


def act_subtract_floor(activation: np.ndarray) -> np.ndarray:
    window = max(3, int(ACT_WINDOW_SEC * ACTIVATION_HZ))
    return np.maximum(activation - running_statistic(activation, window, 50.0),
                      0.0)


def act_normalise(activation: np.ndarray) -> np.ndarray:
    window = max(3, int(ACT_WINDOW_SEC * ACTIVATION_HZ))
    scale = running_statistic(activation, window, ACT_PERCENTILE)
    return np.clip(activation / np.maximum(scale, 1e-6), 0.0, 1.0)


def act_sharpen(activation: np.ndarray) -> np.ndarray:
    decay = float(np.exp(-1.0 / (ACT_SHARPEN_TAU * ACTIVATION_HZ)))
    trace = np.zeros_like(activation)
    for index in range(1, len(activation)):
        trace[index] = max(trace[index - 1] * decay, activation[index - 1] * decay)
    return np.maximum(activation - ACT_SHARPEN_STRENGTH * trace, 0.0)


ACTIVATION_ARMS = {
    "act_subtract_floor": act_subtract_floor,
    "act_normalise": act_normalise,
    "act_sharpen": act_sharpen,
}


# --------------------------------------------------------------------------


def run_live(binary: pathlib.Path, audio: pathlib.Path, model: pathlib.Path
             ) -> dict:
    done = subprocess.run(
        [str(binary), str(audio), "--live", "--live-model", str(model),
         "--live-sample-hz", repr(SAMPLE_HZ)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{audio.name}: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout)


def dump(binary: pathlib.Path, audio: pathlib.Path, model: pathlib.Path) -> dict:
    done = subprocess.run(
        [str(binary), str(audio), "--dump-activation", "--live-model", str(model)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{audio.name}: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout)


def score(item: dict, payload: dict, binary: pathlib.Path, model: pathlib.Path
          ) -> dict:
    scored = _score_one(item, "model", binary, model,
                        estimate=Estimate.from_json(payload))
    return {"f_measure": scored.get("f_measure"), "p70": scored.get("p70"),
            "r70": scored.get("r70"), "usable": bool(scored.get("usable", False)),
            "reasons": list(scored.get("reasons", [])),
            "beats": int(len(scored.get("live_times", []) or []))
            or int(payload.get("live_beats", 0))}


def measure_one(item: dict, aligned: pathlib.Path, binary: pathlib.Path,
                model: pathlib.Path) -> dict:
    import soundfile

    out: dict = {"name": item["name"]}
    baseline_payload = run_live(binary, aligned, model)
    out["baseline"] = score(item, baseline_payload, binary, model)
    baseline_beats = np.asarray(baseline_payload.get("live_times", []),
                                dtype=np.float64)

    audio, rate = soundfile.read(str(aligned), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float64)

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)

        for arm, steps in AUDIO_ARMS.items():
            spectrum = stft(mono)
            for step in steps:
                spectrum = step(spectrum)
            repaired = istft(spectrum, len(mono))
            peak = float(np.max(np.abs(repaired)))
            if peak > 1.0:
                repaired = repaired / peak * 0.98
            path = root / f"{arm}.wav"
            soundfile.write(str(path), repaired.astype(np.float32), int(rate))
            out[arm] = score(item, run_live(binary, path, model), binary, model)

        dumped = dump(binary, aligned, model)
        beat = np.asarray(dumped["activation_beat"], dtype=np.float64)

        # Parity first: the same numbers back through the replay path have to
        # give the same beats, or the arms below are measuring the replay.
        activation_path = root / "beat.txt"
        emit_path = root / "emit.txt"
        times_path = root / "times.txt"
        write_activation_cache(dumped, activation_path, emit_path, times_path)
        replayed = run_activation(binary, aligned, activation_path,
                                  sample_hz=SAMPLE_HZ, emit_path=emit_path,
                                  times_path=times_path)
        replay_beats = np.asarray(replayed.get("live_times", []),
                                  dtype=np.float64)
        same = (len(replay_beats) == len(baseline_beats)
                and (len(baseline_beats) == 0
                     or float(np.max(np.abs(replay_beats - baseline_beats))) == 0.0))
        out["parity"] = {
            "ok": bool(same),
            "baseline_beats": int(len(baseline_beats)),
            "replay_beats": int(len(replay_beats)),
            "max_abs_diff": (float(np.max(np.abs(
                replay_beats[:min(len(replay_beats), len(baseline_beats))]
                - baseline_beats[:min(len(replay_beats), len(baseline_beats))])))
                if len(replay_beats) and len(baseline_beats) else float("nan")),
        }
        out["replay"] = score(item, replayed, binary, model)

        for arm, transform in ACTIVATION_ARMS.items():
            repaired = transform(beat.copy())
            path = root / f"{arm}.txt"
            np.savetxt(path, repaired, fmt="%.17g")
            payload = run_activation(binary, aligned, path, sample_hz=SAMPLE_HZ,
                                     emit_path=emit_path, times_path=times_path)
            out[arm] = score(item, payload, binary, model)
    return out


def summarise(records: list[dict], ceiling: dict) -> dict:
    arms = ["baseline", "replay", *AUDIO_ARMS, *ACTIVATION_ARMS]
    base = float(np.mean([r["baseline"]["f_measure"] for r in records]))
    top = float(np.mean([ceiling[r["name"]] for r in records]))
    out: dict = {"n": len(records), "baseline_mean_f": base,
                 "ceiling_mean_f": top, "half_gap_target": (base + top) / 2.0,
                 "arms": {}}
    for arm in arms:
        rows = [r[arm] for r in records if arm in r]
        if not rows:
            continue
        survivor = next((r[arm]["f_measure"] for r in records
                         if r["name"] == SURVIVOR and arm in r), float("nan"))
        mean_f = float(np.mean([x["f_measure"] for x in rows]))
        out["arms"][arm] = {
            "mean_f": mean_f,
            "delta_vs_baseline": mean_f - base,
            "share_of_gap": ((mean_f - base) / (top - base)) if top > base else 0.0,
            "usable_rate": float(np.mean([x["usable"] for x in rows])),
            "survivor_f": survivor,
            # Registered as a disqualification, not a note: a mean lifted by
            # wrecking the recording that already works is an average hiding a
            # regression.
            "disqualified": bool(survivor < SURVIVOR_FLOOR),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", default=["harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--aligned", type=pathlib.Path, required=True)
    parser.add_argument("--scores", type=pathlib.Path, required=True,
                        help="room_recording_phone.json, for the clean ceiling")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=repository).stdout.strip()
    clean_tree = not subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    cwd=repository).stdout.strip()

    items = {i["name"]: i for i in
             load_corpus(args.manifest, args.music, False, frozenset(args.corpora))}
    seen = sorted({i["corpus"] for i in items.values()})
    if not items or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1

    scored = json.loads(args.scores.read_text(encoding="utf-8"))
    ceiling = {r["name"]: r["clean"]["f_measure"] for r in scored["records"]
               if "clean" in r}

    records, failures = [], []
    for path in sorted(args.aligned.glob("*.wav")):
        item = items.get(path.stem)
        if item is None or path.stem not in ceiling:
            failures.append({"name": path.stem, "error": "no corpus entry or ceiling"})
            continue
        print(f"  {path.stem}", file=sys.stderr, flush=True)
        try:
            records.append(measure_one(item, path, args.binary, args.model))
        except Exception as error:  # noqa: BLE001
            failures.append({"name": path.stem, "error": str(error)[:300]})

    payload = {"commit": commit, "clean": clean_tree, "failures": failures,
               "parity_ok": all(r["parity"]["ok"] for r in records) if records else False,
               "summary": summarise(records, ceiling) if records else {},
               "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not payload["parity_ok"]:
        print("\nPARITY FAILED -- the activation arms measure the replay, "
              "not the repair. See records[].parity.", file=sys.stderr)
    print(json.dumps(payload["summary"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
