#!/usr/bin/env python3
"""Beat This! as the observation: in a room, and through the shipped decoder.

Registered in `PREREGISTERED_beat_this_front_end.md`. Two questions, one run,
because they share every mechanism.

**In a room.** The same four statistics `room_activation.py` computed on
BeatNet's beat channel, computed on Beat This!'s, for the same five captures and
the same five clean files. Whether the collapse found there is a fact about
rooms or a fact about BeatNet has never been asked, and it is answerable with
the audio already on disk.

**Through the same decoder.** The causal sweep compared Beat This! decoded by
its own peak picker against BeatNet decoded by `LiveTracker`, so its +0.102
changed the model and the decoder at once. Here the activation goes through
`--live-activation` into the same `LiveTracker`, and the only thing that differs
from the shipped path is the observation.

## The two numbers have different limitations

The room question is **immune** to `final0` having trained on this material: it
compares one model with itself on clean and room versions of one recording, and
memorising the clean file would enlarge the drop rather than hide it.

The matched GTZAN question is held-out for both `final0` and BeatNet `model_1`.
It is nevertheless a system comparison rather than an architectural one, and a
bidirectional model fed through the replay seam without a recorded release
schedule is observed on an analytic availability delay. Its result is a clean,
short-excerpt bound on what causal replay could give, not a room-domain estimate.
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
from eval.beat_this_onnx import FPS, BeatThisOnnx  # noqa: E402
from eval.live_corpus_benchmark import (_score_one, load_corpus,  # noqa: E402
                                        load_reference_beats)
from eval.provenance import experiment_provenance as provenance  # noqa: E402
from eval.room_activation import (auc, between_beat_ratio,  # noqa: E402
                                  half_height_width, salience_and_floor)

SAMPLE_HZ = 50.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260809


def activation_of(session: BeatThisOnnx, audio: pathlib.Path
                  ) -> tuple[np.ndarray, np.ndarray]:
    import soundfile

    samples, rate = soundfile.read(str(audio), dtype="float64", always_2d=True)
    activations = session.activations(samples.mean(axis=1), float(rate))
    beat = activations.beat_probability()
    return np.arange(len(beat)) / FPS, beat


def through_tracker(binary: pathlib.Path, audio: pathlib.Path,
                    activation: np.ndarray) -> dict:
    """The shipped live decoder, driven by this activation instead of its own."""
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "beat.txt"
        np.savetxt(path, activation, fmt="%.17g")
        done = subprocess.run(
            [str(binary), str(audio), "--live-activation", str(path),
             "--activation-fps", repr(float(FPS)),
             "--live-sample-hz", repr(SAMPLE_HZ)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{audio.name}: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout)


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


def statistics(times: np.ndarray, activation: np.ndarray, beats: np.ndarray
               ) -> dict:
    inside = beats[beats <= times[-1]] if len(times) else beats
    salience, floor = salience_and_floor(times, activation, inside)
    return {
        "frames": int(len(times)),
        "auc": auc(salience, floor),
        "mean_salience": float(np.mean(salience)) if len(salience) else float("nan"),
        "median_floor": float(np.median(floor)) if len(floor) else float("nan"),
        "half_height_width_sec": half_height_width(times, activation, inside),
        "between_beat_ratio": between_beat_ratio(times, activation, inside),
    }


def score(item: dict, payload: dict, binary: pathlib.Path, model: pathlib.Path
          ) -> dict:
    scored = _score_one(item, "model", binary, model,
                        estimate=Estimate.from_json(payload))
    return {"f_measure": scored.get("f_measure"), "p70": scored.get("p70"),
            "r70": scored.get("r70"),
            "usable": bool(scored.get("usable", False)),
            "reasons": list(scored.get("reasons", []))}


def room_pass(session: BeatThisOnnx, items: dict, aligned: pathlib.Path,
              binary: pathlib.Path, model: pathlib.Path) -> list[dict]:
    records = []
    for path in sorted(aligned.glob("*.wav")):
        item = items.get(path.stem)
        if item is None:
            continue
        print(f"  room {path.stem}", file=sys.stderr, flush=True)
        beats = load_reference_beats(item["annotation"])
        row: dict = {"name": item["name"]}
        for arm, audio in (("clean", pathlib.Path(item["audio"])),
                           ("room", path)):
            times, activation = activation_of(session, audio)
            row[arm] = statistics(times, activation, beats)
            row[arm]["through_tracker"] = score(
                item, through_tracker(binary, audio, activation), binary, model)
        row["auc_drop"] = row["clean"]["auc"] - row["room"]["auc"]
        records.append(row)
    return records


def corpus_pass(session: BeatThisOnnx, items: list[dict], binary: pathlib.Path,
                model: pathlib.Path) -> list[dict]:
    """Both front ends, one decoder, same audio: the model's share alone."""
    records = []
    for index, item in enumerate(items):
        if index % 10 == 0:
            print(f"  corpus {index}/{len(items)}", file=sys.stderr, flush=True)
        audio = pathlib.Path(item["audio"])
        try:
            _, activation = activation_of(session, audio)
            row = {
                "name": item["name"], "corpus": item["corpus"],
                "beatnet": score(item, run_live(binary, audio, model), binary, model),
                "beat_this": score(item, through_tracker(binary, audio, activation),
                                   binary, model),
            }
        except Exception as error:  # noqa: BLE001
            records.append({"name": item["name"], "corpus": item["corpus"],
                            "error": str(error)[:200]})
            continue
        row["delta_f"] = row["beat_this"]["f_measure"] - row["beatnet"]["f_measure"]
        records.append(row)
    return records


def paired_bootstrap(records: list[dict], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired percentile CI, resampling recordings/compositions as clusters."""
    values = np.asarray([row["delta_f"] for row in records
                         if "delta_f" in row], dtype=np.float64)
    if not len(values):
        raise ValueError("paired bootstrap requires at least one scored recording")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    # Batch the index matrix so a full 999-track run stays comfortably small.
    batch = 256
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "unit": "recording/composition",
        "method": "paired percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
        "mean_delta_f": float(values.mean()),
        "ci95": [float(low), float(high)],
    }


def corpus_summary(rows: list[dict]) -> dict:
    good = [row for row in rows if "delta_f" in row]
    if not good:
        raise ValueError("corpus pass produced no scored recordings")
    return {
        "n": len(good),
        "failures": [row for row in rows if "delta_f" not in row],
        "beatnet_mean_f": float(np.mean(
            [row["beatnet"]["f_measure"] for row in good])),
        "beat_this_mean_f": float(np.mean(
            [row["beat_this"]["f_measure"] for row in good])),
        "mean_delta_f": float(np.mean([row["delta_f"] for row in good])),
        "beatnet_usable": float(np.mean(
            [row["beatnet"]["usable"] for row in good])),
        "beat_this_usable": float(np.mean(
            [row["beat_this"]["usable"] for row in good])),
        "paired_bootstrap": paired_bootstrap(good),
    }


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--beat-this", type=pathlib.Path, required=True)
    parser.add_argument("--aligned", type=pathlib.Path,
                        help="room captures; omit to run the corpus pass only")
    parser.add_argument("--corpus-limit", type=int, default=0,
                        help="diagnostic stride limit; 0 runs no corpus pass")
    parser.add_argument("--full-corpus", action="store_true",
                        help="run every available recording (P0 acceptance path)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if args.full_corpus and args.corpus_limit:
        parser.error("--full-corpus and --corpus-limit are mutually exclusive")
    if args.corpus_limit < 0:
        parser.error("--corpus-limit cannot be negative")
    try:
        args.output.resolve().relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        parser.error("--output must be outside the evaluation worktree")

    run_provenance = provenance(
        repository,
        {"manifest": args.manifest, "binary": args.binary,
         "model": args.model, "beat_this": args.beat_this},
    )

    loaded = load_corpus(args.manifest, args.music, False, frozenset(args.corpora))
    seen = sorted({i["corpus"] for i in loaded})
    if not loaded or seen != sorted(set(args.corpora)):
        print(f"asked for {sorted(set(args.corpora))}, loaded {seen}",
              file=sys.stderr)
        return 1
    items = {i["name"]: i for i in loaded}

    session = BeatThisOnnx(args.beat_this)

    payload: dict = {"provenance": run_provenance,
                     "corpora": args.corpora}
    if args.aligned:
        payload["room"] = room_pass(session, items, args.aligned, args.binary,
                                    args.model)
    if args.full_corpus or args.corpus_limit:
        chosen = loaded
        selection = {
            "available": len(loaded),
            "mode": "full" if args.full_corpus else "diagnostic_stride",
            "requested_limit": None if args.full_corpus else args.corpus_limit,
            "stride": 1,
            "excluded": [],
        }
        if args.corpus_limit and len(chosen) > args.corpus_limit:
            stride = -(-len(chosen) // args.corpus_limit)
            chosen = chosen[::stride]
            kept = {(item["corpus"], item["name"]) for item in chosen}
            selection["stride"] = stride
            selection["excluded"] = [
                {"corpus": item["corpus"], "name": item["name"]}
                for item in loaded
                if (item["corpus"], item["name"]) not in kept
            ]
        selection["selected"] = len(chosen)
        rows = corpus_pass(session, chosen, args.binary, args.model)
        summary = corpus_summary(rows)
        summary["selection"] = selection
        summary["by_corpus"] = {
            corpus: corpus_summary([row for row in rows
                                    if row.get("corpus") == corpus])
            for corpus in sorted({row.get("corpus") for row in rows
                                  if row.get("corpus")})
        }
        summary["records"] = rows
        payload["corpus"] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_name(f".{args.output.name}.tmp")
    staged.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    staged.replace(args.output)

    for row in payload.get("room", []):
        print(f"\n{row['name']}")
        for arm in ("clean", "room"):
            a = row[arm]
            print(f"  {arm:5s} AUC {a['auc']:.3f}  salience {a['mean_salience']:.3f}"
                  f"  floor {a['median_floor']:.4f}  width "
                  f"{a['half_height_width_sec']:.3f}s  between "
                  f"{a['between_beat_ratio']:.3f}   F(tracker) "
                  f"{a['through_tracker']['f_measure']:.3f}")
    if "corpus" in payload:
        summary = {k: v for k, v in payload["corpus"].items() if k != "records"}
        summary["failures"] = len(summary["failures"])
        print("\n" + json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
