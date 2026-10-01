"""What the bench's click gains mean against each recording's own level.

Post hoc, for `PREREGISTERED_click_gate_and_antialias.md` and its loud-ungated
addendum. `dump_analysis --live-click-db` scales the click's nominal amplitude;
nothing measures the music. This measures it. Per recording, it compares:

* the click's power averaged over one annotated beat period against the
  music's power over the whole file, mono, as the bench feeds it;
* the click's loudest 50 ms against the music's 99th-percentile 50 ms RMS.

Both figures are for the 0 dB arm; a -12 dB arm is 12 dB lower.

    cd research
    .venv/Scripts/python -m eval.click_music_ratio --music ../music         --output results/click_gate_and_antialias/click_music_ratio.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.live_corpus_benchmark import load_corpus, load_reference_beats  # noqa: E402

# ClickConfig::beat at 0 dB: a sine falling 60 dB over its length.
GAIN, LENGTH = 0.75, 0.060
TAU = LENGTH / math.log(1000.0)
CLICK_ENERGY = GAIN ** 2 * TAU / 4.0 * (1.0 - math.exp(-2.0 * LENGTH / TAU))
WINDOW_SEC = 0.05


def music_levels(path: pathlib.Path) -> tuple[float, float]:
    """Mean square over the file, and the 99th percentile of 50 ms RMS."""
    import soundfile

    total, count, windows = 0.0, 0, []
    with soundfile.SoundFile(str(path)) as handle:
        hop = int(WINDOW_SEC * handle.samplerate)
        for block in handle.blocks(blocksize=hop * 200, dtype="float32", always_2d=True):
            mono = block.mean(axis=1).astype(np.float64)
            total += float(np.dot(mono, mono))
            count += mono.size
            whole = mono.size // hop
            if whole:
                windows.append(np.sqrt((mono[:whole * hop].reshape(whole, hop) ** 2).mean(axis=1)))
    loud = float(np.percentile(np.concatenate(windows), 99)) if windows else 0.0
    return total / max(count, 1), loud


def ratios(item: dict) -> dict | None:
    beats = load_reference_beats(pathlib.Path(item["annotation"]))
    if len(beats) < 3:
        return None
    period = float(np.median(np.diff(beats)))
    mean_square, loud = music_levels(pathlib.Path(item["audio"]))
    if mean_square <= 0.0 or loud <= 0.0:
        return None
    return {"corpus": item["corpus"], "name": item["name"],
            "music_rms_dbfs": 10.0 * math.log10(mean_square),
            "period_sec": period,
            "click_to_music_beat_db": 10.0 * math.log10(CLICK_ENERGY / period / mean_square),
            "click_to_music_loud_db": 10.0 * math.log10(CLICK_ENERGY / WINDOW_SEC) - 20.0 * math.log10(loud)}


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--music", type=pathlib.Path, default=repository / "music",
                        help="the corpora's root; a worktree has none of its own")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    sets = {"rwc": (args.music / "rwc2" / "manifest.csv", None),
            "gtzan": (args.music / "ground-truth" / "manifest.csv", {"gtzan"})}
    result: dict = {"click": {"gain": GAIN, "length_sec": LENGTH, "energy": CLICK_ENERGY},
                    "corpora": {}, "unreadable": []}
    for label, (manifest, corpora) in sets.items():
        rows = []
        for item in load_corpus(manifest, manifest.parent.parent, False, corpora):
            try:
                row = ratios(item)
            except Exception as error:  # noqa: BLE001 - listed, not hidden
                result["unreadable"].append(f"{item['name']}: {type(error).__name__}")
                continue
            if row is not None:
                rows.append(row)
        beat = np.array([r["click_to_music_beat_db"] for r in rows])
        loud = np.array([r["click_to_music_loud_db"] for r in rows])
        result["corpora"][label] = {
            "n": len(rows),
            "beat_db_median": float(np.median(beat)),
            "beat_db_iqr": [float(v) for v in np.percentile(beat, [25, 75])],
            "loud_db_median": float(np.median(loud)),
            "loud_db_iqr": [float(v) for v in np.percentile(loud, [25, 75])],
            "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "records"}
                      for k, v in result["corpora"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
