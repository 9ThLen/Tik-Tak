"""What 2048 particles cost against 512, one process at a time.

Registered in `PREREGISTERED_particles_2048.md`. Five GTZAN and five Harmonix
recordings, the first by name in each corpus, each run through
`dump_analysis --live` single-process, interleaved 512, 2048, 512, 2048, 512,
2048, so that background load drifts equally over both configurations. The
statistic is the median over recordings of the ratio of wall time per second
of audio.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.live_corpus_benchmark import load_corpus  # noqa: E402
from eval.provenance import experiment_provenance  # noqa: E402

CONFIGS = {"512": (), "2048": ("--live-particles", "2048")}
REPEATS = 3
PER_CORPUS = 5


def timed(binary: pathlib.Path, model: pathlib.Path, audio: pathlib.Path,
          flags: tuple[str, ...]) -> tuple[float, float]:
    started = time.perf_counter()
    done = subprocess.run([str(binary), str(audio), "--live", "--live-model", str(model),
                           "--live-sample-hz", "50", *flags],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True)
    wall = time.perf_counter() - started
    duration = float(json.loads(done.stdout).get("duration_sec", 0.0))
    return wall, duration


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    chosen = []
    for corpus in ("gtzan", "harmonix"):
        items = sorted(load_corpus(args.manifest, args.music, False, frozenset({corpus})),
                       key=lambda item: item["name"])
        chosen.extend(items[:PER_CORPUS])
    record = experiment_provenance(repository, {"binary": args.binary, "model": args.model},
                                   experiment="particle_cost")
    rows = []
    for item in chosen:
        runs: dict[str, list[float]] = {name: [] for name in CONFIGS}
        duration = 0.0
        for _ in range(REPEATS):
            for name, flags in CONFIGS.items():
                wall, duration = timed(args.binary, args.model, pathlib.Path(item["audio"]), flags)
                runs[name].append(wall)
        per_second = {name: float(np.median(walls)) / max(duration, 1e-9)
                      for name, walls in runs.items()}
        rows.append({"corpus": item["corpus"], "name": item["name"], "duration_sec": duration,
                     "wall_sec": runs, "wall_per_audio_sec": per_second,
                     "ratio": per_second["2048"] / per_second["512"]})
    result = {"provenance": record, "repeats": REPEATS, "records": rows,
              "median_ratio": float(np.median([r["ratio"] for r in rows])),
              "median_wall_per_audio_sec": {
                  name: float(np.median([r["wall_per_audio_sec"][name] for r in rows]))
                  for name in CONFIGS}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("provenance", "records")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
