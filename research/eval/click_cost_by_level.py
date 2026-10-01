"""Post hoc: does the ungated click cost more where it is louder against the music?

Reads `click_music_ratio.json` and the click-gate per-track records, and for
G3 (-12 dB) and G7 (0 dB) against G0 reports the Spearman correlation between
each recording's click-to-music ratio and its change in beat F, with the mean
change by tercile of the ratio. Observational: quiet recordings, where the
click stands out most, differ from loud ones in other ways too.

    cd research
    .venv/Scripts/python -m eval.click_cost_by_level
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
ARMS = {"g3_nogate": -12.0, "g7_loud_nogate": 0.0}


def f_by_recording(path: pathlib.Path) -> dict[tuple[str, str], float]:
    return {(r["corpus"], r["name"]): float(r["f_measure"])
            for r in json.loads(path.read_text(encoding="utf-8")) if r.get("ok")}


def main() -> int:
    levels = json.loads((RESULTS / "click_gate_and_antialias" / "click_music_ratio.json")
                        .read_text(encoding="utf-8"))
    tracks = RESULTS / "per_track" / "click_gate_and_antialias"
    out = {}
    for corpus in ("rwc", "gtzan"):
        ratio = {(r["corpus"], r["name"]): r["click_to_music_beat_db"]
                 for r in levels["corpora"][corpus]["records"]}
        base = f_by_recording(tracks / f"g0_baseline_{corpus}.model.json")
        for arm, gain in ARMS.items():
            arm_f = f_by_recording(tracks / f"{arm}_{corpus}.model.json")
            keys = [k for k in ratio if k in base and k in arm_f]
            x = np.array([ratio[k] + gain for k in keys])
            d = np.array([arm_f[k] - base[k] for k in keys])
            rho, p = spearmanr(x, d)
            cut = np.percentile(x, [100 / 3, 200 / 3])
            terciles = [float(d[x <= cut[0]].mean()), float(d[(x > cut[0]) & (x <= cut[1])].mean()),
                        float(d[x > cut[1]].mean())]
            out[f"{arm}_{corpus}"] = {"n": len(keys), "spearman": float(rho), "p": float(p),
                                      "ratio_db_median": float(np.median(x)),
                                      "delta_f_by_tercile_low_mid_high": terciles}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
