#!/usr/bin/env python3
"""Where the bar line is actually lost: an oracle decomposition.

A downbeat result is the product of three separate things — the beat grid, the
metre, and which beat of the bar is the ``1``. A single F-measure cannot say
which of them failed, and the three call for entirely different work. This hands
each of them to the tracker correct, one at a time, and reports what is left.

    ceiling       annotated beats, annotated metre, phase chosen by the answer
    phase_scored  annotated beats, annotated metre, phase chosen by the salience
    grid_ceiling  the tracker's beats, annotated metre, phase chosen by the answer
    end_to_end    the model's own beats and its own downbeats

`ceiling` minus `phase_scored` is the cost of the phase decision alone.
`ceiling` minus `grid_ceiling` is the cost of the grid alone, since everything
else in that row is free. Reading the two together is the point of the file.

**Which checkpoint matters, and it is not a detail.** Beat This!'s `final*`
models are trained on everything except GTZAN, so a number measured on GTZAN
with `final0` is honest and the same number measured on Ballroom is recall of
the training set. Ballroom has to be scored out of fold: `annotations/ballroom/
8-folds.split` assigns every recording a fold, and `fold0`..`fold7` are the
checkpoints that each held one fold out. Pass ``--checkpoint fold-matched`` and
this file will use the right one per recording.

Measured, everything below out of fold or held out — no row is scored by a model
that saw the recording. Ballroom is 685 recordings (13 of the 698 are absent
from the split file and are skipped, since a recording whose fold is unknown
cannot be shown to be held out); GTZAN is 992 of 999.

    Ballroom, out of fold        AUPRC 0.993   phase chosen correctly 98.1%
                     beat F   CMLt   AMLt   db F  db acc  missing  extra  octave
      ceiling         1.000  1.000  1.000  0.999   1.000    0.000  0.000    0.0%
      phase_scored    1.000  1.000  1.000  0.982   0.991    0.000  0.000    0.0%
      grid_ceiling    0.762  0.578  0.847  0.680   0.916    0.237  0.195   27.0%
      end_to_end      0.976  0.965  0.971  0.954   0.984    0.018  0.028    0.4%

    GTZAN, held out from final0  AUPRC 0.869   phase chosen correctly 85.8%
      ceiling         1.000  1.000  1.000  0.995   0.997    0.000  0.000    0.0%
      phase_scored    1.000  1.000  1.000  0.855   0.925    0.000  0.000    0.0%
      grid_ceiling    0.784  0.653  0.849  0.729   0.919    0.190  0.207   19.7%
      end_to_end      0.892  0.799  0.903  0.787   0.878    0.091  0.101   12.1%

    GTZAN, held out from small0  AUPRC 0.856   phase chosen correctly 84.8%
      ceiling         1.000  1.000  1.000  0.995   0.997    0.000  0.000    0.0%
      phase_scored    1.000  1.000  1.000  0.844   0.920    0.000  0.000    0.0%
      grid_ceiling    0.784  0.653  0.849  0.729   0.919    0.190  0.207   19.7%
      end_to_end      0.886  0.791  0.894  0.772   0.868    0.097  0.106   12.6%

`grid_ceiling` comes out identical in the two GTZAN tables, as it must: that row
uses no model at all. It is the cheapest available check that the columns have
not been crossed.

**small0 is the default here, not the library's final0**, because small0 is the
model this project intends to ship and a default that measures something else
is how a project ends up with numbers for a model it does not use. Ten times
smaller costs 1.5 points of downbeat F end to end — 0.772 against 0.787 — on the
same held-out corpus.

Note what cannot be done, so it is not attempted later and mistaken for an
oversight: CPJKU publish fold checkpoints for the full model only, so Ballroom
out of fold exists for `final*` and not for `small*`. Their README puts the
limit plainly — an evaluation with `final*` or `small*` is fair on GTZAN and
nowhere else they trained on. GTZAN is therefore the only corpus on which the
shipping candidate can be scored honestly at all.

Three readings, none of which is visible in a single F-measure.

*The phase is not free, and how unfree depends on the material.* On Ballroom it
costs 0.017 F, which is what made an earlier measurement on that corpus alone
conclude it cost nothing. On GTZAN it costs 0.140. Ballroom is programmed dance
music with the bar line played on the kick; GTZAN is not, and the difference
between 98.1% and 85.8% of phases chosen correctly is the whole gap. Work on the
phase decision is worth doing, and Ballroom cannot be the corpus it is judged on.

*The grid costs more than everything else together.* `grid_ceiling` hands the
tracker's own beats a free metre and a free phase and still lands at 0.680 and
0.729 — below `end_to_end`, where the model decides all three for itself. Our
grid is far enough behind that no help with the other two decisions catches up.
The `missing` and `extra` columns say it plainly: a fifth to a quarter of the
annotated beats have no estimate near them.

*Ballroom is easy, not memorised.* An earlier run scored it with `final0`, which
was trained on it, and read 0.999 AUPRC as recall of the training set. Out of
fold it is 0.993. The corpus was never the problem; it is simply steady dance
music, and it flatters any tracker. GTZAN's 0.869 is what a corpus that does not
flatter looks like.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np

if __package__ in (None, ""):  # running as a script
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mir_eval.util  # noqa: E402

from eval.annotations import load_annotation  # noqa: E402
from eval.backends import sample_at_beats  # noqa: E402
from eval.harness import (F_MEASURE_TOLERANCE, TRIM_MIN_BEAT_TIME, evaluate,  # noqa: E402
                          evaluate_downbeats)

__all__ = [
    "CONFIGURATIONS",
    "fold_map",
    "octave_label",
    "matched_beat_stats",
    "auprc",
]

CONFIGURATIONS = ("ceiling", "phase_scored", "grid_ceiling", "end_to_end")

# The metrical ratios worth naming. Anything else is not an octave error, it is
# a different answer, and lumping the two together hides which one happened.
_RATIOS = ((1 / 3, "1/3"), (0.5, "1/2"), (2 / 3, "2/3"), (1.0, "1x"),
           (1.5, "3/2"), (2.0, "2x"), (3.0, "3x"))


def fold_map(path: pathlib.Path) -> dict[str, int]:
    """Recording stem to fold index, from a `<name>\\t<fold>` split file.

    The split file carries the corpus name as a prefix (`ballroom_Media-1234`)
    and the audio does not, so the prefix is dropped. Both spellings are kept in
    the map, because which one a dataset folder uses is not this file's business.
    """
    folds: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        # Tab first, and only then any whitespace: the file is tab separated,
        # and splitting on spaces would quietly reject any recording whose name
        # contains one rather than reading its fold.
        parts = line.split("\t") if "\t" in line else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
            continue
        name, fold = parts[0], int(parts[1])
        folds[name] = fold
        if "_" in name:
            folds[name.split("_", 1)[1]] = fold
    return folds


def octave_label(estimate: np.ndarray, reference: np.ndarray) -> str:
    """Which metrical level the estimate landed on, by median beat interval."""
    if len(estimate) < 2 or len(reference) < 2:
        return "none"
    est = float(np.median(np.diff(estimate)))
    ref = float(np.median(np.diff(reference)))
    if not (np.isfinite(est) and np.isfinite(ref)) or est <= 0 or ref <= 0:
        return "none"
    # Period ratio inverted into a tempo ratio, so "2x" reads as "twice as fast"
    # here and in every other table in this repository.
    ratio = ref / est
    for value, name in _RATIOS:
        if abs(ratio / value - 1.0) < 0.04:
            return name
    return "other"


def _trim(times: np.ndarray) -> np.ndarray:
    return np.asarray(times, dtype=np.float64)[
        np.asarray(times, dtype=np.float64) >= TRIM_MIN_BEAT_TIME]


def matched_beat_stats(ref_beats: np.ndarray, est_beats: np.ndarray,
                       ref_downbeats: np.ndarray,
                       est_downbeats: np.ndarray) -> dict:
    """Grid errors and label errors, separated.

    `missing` and `extra` are the grid's own failures: an annotated beat with no
    estimate near it, and an estimate with no annotated beat near it. They are
    reported as fractions of the annotation and of the estimate respectively, so
    a recording with twice as many beats does not weigh twice as much.

    `downbeat_accuracy` is asked only of the beats that did match. It answers a
    question the F-measure cannot: when the grid is right, is the *label* right?
    A tracker can score badly on downbeats purely because its beats are wrong,
    and this is what tells the two apart.
    """
    ref, est = _trim(ref_beats), _trim(est_beats)
    if len(ref) == 0 or len(est) == 0:
        return {"missing": float("nan"), "extra": float("nan"),
                "downbeat_accuracy": float("nan"), "matched": 0}

    pairs = mir_eval.util.match_events(ref, est, F_MEASURE_TOLERANCE)
    matched = len(pairs)

    def is_downbeat(times: np.ndarray, marks: np.ndarray) -> np.ndarray:
        flags = np.zeros(len(times), dtype=bool)
        marks = _trim(marks)
        for t in marks:
            if len(times) == 0:
                break
            index = int(np.argmin(np.abs(times - t)))
            if abs(times[index] - t) <= F_MEASURE_TOLERANCE:
                flags[index] = True
        return flags

    ref_flags = is_downbeat(ref, ref_downbeats)
    est_flags = is_downbeat(est, est_downbeats)
    agree = sum(1 for i, j in pairs if ref_flags[i] == est_flags[j])

    return {
        "missing": 1.0 - matched / len(ref),
        "extra": 1.0 - matched / len(est),
        "downbeat_accuracy": agree / matched if matched else float("nan"),
        "matched": matched,
    }


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision, written out so sklearn is not pulled in for one number."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    y = np.asarray(labels, dtype=bool)[order]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / positives)


def _downbeat_flags(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(beats), dtype=bool)
    for t in downbeats:
        if len(beats) == 0:
            break
        flags[int(np.argmin(np.abs(beats - t)))] = True
    return flags


def _phase_by_salience(salience: np.ndarray, metre: int) -> int:
    return int(np.argmax([float(np.mean(salience[p::metre])) for p in range(metre)]))


def _phase_by_answer(beats: np.ndarray, metre: int,
                     reference_downbeats: np.ndarray) -> tuple[int, float]:
    best, best_f = 0, -1.0
    for phase in range(metre):
        f = evaluate_downbeats(reference_downbeats,
                               beats[phase::metre])["downbeat_f_measure"]
        if f > best_f:
            best, best_f = phase, f
    return best, float(best_f)


def _score_clip(wav: pathlib.Path, frames, postprocess, tracker_beats):
    """Every configuration for one recording, or None when it cannot be scored."""
    import soundfile as sf

    try:
        audio, rate = sf.read(wav, dtype="float64")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    reference = load_annotation(wav.with_suffix(".beats"))
    ref_beats, ref_downbeats = reference.beats, reference.downbeats
    metre = int(reference.beats_per_bar)
    if len(ref_beats) < 8 or len(ref_downbeats) < 2 or metre < 2:
        return None

    beat_logits, downbeat_logits = frames(audio, float(rate))
    activation = 1.0 / (1.0 + np.exp(-np.asarray(downbeat_logits, dtype=np.float64)))
    frame_times = np.arange(len(activation)) / 50.0  # the model's frame rate

    salience = sample_at_beats(activation, frame_times, ref_beats)
    labels = _downbeat_flags(ref_beats, ref_downbeats)
    if labels.sum() < 2 or labels.all():
        return None

    row: dict = {"stem": wav.stem, "metre": metre,
                 "salience": salience.tolist(), "labels": labels.tolist()}

    # ceiling and phase_scored: the annotated grid, so the grid cannot be blamed.
    oracle_phase, ceiling_f = _phase_by_answer(ref_beats, metre, ref_downbeats)
    chosen_phase = _phase_by_salience(salience, metre)
    chosen_f = evaluate_downbeats(
        ref_downbeats, ref_beats[chosen_phase::metre])["downbeat_f_measure"]
    row["ceiling"] = _pack(ref_beats, ref_beats[oracle_phase::metre],
                           ref_beats, ref_downbeats, float(ceiling_f))
    row["phase_scored"] = _pack(ref_beats, ref_beats[chosen_phase::metre],
                                ref_beats, ref_downbeats, float(chosen_f))
    row["phase_correct"] = bool(chosen_phase == oracle_phase)

    # grid_ceiling: our own grid, everything else still free.
    grid = tracker_beats(wav)
    if grid is not None and len(grid) >= metre + 1:
        phase, f = _phase_by_answer(grid, metre, ref_downbeats)
        row["grid_ceiling"] = _pack(grid, grid[phase::metre],
                                    ref_beats, ref_downbeats, float(f))

    # end_to_end: the model deciding everything for itself, from the logits
    # already computed above rather than a second forward pass over the same
    # audio — Audio2Beats would run the network again for nothing.
    model_beats, model_downbeats = postprocess(beat_logits, downbeat_logits)
    row["end_to_end"] = _pack(
        model_beats, model_downbeats, ref_beats, ref_downbeats,
        evaluate_downbeats(ref_downbeats, model_downbeats)["downbeat_f_measure"])
    return row


def _pack(est_beats, est_downbeats, ref_beats, ref_downbeats, downbeat_f) -> dict:
    beat = evaluate(ref_beats, est_beats)
    stats = matched_beat_stats(ref_beats, est_beats, ref_downbeats, est_downbeats)
    return {
        "beat_f": float(beat["f_measure"]),
        "cmlt": float(beat["cmlt"]),
        "amlt": float(beat["amlt"]),
        "downbeat_f": float(downbeat_f),
        "octave": octave_label(np.asarray(est_beats), np.asarray(ref_beats)),
        **stats,
    }


# Beat This!'s shortname for an artifact the manifest files under another name.
# `small0.ckpt` from their share and `beat_this_small.ckpt` here are the same
# bytes — verified, sha256 6074be2c4d49… — so the small0 column is covered by
# the manifest rather than by an unvouched download.
_PINNED_ALIASES = {"small0": "beat_this_small.ckpt"}


def resolve_checkpoint(name: str, directory: pathlib.Path | None) -> str:
    """A local checkpoint file when there is one, otherwise the shortname.

    Beat This! will fetch a shortname over the network and cache it, which is
    fine once and wasteful eight times. A file already pinned in models/ is also
    the one the manifest vouches for, so it is preferred when present.
    """
    if directory is not None:
        for candidate in (directory / f"beat_this_{name}.ckpt",
                          directory / _PINNED_ALIASES.get(name, "")):
            if candidate.name and candidate.is_file():
                return str(candidate)
    return name


def _worker(task):
    """One checkpoint, one slice of the corpus, one model load."""
    checkpoint, paths, binary = task
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor

    frames = Audio2Frames(checkpoint_path=checkpoint, device="cpu")
    postprocess = Postprocessor(type="minimal")

    def tracker_beats(wav: pathlib.Path):
        if binary is None:
            return None
        done = subprocess.run([str(binary), str(wav)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            return None
        return np.asarray(json.loads(done.stdout).get("beats", []), dtype=np.float64)

    out = []
    for path in paths:
        try:
            row = _score_clip(pathlib.Path(path), frames, postprocess, tracker_beats)
        except Exception:
            row = None
        if row is not None:
            row["checkpoint"] = checkpoint
            out.append(row)
    return out


def _report(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 78}\n{title}   ({len(rows)} recordings)\n{'=' * 78}")

    pooled_s = np.concatenate([np.asarray(r["salience"]) for r in rows])
    pooled_y = np.concatenate([np.asarray(r["labels"]) for r in rows])
    prevalence = float(pooled_y.mean())
    print(f"\ndownbeat salience at annotated beats:  AUPRC "
          f"{auprc(pooled_s, pooled_y):.3f}   prevalence {prevalence:.3f}   "
          f"lift {auprc(pooled_s, pooled_y) / prevalence:.2f}x")
    print(f"phase chosen correctly on the annotated grid: "
          f"{np.mean([r['phase_correct'] for r in rows]):.1%}")

    header = (f"\n{'configuration':<14}{'beat F':>8}{'CMLt':>7}{'AMLt':>7}"
              f"{'db F':>7}{'db acc':>8}{'missing':>9}{'extra':>8}")
    print(header)
    print("-" * len(header.strip("\n")))
    for name in CONFIGURATIONS:
        present = [r[name] for r in rows if name in r]
        if not present:
            continue
        def mean(key):
            values = [p[key] for p in present if np.isfinite(p[key])]
            return statistics.mean(values) if values else float("nan")
        print(f"{name:<14}{mean('beat_f'):>8.3f}{mean('cmlt'):>7.3f}"
              f"{mean('amlt'):>7.3f}{mean('downbeat_f'):>7.3f}"
              f"{mean('downbeat_accuracy'):>8.3f}{mean('missing'):>9.3f}"
              f"{mean('extra'):>8.3f}")

    print(f"\n{'configuration':<14}{'octave errors':>15}   distribution")
    for name in CONFIGURATIONS:
        present = [r[name] for r in rows if name in r]
        if not present:
            continue
        counts = collections.Counter(p["octave"] for p in present)
        wrong = sum(counts[k] for k in ("1/2", "2x", "1/3", "3x"))
        spread = "  ".join(f"{k} {counts[k]}" for _, k in _RATIOS if counts[k])
        if counts["other"]:
            spread += f"  other {counts['other']}"
        print(f"{name:<14}{wrong:>8} ({wrong / len(present):>5.1%})   {spread}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--checkpoint", default="small0",
                        help="a Beat This! checkpoint name, or 'fold-matched'. "
                             "Defaults to small0, the model this project intends "
                             "to ship, rather than to the library's own final0")
    parser.add_argument("--folds", type=pathlib.Path, default=None,
                        help="split file, required by 'fold-matched'")
    parser.add_argument("--skip-unsplit", action="store_true",
                        help="leave out recordings the split file does not cover")
    parser.add_argument("--binary", type=pathlib.Path, default=None,
                        help="dump_analysis, for the grid_ceiling row")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2] / "models",
                        help="where beat_this_<name>.ckpt files are, if any")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    wavs = sorted(args.dataset.rglob("*.wav"))
    if args.limit:
        wavs = wavs[:: max(1, len(wavs) // args.limit)][: args.limit]

    if args.checkpoint == "fold-matched":
        if args.folds is None:
            print("fold-matched needs --folds")
            return 2
        folds = fold_map(args.folds)
        missing = [w.stem for w in wavs if w.stem not in folds]
        if missing and not args.skip_unsplit:
            print(f"{len(missing)} recording(s) are not in the split file, "
                  f"e.g. {missing[:3]} — refusing rather than scoring them in "
                  f"fold, which would be the error this mode exists to avoid. "
                  f"Pass --skip-unsplit to leave them out instead.")
            return 2
        if missing:
            # Left out, not guessed at. A recording whose fold is unknown might
            # be one the checkpoint was trained on, and a single such recording
            # in the table would make the whole column mean something else.
            print(f"skipping {len(missing)} recording(s) absent from the split "
                  f"file: their fold is unknown, so they cannot be held out")
            wavs = [w for w in wavs if w.stem in folds]
        groups: dict[str, list] = collections.defaultdict(list)
        for wav in wavs:
            groups[f"fold{folds[wav.stem]}"].append(str(wav))
        tasks = [(resolve_checkpoint(name, args.checkpoint_dir), paths, args.binary)
                 for name, paths in sorted(groups.items())]
        print(f"{len(wavs)} recordings, out of fold across "
              f"{len(tasks)} checkpoints", flush=True)
    else:
        resolved = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
        chunk = max(1, len(wavs) // max(1, args.workers))
        tasks = [(resolved, [str(w) for w in wavs[i:i + chunk]], args.binary)
                 for i in range(0, len(wavs), chunk)]
        print(f"{len(wavs)} recordings, checkpoint {args.checkpoint}", flush=True)

    rows: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, batch in enumerate(pool.map(_worker, tasks), 1):
            rows.extend(batch)
            print(f"  {done}/{len(tasks)} batches, {len(rows)} scored", flush=True)

    if not rows:
        print("nothing scored")
        return 1

    _report(rows, args.title or f"{args.dataset.name} — {args.checkpoint}")

    if args.out:
        args.out.write_text(json.dumps(rows), encoding="utf-8")
        print(f"\nper-recording rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
