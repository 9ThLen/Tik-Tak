#!/usr/bin/env python3
"""Three BeatNet folds, separately and averaged, through the one decoder.

The published BeatNet ships as three checkpoints, each trained with one corpus
held out. Only fold 1 has ever been measured here, and two questions follow from
that which no amount of decoder tuning can answer:

* **Is fold 1 the model, or is fold 1 a draw?** A single checkpoint's score on
  an unseen corpus carries the variance of one training run. If the three folds
  differ by ten points on RWC, every live-path number in this repository has an
  error bar nobody has drawn, and the ranking of anything measured against them
  is not safe.
* **Is an average of the three better than any of them?** Ensembling is the
  cheapest thing left on the observation side — no new architecture, no new
  training data, no licence question — and the front end is where the recall
  budget said the headroom was. Three times the compute for a causal model that
  runs at a small fraction of real time is a price a phone can pay.

Everything is scored through ``--live-activation``, including the individual
folds, so the mean is compared against the folds on the same seam. Running the
folds natively through ``--live-model`` and the mean through the activation file
would confound the ensemble with the difference between two code paths — a real
difference, since ``--live-activation`` carries the beat activation only and the
downbeat probabilities do not survive it. The natively-run folds are a separate
experiment (three plain ``live_corpus_benchmark`` runs) and answer the shipping
question; this one answers the ensemble question.

Synchronisation needs no alignment step and the script asserts as much: all
three folds share one front end, so their activations are the same length, at
the same 50 fps, from the same frame zero. A length mismatch would mean the
streams are not what this script thinks they are, and it is treated as a failure
of the recording rather than quietly truncated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
sys.path.insert(0, str(RESEARCH))

from eval.analysis import DEFAULT_BINARY  # noqa: E402
from eval.live_corpus_benchmark import _score_one  # noqa: E402
from eval.live_corpus_benchmark import load_corpus  # noqa: E402
from eval.provenance import experiment_provenance as provenance  # noqa: E402

FPS = 50.0
MACRO_MIN = 30

# How much of the corpus may go missing before an average stops describing the
# corpus it names. Two percent is roughly one bad file in fifty and is worth a
# warning; a third of the corpus is a different experiment wearing the same
# name. Nothing was chosen by looking at a result — the failure that motivated
# it lost 33%, and any threshold in this range would have caught it.
MAX_DROPPED_SHARE = 0.02
MAX_DROPS_SHOWN = 20

# Fold order is the order the folds are published in, not a ranking.
FOLDS = ("1", "2", "3")


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p for a paired win/loss count.

    Recordings where both arms agree carry no information about which is
    better and are discarded, which is what makes this the sign test rather
    than a comparison of two rates: the question is whether the recordings
    that *moved* moved mostly one way.

    Exact rather than normal-approximate because the discordant counts here
    are in the tens, where the approximation is not reliable, and because
    there is no reason to approximate a sum of thirty binomial terms.
    """
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2 ** total)


def activation_of(binary: pathlib.Path, audio: pathlib.Path,
                  model: pathlib.Path) -> tuple[np.ndarray | None, str]:
    """The activation, or None and the reason there is none.

    Every one of these failures used to be spelled `return None`, and `one`
    turned any of them into a recording that simply was not there. That is how
    a run on 2026-08-04 lost the last 109 of 328 recordings — the machine
    stopped being able to spawn processes partway through — and still produced
    a well-formed table, with every surviving corpus scoring exactly as before
    and the macro average collapsing because the easy corpus had silently
    fallen out of it. The numbers were not wrong; the corpus was, and nothing
    said so. So the reason travels with the failure now.
    """
    try:
        done = subprocess.run(
            [str(binary), str(audio), "--live", "--live-model", str(model),
             "--dump-activation"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as error:
        # The failure that actually happened: not a bad file, an environment
        # that could no longer start a child process.
        return None, f"could not run the binary: {type(error).__name__}"
    if done.returncode != 0:
        return None, (f"exit {done.returncode}: "
                      f"{done.stderr.strip()[:160] or 'no message'}")
    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError:
        return None, "output was not JSON"
    stream = payload.get("activation_beat")
    if not stream:
        return None, "no activation in the output"
    return np.asarray(stream, dtype=np.float64), ""


def one(item: dict, binary: pathlib.Path,
        models: dict[str, pathlib.Path]) -> dict | None:
    streams: dict[str, np.ndarray] = {}
    for fold, model in models.items():
        stream, why = activation_of(binary, item["audio"], model)
        if stream is None:
            return {"__dropped__": f"{item['corpus']}/{item['name']}: "
                                   f"fold {fold} {why}"}
        streams[fold] = stream

    lengths = {len(stream) for stream in streams.values()}
    if len(lengths) != 1:
        # Not a tolerance question. One front end produced all three, so
        # different lengths mean one of the runs is not the run it claims to be.
        return {"__dropped__": f"{item['corpus']}/{item['name']}: the three "
                               f"folds produced {sorted(lengths)} frames"}

    stack = np.vstack([streams[fold] for fold in models])
    arms = dict(streams)
    arms["mean"] = stack.mean(axis=0)
    # The other obvious pooling rule, and the one with the opposite bias: a mean
    # is dragged down by a fold that is unsure, a max by a fold that is wrong
    # and confident. Which of those failures the corpus actually contains is the
    # thing being measured, so both are run rather than argued about.
    arms["max"] = stack.max(axis=0)

    rows: dict[str, dict] = {}
    for arm, activation in arms.items():
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        try:
            handle.write("\n".join(f"{value:.6f}" for value in activation))
            handle.close()
            rows[arm] = _score_one(
                item, "baseline", binary, None,
                extra=("--live-activation", handle.name,
                       "--activation-fps", repr(FPS)))
        finally:
            pathlib.Path(handle.name).unlink(missing_ok=True)
    return rows


def aggregate(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get("ok") and row.get("annotated")]
    if not scored:
        return {"n": 0}
    corpora = sorted({row["corpus"] for row in scored})
    by_corpus = {}
    for corpus in corpora:
        part = [row for row in scored if row["corpus"] == corpus]
        failures: Counter[str] = Counter()
        for row in part:
            failures.update(row.get("reasons", ()))
        by_corpus[corpus] = {
            "n": len(part),
            "usable_rate": sum(row["usable"] for row in part) / len(part),
            "usable_rate_strict":
                sum(row["usable_strict"] for row in part) / len(part),
            "usable_rate_any_octave":
                sum(row["usable_any_octave"] for row in part) / len(part),
            "f_measure": float(np.nanmean([row["f_measure"] for row in part])),
            "cmlt": float(np.nanmean([row["cmlt"] for row in part])),
            "failure_reasons": {key: count / len(part)
                                for key, count in failures.most_common()},
        }
    big = [c for c in by_corpus if by_corpus[c]["n"] >= MACRO_MIN]
    return {
        "n": len(scored),
        "usable_rate_macro":
            float(np.mean([by_corpus[c]["usable_rate"] for c in big])) if big else None,
        "usable_rate_strict_macro":
            float(np.mean([by_corpus[c]["usable_rate_strict"] for c in big])) if big else None,
        "usable_rate_any_octave_macro":
            float(np.mean([by_corpus[c]["usable_rate_any_octave"] for c in big])) if big else None,
        "usable_rate_pooled": sum(row["usable"] for row in scored) / len(scored),
        "usable_rate_strict_pooled":
            sum(row["usable_strict"] for row in scored) / len(scored),
        "usable_rate_any_octave_pooled":
            sum(row["usable_any_octave"] for row in scored) / len(scored),
        "f_measure": float(np.nanmean([row["f_measure"] for row in scored])),
        "cmlt": float(np.nanmean([row["cmlt"] for row in scored])),
        "by_corpus": by_corpus,
        # Per recording, so two arms can be compared as a paired count rather
        # than as two rates. Between a mean activation and the best single fold
        # the difference is a couple of points, which over 328 recordings is a
        # handful of tracks — and two rates two points apart can hide anything
        # from six tracks moving one way to forty moving in both. Only the
        # verdicts, keyed by name: the full metrics are what the run itself is.
        "tracks": {f"{row['corpus']}/{row['name']}":
                   [bool(row["usable"]), bool(row["usable_strict"])]
                   for row in scored},
    }


def paired(report: dict) -> dict | None:
    """Every arm the mean might be quoted against, paired over recordings.

    The rate table cannot answer this. A mean that scores two points higher has
    either won a few recordings outright or won many and lost nearly as many,
    and those are different claims about whether averaging helps: the first is
    a small real gain, the second is churn with a favourable sign.

    Every arm, not only the one the mean happens to beat, and every p corrected
    for all of them. The correction is the point rather than a formality: with
    eight comparisons in front of you the smallest p is not the p you would
    have got had you asked one question, and quoting it as though it were is
    how churn becomes a finding. On the RWC run it decides two of the eight
    rows. Holm rather than a flat Bonferroni — the same family-wise guarantee,
    uniformly less conservative, four lines.
    """
    arms = report.get("by_arm", {})
    if not arms.get("mean", {}).get("n"):
        return None
    mean = arms["mean"]["tracks"]

    comparisons = []
    for arm in [*FOLDS, "max"]:
        other = arms.get(arm, {}).get("tracks")
        if not other:
            continue
        shared = sorted(set(other) & set(mean))
        for index, label in ((0, "usable"), (1, "usable, strictly")):
            won = sum(mean[k][index] and not other[k][index] for k in shared)
            lost = sum(other[k][index] and not mean[k][index] for k in shared)
            comparisons.append({
                "against": arm, "criterion": label, "n": len(shared),
                "mean_wins": won, "mean_loses": lost,
                "p": sign_test(won, lost)})
    if not comparisons:
        return None

    # Holm: sorted ascending, the k-th smallest p is multiplied by (m - k)
    # rather than by m, and the corrected values are made non-decreasing so
    # that no comparison is called significant while a stronger one is not.
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p"])
    running = 0.0
    for rank, index in enumerate(order):
        row = comparisons[index]
        running = max(running, min(1.0, row["p"] * (len(comparisons) - rank)))
        row["p_holm"] = running
        row["significant_at_05"] = running < 0.05

    report["paired"] = {
        "family_size": len(comparisons),
        "correction": "Holm-Bonferroni over every comparison in this table",
        "comparisons": [comparisons[index] for index in order],
    }
    return report["paired"]


def show_paired(report: dict) -> None:
    block = report["paired"]
    print(f"\n   the mean against each arm, paired over recordings "
          f"(family of {block['family_size']}, Holm-corrected)\n")
    print(f"   {'against':8} {'criterion':16} {'won':>4} {'lost':>5} "
          f"{'p':>8} {'corrected':>10}")
    for row in block["comparisons"]:
        mark = " *" if row["significant_at_05"] else ""
        name = row["against"] if row["against"] == "max" else f"fold {row['against']}"
        print(f"   {name:8} {row['criterion']:16} "
              f"{row['mean_wins']:>4} {row['mean_loses']:>5} "
              f"{row['p']:>8.4f} {row['p_holm']:>10.4f}{mark}")
    print("\n   * significant at 0.05 after correction; "
          "everything else is not established")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=REPOSITORY / "music" / "rwc2" / "manifest.csv")
    parser.add_argument("--music", type=pathlib.Path,
                        default=REPOSITORY / "music" / "rwc2")
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--limit", type=int, default=0,
                        help="score a spanning subset rather than everything")
    parser.add_argument("--corpora", nargs="+",
                        help="datasets to score; see the same flag on "
                             "live_corpus_benchmark")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=pathlib.Path)
    # Re-analysis without re-measuring. The paired statistics are the part most
    # likely to change — a correction chosen differently, a comparison added —
    # and re-running 328 recordings through three networks to try one is both
    # slow and, worse, a second measurement where a second *analysis* was
    # meant. The saved report carries every per-recording verdict, so the whole
    # of `paired` can be rebuilt from it.
    parser.add_argument("--from", dest="reanalyse", type=pathlib.Path,
                        help="recompute the paired statistics from a saved "
                             "report instead of measuring again")
    args = parser.parse_args(argv)

    if args.reanalyse:
        report = json.loads(args.reanalyse.read_text(encoding="utf-8"))
        if paired(report) is None:
            parser.error(f"{args.reanalyse} has no per-recording verdicts to "
                         "pair — it predates --from")
        show_paired(report)
        target = args.output or args.reanalyse
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    models = {fold: REPOSITORY / "models" / f"beatnet_model_{fold}.ttw"
              for fold in FOLDS}
    missing = [str(path.name) for path in models.values() if not path.is_file()]
    if missing:
        parser.error("missing weights: " + ", ".join(missing))

    items = [item for item in load_corpus(args.manifest, args.music, False,
                                          corpora=set(args.corpora)
                                          if args.corpora else None)
             if item["annotated"]]
    if args.limit and len(items) > args.limit:
        items = items[:: -(-len(items) // args.limit)]

    def work(item):
        return one(item, args.binary, models)

    collected: dict[str, list[dict]] = {}
    dropped: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(work, items):
            if "__dropped__" in result:
                dropped.append(result["__dropped__"])
                continue
            for arm, row in result.items():
                collected.setdefault(arm, []).append(row)

    # Loud, and fatal past a threshold. A dropped recording is not a corpus
    # that is one smaller — it is a different corpus, and if the drops are not
    # spread evenly they change the macro average without changing a single
    # per-corpus rate. That is exactly what happened when this was a silent
    # counter: every surviving corpus scored the same and the headline moved
    # eight points, because the easy corpus was the part that vanished.
    for line in dropped[:MAX_DROPS_SHOWN]:
        print(f"   dropped {line}")
    if len(dropped) > MAX_DROPS_SHOWN:
        print(f"   ... and {len(dropped) - MAX_DROPS_SHOWN} more")
    if dropped:
        share = len(dropped) / len(items)
        print(f"\n   {len(dropped)} of {len(items)} recordings dropped "
              f"({share:.1%})")
        if share > MAX_DROPPED_SHARE:
            print(f"   REFUSING to aggregate: more than "
                  f"{MAX_DROPPED_SHARE:.0%} of the corpus is missing, so any "
                  f"average here describes a corpus nobody chose. Fix the "
                  f"cause and run it again.")
            return 1

    report = {
        "provenance": provenance(
            REPOSITORY,
            {"binary": args.binary,
             **{f"model_{fold}": path for fold, path in models.items()}},
            arms=sorted(collected), attempted=len(items),
            dropped=len(dropped), dropped_detail=dropped),
        "fps": FPS,
        "by_arm": {arm: aggregate(rows) for arm, rows in collected.items()},
    }

    order = [*FOLDS, "mean", "max"]
    # A subset small enough that no corpus reaches MACRO_MIN has no macro
    # average, and printing nothing there makes a smoke test look like a
    # failure. Pooled is the honest fallback as long as the header says so.
    macro = any(summary.get("usable_rate_macro") is not None
                for summary in report["by_arm"].values())
    suffix = "" if macro else " (pooled — no corpus reaches the macro minimum)"
    print(f"\n{len(items) - len(dropped)} of {len(items)} recordings, "
          f"every arm through --live-activation{suffix}\n")
    print(f"   {'arm':8} {'usable':>8} {'strict':>8} {'any lvl':>8} "
          f"{'F':>7} {'CMLt':>7}")
    for arm in order:
        summary = report["by_arm"].get(arm)
        if not summary or not summary.get("n"):
            continue

        def rate(key: str) -> float:
            value = summary.get(f"{key}_macro")
            return summary[f"{key}_pooled"] if value is None else value

        print(f"   {arm:8} {rate('usable_rate'):>8.1%} "
              f"{rate('usable_rate_strict'):>8.1%} "
              f"{rate('usable_rate_any_octave'):>8.1%} "
              f"{summary['f_measure']:>7.4f} {summary['cmlt']:>7.4f}")

    if paired(report):
        show_paired(report)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
