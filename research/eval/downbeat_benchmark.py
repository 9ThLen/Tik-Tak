#!/usr/bin/env python3
"""Bar-line benchmark: metre, phase, and where to put the margin threshold.

    cd research
    .venv/bin/python -m eval.downbeat_benchmark                    # synthetic
    .venv/bin/python -m eval.downbeat_benchmark --dataset ~/clips  # real files

Build the analysis tool first — see eval/analysis.py.

**What the synthetic run does and does not prove.** The clips in
``tiktak.synth`` are percussion and sustained tones with no chord changes in
them, so a synthetic run exercises the low-band onset cue alone and says
nothing at all about the harmony cue, which is the one that carries material
with no drums. Onsets are also exact, the tempo is exact, and there is no room.
Treat it as a regression signal: it will notice the day a change breaks metre
detection, and it will flatter any tracker's absolute numbers.

The margin threshold in particular **must not** be read off a synthetic run.
That is what ``--dataset`` is for, and until a real annotated set exists the
threshold in the harness stays a placeholder that says so.

A dataset folder holds audio with an annotation of the same name beside it::

    clips/
      band-take-3.wav
      band-take-3.beats
      waltz.mp3
      waltz.beats

See eval/annotations.py for the annotation format. Thirty to fifty recordings
covering what the app will actually meet — full band, drumless takes, vocals
with guitar, waltzes, 6/8 — is the useful size; a dozen will move the threshold
around by more than the threshold matters.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import numpy as np

from eval.analysis import Analyser, DEFAULT_BINARY
from eval.annotations import Reference, find_pairs
from eval.downbeat import (
    DEFAULT_MIN_METER_MARGIN,
    DEFAULT_MIN_PHASE_MARGIN,
    Verdict,
    choose_margins,
    eligible,
    format_scores,
    format_sweep,
    frontier,
    score,
    sweep,
)
from tiktak.synth import make_clip



def synthetic_cases() -> list[tuple[Reference, np.ndarray, int]]:
    """Clips with exact ground truth, built to span the metres we claim to find."""
    cases = []
    recipes = [
        ("four 120", dict(bpm=120, beats_per_bar=4, seed=1)),
        ("four 96", dict(bpm=96, beats_per_bar=4, seed=2)),
        ("four 144", dict(bpm=144, beats_per_bar=4, seed=3)),
        ("waltz 150", dict(bpm=150, beats_per_bar=3, seed=4)),
        ("waltz 100", dict(bpm=100, beats_per_bar=3, seed=5)),
        ("two 120", dict(bpm=120, beats_per_bar=2, seed=6)),
        ("six 168", dict(bpm=168, beats_per_bar=6, seed=7)),
        ("four, lead-in", dict(bpm=120, beats_per_bar=4, silence_lead=3.0, seed=8)),
        ("four, noisy", dict(bpm=120, beats_per_bar=4, noise_db=6.0, seed=9)),
        ("four, swung", dict(bpm=120, beats_per_bar=4, swing=0.3, seed=10)),
        ("four, drifting", dict(bpm=120, beats_per_bar=4, tempo_drift=15.0, seed=11)),
        ("waltz, sparse", dict(bpm=120, beats_per_bar=3, sparse=True, seed=12)),
        ("four, sparse", dict(bpm=110, beats_per_bar=4, sparse=True, seed=13)),
    ]
    for name, kwargs in recipes:
        clip = make_clip(duration_sec=25, **kwargs)
        reference = Reference(
            beats=clip.beats,
            downbeats=clip.downbeats,
            beats_per_bar=clip.beats_per_bar,
            name=name,
            bar_lengths=(clip.beats_per_bar,) * max(len(clip.downbeats) - 1, 0),
        )
        cases.append((reference, clip.audio, clip.sample_rate))
    return cases


def split_of(name: str, test_fraction: float, salt: str = "tiktak") -> str:
    """Which split a clip belongs to, decided by its name and nothing else.

    Hashing the name rather than shuffling with a seed means a clip stays in the
    same split when the set grows, so a threshold chosen last month and a test
    score computed today are still about disjoint material.
    """
    digest = hashlib.sha256(f"{salt}:{name}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:4], "big") / 2**32
    return "test" if position < test_fraction else "validation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path,
                        help="folder of audio with annotations beside it; "
                             "omit for synthetic clips")
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY,
                        help="path to the dump_analysis executable")
    parser.add_argument("--max-wrong-rate", type=float, default=0.05,
                        help="wrong accents as a share of all clips (default 0.05)")
    parser.add_argument("--max-conditional-error", type=float, default=0.10,
                        help="wrong accents as a share of the accents actually "
                             "shown (default 0.10)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="score the clips that loaded even when others in "
                             "the dataset could not be read")
    parser.add_argument("--test-fraction", type=float, default=0.35,
                        help="share of clips held out; the threshold is chosen "
                             "on the rest (default 0.35)")
    parser.add_argument("--no-split", action="store_true",
                        help="score every clip together — for a quick look, not "
                             "for choosing a threshold")
    parser.add_argument("--expect-correct", type=int, default=None,
                        help="fail unless at least this many clips come back "
                             "with the right metre and phase; how CI notices a "
                             "regression")
    args = parser.parse_args(argv)

    analyser = Analyser(args.binary)
    if not analyser.available:
        print(f"not found: {args.binary}")
        print("Build it first — see eval/analysis.py.")
        return 2

    scores = []
    if args.dataset:
        references, problems = find_pairs(args.dataset)
        for complaint in problems:
            print(f"  ! {complaint}")
        if not references:
            print("nothing to score.")
            return 2
        if problems and not args.allow_partial:
            # Refusing by default rather than scoring what happens to have
            # loaded. A run that silently covered 12 of someone's 40 recordings
            # reports a healthy-looking mean over a set that mostly never ran,
            # and the threshold that comes out of it is calibrated on a
            # different dataset than the one they think they built.
            print(f"\n{len(problems)} problem(s) above. Fix them, or pass "
                  f"--allow-partial to score the {len(references)} that loaded.")
            return 2
        print(f"scoring {len(references)} annotated recording(s) from {args.dataset}\n")
        for reference in references:
            estimate = analyser.analyse_file(reference.audio_path)
            scores.append(score(reference, estimate))
    else:
        print("scoring synthetic clips — a regression signal, not a calibration.")
        print("The harmony cue is untested here: synthetic clips have no chord "
              "changes.\n")
        for reference, audio, rate in synthetic_cases():
            estimate = analyser.analyse_audio(audio, rate)
            scores.append(score(reference, estimate))

    unscorable = [s for s in scores if not s.scorable]
    if unscorable:
        print(f"{len(unscorable)} clip(s) have no bar lines annotated and are "
              f"scored on beats only.\n")
    changing = [s for s in scores if not s.meter_is_stable]
    if changing:
        print(f"{len(changing)} clip(s) change metre; no single answer is right "
              f"for the whole take.\n")

    print(format_scores(scores))
    beat_f = np.array([s.beat_f for s in scores], dtype=np.float64)
    print(f"\nbeat F-measure: mean {np.nanmean(beat_f):.3f}, "
          f"median {np.nanmedian(beat_f):.3f}")
    downbeat_f = np.array([s.downbeat_f for s in scores], dtype=np.float64)
    if not np.all(np.isnan(downbeat_f)):
        print(f"downbeat F-measure: mean {np.nanmean(downbeat_f):.3f}, "
              f"median {np.nanmedian(downbeat_f):.3f}")

    if args.no_split:
        splits = [("all clips", scores)]
    else:
        splits = [
            ("validation", [s for s in scores
                            if split_of(s.name, args.test_fraction) == "validation"]),
            ("test", [s for s in scores
                      if split_of(s.name, args.test_fraction) == "test"]),
        ]

    # What the thresholds the app currently ships would have done, so a run can
    # be read against the code as it stands rather than only against its own
    # best case.
    usable = eligible(scores)
    shipped = [s.verdict(DEFAULT_MIN_PHASE_MARGIN, DEFAULT_MIN_METER_MARGIN)
               for s in usable]
    shown = sum(v in Verdict.SHOWN for v in shipped)
    wrong = sum(v in Verdict.WRONG for v in shipped)
    print(f"\nat the shipped thresholds "
          f"(phase {DEFAULT_MIN_PHASE_MARGIN}, metre {DEFAULT_MIN_METER_MARGIN}): "
          f"{shown}/{len(usable)} accented, {wrong} wrong")

    chosen = None
    for label, subset in splits:
        rows = sweep(subset)
        print(f"\n--- {label} ({len(eligible(subset))} clip(s) usable) ---")
        if not any(r.n for r in rows):
            print("no clip in this split can be calibrated on.")
            continue
        print(format_sweep(frontier(rows)))

        if label != "test":
            chosen = choose_margins(rows, args.max_wrong_rate,
                                    args.max_conditional_error)
            if chosen is None:
                print(f"\nNo threshold pair keeps the wrong rate at or under "
                      f"{args.max_wrong_rate:.0%} and the conditional error "
                      f"under {args.max_conditional_error:.0%}. On this material "
                      f"the accent should stay off rather than the budget be "
                      f"raised.")
            else:
                print(f"\nchosen on {label}: phase >= {chosen.min_phase_margin:.2f}, "
                      f"metre >= {chosen.min_meter_margin:.2f} — coverage "
                      f"{chosen.coverage:.0%}, wrong rate {chosen.wrong_rate:.0%}, "
                      f"conditional error {chosen.conditional_error:.0%}")

    # The one number the whole exercise is for: the chosen thresholds applied
    # once to material they were not chosen on. Reported whatever it says.
    if chosen is not None and not args.no_split:
        held_out = [s for s in eligible(scores)
                    if split_of(s.name, args.test_fraction) == "test"]
        if held_out:
            verdicts = [s.verdict(chosen.min_phase_margin, chosen.min_meter_margin)
                        for s in held_out]
            shown = sum(v in Verdict.SHOWN for v in verdicts)
            wrong = sum(v in Verdict.WRONG for v in verdicts)
            print(f"\nheld out, at phase >= {chosen.min_phase_margin:.2f} and "
                  f"metre >= {chosen.min_meter_margin:.2f}: "
                  f"{shown}/{len(held_out)} accented, {wrong} wrong "
                  f"({wrong / len(held_out):.0%})")
        else:
            print("\nheld-out split is empty — too few clips to hold any out.")

    if args.expect_correct is not None:
        # Counted with the thresholds out of the way, so this measures the
        # analysis and not the thresholds: withholding an accent is a product
        # decision, and a change that starts withholding everything should show
        # up here as clips lost, not be hidden by thresholds that agree.
        correct = sum(s.verdict(0.0, 0.0) == Verdict.CORRECT
                      for s in scores if s.scorable)
        print(f"\n{correct} clip(s) right on metre and phase "
              f"(expected at least {args.expect_correct})")
        if correct < args.expect_correct:
            print("REGRESSION")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
