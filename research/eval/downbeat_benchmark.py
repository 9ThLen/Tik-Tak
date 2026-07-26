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
covering what the app will actually meet are enough for a pilot, not a claim.
With zero failures, the 95% Wilson bound needs 73 independent test groups for a
5% wrong-rate budget and 35 shown groups for a 10% conditional-error budget.
At the default 35% held-out share, 73 test groups correspond to about 209 total
groups **in expectation**, not as a guarantee: the realized held-out count
printed by the run is authoritative. 200–300 grouped recordings are therefore
only a practical target when each is its own independent group. Real
validation/test runs require ``groups.json`` so excerpts of one song, session,
or backing track cannot leak across the split or inflate Wilson's denominator;
see eval/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

from eval.analysis import Analyser, DEFAULT_BINARY
from eval.annotations import Reference, find_pairs
from eval.downbeat import (
    Verdict,
    choose_margins,
    eligible,
    evidence_gap,
    format_scores,
    format_sweep,
    frontier,
    grouped_shipped_verdicts,
    grouped_verdicts,
    score,
    sweep,
    wilson_upper,
)
from tiktak.synth import make_clip


GROUPS_FILENAME = "groups.json"


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
            group_id=name,
            bar_lengths=(clip.beats_per_bar,) * max(len(clip.downbeats) - 1, 0),
        )
        cases.append((reference, clip.audio, clip.sample_rate))
    return cases


def _canonical_id(value: str) -> str:
    """Make path-shaped identifiers stable across Windows and POSIX."""
    return value.replace("\\", "/").strip("/")


def load_groups(path: pathlib.Path) -> dict[str, str]:
    """Read a ``clip name -> independent group id`` JSON object."""
    path = pathlib.Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read group manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: group manifest must be a JSON object")

    groups: dict[str, str] = {}
    for clip, group in raw.items():
        if not isinstance(clip, str) or not clip.strip():
            raise ValueError(f"{path}: every clip id must be a non-empty string")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(
                f"{path}: group for {clip!r} must be a non-empty string"
            )
        canonical = _canonical_id(clip)
        canonical_group = _canonical_id(group.strip())
        if not canonical_group:
            raise ValueError(
                f"{path}: group for {clip!r} must contain an identifier"
            )
        if canonical in groups and groups[canonical] != canonical_group:
            raise ValueError(
                f"{path}: {clip!r} duplicates a clip id after path normalisation"
            )
        groups[canonical] = canonical_group
    return groups


def group_manifest_path(dataset: pathlib.Path,
                        requested: pathlib.Path | None,
                        no_split: bool) -> pathlib.Path | None:
    """Resolve grouping, refusing a real held-out split without it."""
    if requested is not None:
        return requested
    default = pathlib.Path(dataset) / GROUPS_FILENAME
    if default.is_file():
        return default
    if no_split:
        return None
    raise ValueError(
        f"{default}: a group manifest is required for validation/test splitting; "
        "pass --groups or use --no-split for an exploratory all-clips report"
    )


def split_of(name: str, test_fraction: float, salt: str = "tiktak",
             *, group_id: str | None = None) -> str:
    """Which split a clip belongs to, decided by its independent group.

    With no group id this preserves the old name-based behaviour. Supplying a
    song, recording-session, or backing-track id keeps every related excerpt in
    one split. Hashing rather than shuffling means a group stays in the same
    split when the set grows.
    """
    key = group_id if group_id is not None else name
    digest = hashlib.sha256(
        f"{salt}:{_canonical_id(key)}".encode("utf-8")
    ).digest()
    position = int.from_bytes(digest[:4], "big") / 2**32
    return "test" if position < test_fraction else "validation"


def held_out_error_report(wrong: int, total: int, shown: int,
                          max_wrong_rate: float = 0.05,
                          max_conditional_error: float = 0.10) -> str:
    """Both held-out point estimates and independent-group Wilson bounds."""
    wrong_rate = wrong / total if total else float("nan")
    conditional = f"{wrong / shown:.2%}" if shown else "n/a"
    wrong_upper = wilson_upper(wrong, total)
    conditional_upper = wilson_upper(wrong, shown)
    demonstrated = (
        total > 0
        and shown > 0
        and wrong_upper <= max_wrong_rate
        and conditional_upper <= max_conditional_error
    )
    conclusion = "DEMONSTRATED" if demonstrated else "NOT DEMONSTRATED"
    return (
        f"wrong rate {wrong_rate:.2%} "
        f"(95% upper {wrong_upper:.2%}), "
        f"conditional error {conditional} "
        f"(95% upper {conditional_upper:.2%}); both budgets {conclusion}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path,
                        help="folder of audio with annotations beside it; "
                             "omit for synthetic clips")
    parser.add_argument(
        "--groups",
        type=pathlib.Path,
        help=f"JSON object mapping clip names to independent group ids; "
             f"defaults to DATASET/{GROUPS_FILENAME} when that file exists",
    )
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY,
                        help="path to the dump_analysis executable")
    parser.add_argument("--max-wrong-rate", type=float, default=0.05,
                        help="wrong accents as a share of independent groups "
                             "(default 0.05)")
    parser.add_argument("--max-conditional-error", type=float, default=0.10,
                        help="wrong accents as a share of the accents actually "
                             "shown (default 0.10)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="score the clips that loaded even when others in "
                             "the dataset could not be read")
    parser.add_argument("--test-fraction", type=float, default=0.35,
                        help="share of independent groups held out; the threshold "
                             "is chosen on the rest (default 0.35)")
    parser.add_argument("--no-split", action="store_true",
                        help="score every clip together — for a quick look, not "
                             "for choosing a threshold")
    parser.add_argument("--expect-correct", type=int, default=None,
                        help="fail unless at least this many clips come back "
                             "with the right metre and phase; how CI notices a "
                             "regression")
    args = parser.parse_args(argv)

    if (not np.isfinite(args.test_fraction)
            or not 0.0 < args.test_fraction < 1.0):
        parser.error("--test-fraction must be finite and strictly between 0 and 1")
    if args.groups is not None and args.dataset is None:
        parser.error("--groups requires --dataset")

    analyser = Analyser(args.binary)
    if not analyser.available:
        print(f"not found: {args.binary}")
        print("Build it first — see eval/analysis.py.")
        return 2

    scores = []
    group_ids: dict[str, str] = {}
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

        try:
            groups_path = group_manifest_path(
                args.dataset, args.groups, args.no_split
            )
        except ValueError as exc:
            print(exc)
            return 2
        if groups_path is not None:
            try:
                group_ids = load_groups(groups_path)
            except ValueError as exc:
                print(exc)
                return 2
            missing = [
                reference.name for reference in references
                if _canonical_id(reference.name) not in group_ids
            ]
            if missing:
                print(f"{groups_path}: no group id for "
                      f"{', '.join(repr(name) for name in missing)}")
                return 2
            print(f"splitting by {len(set(group_ids.values()))} independent "
                  f"group(s) from {groups_path}")

        print(f"scoring {len(references)} annotated recording(s) from {args.dataset}\n")
        for reference in references:
            reference.group_id = group_ids.get(
                _canonical_id(reference.name), reference.name
            )
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

    def split_for(item) -> str:
        return split_of(
            item.name, args.test_fraction, group_id=item.group_id
        )

    usable = eligible(scores)
    if not args.no_split:
        realized = {
            label: {
                item.group_id for item in usable if split_for(item) == label
            }
            for label in ("validation", "test")
        }
        if not realized["validation"] or not realized["test"]:
            print("\ncannot calibrate: the realized split must contain eligible "
                  "independent groups on both sides; got "
                  f"{len(realized['validation'])} validation and "
                  f"{len(realized['test'])} test group(s). Add independent "
                  "groups or adjust --test-fraction.")
            return 2

    if args.no_split:
        splits = [("all clips", scores)]
    else:
        splits = [
            ("validation", [s for s in scores
                            if split_for(s) == "validation"]),
            ("test", [s for s in scores
                      if split_for(s) == "test"]),
        ]

    # What the thresholds the app currently ships would have done, so a run can
    # be read against the code as it stands rather than only against its own
    # best case.
    shipped = grouped_shipped_verdicts(usable)
    shown = sum(v in Verdict.SHOWN for v in shipped)
    wrong = sum(v in Verdict.WRONG for v in shipped)
    print(f"\nat the thresholds reported by the C++ core: "
          f"{shown}/{len(shipped)} independent group(s) accented, {wrong} wrong")

    chosen = None
    for label, subset in splits:
        rows = sweep(subset)
        independent = len({
            s.group_id for s in eligible(subset)
        })
        print(f"\n--- {label} ({independent} independent group(s) usable) ---")
        if not any(r.n for r in rows):
            print("no independent group in this split can be calibrated on.")
            continue
        print(format_sweep(frontier(rows)))

        if args.no_split:
            print("\nexploratory --no-split run: no threshold is calibrated "
                  "without independent validation/test groups.")
            continue

        if label != "test":
            chosen = choose_margins(rows, args.max_wrong_rate,
                                    args.max_conditional_error)
            if chosen is None:
                # Two reasons produce the same None and they are not the same
                # news. Either the cues genuinely cannot separate right from
                # wrong on this material, or they might and the split is too
                # small to tell. Saying which is the difference between "leave
                # the feature off" and "go and record more".
                unbounded = choose_margins(rows, args.max_wrong_rate,
                                           args.max_conditional_error,
                                           bounded=False)
                if unbounded is None:
                    print(f"\nNo threshold pair keeps the wrong rate at or under "
                          f"{args.max_wrong_rate:.2%} and the conditional error "
                          f"under {args.max_conditional_error:.2%}. On this "
                          f"material the accent should stay off rather than the "
                          f"budget be raised.")
                else:
                    print(f"\nnothing calibrated on {label}. Where it would land "
                          f"on the observed rates alone: phase >= "
                          f"{unbounded.min_phase_margin:.2f}, metre >= "
                          f"{unbounded.min_meter_margin:.2f} — "
                          f"{unbounded.shown} shown, {unbounded.wrong} wrong.")
                    gap = evidence_gap(unbounded, args.max_wrong_rate,
                                       args.max_conditional_error)
                    if gap is not None:
                        print(gap)
                    else:
                        print("not demonstrated: the 95% Wilson bounds reject "
                              "this threshold.")
            else:
                print(f"\nchosen on {label}: phase >= {chosen.min_phase_margin:.2f}, "
                      f"metre >= {chosen.min_meter_margin:.2f} — coverage "
                      f"{chosen.coverage:.2%}, wrong rate {chosen.wrong_rate:.2%}, "
                      f"conditional error {chosen.conditional_error:.2%} "
                      f"(upper bounds {chosen.wrong_rate_upper:.2%} / "
                      f"{chosen.conditional_error_upper:.2%})")

    # The one number the whole exercise is for: the chosen thresholds applied
    # once to material they were not chosen on. Reported whatever it says.
    if chosen is not None and not args.no_split:
        held_out = [s for s in eligible(scores)
                    if split_for(s) == "test"]
        if held_out:
            verdicts = grouped_verdicts(
                held_out, chosen.min_phase_margin, chosen.min_meter_margin
            )
            shown = sum(v in Verdict.SHOWN for v in verdicts)
            wrong = sum(v in Verdict.WRONG for v in verdicts)
            error_report = held_out_error_report(
                wrong, len(verdicts), shown,
                args.max_wrong_rate, args.max_conditional_error,
            )
            print(f"\nheld out, at phase >= {chosen.min_phase_margin:.2f} and "
                  f"metre >= {chosen.min_meter_margin:.2f}: "
                  f"{shown}/{len(verdicts)} independent group(s) accented, "
                  f"{wrong} wrong "
                  f"— {error_report}")
        else:
            print("\nheld-out split is empty — too few groups to hold any out.")

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
