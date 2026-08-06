---
spike: 001
name: octave-veto-a1-a4
type: standard
validates: "Given real live octave proposals, when a causal downbeat veto is replayed at matched cost, then A1--A4 and all standing gates hold"
verdict: PENDING
related: []
tags: [live, tempo, downbeat, replay]
---

# Spike 001: Octave veto A1--A4

## What This Validates

The executable contract is
[`research/eval/PREREGISTERED_octave_veto.md`](../../../research/eval/PREREGISTERED_octave_veto.md).
This spike is complete only after RWC fixes the threshold and matched policies,
that selection is committed, and one Harmonix transfer run evaluates A1--A4.

## Research

No external library is introduced. Existing pieces are reused:

| Approach | Reused component | Advantage | Risk | Status |
|---|---|---|---|---|
| Python particle-filter replay | none | Easy to sweep | Would measure a second tracker | rejected |
| C++ copy of the decoder | live core | One-pass stateful run | Two implementations of the registered statistic | rejected |
| Fixed-point schedule through core | Python decoder + `LiveTracker` anchor seam | One formula and real filter consequences | Must prove convergence and cached-activation parity | chosen |

## How to Run

From `research/`, after building `dump_analysis` and providing the fold-1 model:

```powershell
python -m eval.octave_veto_experiment rwc `
  --binary ..\tools\eval\build\RelWithDebInfo\dump_analysis.exe `
  --model <fold1.weights> `
  --manifest <music>\ground-truth\manifest.csv --music <music> `
  --output <outside-repo>\octave-veto-rwc.json
```

The RWC command requires the implementation to be committed and the worktree
to be clean. Copy its JSON result into the repository and commit that file by
itself as the next commit. Only then:

```powershell
python -m eval.octave_veto_experiment harmonix `
  --binary ..\tools\eval\build\RelWithDebInfo\dump_analysis.exe `
  --model <fold1.weights> --selection <committed-rwc-selection.json> `
  --manifest <music>\ground-truth\manifest.csv --music <music> `
  --output <outside-repo>\octave-veto-harmonix.json
```

The Harmonix command refuses a dirty worktree or a selection file not committed
at the current `HEAD`. It also refuses to open Harmonix unless RWC passed A2,
A3 and produced a within-0.5-point match in every simple-policy family.

## What to Expect

- RWC evaluates the exact 21-value threshold grid and every registered
  debounce, margin, rate-limit and total-ban candidate.
- Every scheduled arm reaches a fixed point in at most eight passes.
- The cached activation path matches the direct model run before policy replay.
- Harmonix prints one final `accepted: true|false`; no retuning path exists.

## Observability

The JSON artifact includes candidate metrics, event coverage, A2/A3, maximum
fixed-point passes, selected event decisions, the three-test Holm family and
all five standing cost gates. It also carries P2 sign agreement, ambiguity,
D1 and coverage by annotated metre. `dump_analysis` reports applied interval
and frame counts so an empty or ignored schedule cannot look like a successful
arm.

## Investigation Trail

1. Reused the existing Python decoder and live replay instead of rebuilding
   either algorithm.
2. Added an allocation-free, null-by-default anchor resolver to the C++ live
   core and a schedule-only CLI seam in `dump_analysis`.
3. Found two event-state bugs before corpus access: close time was the start of
   the one-second confirmation window, and merged events did not extend the
   first event's close. Both now have regression tests.
4. Defined the previously unnamed shifted-null control and the exact
   recording-balanced aggregation before seeing RWC.
5. Built a two-stage CLI that enforces RWC selection before Harmonix transfer.

## Results

PENDING. Synthetic, core and integration verification passes. RWC has not been
opened yet; its corpus and fold-1 model are supplied explicitly because they
live outside this worktree.
