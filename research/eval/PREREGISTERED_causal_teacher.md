# Pre-registered: how much of Beat This!'s advantage is the model, and how much is the lookahead?

Written before the script exists. Nothing below was chosen after seeing a number
it is measured against.

## Why now, and why this is the last gate before training

The decision to train a causal front end was gated on three conditions. Two are
now measured:

1. **the real↔oracle gap survives precision and F1** — Harmonix, 581
   recordings: `usable` 0.365 → 0.952, `p70` 0.798 → 0.973, `r70` 0.807 →
   0.993 under a perfect observation, and precision failures disappear
   entirely;
2. **the gap is not closed by tracker settings** — the matched-corpus agility
   sweep on all 328 RWC recordings: the best any raised setting buys on the
   real observation is +0.020 on one sub-corpus of 102, while costing pop 0.040
   to 0.060. The limit is structural.

The third is open: **is a causal teacher actually better than BeatNet?** Without
it, training aims at a target nobody has shown a causal model can reach.

Two numbers exist and cannot be composed. Beat This! through the *shipped
decoder* is +0.138 mean F on GTZAN — but unbounded. Beat This! bounded to one
second of lookahead loses 3.1 points — but measured through *its own*
postprocessor, at one-second granularity, which `eval/beat_this_causal.py`
itself says is "more than the whole margin being measured". One is the right
decoder with the wrong causality, the other the reverse.

## The measurement

Both arms enter the same `LiveTracker` the same way and differ only in the
activation.

- **seam** — every arm through `--live-activation` at 50 fps, sampled at 50 Hz.
  BeatNet's own activation is obtained with `--dump-activation` and replayed
  through the identical path, so no arm gets a different delivery. That seam was
  already measured at +0.009, which is the resolution this comparison has.
- **bounded context** — activations computed on prefixes at **0.1 s**
  granularity. Frame `t` takes its value from the shortest prefix ending at or
  after `t + L`. One set of prefix passes serves every `L`.
- **arms** — `beatnet`; `at_most_0.1s`, `at_most_0.2s`, `at_most_0.3s`,
  `at_most_0.5s`; and `offline`, the whole file.
- **corpus** — 40 GTZAN clips on a fixed stride across the genre ordering, so
  no genre is absent. Named in the artifact.

**The bound is a range and the labels say so.** A frame at `t` read from the
prefix `t + L` saw exactly `L` of its future; a frame 0.1 s earlier in the same
window saw `L + 0.1`. Arms are therefore named for `L + 0.1`, the upper bound,
never for `L`. Calling the tightest arm "strictly causal" would be wrong by up
to 100 ms, and the live metronome's own lookahead is 50 ms.

## Primary readout

Mean F through the tracker, per arm, and the share of the advantage that
survives each bound:

    survival(L) = (F(L) - F_beatnet) / (F_offline - F_beatnet)

## The registered prediction, so this can fail

**At the tightest bound, `at_most_0.1s`, at least half the advantage survives**
— `survival >= 0.50`.

If it does, most of Beat This!'s advantage is the model rather than the
lookahead, and a causal student has something real to be aimed at. If less than
half survives, the advantage is mostly future context, no causal model can
inherit it, and the front-end gain has to come from training distribution —
room-matched data — rather than from architecture. That is a different project
plan, and it is better to learn it before any weights exist than after.

## What no outcome licenses

**Not the absolute level.** `models/beat_this.onnx` is `final0`, trained on
sixteen sets including GTZAN, and GTZAN is BeatNet `model_1`'s held-out fold.
The comparison is therefore maximally unfavourable to BeatNet in both
directions at once. The *shape* — how F falls as the bound tightens — is a
comparison within one model and survives that; the height of the curve does not
and must not be quoted as an achievable target.

**Not a claim about a trained causal model.** This measures what a non-causal
model retains under a causal constraint. That is an upper bound on what a causal
architecture could inherit, not a prediction of what one would achieve.

**Not a room result.** Every clip is a clean 30-second excerpt.

## Bound in advance

Granularity, arms, corpus size and the primary readout are fixed here. If the
run must be cut short, clips are dropped from the end of the fixed stride and
the artifact says how many — never re-sampled to a different selection.
