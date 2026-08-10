# Pre-registered: where the metrical deficit lives, and whether long recurrent state is worth training for

Written before either experiment exists. Nothing below was chosen after seeing a
number it is measured against.

Two experiments, registered together because both run before any training, both
gate work downstream of them, and both would otherwise be read without a rule.

* **M0a** — is the bar phase lost in the decoder, or upstream of it?
* **S0** — does BeatNet's recurrent state actually carry bar-level structure at
  inference, before we pay to train for it?

## Why these two need a registration at all

The product goal moved from BPM to metrical structure. Every gate in
`plan.md` §8 is fixed after the pilot; these two run **before** it, and both
carry binding consequences:

* a negative M0a stops neural front-end work and redirects to the decoder;
* a negative S0 removes stateful block training from the top of the queue.

A binding consequence read without a threshold is not binding. The repository
already has a recorded instance of this failure mode — the original downbeat
audit compared 0.608 against a shuffled 0.235 and never against the 0.949
constant, and the constant was what decided it. This document exists so the
same thing cannot happen to the first two experiments of a new programme.

---

# M0a — the oracle ladder

## The question

`BarTracker`/`resolveMeter` turns a beat grid plus a downbeat channel into a bar
line. The end-to-end level of that is already measured: **bar-line agreement F1
of 0.522 on GTZAN and 0.606 on Harmonix**, against a `random_phase` null of
0.209 and 0.217 (`causal_metre_gtzan.json`, `research/results/README.md`).

What is not measured is where the missing 0.48 and 0.39 go. Two candidates, and
they lead to opposite plans:

* the downbeat evidence is too weak → the work is in the front end, and the rest
  of `plan.md` is correct;
* the decoder cannot use even good evidence → a bigger network is not warranted
  and the work is in `BarTracker` or the label contract.

## The arms

One decoder, one scorer, four inputs. The decoder is the shipped causal
`BarTracker/resolveMeter` at its shipped configuration; nothing about it is
tuned in this experiment.

| Arm | Tactus grid | Downbeat evidence | Isolates |
|---|---|---|---|
| **A1** | reference | reference | the decoder alone |
| **A2** | reference | predicted | the downbeat channel |
| **A3** | predicted | oracle phase | whether the grid is good enough to hold a phase |
| **A4** | predicted | predicted | end to end |

A4 is the arm already measured as `beat_sync` restricted to recordings tracked
at the annotated level. It is re-run rather than quoted, because the restriction
set must be identical across all four arms or the columns are not comparable.

**All four arms must publish byte-identical tactus grids wherever the grid is
the same input.** A1 and A2 share the reference grid; A3 and A4 share the
predicted one. If they diverge, something other than the registered factor
changed and the run is void.

## What is measured

**F1 — bar-line agreement at 70 ms**, over the beats after the metre settled.
This is the existing definition in `causal_metre.py` and is not redefined here.

**The null is `random_phase`**: the mean over all rotations of the same settled
grid, which is the exact expectation of a uniformly random bar line. It is not a
shuffle. With the metre almost always four, a shuffled channel and a rotated one
are different objects and the rotation is the honest one.

Secondary, reported but not deciding: F2 phase-correct share per beat; time to
first correct bar line; `tactus_beats_per_bar` accuracy against the `always 4`
constant.

**Not measured: denominator, `meter_family`, `canonical_time_signature`.**
`BarTracker` returns grouping only. Scoring it on outputs it does not produce
would measure the absence of a feature, not the quality of one.

## Acceptance

Stated as three bands, because the consequences are asymmetric and the material
supports a hard conclusion in only one direction.

**Band 1 — hard negative, binding.** A1 fails to exceed A4 by **20 points of F1
on either corpus**. Given the measured A4 of 0.522 and 0.606, that is A1 below
**0.722 on GTZAN or below 0.806 on Harmonix**.

Reading: perfect beats and perfect bar lines, handed to the decoder, buy less
than 20 points over what the predicted channel already achieves. The decoder is
then discarding evidence it is given, and no improvement to the front end can be
worth more than what the decoder throws away. **This stops neural front-end work
for the metrical goal** and redirects to `BarTracker` and the label contract.
The 20-point margin is the same one the earlier phase registration used to
separate a real effect from its control, and is adopted rather than invented.

**Band 2 — not falsified, preliminary only.** A1 clears Band 1 but stays below
0.90 on either corpus. The decoder is not the dominant loss, but it is not
transparent either. This licenses continuing, and licenses nothing else — no
threshold anywhere else in the plan may be set from this run.

**Band 3 — decoder transparent.** A1 reaches **0.90 or above on both corpora**.
The metrical deficit is then upstream in its entirety, and A2 measures how much
of it the downbeat channel owns.

**A1 must clear `random_phase` by at least 20 points as a precondition of the
run being interpretable at all.** An oracle arm that cannot beat a rotated grid
means the harness is wrong, not that the decoder is.

**Interpretation of the remaining arms, fixed now:**

* A1 high, A2 low → the downbeat channel is the constraint. `plan.md` proceeds
  as written.
* A2 high, A3 low → the tactus grid is the constraint, not the downbeat channel,
  and the front-end work is on beats rather than on bar lines.
* A3 high, A4 low → grid and phase are each recoverable but not jointly, which
  points at the interaction and not at either component.

## Corpora

**GTZAN and Harmonix.** GTZAN 991 of 999 (`jazz.00054` is not a WAV), Harmonix
579 of 581 — the counts from the downbeat audit, and any further exclusion must
be listed with its reason before the run.

Ballroom is excluded: shipped fold 1 trained on it.

**Both corpora are development ground and this run does not change that.**
Harmonix was spent on the ensemble test; RWC was spent on the anchor width. The
binding output of M0a is a *negative*, and a negative on development material is
still a negative — that is why Band 1 is the only band with a consequence
attached. Bands 2 and 3 set no thresholds anywhere.

**What this material cannot answer**, stated so it is not attempted later: 95%
of GTZAN and 98% of Harmonix are in four. Grouping and metre are unanswerable
here and are M0b's job on meter-diverse data. **Phase is the half this material
does vary**, which is why phase is the endpoint.

## Predictions

Recorded so that being wrong is visible.

* **P1.** A1 lands in Band 3. The decoder is handed the bar lines; reproducing
  them is close to a copy.
* **P2.** A2 lands between A1 and A4 and nearer A4, because the downbeat channel
  is the weak link and A4's 0.522/0.606 is mostly its doing.
* **P3.** A3 exceeds A4 on Harmonix by more than on GTZAN, because full-length
  songs lose the level and excerpts lose beats.

## What would sink this

* A1 and A4 come out close **and high**. Then the 20-point rule fires a hard
  negative on material where the decoder was in fact fine, and the rule is
  wrong rather than the decoder. Guard: the Band 1 test is a margin *and* the
  precondition above; if A4 itself exceeds 0.80 the margin rule is suspended and
  the run is reported as inconclusive rather than negative.
* The reference grid and the predicted grid settle at different times, so
  "beats after the metre settled" selects different beats per arm. Guard: the
  settle point is taken from the arm's own grid and the per-arm beat counts are
  published; a difference above 5% voids the comparison.
* Reference downbeats are injected in a form the decoder treats differently from
  a model channel — a hard 1.0 where it expects a distribution. Guard: the
  injection format is fixed before the run and reported.

## Not in scope

Tuning `BarTracker`. Changing the settle rule. Any claim about metre, grouping,
denominator, or about rooms. Any threshold for the model gate.

---

# S0 — reset horizon

## The question

`plan.md` promotes stateful block training (S1) on the argument that our C++
path carries recurrent state through a whole recording while upstream training
resets per excerpt, so the state never learns to carry bar-level structure.

That argument is plausible and untested. If long state does not help **at
inference on the frozen model**, training for it is not the first thing to buy.

BEAST does not settle this either way: it trains on whole sequences or 30 s
clips, and carrying LSTM state across truncated-BPTT blocks is our hypothesis,
not its result.

## The arms

Frozen `beatnet_model_1`, shipped configuration, no training of any kind.

| Arm | `BeatNetModel` state |
|---|---|
| R2 | reset every 2 s |
| R4 | reset every 4 s |
| R8 | reset every 8 s |
| R16 | reset every 16 s |
| R32 | reset every 32 s |
| R∞ | never reset within a recording |

## What is isolated, and what must not move

This is the whole experiment: a reset that touches more than the recurrent state
measures the damage of an artificial discontinuity, not the value of memory.

* **only** `BeatNetModel` recurrent state is reset;
* feature history, `LiveTracker` and `BarTracker` are **not** reset;
* the post-reset transient is reported **separately**, and the deciding metric
  is also computed with a masked warm-up after each reset;
* reset points are identical across arms and at fixed wall-clock offsets —
  **never** chosen at musical boundaries, or the experiment contains a choice of
  boundaries rather than a horizon;
* the same recordings, the same restriction set and the same scorer as M0a.

## What is measured

Primary: **bar-phase F1**, the same definition as M0a. Secondary: downbeat F,
beat F, `usable_strict`.

The comparison is paired per recording, and the interval is a
composition-level bootstrap.

## Acceptance

**S0 positive** — all three hold:

1. **R∞ exceeds R2 by at least 5 points of bar-phase F1 on both corpora**, with
   the paired bootstrap lower bound above zero;
2. the horizon series is **monotone non-decreasing** from R2 to R∞ within
   overlapping intervals — an effect that appears only at one horizon and
   reverses elsewhere is noise, not memory;
3. the effect survives warm-up masking, i.e. it is not the reset transient.

**S0 negative** — condition 1 fails. S1 leaves the top of the queue and becomes
a late ablation. It is not deleted: a training-time benefit could exist without
an inference-time one, but it stops being the first thing bought.

**S0 ambiguous** — condition 1 holds and 2 or 3 fails. Reported as ambiguous and
S1 stays where a negative would put it. Ambiguity resolves downward, not upward.

The 5-point margin is smaller than M0a's 20 because the question is different:
M0a asks whether an effect is large enough to redirect a programme, S0 asks
whether an effect exists at all before we pay for it.

## Predictions

* **P4.** R2 is clearly worst and the series rises. A 2 s window is under one bar
  at most tempi.
* **P5.** The series flattens by R8 or R16, so R∞ − R2 is real but R∞ − R16 is
  small. If so, S1's ceiling is small even when S0 is positive, and that is
  worth knowing before F1 of the executive plan is built.
* **P6.** Beat F moves less than bar-phase F1. Beats are local; bar lines are not.

## What would sink this

* Frozen BeatNet is not the model S1 would produce, so a null here bounds the
  inference-time value of state in **this** model and not the trainable value of
  it in another. This is the same class of limit as truncating a bidirectional
  model, and it is why a negative deprioritises S1 rather than closing it.
* The 50 fps stream means R2 is 100 frames; if the network's effective memory is
  shorter than that, every arm is R∞ and the experiment measures nothing.
  Guard: report the frame counts and treat a flat series across all horizons as
  a null result about the *network*, not about the hypothesis.

## Not in scope

Any training. Any change to reset semantics in the shipped path. Any claim about
what a model trained with carried state would do.

---

## Provenance and execution, both experiments

* Run from a clean eval worktree; artifacts written outside it.
* `provenance.provenance()` for every artifact — `tree_clean` must be `true`,
  not `null`. A run whose provenance is unknown is repeated, not annotated.
* SHA-256 of the binary, the weights and every annotation file in the artifact.
* The phase scorer, the `random_phase` null and the Wilson intervals are reused
  from `causal_metre.py` unchanged. The arms are new: injecting reference
  downbeats and oracle phase is not something that harness does today.
* Registered before any of it exists. Deviations found during the run are
  appended below with their direction, never folded into the text above.
