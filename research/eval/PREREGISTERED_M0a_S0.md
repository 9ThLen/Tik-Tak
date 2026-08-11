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
`tracking::BarTracker` at its shipped configuration; nothing about it is tuned
in this experiment. `dump_analysis --beats --salience` is **not** this seam: it
calls the batch `analysis::resolveMeter` after the live run has finished, while
the registered A4 is produced by `LiveTracker → BarTracker`. M0a therefore uses
a causal BarTracker replay for all four arms. The replay consumes the exact
frame timestamps and frame-release schedule recorded from the live run. For the
predicted grid it also consumes the recorded beat-publication schedule. A4 must
reproduce the ordinary live `beat_sync` arm before any oracle comparison is
accepted.

| Arm | Tactus grid | Downbeat evidence | Isolates |
|---|---|---|---|
| **A1** | reference | reference | the decoder alone |
| **A2** | reference | predicted | the downbeat channel |
| **A3** | predicted | oracle phase | whether the grid is good enough to hold a phase |
| **A4** | predicted | predicted | end to end |

A4 is the arm already measured as `beat_sync` restricted to recordings tracked
at the annotated level. It is re-run through the causal replay rather than
quoted. Its tactus grid, per-beat meter and bar position must be byte-identical
to the ordinary live run; any mismatch voids the recording and the replay must
be fixed before the experiment continues.

**All four arms must publish byte-identical tactus grids wherever the grid is
the same input.** A1 and A2 share the reference grid; A3 and A4 share the
predicted one. If they diverge, something other than the registered factor
changed and the run is void.

## What is measured

**F1 — bar-line agreement at 70 ms**, over one common time span per recording.
The first-decision time is the first beat where an arm's meter is non-zero; it
is deliberately not called stability because `score_phase` does not test a
stable suffix. The deciding suffix begins at the **latest** first-decision time
among A1–A4. All arms are rescored on that same suffix. First-decision time and
coverage before and after the common cut are reported per arm.

**The null is `random_phase`**: the same metre and the same settled grid with the
bar line rotated, averaged over all rotations — the exact expectation of a
uniformly random bar line. It is not a shuffle.

`causal_metre.py` builds that null for `beat_sync` only. **Each arm gets its own
null, built by rotating that arm's own settled grid**, and every arm's null is
reported. A null taken from a different arm's grid would compare against a
different denominator.

Secondary, reported but not deciding: F2; time to first correct bar line;
`tactus_beats_per_bar` accuracy against the `always 4` constant.

**F2 is binary bar-line agreement, not position accuracy.** As implemented it
scores `(shifted == 0) == should_be_zero`, i.e. whether each beat agrees about
being a downbeat or not. It does **not** check that a beat at position 2 is
called 2. Exact `tactus_position_in_bar` accuracy is a different metric, is not
implemented, and is not an endpoint of this run. Reporting F2 as "position
accuracy" would overstate what was measured.

## Definitions fixed before the run

Everything below was left implicit in the first draft of this document and is
fixed now, before any number exists. An earlier version said the oracle
injection format would be "fixed before the run and reported", which is a
loophole: fixed outside the registration is not registered.

**Oracle input construction.**

* Reference downbeats become a 50 fps channel by placing a **single frame at
  1.0** at the frame nearest each reference downbeat, zero elsewhere — no
  smoothing, no Gaussian. Ties between two equidistant frames resolve to the
  earlier frame.
* The reference tactus grid is fed as the beat sequence directly; it is not
  re-derived from an activation. Reference beats have no product publication
  schedule to record, so their schedule is fixed here: each is published on the
  first 512-sample device block whose ending clock puts it within the shipped
  50 ms lookahead (`beat <= now + 0.05`). This is the earliest schedule the
  product could publish that timestamp and is generated without consulting
  meter or phase. A1 and A2 use the same generated block indices.
* **A hard impulse is not what the decoder normally sees.** The earlier
  three-frame triangular sensitivity arm is withdrawn because BarTracker takes
  a maximum over ±70 ms: both constructions have the same peak of 1.0 and are
  almost the same input to this decoder. Sensitivity is instead measured with
  amplitudes 0.5 and 1.0 plus a deterministic timing-jitter arm of alternating
  −20/+20 ms. If either changes A1 by more than 5 points of F1, the oracle input
  format dominates and the M0a verdict is withheld.
* **A3's oracle phase** is the rotation of the arm's own predicted grid that
  maximises bar-line F1 against the reference downbeats. After the rotation is
  chosen, A3 receives an explicit 50 fps evidence channel: value 1.0 at the
  nearest frame to every predicted beat selected as a bar line by that rotation,
  zero elsewhere, with the same earlier-frame tie rule. The causal BarTracker,
  not the scorer, still produces A3's meter and positions. Recordings with a
  reference meter change are outside M0a and are excluded by annotation before
  any arm is run; M0b owns changing meter.
* Unmatched beats and unmatched downbeats count as they already do in
  `score_phase` — a claimed beat with no reference downbeat within 70 ms is a
  precision miss, a reference downbeat with no claimed beat within 70 ms is a
  recall miss. Nothing is dropped.

**The primary set is intention-to-treat over technically valid recordings.** A
recording that loads, has valid annotations and reaches all four arms remains in
the denominator. If an arm never produces a phase, its F1 is 0.0; decoder
abstention is a failure of that arm, not a reason to remove the recording.
Technical exclusions (malformed audio, missing annotation, failed invariant)
are fixed before arm output is inspected and published with reasons. A
four-arm complete-case intersection is reported only as a sensitivity analysis.

**Interval method.** F1 here is a mean of per-recording F-measures, not a
proportion, so **Wilson does not apply to it** — `causal_metre.py` already says
so and bootstraps instead. An earlier version of this document named Wilson for
F1 and was wrong. Wilson stays where it belongs: the accuracy proportions.

**Resampling unit.** The bootstrap resamples **recordings**, because neither
GTZAN nor Harmonix carries a composition grouping in this harness. That is a
weaker unit than the composition-level bootstrap the model gate will need, and
it is recorded as a limitation of M0a rather than quietly called composition
level. Two GTZAN excerpts of the same piece would be treated as independent;
the corpora are not known to contain such pairs, and that is an assumption, not
a verified fact.

**Not measured: denominator, `meter_family`, `canonical_time_signature`.**
`BarTracker` returns grouping only. Scoring it on outputs it does not produce
would measure the absence of a feature, not the quality of one.

## Acceptance

Stated as three bands, because the consequences are asymmetric and the material
supports a hard conclusion in only one direction.

**The deciding quantity is the paired difference A1 − A4 measured in this run.**
Both arms are scored on the same intention-to-treat recordings, so the
difference is computed **per recording**, and the interval is a bootstrap over
the same resampling unit with 2000 draws and the same seed sequence as
`phase_block`. Both bounds of the 95% interval decide. Two per-arm intervals
overlapping or not is not the test and is not reported as one.

The historical A4 of 0.522 and 0.606 is **not** the baseline. It is the reason
20 points was chosen as the margin, and nothing else. If this run's A4 differs
from it, the difference rule is unchanged and the historical figures are not
substituted back in. An earlier version of this document also gave absolute
thresholds of 0.722 and 0.806 derived from those historical numbers; that made
the gate two rules at once with no rule for disagreement between them, and the
absolute form is **withdrawn**.

**Band 1 — hard negative, binding.** The **upper** bound of the paired A1 − A4
interval is **below 20 points of F1 on either corpus**. This is evidence that
the gain is smaller than the programme-changing margin, rather than merely a
failure to prove that it is larger.

Reading: perfect beats and perfect bar lines, handed to the decoder, buy less
than 20 points over what the predicted channel already achieves. The decoder is
then discarding evidence it is given, and no improvement to the front end can be
worth more than what the decoder throws away. **This stops neural front-end work
for the metrical goal** and redirects to `BarTracker` and the label contract.
The 20-point margin is the same one the earlier phase registration used to
separate a real effect from its control, and is adopted rather than invented.

**Band 2 — inconclusive or preliminary.** If the interval straddles 0.20, the
margin is not resolved and no binding redirect follows. If its lower bound is
at least 0.20 but A1 stays below 0.90 on either corpus, the decoder is not shown
to be transparent. Either case licenses no threshold elsewhere in the plan.

**Band 3 — decoder not falsified on these corpora.** The A1−A4 lower bound is
at least 0.20 and A1 reaches **0.90 or above on both corpora**. The decoder is
then not the dominant measured loss for fixed-meter phase on these development
corpora; this does not establish transparency for meter-diverse material.

**A1 must clear `random_phase` by at least 20 points as a precondition of the
binding bands.** Synthetic fixtures must first prove that the harness recovers
planted meter and phase. If those fixtures pass but real A1 does not clear the
null, the result is a severe decoder/label-contract failure, not automatically
declared a harness bug.

**Interpretation of the remaining arms, fixed now:**

* A1 high, A2 low → the downbeat channel is the constraint. `plan.md` proceeds
  as written.
* A2 high, A3 low → the tactus grid is the constraint, not the downbeat channel,
  and the front-end work is on beats rather than on bar lines.
* A3 high, A4 low → grid and phase are each recoverable but not jointly, which
  points at the interaction and not at either component.

## Corpora

**GTZAN and Harmonix.** The accepted P0 preflight selected 1000 GTZAN files and
scored 999; `jazz.00054` is malformed and is the one technical exclusion.
Harmonix availability is re-counted by the harness before any arm output is
read. The artifact records the selected/scored counts, input manifest and every
technical exclusion; historical downbeat-audit counts are not silently reused.

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
* The reference and predicted grids reach their first decisions at different
  times. Guard: the deciding score uses the latest first-decision time as one
  common cut; arm-specific spans are diagnostic only.
* Reference downbeats are injected in a form the decoder treats differently from
  a model channel — a hard 1.0 where it expects a distribution. Guard: amplitude
  and timing-jitter sensitivity arms are fixed above and can withhold verdict.

## Not in scope

Tuning `BarTracker`. Any claim about metre, grouping, denominator, or rooms. Any
threshold for the model gate.

## Execution deviation: annotation tail validation

The first full M0a attempt at commit `531af40` aborted fail-closed after 775
completed futures, before writing an artifact or reading a corpus verdict.
`gtzan/reggae.00002` supplied 104 reference beats but its last annotation was at
`30.117052 s`, after the decoded audio ended at `30.013333 s`. Its causal
publication block therefore occurred after the stream ended, and A1 correctly
returned only 103 bar positions. The generic length invariant stopped the run.

This is now classified explicitly as the technical annotation exclusion that
the intention-to-treat contract already permits, before any arm score is read.
A corpus-wide preflight found exactly one such tail-invalid recording among the
1,581 selected inputs; `jazz.00054` remains the separate unreadable-audio
exclusion. No arm construction, scorer, threshold, complete-case rule, or
bootstrap rule changed. The full run restarts from the beginning on the commit
that adds this check; the aborted attempt contributes no numbers.

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
* the post-reset transient is reported **separately** and is **not masked out of
  the primary metric**: it is one mechanism by which short state can hurt;
* reset points are identical across arms and at fixed wall-clock offsets —
  **never** chosen at musical boundaries, or the experiment contains a choice of
  boundaries rather than a horizon;
* the same intention-to-treat recordings, common initial cut and scorer as M0a.

**The numbers, fixed here rather than in the harness.**

| Quantity | Value | Why this one |
|---|---|---|
| Reset frame | the frame whose timestamp first reaches `k · H` for horizon `H`, `k ≥ 1` | a wall-clock rule with no musical input |
| First reset | at `H`, not at 0 | every arm starts from the same initialised state |
| Common initial cut | **2.0 s** from the start of every arm | removes only the shared cold start; unlike per-reset masking it leaves temporal support in R2 |
| Transient window | **2.0 s after each reset**, clipped at the next reset and recording end, reported separately | exposes activation recovery without deleting it from the primary endpoint |
| Primary support | every scored beat after the common initial cut | R2 and R∞ retain common temporal support |

The withdrawn rule masked 2.0 s after every reset. For R2 that masked every
sample after the first reset, leaving no overlap with R∞ and almost certainly no
phase after BarTracker's 12-beat minimum. It could not answer S0. The primary is
therefore unmasked apart from the common initial cut; transient diagnostics are
descriptive and cannot replace the primary verdict.

## What is measured

Primary: **bar-phase F1**, the same definition as M0a. Secondary: downbeat F,
beat F, `usable_strict`.

The comparison is paired per recording, and the interval bootstraps recordings,
the same resampling unit and limitation as M0a.

## Acceptance

**S0 positive** — both hold:

1. **the paired R∞ − R2 difference has a 95% bootstrap lower bound of at least
   0.05 of bar-phase F1 on both corpora** — paired per recording, same
   resampling unit and draw count as M0a;
2. the primary series is **monotone within tolerance**: for every adjacent pair in
   `R2 → R4 → R8 → R16 → R32 → R∞`, the paired difference has a 95% lower bound
   **above −0.01**. One step allowed to fall below that and no further; two or
   more, or any step whose lower bound is below −0.03, fails the condition. A
   flat step is not a violation — the claim is non-decreasing, not increasing.

The −0.01 tolerance exists because a strictly monotone requirement would fail on
noise alone across five comparisons, and the one-exception allowance because
the series has five steps and demanding all five is a multiple-comparison trap
in the strict direction.

**S0 negative** — the **upper** 95% bound of R∞−R2 is below 0.05 on either
corpus. S1 leaves the top of the queue and becomes a late ablation. It is not
deleted: a training-time benefit could exist without an inference-time one, but
the frozen model excludes a five-point inference benefit there.

**S0 inconclusive** — the interval contains 0.05, or the lower bound clears 0.05
but monotonicity fails. No claim of no effect is made. S1 is not promoted by S0
until a positive result satisfies both conditions.

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

* Run from a clean eval worktree; artifacts written outside it. The primary
  checkout carries unreadable `.pytest_tmp_codex_*` directories, so a run from
  it fails the provenance gate by design.
* **`provenance.experiment_provenance()`**, not `provenance()`. The plain
  function is the diagnostic path and returns a record with `tree_clean: None`;
  the fail-closed wrapper refuses to return at all unless the commit describes
  the run. An earlier version of this document named the wrong one.
* SHA-256 of the binary, the weights and every annotation file in the artifact.
* The phase scorer and `random_phase` construction are reused from
  `causal_metre.py`, but their sampling contract changes as registered above:
  common suffix, intention-to-treat zero for no phase, and paired bootstrap.
  Wilson intervals do not apply to mean per-recording F1.
* Registered before any of it exists. Deviations found during the run are
  appended below with their direction, never folded into the text above.

## Revision, 2026-08-10, before any run

The first draft was audited and found to leave eight rules to be settled after
the numbers appeared. It is revised rather than annotated because **nothing has
been run**: no arm exists, no artifact exists, and there is therefore no result
that any of these choices could have been fitted to.

Changed: the deciding quantity is a paired bootstrap difference rather than two
per-arm figures; the absolute 0.722/0.806 thresholds are withdrawn; hard
negatives use the upper confidence bound; no-phase arms remain in the primary
denominator; every arm is scored on a common suffix; A3's frame-level oracle
channel and the causal BarTracker replay are specified; the redundant triangle
sensitivity is replaced; the S0 primary keeps R2 temporal support and reports
transients separately; bootstrap units are recordings; F2 is described as
binary bar-line agreement; `experiment_provenance` replaces `provenance`.

The earlier text called the first non-zero meter "settled" because that is the
word in `score_phase`'s docstring. The code only finds the first decision and
computes stability separately, so this revision uses the operationally accurate
name and removes arm-dependent scoring spans.

Any further change after a number exists goes below this line as a deviation,
not into the text.

## Pre-run implementation clarification, 2026-08-10

No corpus artifact or binding verdict existed when the implementation exposed
these contracts. Diagnostic smoke runs are not accepted results and are not
written to `research/results/`.

* The mandatory synthetic M0a preflight is now concrete: 32 seconds at 120 BPM,
  a one-beat phase offset, and planted fixed meters 3, 4 and 6. Every fixture
  must end on its planted meter and reach bar-phase F1 >= 0.90 through the same
  causal C++ replay, or the corpus run aborts.
* A binding M0a or S0 verdict requires scored blocks for both registered corpora;
  `--limit`, a single-corpus diagnostic, or complete technical failure of one
  corpus can only report `inconclusive`.
* S0 aborts unless Rinf activation replay reproduces the ordinary live beat,
  bar-position and meter sequences exactly, and unless every finite arm's
  recorded reset frames equal the first model frames at or after `k * H`.
* Fixed meter is fail-closed at the manifest boundary. A recording whose
  manifest has no meter label is a published technical exclusion rather than a
  meter inferred after inspecting arm output. The label itself is retained in
  the per-record result; tactus grouping is still derived from the reference
  beat/downbeat times and denominator remains outside the scored target.
