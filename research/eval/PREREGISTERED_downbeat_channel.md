# Pre-registered: giving the live tracker the downbeat it already computes

Written before any of it exists. Nothing below was chosen after seeing a number
it is measured against.

## The question

`core/src/tracking/live.cpp` runs BeatNet and takes one of its three outputs:

    model_->process(samples, n, [&](double frame_sec, double beat, double) {

The third parameter has no name. The network emits **beat, downbeat, null** at
50 fps, the beat channel drives everything, and the downbeat probability is
dropped on the floor of that lambda.

A bar line every N beats is a direct statement about which of half, one and
double is the beat. The octave is half of all live failure. This asks whether
using the channel that already answers it reduces wrong-level episodes.

## Why this is not the experiment that just failed

`eval/PREREGISTERED_octave_freeze.md` held the last confidently chosen octave
when the estimator's margin weakened. It came back **inert**: 41.82% against a
46.5% bound, 17 recordings won to 15 lost, p = 0.86 — while demonstrably acting,
since the switch rate fell 4.21 → 3.81 and the same trigger wired to the old
`clearAnchor` response was catastrophic.

The reason that matters here is the reason it failed. The freeze introduced **no
new evidence**. It re-used the same estimator's answer and changed only what was
done with it — a better search over the same evidence, which
`core/src/tracking/live.hpp` has already said is not the shape of this problem.
The downbeat channel is a different quantity, computed by a different head of
the network, and it measures the bar rate rather than re-reading the beat rate.

That is the whole claim, and it is falsifiable: if this fails too, the octave is
not recoverable from BeatNet's outputs at all, and the next move is a different
front end rather than another reading of this one.

## The mechanism, stated before the measurement

Not "feed it in and see". The octave-freeze result says plainly that a policy
without a mechanism for *why* it repairs the failure is not worth running, so:

Beats and bars are locked to each other by a small integer. If the beat period
is `P` and the bar period is `B`, then `B / P` must be near 2, 3, 4 or 6 for any
music this product is for. A tracker at double the true tempo reports `P/2`, and
the ratio it implies becomes 8 or 12 — which is not a plausible bar. A tracker
at half reports `2P` and implies 1 or 1.5, equally implausible.

So the bar rate is an **independent constraint on the octave**, and unlike the
anchor margin it does not have to be believed on its own: it only has to be
approximately right to rule an octave out.

## The system under test

**Shipped fold 1**, not `EnsembleMean`, whose adoption was not approved. Fold 1
holds GTZAN out of training, so GTZAN is honest ground here.

## The arms

| arm | what it does |
|---|---|
| **baseline** | shipped fold 1, downbeat channel discarded as today |
| **bar-rate** | a second `ActivationTempo` over the downbeat channel; the beat octave whose implied `B / P` is nearest a plausible integer is anchored |
| **oracle-bar** | the same, with the bar period taken from the annotation — the ceiling on any use of this channel, not a candidate |

`oracle-bar` is a **diagnostic bound and not shippable**. It answers the question
that decides what to do if `bar-rate` fails: whether the channel is uninformative
or merely hard to read. Those need different responses and the corpus cannot
separate them without it.

## Exact semantics of `bar-rate`

1. A second `ActivationTempo`, identical configuration, observes the **downbeat**
   probability at the same 50 fps. It reports a bar period `B` and its own
   confidence and octave margin, exactly as the beat one does.
2. When either estimator has not answered, the arm is **byte-identical to
   baseline**. No answer is not evidence.
3. Otherwise, for each candidate beat period `P * 2^k` with `k` in {-1, 0, +1},
   compute `r_k = B / (P * 2^k)` and score it by distance to the nearest element
   of {2, 3, 4, 6} in log space.
4. Anchor at the `k` with the best score, at the shipped `anchor_width_octaves`,
   **only if** that score beats the runner-up by `bar_ratio_margin`. Otherwise
   anchor exactly as baseline does. A bar rate that cannot choose does not get to
   vote.
5. The beat **phase is never touched**. This writes an anchor, which is a prior
   over period. Where a beat falls is not this channel's business.

`bar_ratio_margin` is **0.15** in log2 units, fixed here, and swept on nothing.

Note what this deliberately does not do: it does not use the downbeat channel to
place bar lines. That is a second, separate use of the same evidence and mixing
the two would leave a result nobody can attribute.

## Corpora

**Harmonix (581) carries the primary comparison**, because the gates below are
the two previous experiments' and were measured there, and because full-length
songs are where an episode means something.

**GTZAN (999) is the replication**, and it is the fresher of the two: fold 1
holds it out of training and it has not been used to choose anything in this
line of work. A result on Harmonix that does not repeat on GTZAN is not a result.

RWC (328) and SMC (217) are reported beside them without gates.

## Acceptance gates

Unchanged from `PREREGISTERED_ensemble_in_core.md` and
`PREREGISTERED_octave_freeze.md`, so that a third proposal is judged against the
goalposts the first two were. Baselines are fold 1's own, re-measured in the
same run.

| Harmonix | baseline | to accept |
|---|---:|---:|
| no wrong-level episode >4 s | 41.5% | ≥ 46.5%, p < .05 |
| usable, strictly | 26.2% | ≥ 30% |
| correct time (eligible, mean) | 77.5% | ≥ 75% |
| switches / eligible 5 min | 4.21 | ≤ 4.21 |
| settle P90 | 36.61 s | ≤ 36.61 s |
| beat F | 0.7953 | ≥ 0.785 |

**Failing any acceptance gate means "adoption not approved".** Not "mixed", not
"promising". The gates decide.

## Primary comparison

**`bar-rate` against `baseline`, on Harmonix, paired per recording, on
`no_wrong_level_episode_fraction`.** Exact two-sided binomial sign test on the
discordant pairs, α 0.05, Holm-corrected over the family reported.

`oracle-bar` is descriptive and is **not** in the correction family: it cannot be
adopted whatever it shows, so buying a significance test for it would only
inflate the correction on the one comparison being made.

## Predictions

- **P1.** `bar-rate` clears the episode gate on Harmonix: ≥ 46.5%, p < .05.
- **P2.** The gain is concentrated in `double`-time recordings rather than
  `half`. Harmonix doubles seven to one, and a bar constraint rules out an
  implied ratio of 8 more decisively than one of 1.5, which is only one integer
  away from 2.
- **P3.** `oracle-bar` clears the gate comfortably — ≥ 55%. If the *annotated*
  bar rate cannot repair the octave, the channel is not the answer and neither is
  a better reading of it.
- **P4.** `bar-rate` recovers at least a third of the distance from `baseline`
  to `oracle-bar`. Below that the estimator, not the evidence, is the limit.
- **P5.** Beat F moves by less than a point in either direction. This changes
  which octave is anchored, not where beats sit inside it; a larger move means
  the anchor is disturbing phase and is a bug rather than a result.
- **P6.** GTZAN moves in the same direction as Harmonix. Excerpts are shorter
  than the bar estimator's own window, so the effect there should be smaller,
  not absent.

## What would sink this

- The episode gate missed, or met without significance.
- Any acceptance gate failed — the same sentence, restated because the ensemble
  experiment showed how readily a shorter list of ways to fail gets read as a
  softer alternative to the gates.
- `oracle-bar` failing to clear its own bound. That is the informative failure:
  it would say the downbeat channel cannot decide the octave even when read
  perfectly, and it would retire this direction rather than this policy.
- Beat F down by more than a point, or the switch rate up.

## Before any corpus is touched

Synthetic tests, on a `LiveTracker` driven by constructed activations, all
passing first:

1. **No downbeat answer.** With the downbeat channel silent, every anchor
   decision is identical to baseline, byte for byte.
2. **Ratio arithmetic.** A bar period of 2.0 s against beat periods of 0.5, 0.25
   and 1.0 s scores `k` = 0, −1 and +1 respectively, and a bar period of 1.5 s
   against 0.5 s scores `k` = 0 at ratio 3.
3. **An implausible ratio does not vote.** With `B / P` near 5 or 7 at every
   candidate, the margin is not met and the arm falls back to baseline.
4. **Double-time rescue.** A tracker anchored at 2× with a correct bar rate
   moves to 1×, on the same frame the margin is met.
5. **Phase is untouched.** Beat times under `bar-rate` and under baseline are
   identical on material where the downbeat channel never answers.
6. **Three and six.** A waltz — bar over beat of 3 — is not pulled towards 4.

Test 6 is the one to write first: a rule that quietly assumes four beats to the
bar would show up as a gain on this corpus and a defect on the material it was
not measured on.

---

## Deviations found during implementation, 2026-08-05

Appended, not edited into the text above. Everything before this line is what
was registered; everything below is what implementing it revealed, recorded
before any corpus was touched.

### 1. The bar estimator cannot share the beat estimator's configuration

"a second `ActivationTempo`, identical configuration" is not implementable.
`ActivationTempoConfig` runs from 40 to 220 BPM, and a four-beat bar at 120 BPM
is 30 a minute — below its floor, so the shipped configuration cannot represent
the quantity at all.

`barTempoDefaults()` in `tracking/live.hpp` sets 10 to 120 BPM, from the
arithmetic (a six-beat bar at 60, a two-beat bar at 240), a prior centre of 35
(the beat prior's 140 at four beats to the bar, so the two cannot disagree about
which octave is a priori plausible) and a 12-second window. The window is the
one judgement: `min_window_sec` must equal `window_sec` because a partly filled
ring is zero-padded and the padding reads as evidence, so the window is also how
long the arm is inert at the start of a recording. Twelve seconds is about six
bars, the fewest an autocorrelation peak can be believed on, and it leaves
eighteen seconds of a thirty-second excerpt rather than six. None of it was
chosen against a corpus.

### 2. The mechanism is one-sided, and the registered text missed why

**The set of plausible bars is closed under doubling** on (2, 4) and on (3, 6).
An octave shift therefore carries one plausible bar onto another, the two tie,
and the arm abstains. Measured on the arithmetic, gap to the runner-up:

| case | best k | gap | outcome |
|---|---:|---:|---|
| 4/4 at 120, correct | 0 | 0.000 | abstains |
| 4/4 at 120, **doubled** | +1 | **0.415** | **rescued** |
| 4/4 at 120, halved | −1 | 0.000 | abstains |
| waltz, 3 to the bar | −1 | 0.000 | abstains |
| 6/8 | 0 | 0.000 | abstains |

The registered mechanism said a tracker at half implies a bar of one and a half,
which is true, and did not notice that the *correct* reading is simultaneously
ambiguous — 2 against 4 — so the tie kills the vote there as well.

What the arm actually is, then: **a one-sided double-time corrector that is
silent everywhere else.** It cannot confirm a correct level and cannot rescue a
halved one. It is safe, because it never moves an answer it cannot decide, and
it is pointed at the dominant error — Harmonix doubles seven times for every
once it halves.

**This changes two predictions and they are restated rather than quietly
dropped:**

- **P2** is no longer a prediction. "The gain is concentrated in double-time
  rather than half" is now true by construction, not by measurement, and must
  not be reported as a confirmed prediction.
- **P4** stands, but its interpretation narrows: the distance to `oracle-bar`
  is now bounded by the share of episodes that are doubling in the first place,
  not by how well the bar rate is estimated.

**The tie was not broken with the tempo prior, deliberately.** That would work,
and it would also destroy the only reason this arm exists: the prior is already
measured as zero-sum on the octave, and folding it back in would make the bar
rate a re-reading of the same belief rather than independent evidence — which
is exactly what sank the octave freeze.
