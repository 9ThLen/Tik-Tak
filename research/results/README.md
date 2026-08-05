# What is in here, and what each number is allowed to say

Artifacts written by the scripts in `research/eval/`. Every one carries a
`provenance` block: the commit, whether the tree was clean, SHA-256 of the
binary and of each model, and the per-corpus counts. **`tree_clean: false` means
the commit does not identify the binary that produced the number.** Committing
afterwards does not repair that — the run has to be repeated.

Which corpus a number came from decides what it can be used for, and this is not
a formality here:

| corpus | status | what it can answer |
|---|---|---|
| ballroom | in `beatnet_model_1`'s training set | nothing about performance |
| GTZAN | out of fold 1's training, in folds 2 and 3's | fold 1 only |
| SMC | out of all three | anything, but at 3.2% usable it has no resolution |
| RWC | out of all three — **development corpus since 2026-08-03** | debugging, factorization, regression; not confirmation |
| Harmonix | out of all three — **spent 2026-08-04** on the pre-registered ensemble test | that one hypothesis, honestly; a development corpus from now on |

RWC became a development corpus the moment `anchor_width_octaves` was chosen by
looking at it and the averaged activation was taken seriously after seeing its
scores. It is still the most useful corpus here for finding out *why* something
fails. It can no longer say that a chosen configuration is good.

**An averaged activation narrows that table further, and permanently.** The row
for GTZAN says "fold 1 only" for a reason: folds 2 and 3 were trained on GTZAN,
and folds 1 and 3 on Ballroom. So a mean of the three is train-on-test on both,
and the shipped single fold may be quoted on GTZAN where the ensemble may not —
they are not comparable there at all. Adopting `EnsembleMean` retires 1,697 of
the 2,760 annotated recordings here as evaluation ground, leaving Harmonix,
RWC and SMC. That is a cost of the ensemble, not merely of testing it, and it is
the strongest argument for recording new material.

## The baseline every later arm is measured against

`live_baseline_gtzan_family.json`, `live_baseline_rwc.json`,
`live_baseline_harmonix.json`. Commit `4422afc`, clean tree, nothing dropped
(1914 of 1915, 328 of 328, 581 of 581). The shipped configuration: fold 1,
`anchor_width_octaves` 0.02, the core's own front end.

```bash
research/.venv/Scripts/python.exe -m eval.live_corpus_benchmark --mode model --model models/beatnet_model_1.ttw --workers 8 --output results/live_baseline_gtzan_family.json
```

with `--manifest music/rwc2/manifest.csv --music music/rwc2` for RWC and
`--corpora harmonix` for Harmonix.

| corpus | n | no episode >4 s | usable | strict | correct time | longest run | settle P50/P90 | sw / 5 min | never settled | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ballroom¹ | 698 | 78.1% | 60.9% | 58.2% | 80.8% | 24 s | 7.0 / 13.6 | 9.14 | 10.5% | 0.820 |
| GTZAN | 999 | 67.9% | 44.5% | 43.6% | 67.7% | 23 s | 5.0 / 14.0 | 5.34 | 24.0% | 0.685 |
| SMC | 217 | 28.1% | 3.2% | 3.2% | 13.9% | 0 s | 12.0 / 31.0 | 3.82 | 80.2% | 0.228 |
| RWC-Pop | 100 | 47.0% | 39.0% | 38.0% | 81.0% | 147 s | 7.0 / 32.5 | 5.74 | 4.0% | 0.799 |
| RWC-Genre | 102 | 19.6% | 12.7% | 12.7% | 55.1% | 65 s | 8.0 / 66.8 | 4.62 | 22.5% | 0.583 |
| RWC-Jazz | 50 | 16.0% | 8.0% | 8.0% | 49.5% | 30 s | 15.0 / 63.4 | 6.27 | 14.0% | 0.539 |
| RWC-Classical | 61 | 1.6% | 0.0% | 0.0% | 20.2% | 7 s | 31.0 / 204.8 | 6.24 | 39.3% | 0.352 |
| **Harmonix** | 581 | **41.5%** | **31.0%** | **26.2%** | **77.5%** | 114 s | 8.0 / 36.6 | 4.21 | 4.5% | 0.795 |

¹ in fold 1's training set — not quotable as performance, present only because
the GTZAN-family run produces it.

**Average correctness is not the binding constraint, and has not been for some
time.** On Harmonix the tracker is right for 77.5% of the time after warm-up and
usable on 31.0% of recordings; on RWC-Pop, 81.0% against 39.0%. Those describe
the same runs. The reconciliation is in the same table: the median longest
correct run is 114 seconds, and 58.5% of Harmonix has at least one slip to the
wrong level lasting more than four seconds. A recording that is right for 95% of
its length fails on the other 5% if that 5% is contiguous. So a target of
"raise correct time above 70%" would have been asking for a number that already
passes by seven points.

`no_wrong_level_episode_fraction` is therefore the primary endpoint from here,
and the target table is written around episode-freeness rather than averages.

Three denominators are reported for correct time and they are not
interchangeable — the column above is the mean over recordings. On Harmonix the
three read 77.5% (mean over recordings), 76.8% (pooled over seconds) and 82.1%
(over *locked* time only, which is the old `active_state_shares.same`). The last
is the flattering one, because silence leaves its own denominator; a plan was
recently built around a `64.6%` whose denominator nobody could name.

SMC is not a hard corpus so much as one the tracker never starts on: 80.2% never
settle and the median longest correct run is zero seconds. Its oracle ceiling is
4.1%, so it has no resolution to lend any comparison.

## The pre-registered test: does averaging the three folds hold up out of sample

`beatnet_ensemble_harmonix.json`. 581 full-length recordings, `attempted 581,
dropped 0`, commit `e6cf8bd`, clean tree. The protocol and the four predictions
were fixed in `eval/PREREGISTERED_harmonix_ensemble.md` before the corpus was
looked at; primary endpoint `usable_strict`, primary comparison mean against
fold 1, α 0.05 uncorrected because the hypothesis was fixed in advance.

| arm | usable | strict | any level | F | CMLt |
|---|---|---|---|---|---|
| fold 1 — ships today | 31.7% | 27.5% | 52.7% | 0.8027 | 0.6911 |
| fold 2 | 31.2% | 25.8% | 51.5% | 0.7970 | 0.7008 |
| fold 3 | 32.5% | 25.6% | 52.7% | 0.7872 | 0.6809 |
| **mean** | **38.7%** | **33.0%** | **60.2%** | **0.8445** | **0.7404** |
| max | 28.6% | 22.2% | 41.1% | 0.7564 | 0.6239 |

Every comparison the mean makes is significant after Holm correction over all
eight:

| the mean against | criterion | won | lost | p |
|---|---|---|---|---|
| max | strictly | 72 | 9 | <1e-6 |
| max | usable | 79 | 20 | <1e-6 |
| fold 2 | strictly | 59 | 17 | 1e-6 |
| fold 3 | strictly | 62 | 19 | 2e-6 |
| fold 2 | usable | 70 | 26 | 8e-6 |
| fold 1 | usable | 69 | 28 | 3.8e-5 |
| **fold 1** | **strictly (primary)** | **54** | **22** | **3.1e-4** |
| fold 3 | usable | 67 | 31 | 3.6e-4 |

**Two of the four predictions were wrong**, both because the ensemble did better
than expected:

| | prediction | result | |
|---|---|---|---|
| P1 (primary) | mean beats fold 1 on strict, p<0.05 | p = 3.1e-4 | ✅ |
| P2 | margin smaller than RWC's 5.3 pts | 5.5 pts | ❌ |
| P3 | mean strict in 30–50% | 33.0% | ✅ |
| P4 | mean does not beat fold 3 | p = 2e-6 | ❌ |

P4's failure retires a claim: **"fold 1 is the weakest" does not replicate.** On
RWC the folds spread 14.7 / 18.2 / 18.7; here they sit inside 1.3 points with
fold 1 in the middle. That ranking was corpus-specific noise. What replicates,
and more strongly out of sample, is that the mean beats all of them — and since
there is no best fold to pick, no corpus need be spent picking one.

P2's failure retires a worry rather than a claim. The margin was expected to
shrink because RWC had chosen the width; it grew from 5.3 to 5.5 points. The
premise was wrong: a width chosen on RWC moves every arm together, so it biases
the absolute level and not a fold-against-mean contrast.

### Where the remaining distance is

Same run, same best configuration, why the 581 recordings fail — shares of the
whole corpus, so they overlap:

| | mean | fold 1 |
|---|---|---|
| wrong metrical level over 4 s | **49.4%** | 59.2% |
| wrong beats (precision) | 24.1% | 32.9% |
| too few beats (recall) | 24.1% | 31.7% |
| slow to acquire | 17.4% | 15.0% |

The level is twice the next failure, and forgiving it outright is worth
**21.5 points** (38.7% → 60.2%). See `core/src/tracking/live.hpp` for why the
next thing to try is giving `LiveTracker` the downbeat channel it currently
discards.

## The three BeatNet folds and their average, on RWC

`beatnet_ensemble_rwc.json`, produced by `eval/beatnet_ensemble.py`. 328
full-length recordings, `anchor_width_octaves` 0.02.

BeatNet publishes three checkpoints, each withholding a different corpus. Only
fold 1 had ever been measured here, for no better reason than that it was
fetched first.

Run natively, through `--live-model`, which is what would ship:

| fold | usable | strict | any level | F |
|---|---|---|---|---|
| 1 — ships today | 14.9% | 14.7% | 24.6% | 0.6015 |
| 2 | 18.1% | 16.9% | 28.8% | 0.6140 |
| 3 | 18.7% | 16.2% | 27.4% | 0.6033 |

Every arm through one seam, `--live-activation`, so the mean is compared against
the folds on the same code path rather than across two. Each fold lands within
0.4 points of its native run above, which is the control that makes the last two
rows readable:

| arm | usable | strict | any level | F | CMLt |
|---|---|---|---|---|---|
| fold 1 | 14.7% | 13.9% | 24.6% | 0.6127 | 0.4786 |
| fold 2 | 18.2% | 16.2% | 29.3% | 0.6238 | 0.4776 |
| fold 3 | 18.7% | 16.4% | 28.9% | 0.6120 | 0.4737 |
| **mean** | **20.6%** | **19.2%** | **33.7%** | **0.6499** | **0.5100** |
| max | 15.2% | 14.4% | 21.9% | 0.5899 | 0.4286 |

Per corpus, the mean is ahead on all three that reach the macro minimum —
genre 20.6% against the best fold's 19.6%, jazz 18.0 against 16.0, pop 44.0
against 41.0 — so it is not one corpus carrying the average. Classical is 0.0%
for every arm.

### What the rate table cannot say

Two rates two points apart over 328 recordings can be six tracks moving or forty
moving both ways. Paired over recordings, exact two-sided sign test, Holm-
corrected over the whole family of eight the harness runs:

| the mean against | criterion | won | lost | p | corrected |
|---|---|---|---|---|---|
| fold 1 | usable | 25 | 7 | .0021 | **.0168** |
| fold 1 | strictly | 25 | 9 | .0090 | .0633 |
| max | usable | 25 | 9 | .0090 | .0633 |
| max | strictly | 23 | 9 | .0201 | .1003 |
| fold 3 | strictly | 18 | 9 | .1221 | .4883 |
| fold 2 | strictly | 19 | 10 | .1360 | .4883 |
| fold 2 | usable | 19 | 12 | .2810 | .5621 |
| fold 3 | usable | 18 | 12 | .3616 | .5621 |

One row is established. **The mean beats fold 1 on the headline criterion**, and
that is what shipping fold 1 costs. The same comparison read strictly does not
survive the correction, at .0633; an earlier revision of `live.hpp` claimed it
did by quoting the loose column's correction for both. That the mean beats the
*best* single fold is not established either — 18 against 12 is churn with a
favourable sign. Folds 2 and 3 are indistinguishable from each other (+17 −16,
p 1.00 uncorrected). `max` is worse than the mean, so this is not "any pooling
helps": what a mean suppresses and a max keeps is one fold being confident and
wrong.

Recompute any of this without re-measuring:

```bash
research/.venv/Scripts/python.exe -m eval.beatnet_ensemble --from research/results/beatnet_ensemble_rwc.json
```

### Why not simply ship the best fold

Each fold withholds a *different* one of BeatNet's five training corpora — 1 →
GTZAN, 2 → Ballroom, 3 → Rock Corpus — so on any of those five the folds are not
comparable, because each has a different subset memorised. That leaves RWC and
SMC as the only corpora equally unseen by all three, and SMC has no resolution.
Choosing a fold on RWC would spend RWC on that choice.

The honest position is that RWC has *already* been spent, on the width and on
taking the mean seriously, and neither the mean nor any fold can be confirmed
there now. That is what `eval/PREREGISTERED_harmonix_ensemble.md` exists to fix.

## The pre-registered test: does the core reproduce it, and at what cost

`ensemble_in_core_{harmonix,rwc,smc}.json` against
`fold1_in_core_{harmonix,rwc,smc}.json`, verdict in
`ensemble_in_core_verdict.json`, produced by `eval/ensemble_in_core.py`. Six
arms, commit `fa781bc`, `tree_clean` true on all six, nothing dropped (581 of
581 Harmonix, 328 of 328 RWC, 217 of 217 SMC). Protocol and predictions were
fixed in `eval/PREREGISTERED_ensemble_in_core.md` before `EnsembleMean` existed.

The seam experiment above handed the tracker a pre-computed activation. This
asks the different question: **does the core, running three networks itself over
one shared front end, reproduce that gain** — and is what it costs worth what it
buys.

Read the reproduction first, because it is what makes the rest mean anything.
Every fold-1 arm reproduced the `4422afc` baselines **exactly**, to the last
printed digit, on all three corpora. The harness is deterministic and averaging
did not disturb the single-checkpoint path it shares code with.

### The six gates

| Harmonix | fold 1 | ensemble | required | |
|---|---:|---:|---:|---|
| no wrong-level episode >4 s | 41.5% | **48.2%** | ≥ 46.5%, p<.05 | ✅ |
| usable, strictly | 26.2% | 28.9% | ≥ 30% | ❌ |
| correct time (eligible, mean) | 77.5% | 79.0% | ≥ 75% | ✅ |
| switches / eligible 5 min | 4.21 | 4.46 | not above baseline | ❌ |
| settle P90 | 36.61 s | 36.81 s | not above baseline | ❌ |
| beat F | 0.7953 | **0.8300** | ≥ 0.785 | ✅ |

### Paired over recordings, Holm-corrected over all six comparisons

| corpus | endpoint | n | fold 1 | ensemble | won | lost | p | p Holm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| harmonix | no wrong-level episode >4 s | 581 | 41.5% | 48.2% | 70 | 31 | 0.0001 | **0.0008** |
| harmonix | usable, strictly | 581 | 26.2% | 28.9% | 45 | 29 | 0.0805 | 0.2415 |
| rwc | no wrong-level episode >4 s | 328 | 25.3% | 31.1% | 28 | 9 | 0.0026 | **0.0128** |
| rwc | usable, strictly | 328 | 18.0% | 22.3% | 23 | 9 | 0.0201 | 0.0802 |
| smc | no wrong-level episode >4 s | 217 | 28.1% | 25.8% | 26 | 31 | 0.5966 | 1.0000 |
| smc | usable, strictly | 217 | 3.2% | 2.8% | 0 | 1 | 1.0000 | 1.0000 |

The correction is not a formality here. RWC's strict-usability row is p = 0.020
raw and 0.080 corrected, so it is reported as having moved and **not** as having
been shown.

### The five predictions

| | prediction | outcome | |
|---|---|---|---|
| P1 | episode gate cleared on Harmonix, ≥46.5%, p<.05 | 48.2%, p_holm 0.0008 | ✅ |
| P2 | episode gain larger than the strict-usability gain | +6.7 against +2.7 pts | ✅ |
| P3 | RWC-Pop the same direction, by less than Harmonix | +5.0 against +6.7 pts | ✅ |
| P4 | SMC does not improve | −2.3 pts, p_holm 1.0 | ✅ |
| P5 | the core within 2 points of the seam | 48.2% against ~51% | ❌ |

**P5's miss is not the bug it was written to catch.** It said a discrepancy over
two points means the shared front end or the per-frame averaging is not doing
what the offline average did. The averaging is not at fault: the core's averaged
activation agrees with the mean of three separately dumped activations to 8e-6,
while the folds themselves differ by up to 0.99 on the same file. What is left
is the front end underneath, and this file already records that the two paths
differ by about a point. The core reproduces roughly half the seam's gain on
strict usability, +2.7 against +5.5, and that is a property of the ensemble on
the core's front end rather than of the arithmetic.

### The verdict: effect confirmed, adoption not approved

**The effect is real.** Episodes fall on Harmonix and again on RWC, both
surviving correction, and beat F rises 3.5 points. Averaging the folds does
what it was adopted to do.

**Adoption is not approved, because three acceptance gates failed** — strict
usability 28.9% against a 30% bound, switches 4.46 against 4.21, settle P90
36.81 s against 36.61 s. The table those come from is headed "to accept", and a
failed acceptance gate means the thing is not accepted. The separate "what
would sink this" list was hit by nothing, and an earlier version of this section
presented the two readings as an open disagreement; that was wrong. A shorter
list of ways to fail outright cannot retire the gates that were written to
decide the question, and reading it as if it could is exactly the move
pre-registration exists to prevent.

What the "what would sink this" list being clean does mean is narrower and
still worth stating: nothing here disqualifies the approach, so the gates are
worth another attempt rather than the idea being finished.

**The two failed cost gates change sign by corpus.** Switches per five minutes
*fell* on RWC-Pop (5.74 → 5.07), RWC-Classical (6.24 → 3.97) and
RWC-Royalty-Free (5.48 → 4.51), and rose only on Harmonix, RWC-Genre and
RWC-Jazz. A cost that changes sign with the material is not the cost that gate
was written to catch, and "episodes bought with churn" is not supported here as
a general claim.

**None of this is a decision to ship, and the pre-registration says so.** Real-
time factor with three networks on the target phone is out of its scope, and if
three networks do not fit the CPU budget the corpus verdict is moot. That is the
cheapest measurement that can close the question, and it is worth taking before
paying the adoption price — GTZAN and Ballroom retired permanently as evaluation
corpora, 1,697 of the 2,760 annotated recordings here.

The whole protocol re-runs as one command:

```bash
python -m eval.ensemble_in_core --family
```

## The anchor width sweep

`live_usable_width*.json` and `live_usable_rwc_width*.json`. The table and the
reasoning live beside the constant, in `core/src/tracking/live.hpp` at
`anchor_width_octaves`, because that is where someone changing it will look.

## Can a wrong level be seen coming: a documented negative, and an accident

`phase_instability_{rwcpop,harmonix}.json`, from `eval/phase_instability.py`,
where the tables and the reasoning live. The question was whether the settled
phase relationship between the low and high ODF bands comes apart *before* the
tracker slips to the wrong metrical level. RWC-Pop chose the threshold, Harmonix
never chose anything.

Harmonix is the **threshold-transfer corpus, not a held-out one** — it was
already spent on the seam experiment above, and calling it held out would claim
more than it carries.

**It does not.** With the threshold carried across, the phase feature warns of
16.3% of every episode over four seconds — the diagonal. It is not better than
a plain fall in coherence, which leads it on both corpora, and it is one of
only two signals here that cannot even see every episode: no settled phase ever
forms on 10% of them. The mechanism is in the same file — a single band's phase
mostly fails the Rayleigh threshold the core already uses to decide whether a
phase means anything, so there is little reliable relationship there to come
apart.

**What the controls turned up instead is worth more than the hypothesis was.**
`live_anchor_margin`, which the live path computes every frame and uses for
nothing else, warns of **85.9% of every wrong-level episode one to four seconds
before it starts**, at a threshold chosen on RWC-Pop and carried to Harmonix as
a number, with 16.9% of correct locked frames above it. It sees all 1,063
episodes, so that rate has no hidden denominator. Anticipatory, not concurrent:
0.895 AUC on the window ending a second before the onset against 0.932 on the
transition itself.

**That 16.9% is not a cost and must not be quoted as one.** It is the share of
correct locked frames above the threshold. What a gate costs depends on what
the gate does, and freezing the octave while leaving tempo and phase free is
nearly free on a frame where the octave was not going to move. Only replaying
the tracker under the policy turns that column into a cost — and a tracker that
abstains spends no time at the wrong level, so the episode endpoint can always
be improved by saying less. Anything built on this is scored on locked time
kept as well as on episodes avoided.

An earlier version of this section quoted 85.3% / 18.3% from an evaluator that
scored every signal on the *phase* feature's availability, which silently made
the denominator 996 of 1,063 episodes. The numbers above are from the corrected
one; `tests/test_phase_instability.py` pins the difference.

## Acting on that warning: the octave freeze, measured

`arm_{baseline,clear,freeze,abstain}.json` with their per-track files. Four arms
of `eval/PREREGISTERED_octave_freeze.md`, Harmonix, 581 of 581 on each, commit
`a2c18eb`, `tree_clean` true on all four. Shipped fold 1 throughout, τ = 0.5916
carried from RWC-Pop as a number.

| Harmonix | episode-free | strict | correct time | sw / 5 min | settle P90 | F |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 41.48% | 26.16% | 77.54% | 4.21 | 36.61 s | 0.7953 |
| clear | 30.64% | 18.07% | 74.61% | 7.12 | 51.71 s | 0.7798 |
| **freeze** | **41.82%** | **26.85%** | **77.87%** | **3.81** | **38.61 s** | **0.7965** |
| abstain | 84.17% | 18.93% | 64.74% | 0.71 | 50.91 s | 0.6979 |

Paired against baseline, per recording:

| | won | lost | p |
|---|---:|---:|---:|
| freeze, episode-free (**primary**) | 17 | 15 | **0.86** |
| freeze, usable strictly | 7 | 3 | 0.34 |
| clear, episode-free | 17 | 80 | <1e-4 |
| abstain, episode-free | 248 | 0 | <1e-4 |

**Adoption not approved. The freeze is inert on the endpoint it was built for.**
41.82% against a 46.5% bound, and a sign test that could hardly be more null.

**Five of the six predictions held; the primary did not.** The freeze beats
`clear` by eleven points (P2), `clear` at this τ is worse than baseline on F as
its older measurements predicted (P3), `abstain` takes the highest episode-free
and the lowest correct time and fails the correct-time gate (P4), F moves by
0.0012 (P5), and the switch rate falls, 4.21 → 3.81 (P6).

**So the policy did act, and the episodes did not care.** P6 is the important
one: switches fell by a tenth, F did not move, and `clear` at the same trigger
was catastrophic — the arm is demonstrably doing what it is described as doing,
to the right recordings, at the right moments. The episode rate still did not
move. Whatever makes a wrong-level episode on this corpus, it is not an anchor
switching octave while the estimator is unsure.

**A signal that predicts is not a policy that helps, and this is the cleanest
demonstration of it here.** The same `live_anchor_margin` separates episodes at
0.895 AUC one to four seconds ahead. Acting on exactly that warning, at exactly
that threshold, changes nothing. The prediction is real and the lever was the
wrong one.

**`abstain` is why the endpoint needs its guard.** 248 recordings better and
none worse, to 84.17% episode-free — by saying nothing on more than half the
polls, at a cost of thirteen points of correct time and ten of F. It was never a
candidate, and it is the measured size of what silence buys on this metric.

## Everything else

`oracle_activation*.json`, `activation_recall.json`, `octave_blame_*.json`,
`timing_irregularity.json`, `tempo_stress.json`, `live_usable_rough*.json`,
`live_usable_no_anchor.json`, `live_usable_split*.json` — arm-versus-arm
experiments, all measured at `anchor_width_octaves` 0.10, which shipped before
2026-08-03. Each is a contrast between two arms at one width, which is what they
are cited for and what they are still worth. None is an absolute level any more.
