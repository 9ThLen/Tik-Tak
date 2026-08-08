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

## An accurate bar period has strong leverage; reading one off BeatNet is unproven

`{hx,gz}_{baseline,barrate,oracle}.json` with their per-track files. Three arms
of `eval/PREREGISTERED_downbeat_channel.md`, commit `dce26bb`, `tree_clean` true
on all six, 581 of 581 Harmonix and 999 of 999 GTZAN. Shipped fold 1 throughout.

`bar-rate` estimates the bar period from BeatNet's downbeat head and uses it to
pick the beat octave. `oracle-bar` is the same decision rule handed the
annotated bar length — a bound, never a mode, since nothing in a room knows the
bar in advance.

| Harmonix | episode-free | strict | correct time | sw / 5 min | settle P90 | F |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 41.48% | 26.16% | 77.54% | 4.21 | 36.61 s | 0.7953 |
| **bar-rate** | **41.48%** | 25.82% | 78.35% | 4.69 | 35.02 s | 0.7929 |
| oracle-bar | **60.59%** | 33.56% | 84.83% | 1.19 | 23.01 s | 0.7833 |

| GTZAN | episode-free | strict | correct time | sw / 5 min | settle P90 | F |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 67.87% | 43.64% | 67.74% | 5.34 | 14.0 s | 0.6854 |
| bar-rate | 68.37% | 43.54% | 67.87% | 5.55 | 13.0 s | 0.6845 |
| oracle-bar | 76.18% | 44.94% | 72.93% | 2.98 | 12.0 s | 0.6759 |

**Adoption not approved: `bar-rate` is exactly baseline on the primary
endpoint.** 41.48% against 41.48%, and a paired sign test of 10 won to 10 lost,
p = 1.0000. Not "no significant difference" — the same number.

### The diagnosis, which is why the oracle arm was worth its runtime

Both arms fire on almost exactly the same recordings:

| Harmonix | recordings whose tempo it changed | episode-free won | lost | p |
|---|---:|---:|---:|---:|
| bar-rate | 247 of 581 | 10 | 10 | 1.0000 |
| oracle-bar | 243 of 581 | **114** | **3** | <1e-4 |

The decision rule is right, the firing rate is right, and **the estimated bar
period is no better than a coin about which octave to take.** Handed the true
bar length instead, the identical rule wins 114 recordings against 3.

### What this does and does not license, stated carefully

**`oracle-bar` never touches BeatNet's downbeat output.** It reads the bar
length from the annotation. So what these six runs establish is:

> An accurate bar period has strong leverage on the octave — 19.1 points of
> episode-freeness on Harmonix, 8.3 on GTZAN — and one generic autocorrelation
> estimator failed to supply one.

They do **not** establish that the downbeat channel carries recoverable bar
information. Two possibilities remain conflated and this design cannot separate
them: BeatNet's downbeat output may be uninformative, or `ActivationTempo` may
be the wrong way to read it. The matching firing rates do not help — two arms
can fire on the same recordings while disagreeing about the period on every one
of them, which is exactly what 10-against-10 versus 114-against-3 looks like.

An earlier version of this section was headed "the evidence works, the estimator
over it does not", and the commit that added it claimed "the downbeat channel
decides the octave". Both overclaimed: the oracle is silent about the channel.

The question is now its own experiment — see
`eval/PREREGISTERED_downbeat_audit.md`, which asks whether a bar period can be
recovered from the raw beat and downbeat activations at all, before any further
live policy is written.

### The predictions

| | prediction | outcome | |
|---|---|---|---|
| P1 | `bar-rate` clears 46.5%, p<.05 | 41.48%, p = 1.0 | ❌ |
| P2 | — | retired before the run; true by construction, see the pre-registration's deviations | — |
| P3 | `oracle-bar` clears 55% | 60.59% | ✅ |
| P4 | `bar-rate` recovers a third of the way to the oracle | recovers none of it | ❌ |
| P5 | beat F moves under a point | bar-rate −0.24; oracle −1.20 | ⚠ |
| P6 | GTZAN moves the same way | +0.50 against +0.00 | ✅ |

P5 is flagged rather than passed: the oracle gives up 1.2 points of F for its
19.1 of episodes. That is the shape this project has taken before — a recording
that crosses the usable threshold is worth more than one that goes from 0.79 to
0.82 — but it is a cost and not a rounding error, and any successor to the
estimator inherits it.

P6 passes on direction and means little on size, and the per-track counts say
why: `bar-rate` fires on 95 of 999 GTZAN excerpts against 247 of 581 Harmonix
songs. The bar estimator needs a twelve-second window before it answers at all,
which is most of a thirty-second excerpt.

### What this leaves

Not "replace the estimator" — that assumes the answer to the question above.
What is open is whether the bar period is recoverable from the model's own
outputs at all, and the next experiment is the smallest one that closes it: an
offline audit reading the downbeat probability **at the predicted beat
positions** rather than autocorrelating it, since the downbeat head answers
"which of these beats starts a bar", not "how slow is the bar".

No further live policy until that audit reports.

## The downbeat audit: the metre is there, the octave is not

`audit_{gtzan,harmonix}.json`, from `eval/downbeat_audit.py`, answering
`eval/PREREGISTERED_downbeat_audit.md`. Offline, no live core involved: read
BeatNet's downbeat probability at each beat of a grid and score, over every
metre in {2, 3, 4, 6} and every bar phase, the contrast between the beats that
would be downbeats and the beats that would not. Then score the same grid
**doubled** and ask which one the evidence prefers.

**What was run is not what was registered.** Three deviations, none recorded
before publishing, all found on re-reading the pre-registration against the
code:

1. **Annotated beat positions, not predicted ones.** The pre-registration says
   "take the predicted beat positions" twice; `audit_one` calls
   `load_reference_beats`. This one favours `beat-sync` — a perfect grid is
   better than the live tracker's — so it does not rescue a failure.
2. **Whole-recording accumulation, not "over the last 2–4 bars".** The README
   originally defended this as a ceiling: a causal decoder seeing two to four
   bars cannot extract more than an offline one seeing all of them. That is
   true of an *optimal* offline decoder and false of this one, which takes a
   global mean and so is diluted by arrangement change, dropouts and a chorus
   whose downbeat is strong against a verse whose downbeat is not. It was a
   deviation rationalised after the fact, not a registered choice.
3. **The `autocorr` arm was never implemented**, so P1 is unmeasured.

Four of the seven named measurements — bar period accuracy, coverage, false
corrections, bars to a stable decision, share of the oracle gap — were not
reported either.

| GTZAN, n = 991 | metre | octave separation |
|---|---:|---:|
| **beat-sync** | **60.8%** [57.7, 63.9] | 76.2% [73.4, 78.8] |
| shuffled | 23.5% [20.9, 26.3] | **84.2%** [81.7, 86.4] |
| beat-as-downbeat | 38.1% [35.1, 41.2] | 23.6% [21.0, 26.4] |

| Harmonix, n = 579 | metre | octave separation |
|---|---:|---:|
| **beat-sync** | **82.9%** [79.6, 85.9] | 79.6% [76.1, 82.8] |
| shuffled | 30.1% [26.3, 34.0] | **84.1%** [80.9, 87.0] |
| beat-as-downbeat | 60.1% [56.0, 64.1] | 7.8% [5.7, 10.3] |

### The metre is carried, decisively

`beat-sync` clears both controls on both corpora — 37 and 53 points over
shuffled, and 23 points over `beat-as-downbeat` on each. That last comparison is
the one that matters, because the beat channel alone reaches 38% and 60%: some
metre accuracy is available from the grid's own periodicity, and the downbeat
channel adds a large amount on top of it.

### The octave is not, and the control is the only reason we know

**`beat-sync` scores 76.2% and 79.6% on octave separation — and shuffled noise
scores 84.2% and 84.1%.** The intervals do not overlap on GTZAN. The signal is
*behind* its own null on both corpora.

Without the control this would have read as a strong result. It is not one, and
the reason the null sits so high is structural: the doubled grid carries twice
as many points, so the maximum over (metre, phase) of a noise contrast is
systematically smaller there, and the comparison tilts toward the shorter grid
before any evidence is consulted. 84% is what that tilt is worth. Nothing in
the downbeat channel beats it.

`beat-as-downbeat` confirms the instrument from the other side: at 23.6% and
7.8% it prefers the *doubled* grid outright, which is exactly what a decoder
finding periodicity in the grid it was handed looks like — the beat channel is
high at every beat, so doubling it manufactures a clean alternation.

### Verdict

**A2 fails: the pre-registration asked for at least 15 points over shuffled and
the result is 8.0 and 4.5 points behind it.** That rejects **this decoder**, and
it is the load-bearing sentence of the whole run.

It does not, on its own, close the head, and an earlier version of this section
said it did. The claim has to be sized to what was measured:

> An unnormalised global contrast score, maximised over (metre, phase), gets no
> octave advantage from the downbeat channel.

rather than "the downbeat channel contains no octave information". The gap
between those two is the tilt described above — the null is not a clean null,
because the two grids are permuted independently and have different lengths, so
the comparison mixes the channel's information with the decoder's geometry. The
tilt does not save the result (A2 fails by a wide margin either way) but it does
bound what the result is about.

A1, A3 and A4 are unmeasured. A1 and A3 need the live path; A4 needs a threshold
this decoder never earned. So **three of four acceptance conditions were never
taken, and the fourth was taken with an unregistered grid and an unregistered
window.** The protocol was not completed, and a verdict on an incomplete
protocol is a verdict on the decoder, not on the direction.

**P3 predicted exactly this** — "the metre comes back and the `P` against `P/2`
decision does not" — for the reason `analysis/downbeat.hpp` already records
about the half bar: a bar phase repeats at both grids, so a wrong octave that
repeats on the period the evidence is accumulated over cannot be broken by
accumulating more of it.

### What survives

Not the octave. But an 82.9% metre read on full-length songs, against a 30.1%
null, is a large amount of unused information about something else the product
gets wrong: the offline downbeat resolver scores 0.417 F on GTZAN with the
built-in cues. That is a lead for bar-line placement, not a proposal, and it is
a different experiment from this one.

## Everything else

`oracle_activation*.json`, `activation_recall.json`, `octave_blame_*.json`,
`timing_irregularity.json`, `tempo_stress.json`, `live_usable_rough*.json`,
`live_usable_no_anchor.json`, `live_usable_split*.json` — arm-versus-arm
experiments, all measured at `anchor_width_octaves` 0.10, which shipped before
2026-08-03. Each is a contrast between two arms at one width, which is what they
are cited for and what they are still worth. None is an absolute level any more.

## The octave veto: the decoder is below chance wherever it acts

`octave_veto_rwc.json`, from `eval/octave_veto_experiment.py rwc`, answering
`eval/PREREGISTERED_octave_veto.md`. RWC, all 328 recordings, shipped fold 1,
commit `b102324`, tree clean. **Harmonix was never opened.**

The unit is the decision point, not the frame and not the recording: when the
live tracker actually proposes moving to another octave, does beat-synchronous
metre evidence correctly allow or veto *that* switch. 1029 proposals on the
baseline arm, 97-98% of them scoreable at every metre, 678 labelled against 346
ambiguous — so the experiment is interpretable and the negative is not an
artefact of unlabelable events.

### A2 fails, and the shape of the failure is the finding

| τ | switches | balanced accuracy | false veto | episode-free |
|---:|---:|---:|---:|---:|
| baseline | 1913 | — | — | 0.2744 |
| 0 | 1367 | **0.4859** | 52.8% | 0.2927 |
| 0.5 | 1627 | **0.4951** | 30.1% | 0.2835 |
| 1 | 1844 | **0.4902** | 9.2% | 0.2805 |
| 1.5 (selected) | 1906 | **0.4998** | 0.6% | 0.2744 |
| shift control | 1913 | 0.5000 | 0.0% | 0.2744 |

Read the first two columns together. **Balanced accuracy is below chance at
every threshold where the decoder acts, and reaches exactly 0.5 only as its
action goes to zero.** The more it does, the worse than chance it is. That is
not a decoder with a weak signal; it is a decoder with none, whose apparent
neutrality at the selected threshold is the neutrality of doing nothing.

A2 required a **15-point** margin over the shift-driven control with the
interval's lower bound above zero. Measured: **−0.0002, 95% CI [−0.0060,
+0.0050]**, p = 1.0, 10 000 cluster-bootstrap resamples over recordings. The
interval is tight enough to exclude anything above half a point.

**A3 passes, and that is not a defence.** §7 selects τ by episode-freeness
subject to A3 and the cost gates, and A3's 5% bound on blocked correct escapes
eliminates every threshold below 1.5 — 52.8%, 30.1%, 9.2% all violate it. What
survives selection is the threshold at which the decoder vetoes 7 switches out
of 1913 and leaves every published number equal to baseline to four decimals.
The constraint that protects correct escapes rules out every setting at which
this decoder does anything at all.

It is also **behind the best matched-cost policy on the endpoint**: `margin_0.3`
reaches 0.2927 episode-freeness against the decoder's 0.2744.

### What the comparison policies show on their own

`total_ban` has the best episode rate of any arm, 0.3293, and buys it by cutting
switches from 1913 to 730 while losing correct locked time, 0.5623 → 0.5286.
That is exactly the trade the matched-cost design exists to expose, and it is
the reason "better than baseline" was never allowed to be the endpoint.

### Verdict

**A2 fails on the development corpus.** By the terms fixed before the run, that
closes the downbeat head for octave correction — permanently and without
reservation. Every objection raised against the previous audit was answered
first: predicted grids, a matched null that shifts one track and resamples both
nested grids from it, real decision points, matched-cost alternatives, a
standardised score whose null does not depend on grid length. The protocol was
executed as written. There is no further document.

Two registered predictions also failed, and both are recorded rather than
dropped. **P2** asked for over 90% sign agreement between `Δ` and `Δ_raw`;
measured 80.7% on 996 events, which says the null subtraction is carrying
weight rather than correcting a bias. **P7** predicted D1 above 5% — the share
of doubling proposals on a committed-correct state where the committed grid is
constant, the mechanism I2's failure exposed. Measured **0 of 170**. That
limitation, registered as the thing most likely to sink A1, never occurred on
real music at all.

### What this leaves

Not the octave. The line that ran from the anchor margin through the octave
freeze, the bar-rate arm, the downbeat audit and now this one is closed: **the
metrical level is not recoverable from BeatNet's own outputs**, by any reading
of them that has been tried, and what remains is a different front end.

The metre survives untouched — 82.9% on full-length songs against a 30.1% null —
and it is evidence about bar-line placement, where the offline resolver scores
0.417 F on GTZAN. A different experiment, and one that must not touch BPM.

## What a perfect octave would actually buy

`octave_ceiling_per_track_{rwc,harmonix}.json`, from
`eval.live_corpus_benchmark --per-track` on the shipped configuration. No new
mechanism, no hypothesis: a re-cut of what the benchmark already computes.

`usable_any_octave` scores each recording at **whichever octave reading came
closest**. It is therefore the exact ceiling on everything the closed octave line
was reaching for — anchor margin, octave freeze, bar rate, downbeat audit,
octave veto. Usable means precision and recall both at or above 0.80 with
acquisition inside the limit.

| corpus | n | usable | at best octave | the octave buys | still fails |
|---|---:|---:|---:|---:|---:|
| RWC-Pop | 100 | 39.0% | 60.0% | +21.0 | **40.0%** |
| Harmonix | 581 | 31.0% | 51.3% | +20.3 | **48.7%** |
| GTZAN | 999 | 44.5% | 49.2% | +4.7 | 50.8% |
| RWC-Genre | 102 | 12.7% | 24.5% | +11.8 | 75.5% |
| RWC-Jazz | 50 | 8.0% | 14.0% | +6.0 | 86.0% |
| **RWC-Classical** | 61 | 0.0% | 0.0% | **+0.0** | **100%** |
| SMC | 217 | 3.2% | 4.1% | +0.9 | 95.9% |

**The 23 points quoted from RWC is the best case, not the typical one.** It is
roughly RWC-Pop's 21. On GTZAN a perfect octave is worth 4.7 points, and on
RWC-Classical it is worth **nothing at all**: not one of those 61 recordings
becomes usable at any reading of the level.

### What survives it

Among recordings that fail at their own best octave — 283 of 581 on Harmonix,
230 of 328 on RWC:

| reason set | Harmonix | RWC |
|---|---:|---:|
| too few beats **and** wrong beats | 44.2% | 63.9% |
| both, plus slow acquisition | 14.8% | 17.8% |
| too few beats alone | 18.7% | 8.7% |
| slow acquisition alone | 13.8% | 3.9% |
| wrong beats alone | 1.1% | 2.6% |

**Recall is the dominant survivor: `too_few_beats` appears in 84.8% of Harmonix's
and 93.5% of RWC's.** Precision follows at 60.4% and 84.3%, and precision alone
is almost nonexistent — 1.1% and 2.6%. The two fail together far more often than
either fails apart.

### What this settles

The metrical level was never the binding constraint on most material. Solving it
perfectly takes Harmonix from 31.0% to 51.3% and leaves half the corpus failing
because the beat grid is simultaneously too sparse and in the wrong places.

So the closing sentence of the octave-veto section above — "what remains is a
different front end" — is right in form and wrong in aim. A front end better at
the **octave** has a 51.3% ceiling on Harmonix and a 0.0% ceiling on classical.
Any successor should be pre-registered against **beat-grid recall**, which is a
different question with different acceptance conditions.

One tractable piece is separable: **13.8% of Harmonix's surviving failures are
slow acquisition alone**, with precision and recall both already good. That is
39 recordings failing on a stopwatch rather than on the tracking, and it is the
one part of this picture that does not need a new observation.

## Acquisition was measured on a grid too coarse to see it

`acquisition_50hz_per_track_harmonix.json` beside
`octave_ceiling_per_track_harmonix.json`. Same binary, same model, same
recordings; the only difference is `--live-sample-hz`, which changes how often
the harness reads the tracker and nothing about the tracker. **Beat counts are
identical on all 581 recordings**, which is what makes the comparison a
measurement question rather than a behavioural one.

`acquired_at` is reconstructed in Python from the polled confidence series, and
every live number in this repository was polled **once a second**. The bar for
`slow_acquisition` is eight seconds. Re-read at 50 Hz:

| | 1 Hz | 50 Hz |
|---|---:|---:|
| Harmonix usable | 30.98% | **36.49%** |
| of the 39 slow-acquisition-only failures, now under 8 s | — | **29** |
| median shift in `acquired_at` on those | — | **−4.74 s** |

**This is not quantisation, and the first diagnosis of it was wrong.** A
one-second grid can misplace a threshold crossing by at most one second; these
move by nearly five, and one by fifteen. The mechanism is aliasing: confidence
fluctuates across the 0.25 threshold, and a once-a-second sampler keeps landing
in the gaps. On `0925_sweetdisposition`, **zero of the six polls between 4 s and
10 s** catch it at 1 Hz and the first catch is at 16.02 s; at 50 Hz, 33 of 258
polls in the same window are over threshold and the first is at 4.18 s.

### Verified against something that needs no sampling at all

The tracker's beat list is exact. On ten of the moved recordings, when the first
beat was actually handed out:

| track | 1 Hz | 50 Hz | first beat |
|---|---:|---:|---:|
| 0099_forgetyou | 9.01 | 4.272 | **4.281** |
| 0132_iceicebaby | 10.01 | 4.133 | **4.443** |
| 0418_inthedark | 12.00 | 4.133 | **4.401** |
| 0344_beautifullife | 8.01 | 4.087 | **4.111** |
| 0925_sweetdisposition | 16.02 | 4.180 | **5.832** |
| 0324_yeah3x | 12.00 | 4.830 | 11.583 |
| 0434_lights | 8.01 | 7.825 | 8.276 |

Eight of ten start playing under the bar. So the 1 Hz figure was wrong by
seconds, and the 50 Hz figure is close but still an approximation — it reports a
confidence crossing, and two of the ten crossed without publishing.

### What this costs and what it implies

Every live result here inherits a `usable_rate` that is too low, by 5.5 points
on Harmonix, for a reason that has nothing to do with the tracker. Comparisons
*between* arms measured the same way are unaffected — the error is common to all
of them — but absolute rates are not, and neither is any claim of the form "the
tracker acquires slowly".

**`acquired_at` should be derived from the beat list rather than from a sampled
confidence series.** The beat list is what a listener hears, it is exact, and it
is independent of how often anything is polled. That changes published numbers
and is a decision rather than a fix, so it is recorded here and not applied.

## The causal bar: phase carries, metre cannot be measured here, and the gate fails

`causal_metre_gtzan.json` and `causal_metre_harmonix.json`, from
`eval.causal_metre`, answering the causal arm registered in
`eval/PREREGISTERED_downbeat_audit.md` on 2026-08-08. Commit `676059d`, clean
tree. GTZAN 991 scored of 999 (one corpus defect, `jazz.00054` is not a WAV),
Harmonix 579 of 581, no failures. **All arms published byte-identical beat lists
on every recording**, which is the invariant that makes them comparable: the bar
decision reads a channel nothing else reads and writes nothing back.

The mechanism under test is `44c8c56` — `analysis::resolveMeter` over the last
32 beats the live tracker handed out, re-resolved every beat, inside the
shipping core. The arms differ in one file: the downbeat channel.

### The metre, and why these corpora cannot answer it

| | GTZAN | Harmonix |
|---|---:|---:|
| **always answer 4** | **0.949** | **0.976** |
| `beat_sync` | 0.867 | 0.894 |
| `beat_as_downbeat` | 0.791 | 0.729 |
| `shuffled` | 0.492 | 0.499 |

Restricted to recordings the tracker tracked at the annotated level, which is
what the registration made the answer.

C1 passes on both by 37.5 and 39.5 points, so the decoder is reading structure
and not level. C3 passes, and the causal figure is *above* the whole-recording
audit's 0.608 and 0.829 — which falsifies prediction P5 outright.

**None of that survives the constant.** 690 of 727 restricted GTZAN recordings
and 479 of 491 Harmonix ones are in four, so answering "4" and nothing else
beats the decoder by eight points on both. Off the majority metre there are
**49 recordings in 1218**, and there `beat_sync` scores 0.189 and 0.250 against
`shuffled`'s 0.108 and 0.333. Nothing is distinguishable from anything at those
counts.

That baseline was missing from C1–C3, which were written the same day and
compare only against shuffles and substitutions — both of which a metre prior
clears without deciding anything. The same reading applies backwards: the
original audit's 0.608 was compared against a shuffled 0.235 and never against
the 0.949.

### The phase, which is what the material actually varies

Registered as an addition after the metre result, for a reason stated in the
protocol: the corpus composition that makes metre unanswerable was discovered
in the run. F1 is bar-line agreement at 70 ms over the beats after the metre
settled; the null is the mean over all rotations of the same grid, which is the
exact expectation of a uniformly random bar line.

| | GTZAN F1 | Harmonix F1 |
|---|---:|---:|
| `beat_sync` | **0.522** [0.492, 0.552] | **0.606** [0.581, 0.631] |
| `beat_as_downbeat` | 0.329 [0.305, 0.353] | 0.516 [0.491, 0.541] |
| `random_phase` | 0.209 [0.203, 0.214] | 0.217 [0.212, 0.221] |
| `shuffled` | 0.193 [0.178, 0.208] | 0.207 [0.200, 0.214] |

**The contrast inside a single run is the finding.** Same recordings, same
decoder, same channel: the metre cannot separate from a constant because the
corpus has almost no metre variation, and the phase separates from its own null
by 31.3 and 38.9 points because a bar line has four places to be and the corpus
does not decide which. Both the original audit and the causal metre arm measured
the half of the problem this material holds fixed.

P8 predicted 0.4 to 0.6 and it came in at 0.522 and 0.606.

### Verdict: the flag stays off

The registered condition was "clears `random_phase` by at least 20 points on
both corpora *and* clears `beat_as_downbeat` by at least 10. Failing either
leaves the flag off." The first holds by 31.3 and 38.9. The second holds on
GTZAN at 19.3 and **fails on Harmonix at 9.0**. So the condition fails, and
`bar_tracking` stays off with a documented negative.

**A fact about that control, recorded and not used to overturn the result.**
`ml/beatnet.hpp` computes `beat = p[0] + p[1]` and `downbeat = p[1]`, so the
beat channel *contains* the downbeat channel additively. `beat_as_downbeat` is
therefore not the wrong evidence with the right shape; it is the right evidence
with the rest of the beat channel added to it, and a large gap was never
available. That was knowable from the code before the run and it is a fault in
the 10-point bar rather than a reason to move it now.

**What a clean control would be**, if this is picked up again: `p[0]` alone —
beat-but-not-downbeat — which is `beat − downbeat` and computable from the two
channels already dumped, with no new model pass. It would need its own
registration, precisely because it is being named after a failure.

### What this leaves

The bar mechanism ships, off, tested, and costing nothing. What is now known:

- a causal 32-beat window is **not** the limitation — it beats whole-recording
  reads of the same channel on both corpora;
- the phase signal is real and roughly half of what a perfect bar line would be;
- **GTZAN and Harmonix cannot evaluate a metre decision at all**, and any future
  claim about metre needs a corpus with metre in it;
- the click gate remains untested here, because the harness plays no click, so
  every figure above is an upper bound for a shell with audible output.
