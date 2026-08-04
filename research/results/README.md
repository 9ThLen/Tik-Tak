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

## The anchor width sweep

`live_usable_width*.json` and `live_usable_rwc_width*.json`. The table and the
reasoning live beside the constant, in `core/src/tracking/live.hpp` at
`anchor_width_octaves`, because that is where someone changing it will look.

## Everything else

`oracle_activation*.json`, `activation_recall.json`, `octave_blame_*.json`,
`timing_irregularity.json`, `tempo_stress.json`, `live_usable_rough*.json`,
`live_usable_no_anchor.json`, `live_usable_split*.json` — arm-versus-arm
experiments, all measured at `anchor_width_octaves` 0.10, which shipped before
2026-08-03. Each is a contrast between two arms at one width, which is what they
are cited for and what they are still worth. None is an absolute level any more.
