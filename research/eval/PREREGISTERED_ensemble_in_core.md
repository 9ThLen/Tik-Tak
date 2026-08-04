# Pre-registered: the averaged activation, computed by the core

Written before `EnsembleMean` exists. Nothing below was chosen after seeing a
number it is measured against.

## The question

`research/results/beatnet_ensemble_harmonix.json` established that averaging the
three fold activations beats every single fold, measured by handing a
pre-computed activation to the tracker through `--live-activation`. That is a
seam, not a product. This asks the different question: **does the core, running
three networks itself over its own front end, reproduce that gain** — and is
what it costs worth what it buys.

Two things can go wrong between the seam and the core, and only a run can say
which: the core's front end is shared across the three networks where the seam
ran three whole binaries, and the core averages per frame in real time where the
seam averaged a finished array.

## The endpoint, and why it changed

The endpoint is **`no_wrong_level_episode_fraction`**: the share of recordings
that never once spent more than `MAX_WRONG_OCTAVE_SEC` at the wrong metrical
level.

It replaces average correctness, which was the previous headline and is not the
binding constraint. The Stage 0 baseline settles that: on Harmonix the shipped
tracker is right for 77.5% of the time after warm-up and usable on 31.0% of
recordings. Those describe the same runs. A recording that is right for 95% of
its length still fails if it slips once for five seconds, and 59.2% of Harmonix
has such a slip. Raising the average would move a number that already passes.

The endpoint was chosen from the *baseline's* failure structure, before any
per-arm episode rate was looked at. That ordering is the only thing that makes
it a hypothesis rather than a result.

## What is not honest to measure this on

The three folds hold out GTZAN, Ballroom and Rock Corpus respectively. So on
GTZAN, folds 2 and 3 were trained on the corpus; on Ballroom, folds 1 and 3
were. **An average of the three is train-on-test on both.** Fold 1 alone is
clean on GTZAN, which is why the shipped baseline may be quoted there and the
ensemble may not — the two are not comparable on those corpora at all.

That is a cost of shipping the ensemble, not merely of testing it: adopting
`EnsembleMean` permanently retires GTZAN and Ballroom as evaluation corpora for
this product. Between them they are 1,697 of the 2,760 annotated recordings on
this machine. The remaining honest ground is Harmonix (581), RWC (313) and
SMC (217), and the case for spending it has to be made on those alone.

Harmonix has already been spent once, on the seam experiment. It is used again
here because the question is different — whether an implementation reproduces a
known effect, not whether the effect exists — but it cannot serve as evidence
that the effect is real a second time. RWC and SMC carry that.

## Primary comparison

`EnsembleMean` against the shipped configuration (fold 1 alone), both computed
by the core, over the same 581 Harmonix recordings, paired per recording. Exact
two-sided binomial sign test on the discordant pairs, alpha 0.05, Holm-corrected
over the whole family reported.

## Gates

The gate is not "the average went up". It is that episodes went down, at a
tolerable cost:

Baselines are the core's own, from `research/results/live_baseline_harmonix.json`
at commit `4422afc` — not the seam's fold-1 arm, because this A/B is core against
core and the two paths differ by about a point.

| Harmonix                        | baseline | to accept           |
|---------------------------------|---------:|--------------------:|
| no wrong-level episode >4 s     |    41.5% | >= 46.5%, p < .05   |
| usable, strictly                |    26.2% | >= 30%              |
| correct time (eligible, mean)   |    77.5% | not below 75%       |
| switches / eligible 5 min       |     4.21 | not above baseline  |
| settle P90                      |   36.6 s | not above baseline  |
| beat F                          |    0.795 | not below 0.785     |

46.5%, not the 44% first proposed, and the reason is arithmetic rather than
ambition. 44% is a 2.5-point gain on this baseline, and on n = 581 a shift that
small lands at p ~ 0.1-0.3 whatever the discordance — so a gate of "44% *and*
significant" cannot be met by any result that only just reaches 44%. Five points
is about where the sign test starts to see a shift reliably at this n, so the
threshold is set there. The seam predicts roughly 51%, so this is a gate a
faithful implementation should clear comfortably: it is set to catch one that is
not, rather than to be hard.

## Predictions

Recorded so a wrong one cannot be reread as a right one afterwards.

- **P1.** `EnsembleMean` clears the episode gate on Harmonix: >= 46.5%, p < .05.
- **P2.** The gain on the episode endpoint is larger than the gain on strict
  usability. Averaging cancels the octave evidence the folds disagree about,
  which is an episode-shaped fix.
- **P3.** RWC-Pop moves in the same direction, and by less than Harmonix does.
- **P4.** SMC does not improve. Its failure is that the tracker never starts
  (80.2% never settle, median longest correct run 0 s), and a cleaner activation
  does not address a tracker that has nothing to lock to.
- **P5.** The core reproduces the seam to within 2 points on the primary
  endpoint. A larger discrepancy means the shared front end or the per-frame
  averaging is not doing what the offline average did, and is a bug to find
  rather than a result to report.

## What would sink this

- The episode gate missed, or met without significance.
- Beat F down by more than a point: the ensemble would be trading the metric it
  was adopted for against the one already shipped.
- Real-time factor on the target phone above budget with three networks running.
  Measured separately (task #20); this pre-registration does not cover it, and a
  pass here is not a decision to ship.
