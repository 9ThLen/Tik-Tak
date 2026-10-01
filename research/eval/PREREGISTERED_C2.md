# C2 — the next data-scaling curve: a registered decision not to run it yet

Written 2026-08-20, before any C2 training output exists, as the independent C1
audit required. It registers a **threshold and a stop**, not a run.

The audit's consequence was that a stronger curve "must be registered before
more training output exists", must measure subset-sampling variance with crossed
draws, must add enough independent seeds to make a +0.03 decision informative,
and that **power should be simulated from the C1 seed/work matrix before fixing
the run count**, because simply repeating three seeds is not guaranteed to
resolve the interval.

The simulation was run first. It says something stronger than the audit
supposed: on the current development population, **no number of seeds resolves
the interval at all**. Registering that now is the point of this document —
otherwise someone later runs six seeds, gets `inconclusive`, and reports it as
news.

## What was simulated, and how it was checked

Variance components were estimated from C1's own deciding matrix — the paired
non-Candombe `F1_100 − F1_50`, three seeds by seventy-seven works — by ANOVA
expectations separating seed, work and residual terms:

| component | sd |
|---|---:|
| between seeds | 0.0117 |
| between works | 0.0498 |
| residual (seed × work) | 0.1390 |

Synthetic curves were then generated from those components and passed through
**the registered two-way resample itself**, not a normal approximation.

That check matters and it is why two earlier attempts were discarded. A normal
approximation using the raw variance of work means gave a width of 0.0614 —
close to C1's measured 0.0628, but only because it overstated the work term by
carrying seed noise inside it. A corrected analytic version gave 0.0498, which
is theoretically right and 21% too narrow, because resampling three seeds with
replacement has far heavier tails than the normal it was being compared to.
Neither was fit to size a design. The simulation reproduces **0.0626** against a
measured 0.0628, so what follows can be extrapolated.

## What a design could decide

Mean interval width, and the resulting upper bound if the true effect is C1's
point estimate of +0.0166:

| seeds | dev works | width | upper bound | class at +0.03 |
|---:|---:|---:|---:|---|
| 3 | 77 *(C1)* | 0.0616 | 0.0474 | inconclusive |
| 6 | 77 | 0.0495 | 0.0413 | inconclusive |
| 10 | 77 | 0.0416 | 0.0374 | inconclusive |
| 20 | 77 | 0.0338 | 0.0335 | inconclusive |
| **40** | **77** | 0.0286 | 0.0309 | **inconclusive** |
| 10 | 154 | 0.0307 | 0.0319 | inconclusive |
| **20** | **154** | 0.0248 | 0.0290 | **saturated** |
| **10** | **308** | 0.0238 | 0.0285 | **saturated** |
| 20 | 308 | 0.0187 | 0.0259 | saturated |

Forty seeds on the present development set — sixty training runs across three
fractions, on the order of three hundred GPU-hours — still returns
`inconclusive`. The binding constraint is **the development population, not the
seed count**, and no amount of the cheaper resource substitutes for the dearer
one.

Nor is a wider span a way out. Every contrast available in C1 is inconclusive at
+0.03, because the interval widens with the effect:

| contrast | non-Candombe mean | 95% CI |
|---|---:|---|
| `F1_100 − F1_50` | +0.0166 | [−0.0143, +0.0485] |
| `F1_50 − F1_25` | +0.0279 | [−0.0112, +0.0651] |
| `F1_100 − F1_25` | +0.0445 | [−0.0027, +0.0911] |

## The registered decision

**C2 is not run on the current corpus.** No training run may be started for the
purpose of tightening this curve until one of the entry conditions below holds.
An `inconclusive` obtained by re-running a design this analysis has already
shown to be underpowered is not a result and may not be reported as one.

**Entry conditions.** A curve becomes worth running when the development
population reaches at least **154 works with 20 seeds**, or **308 works with 10
seeds** — the two cheapest decisive cells above. Either must be checked against
a re-run of this simulation with the variance components re-estimated from
whatever data then exists, because the components below are estimated from three
seeds and are themselves poorly determined.

**What C2 must carry when it does run**, from the audit and from C1's own
recorded limitations:

- **crossed subset draws.** C1 used one nested subset hierarchy, so
  subset-sampling variance was never measured — it is not in the table above and
  the real width is therefore wider than shown, by an unknown amount. C2 must
  draw more than one subset per fraction and resample subsets as a third factor.
- **`under_fixed_recipe` retained**, unless the design equalises optimiser
  updates across fractions. At a fixed epoch cap the larger fraction receives
  proportionally more updates and data volume is confounded with update count.
- **`provenance.experiment` naming the running experiment.** Fixed in
  `experiment_label` after all six C1 artifacts shipped labelled `S1`.
- **the power simulation re-run and recorded** before the run count is fixed,
  not after.

## What would void this decision

- Starting a scaling run without meeting an entry condition.
- Re-estimating the variance components from data that does not exist yet, or
  reusing the components in this document as though they were measured on the
  new population.
- Treating the `F1_100 − F1_25` span as a substitute endpoint. It was not
  registered as primary in C1 and is reported here only to show that it is
  equally undecided; using it as a verdict would be choosing an endpoint after
  seeing it.
- Lowering the +0.03 MCID to fit the achievable precision. It is inherited from
  a document fixed before any S1 output existed; tuning it to the interval would
  invert the entire purpose of registering it.

## Where the development population comes from

There is no way to enlarge it from the existing corpus: the development set is
84 works carved from the same 980-record cache that supplies training, so taking
more works for development removes them from the thing being measured.

Enlarging it means new annotated material — which is the same work as
[`PREREGISTERED_P1B0.md`](PREREGISTERED_P1B0.md). That pilot is not a substitute
for this curve and does not answer its question: it changes the *distribution*
rather than its size. But it is the only route by which this curve becomes
answerable at all, and that is a stronger reason to run it than the one the
pilot was registered with.
