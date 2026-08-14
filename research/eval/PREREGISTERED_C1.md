# C1 — first data-scaling curve for the A3 stateful recipe

Status: **pre-run registration**, fixed 2026-08-14, revised the same day after
independent review. The subset generator, runner filter, summariser, launcher
and their tests exist; **no 25% or 50% training output does**, and none may be
produced until this document and that implementation have been reviewed
together. The 100% arm is an already-observed anchor and is declared as
such below, with everything already known about it written down before the new
runs start.

## Question

S1 closed its own hypothesis — carrying recurrent state between TBPTT blocks does
not raise bar-phase F1 — and left one number unexplained. Both trained arms beat
the frozen published model by about +0.10 of bar-phase F1, of which 72.4% came
from seven Candombe works. C1 asks the only question that decides whether more
data of this kind is worth collecting:

> Trained with the fixed A3 stateful recipe, does bar-phase F1 on the fixed 84
> development works still rise materially between 50% and 100% of the available
> training works — overall, and separately on the corpora that are not Candombe?

This is `plan.md`'s "first curve", the one whose result is supposed to size
`P1-B1`. It is **not** the `P1-B1` curve itself, which runs on a separate
composition-grouped corpus that does not yet exist.

## What this cannot establish

- **Not a generalisation estimate.** The 84 development works were held out of
  gradient training but were used to select every checkpoint, and the corpus has
  already been diagnosed on repeatedly. C1 produces a development signal.
- **Not a diversity-versus-capacity verdict.** A flat curve shows that the
  current A3 recipe has saturated on the current distribution. It does not
  choose between "needs different data" and "needs more capacity"; separating
  those requires new-domain data or A4, each behind its own gate.
- **Not an argument for A4.** S1's Candombe result — chance-level 0.023 to 0.970
  by retraining the readout alone — is evidence that the frozen convolution and
  LSTM layer 0 already carry that structure. C1 does not revisit that.
- **Not a room, meter-family, denominator or locked-test claim.**

## Bound sources

- S1 summary SHA-256:
  `4c7dd592ce0bce191b2c78b8a59616f6d75dfe1870056b227c2ba9a81010160f`
- S1 run commit: `b12eea828f25df502a157cccc872c5e2000cc2e3`
- Source checkpoint SHA-256:
  `619091bc317ca3e83b45591d46f6de3d5a41588bcb39fe9fe7be30cffa6aca84`
- Split and cache manifest SHA-256:
  `ed0bb52521aea150c2b838d38ec4096c91f0c4a6b62c7e500c53b8fe4b364a00`
- Frozen A0 baseline SHA-256:
  `4db6990164291f078a3cc22e9b31c47715759a557e28130af3396c91c84b3385`
- Product binary SHA-256 (the M0e evaluator):
  `49c47437423f0d79c2f30dde3bcba506f1075099b9f3a7c780efcffe2eed647d`

All are verified before any training work, as `validate_fixed_inputs` does for
M0e. A digest mismatch is fatal, not a warning.

## The 100% arm is an already-observed anchor

C1 does not rerun 100%. Rerunning it could not restore blindness: same seed,
same data, same code is a deterministic computation, and S1 demonstrated
tensor-identical resume on both CPU and CUDA. A blind 100% point would require
new training seeds, and three more runs of it is not worth the GPU time.

The anchor is therefore `A3_stateful` seeds 17, 29 and 43 from the S1 artifact,
at selected epochs 25, 40 and 35. **What is already known about it, recorded
here so that no later reader has to take on trust that the rule below was fixed
before the unknown half of the comparison existed:**

| | macro | Candombe | BPSD | Rubato | RWC2 | KRAISLER |
|---|---:|---:|---:|---:|---:|---:|
| dev works | 84 | 7 | 7 | 3 | 63 | 4 |
| A0 bar-phase F1 | 0.291 | 0.023 | 0.106 | 0.084 | 0.352 | 0.286 |
| 100% stateful | 0.396 | 0.976 | 0.160 | 0.125 | 0.380 | 0.229 |

The deciding quantity below is `F1_100 − F1_50`, and F1_50 does not exist yet.

## Fixed subsets

One hierarchy, generated once, before any C1 run:

- the unit is the **training work**, never the recording; all performances of a
  work move together;
- **stratified by corpus**: each fraction takes the same proportion of every
  corpus's training works, so the corpus mix does not drift along the curve;
- **nested**: `25% ⊂ 50% ⊂ 100%`, so the curve is monotone in data rather than
  three unrelated samples;
- **identical across the three training seeds.**

The 330 training works are BPSD 24, Candombe 28, KRAISLER 16, Rubato 11 and
RWC2 251. The membership order is fixed to the byte, because a rule this cheap
to state ambiguously is a rule two correct-looking implementations will disagree
about:

- key = `SHA256(b"tiktak-c1-v1" + b"\x00" + corpus + b"\x00" + work_id)`;
- `corpus` and `work_id` are the manifest's own strings, **UTF-8 encoded with no
  normalisation, case folding or trimming**;
- the separator is a single `NUL` byte, and the salt is followed by one too;
- works sort by lowercase hex digest **ascending**, ties broken by `work_id`
  ascending as a byte string;
- each fraction takes a prefix of that order.

Prefix lengths are **tabulated rather than rounded**, so rounding semantics
cannot drift:

| fraction | BPSD | Candombe | KRAISLER | Rubato | RWC2 | total |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 6 | 7 | 4 | 3 | 63 | **83** |
| 50% | 12 | 14 | 8 | 6 | 126 | **166** |
| 100% | 24 | 28 | 16 | 11 | 251 | **330** |

The generated subset artifact records, per fraction and per corpus, the exact
work count, record count and frame count.

**This design estimates optimisation variance and not subset-sampling
variance.** With one hierarchy shared by all seeds, a fortunate quarter is
indistinguishable from an effect of size. Separating the two needs a crossed
design — three subset draws by three training seeds at every fraction, eighteen
new runs — which is not worth it for a first curve. The limitation is stated
here rather than discovered afterwards, and it is the first thing a second curve
should buy.

## Held identical to S1

Changing any of these would make the fractions incomparable with each other and
with the anchor, so they are fixed by inheritance and not restated:

- arm A3 with the **stateful** recipe, 400-frame blocks, 100-frame shared
  warm-up mask, `detach` at every block boundary;
- Adam at 5e-4, batch 8, class weights `[50, 400, 5]`, gradient clip 5.0;
- **50 max epochs, product validation every 5 epochs, patience 4 points**;
- lexicographic checkpoint selection against the same frozen A0, rejecting any
  point more than 0.01 of beat F below it;
- the same 84 development works, the same product evaluator, the same 2,000-draw
  work-level bootstrap with seeds `0..1999`;
- training seeds 17, 29, 43.

The validation cadence in particular is not to be reduced. At 25% roughly
six sevenths of a run's wall time is product evaluation rather than training,
and the temptation is real — but checkpoint selection maximises over validation
points, so a different cadence changes the selection opportunity and makes the
fractions incomparable by construction. S1's selection diagnostics exist to
detect exactly that imbalance and would be defeated by introducing it
deliberately.

**Why the stateful recipe.** It matches the runtime, which carries recurrent
state through a whole recording; it costs no bar-phase accuracy against reset
(−0.0045 [−0.0194, +0.0102]); and it showed fewer bar-state switches than reset
(−7.00 [−11.23, −3.36] per five minutes). The third is a **secondary, post-hoc**
motive: S1 registered that condition as a one-sided ceiling on an increase, and
neither arm separated from A0 (reset +5.57 [−4.45, +14.26], stateful −1.43
[−13.03, +7.68]), so nothing here establishes that reset harms churn. It earns
its own registration or it stays a reason for a recipe choice, not a result.

## Runs

Six new runs: `{25%, 50%} × {17, 29, 43}`. Plus the three anchor runs, unchanged
and not recomputed.

## Endpoints

**Primary:** the paired `F1_100 − F1_50` bar-phase difference, with a **two-way
bootstrap that resamples development works and training seeds independently**,
2,000 draws, deterministic seeds `0..1999`.

S1's scheme — average the three seeds within work, then bootstrap the 84 works —
produces an interval *conditional on those three trained models*. It answers
"how would this differ on other works", not "how would this differ on another
training run", and the second is the question a curve is asked. Seeds are
resampled as paired clusters so the fractions stay paired within a seed.

**The draw is specified to the operation, not just described**, because two
correct-looking summarisers would otherwise report different intervals:

```text
works  = the 84 development work ids, sorted ascending as byte strings
seeds  = (17, 29, 43) in that order
for draw in 0 .. 1999:
    rng        = numpy default_rng(draw)
    seed_index = rng.integers(0, 3, 3)     # drawn first
    work_index = rng.integers(0, 84, 84)   # drawn second
    value[draw] = mean over w in work_index of
                      ( mean over s in seed_index of  d[s][w] )
```

`d[s][w]` is the paired per-seed, per-work difference. The seed draw precedes
the work draw and both come from the same generator, so the sequence is fixed;
the interval is the 2.5th and 97.5th percentiles of `value`.

**This is deliberately the weaker-powered choice, and an `inconclusive` from
interval width is a registered, acceptable outcome rather than a failure.** A
bootstrap over three seeds is a crude variance estimate and will widen the
interval substantially — plausibly to the point where neither MCID bound is
crossed. Raising the seed count would break the anchor reuse, which supplies only
three. Recording that here is the difference between a limitation and a surprise.

The work-only interval is **also reported**, unchanged from S1, so the two runs
remain comparable; it is a secondary and does not decide anything. Per-seed
slopes are reported individually.

**Also required, reported together and none substituting for another:**

- `F1_50 − F1_25`, to say whether the curve has a shape or a step;
- the **all-except-Candombe** macro slope over the 77 remaining works. This is a
  **deciding endpoint, not a diagnostic** — see the decision table. Candombe
  produced 72.4% of S1's pooled gain, so a verdict driven by the overall slope
  alone cannot tell a curve from one genre saturating;
- **validation-point counts, eligible-point counts and selected epochs** per
  fraction and seed, plus the **last-common-epoch** secondary endpoint. Equal
  cadence does not give equal selection opportunity: patience truncates, and S1
  itself came out at 10/10/9 against 9/10/10 from early stopping alone. A
  fraction that plateaus sooner draws fewer maxima on the works that carry the
  endpoint, and only the counts show it;
- per-corpus slopes, with their interpretability fixed **now** rather than
  chosen when the numbers arrive:

| corpus | dev works | status |
|---|---:|---|
| RWC2 | 63 | interval-bearing; the only per-corpus confidence interval to be read as evidence |
| Candombe | 7 | exploratory, with a strong caveat; an interval over 7 works is not a measurement |
| BPSD | 7 | exploratory, same caveat |
| KRAISLER | 4 | descriptive only, no interval |
| Rubato | 3 | descriptive only, no interval |

- beat F, downbeat F, coverage, `usable_strict`, stable exact position, false
  switches and wrong episodes at each fraction, against A0 and against each
  other, as diagnostics.

## Decision

The minimum worthwhile gain is **+0.03 of bar-phase F1**, inherited from S1's
registered efficacy margin. It is an **operational MCID carried over from a
document fixed before any S1 output existed, in the same units on the same
endpoint and the same works — not a product-contract number.** `plan.md` states
no product threshold for bar-phase F1; inventing one now, knowing F1_100 =
0.396, would be worse than reusing one registered in ignorance of it.

Each slope is classified by the same non-overlapping rule: **material** if its
lower bound is at least +0.03, **saturated** if its upper bound is below +0.03,
**inconclusive** otherwise, including any interval spanning +0.03.

**The deciding slope is the all-except-Candombe one**, over the 77 works that are
not Candombe:

| non-Candombe | Candombe mean | outcome |
|---|---|---|
| material | any | `data_limited_under_fixed_recipe` |
| saturated | ≥ +0.03 | `candombe_localized_growth` |
| saturated | < +0.03 | `saturated_at_mcid` |
| inconclusive | any | `inconclusive` |

**Why not the overall slope, which an earlier draft made the primary axis.** The
question C1 exists to answer is whether more data of this distribution is worth
collecting, and the overall slope is the one quantity a single genre can
contaminate. It is also insensitive in exactly the case that matters: an effect
confined to seven works of eighty-four moves the mean by `7/84` of its size, and
resampling works varies how many Candombe works a draw contains, so the interval
stays wide. Measured on fixtures, a Candombe step of 0.60 gives +0.050 [+0.021,
+0.086] and 0.75 gives +0.063 [+0.027, +0.107] — both `inconclusive` — and only a
step of 0.90 or more reaches `material`. Since Candombe's whole available climb
from the frozen model is 0.95, the outcome the second axis was added to name was
very nearly unreachable through the axis that gated it. The overall slope is
still computed and reported; it no longer decides.

**Candombe's own term is a label, not a gate.** Its slope is over 7 works, which
this document classifies as exploratory and not interval-bearing, so it may not
turn a saturated result into a growth result. It selects only which of the two
saturated names is used, and **both carry the same consequence**: neither
justifies sizing `P1-B1` to extend this distribution. The distinction is
diagnostic — `candombe_localized_growth` records that one genre the frozen model
could not track at all was still climbing while everything else had stopped,
which is worth knowing and is not worth acting on by itself.

**A `selection_sensitive` override.** If the deciding slope's
selected-checkpoint endpoint and its last-common-epoch endpoint fall in
different MCID classes, the outcome is `selection_sensitive/inconclusive`
regardless of the table. Checkpoint choice may not be what decides a curve.

**Why `under_fixed_recipe`, and why the suffix never comes off here.** At a fixed
50-epoch cap, 100% receives roughly four times the optimiser updates of 25%, so
data volume and update count are confounded and growth cannot be attributed to
data alone.

An earlier draft of this document proposed dropping the suffix when the smaller
fractions stopped early on patience. That was wrong and is withdrawn: early
stopping means the **selected development metric did not improve within the
patience window**, which is not the same as optimisation having converged. It is
a diagnostic and is reported as one. **The suffix is unconditional in C1.**
Removing it requires evidence C1 does not collect — a longer-schedule arm, or a
compute-matched control that equalises updates across fractions — and either is
a separate registration.

`inconclusive` is also forced by: a digest mismatch, a dirty tree, a missing
seed or fraction, a technical exclusion, a subset that is not nested or not
stratified, any change to the held-identical list, or an anchor whose recomputed
values differ from the S1 artifact.

**What a saturated result licenses.** Only that this recipe on this distribution
has stopped improving. It does not choose the next intervention, and it does not
open A4 or S2.

## Implementation, and the one invariant that decides whether the anchor is real

No C1 runner, subset generator, summariser or test exists yet. None of the
numbers above may be produced until they do and have been independently
reviewed.

**The subset filter must be order-preserving, and this is not a style
preference.** `contiguous_batches` draws
`np.random.default_rng(seed).permutation(len(recordings))` — a permutation of
**positional** indices — and then reads `recordings[order[cursor]]`. The batch
schedule is therefore a pure function of the *order of the list it is handed*
and the seed. `fixed_split` walks `manifest["records"]` in manifest order and
appends; `run.py` derives `train_rows` by list comprehension, preserving it.

So the hash ordering has exactly one job — **choosing which works are members**.
The row list emitted afterwards must be rebuilt in manifest order. Handing the
scheduler rows in hash order would change which recording occupies which slot at
which step for the same seed, which would change the SGD trajectory at **every**
fraction including 100% — and a 100% arm that does not reproduce the S1 runs is
not an anchor, which removes the reason C1 is six runs rather than nine.

**Preflight, before any C1 training.** Order equality is necessary but not what
matters; the emitted schedule is. For each of at least the first three epoch
seeds, run `contiguous_batches` over the C1 100% selection and over S1's
`train_rows`, and require the full emitted sequence of
`(slot_id, identity, work_id, block index, reset, end)` to be identical. This
costs seconds, needs no training, and proves exactly the property the anchor
depends on. The 765-record identity list and its order are digested into the
subset artifact and into checkpoint identity.

**The registered fractions are the work-level 25/50/100**, and the frame-level
fractions follow from them because works differ in length. Under the byte-exact
rule and the count table above they are **29.31% and 55.68%**. Since that rule
now admits one reading, a generator producing anything else has a defect rather
than a defensible alternative, and the preflight treats a mismatch as fatal. The
generator still computes and freezes them in the subset artifact; any figure
drawn from this curve states which axis it uses.

## Operational contract

Outputs, checkpoints, caches and the subset artifact live outside the
repository. The run starts from `tree_clean: true` and records code, source
model, split, cache, subset, config and binary digests. Worker count and paths
are operational; fractions, subsets, arm, recipe, seeds, cadence, selection rule
and thresholds are identity, and resume fails closed on any mismatch.

Acceptance follows M0a–M0e and S1: recompute every deciding value from `records`
without reading `summary`, verify the anchor reproduces the S1 artifact exactly,
compute the artifact SHA-256, write `research/results/C1_REVIEW_<date>.md` with
the absolute path and run commit, and add C1 to `research/results/README.md`
before its verdict is used.

Any correction before the first 25% or 50% output is a dated pre-run revision.
After either exists, a changed rule is a deviation.

## What would void the run

- generating or altering the subsets after seeing any C1 training output;
- changing the validation cadence, patience, selection rule, dev population or
  recipe between fractions;
- rerunning or re-selecting the 100% anchor;
- reporting the macro slope without the all-except-Candombe slope;
- reading a per-corpus interval from a corpus this document marks descriptive;
- resampling recordings instead of works;
- choosing the minimum worthwhile gain after seeing `F1_50`;
- emitting training rows in any order but the manifest's, or running C1 before
  the schedule-equivalence preflight passes;
- deciding on the overall slope alone, without classifying the
  all-except-Candombe slope;
- reading `data_limited` without its `under_fixed_recipe` qualifier when any
  fraction reached the 50-epoch cap.
