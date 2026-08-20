# C1 independent review — 2026-08-20

**Independent verdict: `inconclusive`, confirmed.** The registered deciding
endpoint, non-Candombe bar-phase `F1_100 − F1_50`, is **+0.0165769** with a
two-way 95% interval **[−0.0142559, +0.0484985]**. The interval contains both
zero and the registered +0.03 MCID. It is therefore neither `material` nor
`saturated`, and the decision table returns `inconclusive`.

This review is separate from
[`C1_REVIEW_20260816.md`](C1_REVIEW_20260816.md), which was written by the
party that implemented C1. The recomputation used an independent Python
implementation of the operations fixed in
[`PREREGISTERED_C1.md`](../eval/PREREGISTERED_C1.md). It imported no TikTak
summariser code and did not open C1's `summary.json` until after its own result
had been produced. The reviewer had previously seen the headline values in
conversation, so this is computationally independent but not a blinded review.

## Inputs and identity

| item | independently verified value |
|---|---|
| six C1 runs | `C:\Users\sidle\.codex\visualizations\2026\08\09\019fe6c2-2d8c-71c3-9cb5-af836cdd3c86\c1-runs-d5a9510-20260815` |
| C1 run commit | `d5a95101cd8c6e9afde0fa30cea8f703f5729f31` |
| three reused S1 anchors | `C:\Users\sidle\.codex\visualizations\2026\08\09\019fe6c2-2d8c-71c3-9cb5-af836cdd3c86\s1-runs-b12eea8-20260813` |
| anchor commit | `b12eea828f25df502a157cccc872c5e2000cc2e3` |
| portable records bundle | `cf95055e565de8cc512fca97ac91c1bea5521c8fb6d8e6131ed6c72fb21dae70` |
| subset artifact | `d9f04a86f899331c91070bbbd3944ea4e955e368e773b1d150e2be36ae33a626` |
| S1 anchor summary | `4c7dd592ce0bce191b2c78b8a59616f6d75dfe1870056b227c2ba9a81010160f` |
| A0 baseline | `4db6990164291f078a3cc22e9b31c47715759a557e28130af3396c91c84b3385` |
| C1 `summary.json`, read only after recomputation | `c658239938b687ab34549d2cb7ddfc40fd4dc9b1da36665408073e291e84880d` |

The audit authenticated all nine `result.json` files and all eighteen
evaluations used by the two endpoints: nine selected-checkpoint files and nine
last-common-epoch files. Every digest in
[`C1_RECORDS_20260816.json`](C1_RECORDS_20260816.json) matched its raw source.
All six new runs were complete and reported
`identity.commit == provenance.commit == d5a95101…` and `tree_clean: true`;
all three anchors matched the pinned S1 result/evaluation digests and selected
epochs.

Every endpoint contained the same complete development population: 215
recording records grouped into 84 works — RWC2 63, Candombe 7, BPSD 7,
KRAISLER 4 and Rubato 3. No run, seed, work or selected/common evaluation was
missing. All per-work values were finite. The subset preflight contained
exactly seeds `17,18,29,30,43,44` and 26,813 compared blocks for each.

## Independent arithmetic

The implementation followed the registration literally: works sorted as UTF-8
byte strings; seeds `(17, 29, 43)`; 2,000 draws numbered `0..1999`; one NumPy
`default_rng(draw)` per draw; seed indices sampled before work indices. It used
the raw evaluations' per-work metrics after proving them byte-for-byte equal to
the portable bundle.

| endpoint | mean | two-way 95% CI | class |
|---|---:|---|---|
| **100−50%, non-Candombe (deciding, 77 works)** | **+0.0165769** | **[−0.0142559, +0.0484985]** | **inconclusive** |
| 100−50%, all 84 works | +0.0182434 | [−0.0118445, +0.0489647] | inconclusive |
| 100−50%, non-Candombe, last common epoch | +0.0227950 | [−0.0041996, +0.0522430] | inconclusive |
| 50−25%, non-Candombe | +0.0279037 | [−0.0111521, +0.0650710] | descriptive |
| 50−25%, all 84 works | +0.0324786 | [−0.0050137, +0.0741533] | descriptive |

The selected and last-common endpoints have the same class, so
`selection_sensitive = false`. Their point estimates are not equal; the flag
only says checkpoint selection did not change the registered class.

Per-seed deciding slopes were `+0.0231133`, `−0.0055500` and `+0.0321673` for
seeds 17, 29 and 43. The sign disagreement is real. The registered work-only
secondary was +0.0165769 [−0.0041887, +0.0388593], materially narrower than
the primary seed-and-work interval and therefore insufficient as a replacement.

The common validation epochs independently derived as `45`, `50` and `50`.
All nine selected epochs, validation counts and eligible counts reproduced from
the raw candidate evaluations using the frozen A0 beat-noninferiority condition
and lexicographic `(phase, downbeat, beat)` maximum.

After these results were fixed, 155 numeric fields were compared with C1's
`summary.json`: primary and secondary endpoints, intervals, per-seed slopes,
per-corpus results, selection diagnostics and all registered diagnostic
metrics. There were zero discrepancies above `1e-12`; the maximum absolute
difference was `7.99e-15`. Verdict, endpoint classes and
`selection_sensitive` also matched.

## Interpretation

The measured curve bends over in point estimates, but the experiment does not
establish the bend. Overall bar-phase levels were 0.3447, 0.3772 and 0.3955 at
25%, 50% and 100%. The deciding non-Candombe levels were 0.2982, 0.3261 and
0.3426. Both doublings have intervals spanning zero and +0.03.

Candombe's +0.0365754 [+0.0042454, +0.0785222] is exploratory evidence over
seven works and does not gate the result. RWC2, the only interval-bearing
per-corpus endpoint, is +0.0164725 [−0.0165816, +0.0504762] and agrees with the
primary uncertainty.

The sharpest secondary change is the reduction in false switches from 50% to
100%: −12.112 [−26.404, −1.035] per five minutes. This supports a within-curve
churn improvement at the full fraction. It does **not** establish that the 25%
or 50% models are worse than A0: independently computed two-way intervals
against A0 were +6.50 [−7.91, +19.96] and +10.68 [−4.05, +24.74], both spanning
zero.

The result remains development-only, conditional on one nested subset
hierarchy, three optimisation seeds and the fixed training schedule. Data
volume is confounded with optimiser updates. It therefore licenses neither
expanding `P1-B1` from this distribution nor declaring the recipe saturated.

## Findings in the original review text

These findings do not alter the C1 artifact or verdict.

1. `C1_REVIEW_20260816.md` twice describes the 100%−50% comparison as “four
   times the data.” It is approximately twice the work count (330 versus 166)
   and 1.80 times the frame count (10,566,912 versus 5,883,501). Four times
   applies only to 100% versus 25% by nominal work fraction.
2. “A partially trained model is churnier than no training at all” is stronger
   than the evidence. Only the 100%−50% contrast excludes zero; both partial
   fractions' contrasts against A0 do not.
3. “Beat F1 has already saturated” is not a registered verdict. The observed
   100%−50% beat change is +0.00259 [−0.01815, +0.02097], which supports “no
   detectable increase in this run.” C1 registered +0.03 as a bar-phase MCID,
   not a beat-specific saturation threshold.
4. All six new result files carry `provenance.experiment: "S1"`. Their C1
   identity, subset digest, fraction, commit and output paths are correct, so
   this does not invalidate or ambiguate the run. It is nevertheless a metadata
   defect: future C1/P1-B1 launch paths should label the experiment they are
   actually running.

## Consequence

C1 is now independently accepted as **valid but inconclusive**. A stronger
curve must be registered before more training output exists. At minimum it must
measure subset-sampling variance with crossed subset draws, add enough
independent optimisation seeds to make the +0.03 decision informative, and
either equalise update opportunity or retain an explicit
`under_fixed_recipe` interpretation. Power should be simulated from the C1
seed/work matrix before fixing the run count; simply repeating three seeds is
not guaranteed to resolve the interval.
