# Preregistered M0e — paired non-oracle decoder regression

Status: **fixed before any M0e corpus output exists** (2026-08-12).

## Question

M0d isolated a decoder mechanism under a reference tactus grid and oracle
downbeat evidence. Its fixed `L2_latest` arm raised reacquisition within two
new-meter bars by +0.5806 [0.4247, 0.7312] without losing stable exact-position
accuracy. The same experiment also showed why cost zero is not a candidate:
free switching produced 3,829 held-state changes and lost 0.0725 stable
accuracy.

M0e asks the transfer question M0d deliberately did not answer:

> When the beat grid and downbeat channel both come from the frozen BeatNet
> frontend, does the already selected `L2_latest` decoder reacquire annotated
> meter/phase changes faster than the current decoder, without introducing
> static phase switches or long wrong-state episodes?

This is a paired decoder regression, not another parameter sweep. It compares
one frozen candidate with one frozen baseline. Annotations are used only by the
scorer; neither arm receives reference beats, reference phase, reference meter
or oracle salience.

## Bound sources and populations

The source results are fixed by content:

- M0d artifact SHA-256:
  `668b1890e5055ffb4db9d2ecba00fa03f1c4bfdfd80c656e1764b8a95f419991`;
- M0d run commit:
  `b5376267e70a5d98daefb0cf9365e25f60e3cbca`;
- M0c artifact SHA-256:
  `88d7ecc2e2ef655faf475081768218a0dc2467d29d048bfa4b1eafc5d23f74fa`;
- M0c run commit:
  `c4c5b0c52cce12c835c7f5c626820701c7ff5579`;
- M0b artifact SHA-256:
  `142580478abfe0734bc91ac8fdd20c605a392f9ad7334cb63047d98e5135e921`;
- M0b run commit:
  `1b0cb7c6ed71b70e714b208d14f188a0564f165c`;
- canonical manifest SHA-256:
  `484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd`;
- BeatNet model SHA-256:
  `812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1`.

There are two fixed, overlapping analysis populations.

### Transition efficacy population

The efficacy population is the complete M0c/M0d transition cohort: 34 primary
RWC2 works, 123 registered transitions and the 61 transitions over 31 works
whose two-bar endpoint M0c fixed as fully observable. The other 62 transitions
remain right-censored for the primary endpoint and enter only
intention-to-treat and descriptive summaries. Transition IDs, reference
boundaries, previous/new grouping, next-change/end boundaries,
`common_start_sec` and observability are copied from the content-bound M0c
artifact. No M0e arm may rediscover or alter them.

### Static breadth population

The safety population is every primary record successfully scored by M0b: 980
performances representing 414 independent works. It is fixed from M0b's
`records`, not selected from M0e output:

| corpus | performances | independent works | annotated changes |
|---|---:|---:|---:|
| BPSD | 122 | 31 | 20 |
| Candombe | 35 | 35 | 0 |
| Kraisler | 20 | 20 | 1 |
| Rubato | 489 | 14 | 56 |
| RWC2 | 314 | 314 | 123 |
| **total** | **980** | **414** | **200** |

M0b's seven runtime exclusions are not silently retried or promoted into M0e:
they are two Rubato score works with duplicate/non-increasing canonical tactus
timestamps and never produced complete M0b arm output. They remain named source
exclusions. Every one of the fixed 980 records must produce both M0e arms; any
new technical exclusion makes the binding interpretation `inconclusive`.

These corpora are development data. A positive M0e result is a regression pass,
not an out-of-sample performance claim.

## Fixed arms and evidence contract

| arm | held phase anchor | phase-switch cost | role |
|---|---|---:|---|
| `B64_opening` | opening `result.phase` | 64 | current/M0b A4 baseline |
| `L2_latest` | latest decoded path downbeat | 2 | only eligible candidate |

`L2_latest` is copied unchanged from M0d. Cost 8, cost 64 with latest readout
and cost zero are not rerun and cannot be selected. There is no post-result
ladder, tuning or corpus-specific configuration.

For each recording, the harness performs one ordinary frozen BeatNet live pass
and caches:

- beat and downbeat activation values;
- exact activation timestamps and frame-release blocks;
- every beat handed to `BarTracker` and its publication block;
- the subset of beats actually published to the product output.

Both arms replay the same cached arrays through the same `tracking::BarTracker`.
Only the two settings in the arm table differ. The candidate is therefore
non-oracle but still decoder-only: frontend evidence and its causal delivery
clock are identical between arms.

The resolver operates on the complete internal beat sequence, including beats
that were too late to play, because the product increments bar state for those
beats. Product-facing metrics are scored on the published-beat subset. A fixed
greedy monotonic one-to-one match maps published beats to reference tactus
events within +/-70 ms, with earliest predicted index breaking equal-distance
ties. Unmatched reference events are unknown/wrong; they are never dropped.

The scorer uses each record's M0b `common_start_sec`. The baseline and candidate
therefore share one denominator chosen before M0e and cannot improve a metric by
delaying their own first decision.

## Baseline parity and apparatus power

Before any comparison is interpreted, all of the following must pass:

1. `B64_opening` replay positions, groupings and confidence values reproduce
   the ordinary live baseline for every complete internal beat exactly;
2. the baseline's published-grid M0b A4 scalars and integer change counts
   reproduce the fixed source record within absolute tolerance `1e-12`;
3. both arms report identical activation arrays, frame clocks, internal beat
   grid, publication schedule, visible-beat indices and reference matching;
4. all 980 fixed records, all 414 work IDs, all 34 transition works, all 123
   transition IDs and all 61 fully observable flags reproduce exactly;
5. a deterministic synthetic same-meter phase-shift preflight demonstrates
   that `L2_latest` reaches the planted latest path state within two bars,
   differs from `B64_opening` on at least one planted event, and changes no
   neural evidence, beat grid or meter candidate. The neural evidence and beat
   grid are the same arrays for both replays and are identical by construction;
   the meter candidate is a resolver output the two arms configure differently,
   so the preflight compares `bar_replay_meters` between arms and fails if they
   differ.

Failure of any item is fatal to the binding interpretation. If the synthetic
preflight passes but the two corpus arms are identical, that is a valid
non-transfer result rather than an inert-apparatus result.

The historical source baseline is recorded only to calibrate regression
margins. Across the 980 fixed records, M0b A4 reported phase F1 0.1991,
balanced grouping accuracy 0.2003, exact-position accuracy 0.1258, coverage
0.4213 and false-confident share 0.2156. On the 34 RWC2 transition works it
reported phase F1 0.3403 and exact-position accuracy 0.2721. These already
published baseline values are not substituted for the paired M0e replay and do
not enter the candidate verdict directly.

## Fixed transition efficacy endpoint

Candidate and baseline outputs are mapped onto the same reference tactus
indices. At each fixed M0c transition, acquisition is the first reference
position-1 start of one complete new-grouping bar for which every event is
matched, the predicted grouping equals the new grouping and predicted positions
equal the complete reference sequence `1..new_grouping`.

The primary endpoint is acquisition within two new-meter bars among the fixed
61 fully observable transitions, averaged first within work and then across the
31 works. The deciding quantity is the paired work-level difference
`L2_latest - B64_opening`.

Efficacy passes only when both hold:

- mean paired gain is at least **+0.10**;
- the lower bound of its paired 95% bootstrap interval is greater than zero.

Ten points is deliberately smaller than M0d's registered +0.20 gate because
frontend misses now bound what any decoder can recover, but it still requires a
material transfer rather than a merely positive mean.

Always report first-bar acquisition, all-123 intention-to-treat acquisition,
eventual acquisition, latency in tactus events/new-grouping bars, right
censoring and outcome counts by transition pair and work. None replaces the
primary endpoint.

## Fixed static safety denominator

Safety is measured over the 980-record breadth population. Eligible reference
tactus events are supported, at or after the source `common_start_sec`, and
outside a fixed adaptation interval. For every annotated change, that interval
is the half-open reference-index range from the change through
`2 * new_grouping` events, clipped at the next change or segment end. The union
is computed from annotations before either arm runs and is shared by both.

At an eligible event, exact position requires a matched published beat, correct
grouping and correct 1-based position. Unmatched and unknown output is wrong.
Counts are summed over performances belonging to one work before producing one
work value, so Rubato's multiple performances do not act as independent works.

The primary safety condition is:

- the lower bound of the paired work-level stable exact-position difference is
  at least **-0.03**.

This is tighter than M0d's -0.05 oracle margin because M0e is the product-facing
regression and a three-point absolute loss is already material on the low A4
baseline.

Always report paired phase F1, balanced grouping accuracy, coverage,
false-confident share and unnecessary-unknown share over the same fixed start
and work aggregation. These diagnose a failure but do not replace the exact
position safety gate.

## False-switch and wrong-episode vetoes

Two operational safety vetoes prevent a mean accuracy from hiding visible
churn.

For each answered, matched stable event, define the decoder anchor state as
`(predicted_grouping, (reference_index - predicted_position_zero) mod
predicted_grouping)`. A **false switch** is a change between consecutive
answered stable events in the same annotation segment while the reference
grouping and phase anchor are unchanged. Unknown/unmatched output breaks that
comparison and is instead charged to the wrong-episode metric.

A **wrong-state episode** is a maximal contiguous run of non-exact eligible
reference events inside one stable annotation segment. Adaptation intervals,
unsupported events and segment boundaries end an episode. An episode is long
when it lasts at least one local reference bar (`grouping` tactus events).

Eligible stable duration is fixed from annotations: for every eligible event
that has a following tactus event in the same annotation segment, add
`next_time - current_time`; do not bridge adaptation, unsupported or segment
boundaries, and do not invent duration after a segment's final event. Durations
and counts are summed over performances of the same work before division and
are expressed per five minutes. Candidate safety additionally requires:

- the **upper** bound of the paired false-switch-rate difference is at most
  **+1.0 per five minutes**;
- the **upper** bound of the paired long-wrong-episode-rate difference is at
  most **+0.25 per five minutes**.

These margins mean at most one additional unsupported bar-state flip in a
five-minute recording and no more than one additional >=1-bar wrong episode per
twenty minutes. Report all wrong episodes, >=1-bar episodes, >=2-bar episodes,
longest wrong episode and state changes even when the vetoes pass.

## Statistics

All deciding metrics are paired by independent work. Performance values are
combined within work before comparison. Confidence intervals are deterministic
2,000-draw percentile bootstraps over works with seeds `0..1999`. Record- or
transition-level resampling is forbidden.

This is a single-candidate intersection gate: efficacy and every safety
condition must pass. There is no best-arm selection and no metric may substitute
for another, so no post-hoc multiplicity correction or fallback ordering is
defined.

## Registered interpretation

After parity, coverage and synthetic power pass:

1. efficacy pass plus all three safety conditions pass ->
   `non_oracle_decoder_candidate_pass`;
2. efficacy passes but any safety condition fails ->
   `non_oracle_gain_static_cost`;
3. efficacy fails while all safety conditions pass ->
   `oracle_gain_does_not_transfer`;
4. efficacy fails and any safety condition fails ->
   `non_oracle_candidate_regression`.

The interpretation is `inconclusive` if source validation, clean provenance,
baseline parity, evidence identity, synthetic power, exact population,
complete paired output or minimum 30 efficacy works fails.

A pass freezes `L2_latest` for a later locked/real-input confirmation and code
review. It does not make cost 2 the product default, does not create an
out-of-sample claim and does not open S2 or neural training. A transfer failure
does not erase M0d's oracle mechanism; it says frontend errors prevent that
mechanism from buying the registered product-facing gain and returns priority
to the already planned frontend/stateful-training work.

## Operational contract

Implementation must reuse the existing M0b/M0d manifest validation, causal
replay, bounded worker scheduler, atomic outcome files and pause/resume
checkpoint semantics. Output, checkpoint and pause paths remain outside the
repository. Worker count is operational and may change across resumed sessions;
it is not part of experiment identity.

Checkpoint identity includes the run commit, source artifact hashes, binary,
model and manifest digests, both arm definitions, matching rule, transition IDs,
adaptation rule, all thresholds, bootstrap seeds/draw count, metric schema,
population selection hash and code schema. Resume fails closed on any mismatch.

`--limit`, incomplete audio hashing or skipped source verification may produce
diagnostics only and force `inconclusive`. A binding run starts from
`tree_clean: true`; provenance and synthetic preflight are collected before
checkpoint creation. No binding run begins until this registration,
implementation and tests have been reviewed and committed.

The canonical manifest and BeatNet model digests above are verified against the
files actually passed on the command line, before the source artifacts are read
and before any record is measured. They are preconditions, not annotations.

Any review correction made before the first M0e corpus output must be recorded
as a dated pre-run revision in this document and committed before execution.
Once any candidate corpus output exists, a changed rule is a deviation rather
than a preregistration repair and cannot silently replace this decision logic.

After a completed run, acceptance requires the same repository-side procedure
as M0a-M0d:

1. independently recompute every deciding value from `records` without reading
   `summary`;
2. verify baseline parity, evidence identity, synthetic power, coverage and all
   exclusions;
3. compute the artifact SHA-256;
4. create `research/results/M0E_REVIEW_<date>.md` with absolute path, hash and
   run commit;
5. add M0e to `research/results/README.md` before using its verdict.

## Pre-run review revision — 2026-08-12

Independent review before any M0e corpus output found three declared guarantees
that nothing enforced. No arm, threshold, population, metric, bootstrap or
decision rule changes; each item turns an assertion into a check.

1. `SOURCE_MANIFEST_SHA256` and `SOURCE_MODEL_SHA256` were defined in the runner
   and never compared to anything. A wrong model would have failed later at the
   M0b A4 parity assert, reporting parity rather than the model; a manifest
   agreeing on identity, work and corpus profile would have passed
   `select_population` outright. Both are now verified before any other work.
2. The synthetic preflight tested acquisition and baseline difference but not
   the third registered condition. It now compares `bar_replay_meters` between
   arms and fails if the candidate moved the meter, and the by-construction part
   of the claim is named as such instead of being reported as a measurement.
3. The paired bootstrap took the intersection of the two arms' work sets. That
   is unreachable given per-record transition-ID parity, and it now raises
   rather than silently dropping a work.

No M0e record, arm value, aggregate, interval or verdict existed when these were
fixed.

## Boundaries

**In scope:** the frozen clean-audio BeatNet model, its predicted grid and
downbeat channel, the current decoder, `L2_latest`, the fixed development
populations above and paired decoder outcomes.

**Out of scope:** new training, S1/S2, architecture changes, another hysteresis
sweep, room/microphone transfer, click bleed/AEC, denominator prediction,
canonical time-signature naming, user meter input and shipping configuration.
Those require separate evidence and must not be inferred from M0e.

## What would invalidate the run

- any M0e candidate corpus output before this registration is fixed;
- changing cost 2, adding another candidate or selecting a configuration after
  seeing M0e output;
- feeding either arm reference beats, phase, grouping or oracle salience;
- allowing arms to receive different neural frames, beat grids, publication
  schedules, visible-beat subsets, matchings, starts or safety denominators;
- deriving transition boundaries, censoring or adaptation masks from an M0e
  arm;
- dropping unmatched reference events or excluding a record after one arm has
  produced output;
- treating the 980 performances as independent bootstrap units instead of the
  414 works;
- replacing a failed safety component with a favourable diagnostic mean;
- dirty provenance, a digest mismatch, incompatible resume or incomplete arm
  output;
- reading `summary` instead of independently recomputing from `records` when
  accepting the artifact.
