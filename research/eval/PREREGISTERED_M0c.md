# Preregistered M0c — meter-transition trace diagnostic

Status: **fixed before any M0c model/replay output exists** (2026-08-12).

## Question

M0b retained reference phase and grouping on static passages but acquired only
0.0837 of annotated grouping changes within two bars. M0c asks a narrower
question:

> When the unchanged `BarTracker` receives the same A1 reference tactus grid
> and oracle downbeat evidence as M0b, is failed rapid reacquisition dominated
> by stale previous-grouping state, by new-grouping phase/sequence instability,
> or by an endpoint that is not fully observable before the next change/end?

M0c is a diagnostic follow-up, not a new model benchmark and not a replacement
verdict for M0b. It does not train a model, change the decoder, or open S2.

## Bound evidence and population

The source result is fixed by content, not by a mutable path:

- M0b artifact SHA-256:
  `142580478abfe0734bc91ac8fdd20c605a392f9ad7334cb63047d98e5135e921`;
- M0b run commit:
  `1b0cb7c6ed71b70e714b208d14f188a0564f165c`;
- M0b manifest SHA-256:
  `484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd`;
- model SHA-256:
  `812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1`;
- binary SHA-256:
  `e04881ec4344e451cbdbb44c56ffb7c4b98408ba0d1eff2fc129d1ded620b426`.

The binding population is every primary RWC2 record in that artifact whose A1
`changes.total > 0`: 34 records/works and 123 transitions. Selection and each
record's `common_start_sec` come from the source artifact. They are not
recomputed after seeing M0c output.

The input manifest and source artifact are verified before checkpoint creation.
The source artifact must report clean-tree provenance, experiment `M0b`, the
fixed run commit and the same manifest/model/binary digests. Every selected
name must resolve uniquely in the manifest. Full audio and annotation digest
verification is mandatory for a binding run.

## Annotation-only observability finding

Before writing the M0c scorer, the fixed canonical annotations were inspected
without running the model or replay. All 123 transitions contain at least one
complete new-grouping bar. Their available complete bars are:

- exactly one: 56 transitions;
- exactly two: 6 transitions;
- three or more: 61 transitions.

M0b defines acquisition at the reference position-1 tactus from which one
complete correct bar begins, and calls it within two bars when that start is at
most `2 * new_grouping` tactus steps after the change. Fully observing the
boundary case at offset `2 * new_grouping` therefore requires three complete
new-grouping bars. The 62 shorter transitions are right-censored for that
endpoint. Their registered M0b failures remain unchanged, but M0c must not
describe them as observed decoder failures at two bars.

## Arm and parity contract

M0c reruns only M0b A1:

- reference tactus times and reference emission schedule;
- oracle impulses at reference position 1;
- the unchanged causal C++ `BarTracker` replay;
- the same BeatNet activation clock and beat channel obtained from the same
  binary/model/audio inputs.

For every recording, M0c rescoring must reproduce the source M0b A1 scalar
metrics and aggregate change counts at the source `common_start_sec` within
`1e-12`; integer counts must match exactly. A mismatch aborts the run as an
invariant failure. This is the seam/parity gate. M0b already passed the
profiled-oracle and shifted positive controls, so M0c does not rerun them.

## Per-transition trace

One row is persisted for every registered transition, with:

- recording, work, transition ordinal and stable transition ID;
- previous/new grouping, reference index/time and next-change/end boundary;
- available complete new-grouping bars;
- whether the one-bar and full two-bar-latency endpoints are observable;
- the predicted grouping and 1-based position at each reference tactus from
  one previous reference bar through the complete bar needed to verify the
  two-bar latency boundary (at most the first three new bars, or censoring);
- first new-grouping decision, first correctly phased new-grouping downbeat and
  first complete exact new-grouping bar;
- latency in seconds, tactus steps and new-grouping bars when acquired;
- previous/new/unknown/other grouping shares in the first two new bars;
- a mutually exclusive outcome class.

The scorer uses reference-grid indices, not tempo-derived seconds, for bar
latency. It never drops unmatched/unknown decoder outputs.

## Outcome classes

For a transition with full two-bar-latency observability:

1. `acquired_within_two_bars` — an exact complete new-grouping bar begins no
   later than offset `2 * new_grouping`;
2. `acquired_late` — an exact complete bar exists later in the segment;
3. `stale_previous_grouping` — no qualifying acquisition exists and the
   previous grouping occupies at least half of decoder decisions over the
   first `2 * new_grouping` reference tactus events;
4. `new_grouping_wrong_phase_or_unstable` — the new grouping appears in that
   window, but no qualifying exact bar exists and stale state is not dominant;
5. `unknown_or_other` — neither previous nor new grouping explains the failure.

Transitions without full observability receive the same descriptive fields but
the class `right_censored`; they do not enter the dominance denominator.
Precedence is exactly the order above.

## Aggregation and interpretation

Primary summaries are work-level. Transition proportions are first averaged
within work and then across works. Confidence intervals are deterministic
2,000-draw percentile bootstraps over works with seeds `0..1999`.

The following are always reported:

- observability at one bar and at the registered two-bar latency boundary;
- original M0b intention-to-treat acquisition within two bars;
- acquisition within two bars among fully observable transitions;
- first-bar acquisition over all 123 transitions;
- each outcome-class share overall, by transition pair and by work;
- raw event counts alongside work-level estimates.

Among fully observable failures, a failure mechanism is called dominant only
when its work-level mean share is at least 0.60 and its lower 95% confidence
bound is at least 0.50:

- dominant `stale_previous_grouping` prioritises a decoder-state/reset or
  hysteresis counterfactual;
- dominant `new_grouping_wrong_phase_or_unstable` prioritises phase/reacquisition
  logic and acquisition semantics;
- dominant `unknown_or_other` requires seam/state instrumentation before an
  architectural decision;
- otherwise the result is `mixed`.

If A1 parity fails, any selected item is excluded after arm output, or fewer
than 30 fully observable transitions remain, the M0c interpretation is
`inconclusive`. No M0c result directly authorises S2 or a metrical adapter.

## Operational contract

The runner reuses M0b's bounded scheduler and atomic per-item outcome format,
with an M0c-specific checkpoint schema and identity. Output, checkpoint and
pause paths must be outside the repository. A fresh run refuses existing
output/checkpoint paths. Resume fails closed on any change to commit, source
artifact, binary/model/manifest digest, selected order, threshold, or code
schema. Worker count is operational and may change across resume sessions.

`--pause-file` stops new submissions, drains active workers, checkpoints them,
marks the run paused and exits 75. `--limit` or skipped audio verification may
be used only for diagnostics and force an `inconclusive` artifact.

The corpus run must not start until this document and the implementation have
been reviewed. Any post-output change to a metric, category, threshold or
population is a deviation, not a revision of this preregistration.

## What would invalidate the run

- wrong or dirty source provenance, or a digest mismatch;
- selecting records from M0c output rather than the fixed M0b artifact;
- recomputing a different common start;
- failure to reproduce source A1 metrics/counts;
- treating a right-censored transition as an observed two-bar failure in M0c;
- pooled tactus/event confidence intervals that ignore work clustering;
- overwriting or silently reusing an incompatible checkpoint;
- interpreting this development diagnostic as meter-family, room, locked-test
  or product-prevalence evidence.
