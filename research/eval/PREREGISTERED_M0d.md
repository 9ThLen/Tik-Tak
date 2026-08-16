# Preregistered M0d — decoder path-state reacquisition counterfactual

Status: **fixed before any M0d corpus output exists** (2026-08-12).

## Question

M0c found that 48 of 59 fully observable failures acquire the new meter late,
after a median five new-meter bars. In those late cases the decoder often emits
the new grouping during the first two bars but does not emit one complete,
correctly phased bar.

The C++ resolver already returns a dynamic bar-position path in
`DownbeatResult::downbeats`. `BarTracker`, however, anchors its held output to
`DownbeatResult::phase`, which the resolver documents as the **opening** phase
of the trailing window once dynamic switching is enabled. M0d asks:

> With M0c's reference tactus grid and oracle downbeat evidence held byte-for-
> byte fixed, does reading the current end of the resolver's path, or releasing
> that path's phase-switch hysteresis, causally shorten meter-change
> reacquisition without damaging stable-position accuracy?

M0d changes no audio, model, activation, beat grid, oracle channel, meter
candidate, confidence threshold or frontend. It is a decoder-only development
counterfactual. It does not train a model and cannot by itself open S2.

## Bound source and population

The source result is fixed by content:

- M0c artifact SHA-256:
  `88d7ecc2e2ef655faf475081768218a0dc2467d29d048bfa4b1eafc5d23f74fa`;
- M0c run commit:
  `c4c5b0c52cce12c835c7f5c626820701c7ff5579`;
- source M0b artifact SHA-256:
  `142580478abfe0734bc91ac8fdd20c605a392f9ad7334cb63047d98e5135e921`;
- manifest SHA-256:
  `484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd`;
- model SHA-256:
  `812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1`;
- M0c binary SHA-256:
  `e04881ec4344e451cbdbb44c56ffb7c4b98408ba0d1eff2fc129d1ded620b426`.

The binding population is the complete M0c population: 34 primary RWC2 works
and their 123 registered transitions. The primary latency population is the 61
transitions M0c fixed as fully observable at the two-bar boundary. The other 62
remain right-censored for that endpoint and enter only intention-to-treat and
descriptive summaries. Population, `common_start_sec`, transition IDs and
observability are read from the fixed M0c artifact, never rediscovered from an
M0d arm.

An annotation-only denominator check made before implementation found that the
61 fully observable transitions span 31 works. The stable-event definition
below yields 17,175 events across all 34 works; every work contributes, and the
smallest contributes 31 events. A binding run must reproduce all four fixed
coverage counts: 61 transitions, 31 efficacy works, 17,175 stable events and 34
stable works.

The source artifact, manifest, audio, annotations, model and newly built binary
are digest-verified before checkpoint creation. M0c must report clean-tree
provenance, experiment `M0c`, the fixed run commit, 34/34 scored records, no
technical exclusions and exact source digests.

## Fixed arms

Every arm receives the same cached BeatNet activation clock, reference tactus
times/emission schedule and oracle impulses at reference position 1. Arms differ
only in two `BarTracker` decoder settings:

| arm | held phase anchor | phase-switch cost | role |
|---|---|---:|---|
| `B64_opening` | opening `result.phase` | 64 | exact current-path/M0c baseline |
| `L64_latest` | latest path downbeat | 64 | readout-only counterfactual |
| `L8_latest` | latest path downbeat | 8 | moderate state-release candidate |
| `L2_latest` | latest path downbeat | 2 | aggressive state-release candidate |
| `L0_latest_control` | latest path downbeat | 0 | positive apparatus control only |

The values 64, 8 and 2 are existing resolver sweep points documented in
`analysis/downbeat.hpp`; they are not fitted on M0c. Zero is deliberately not a
product candidate: it makes changing phase free and exists only to prove that
the path/readout apparatus can move under oracle evidence.

`latest path downbeat` means the most recent beat in
`DownbeatResult::downbeats`, mapped back to the exact beat index already held by
the same `BarTracker` window. It does not inject a reference phase or meter and
does not reset the tracker. If the resolver returns no downbeat, the previous
held decision is retained exactly as in the baseline.

## Baseline parity and synthetic power

For every work, `B64_opening` must reproduce the fixed M0c record's A1 score,
transition trace and aggregate counts within absolute tolerance `1e-12`, with
integer and categorical fields exact. Any mismatch aborts the run.

Before checkpoint creation, a deterministic synthetic same-meter phase-shift
preflight drives the C++ seam with planted salience. It must show:

1. default settings reproduce the opening-anchor baseline;
2. latest-path readout at zero cost follows the planted new phase;
3. no neural state, beat grid or meter candidate changes between those arms.

Failure is fatal and produces no checkpoint.

The corpus-level positive-control power gate is also fixed. Relative to
`B64_opening`, `L0_latest_control` must improve work-level acquisition within two
bars among fully observable transitions by at least `+0.30`, and the lower bound
of its paired 95% bootstrap interval must be greater than zero. Otherwise the
M0d interpretation is `inconclusive`: a null candidate cannot be distinguished
from an inert path/readout apparatus.

## Endpoints

The primary efficacy endpoint is M0c's exact acquisition within two new-meter
bars among the fixed 61 fully observable transitions. Values are averaged first
within work and then across works. Arm differences are paired by work.

The primary safety endpoint is stable exact bar-position accuracy. For each
work, an adaptation interval is the half-open reference-index range from a
registered change through `2 * new_grouping` tactus events, clipped at the next
change/end. Stable events are supported reference tactus events at or after the
fixed `common_start_sec` that lie outside the union of those intervals. An event
is exact only when both grouping and 1-based position equal the reference.
Unknown output is wrong, not dropped.

A candidate is effective only when its paired work-level improvement on the
primary efficacy endpoint is at least `+0.20` and the lower 95% confidence bound
is greater than zero. It is safe only when the lower bound for its paired stable
accuracy difference is at least `-0.05`. These are joint requirements.

Always report, by arm:

- acquisition within one and two bars, intention-to-treat and fully observable;
- acquisition latency in tactus events and new-grouping bars, with censoring
  counts kept beside it;
- stable exact-position accuracy and its paired difference from baseline;
- outcome counts by transition pair and work;
- the number of resolver path changes and the number that alter held output;
- paired differences and deterministic confidence intervals.

Confidence intervals are deterministic 2,000-draw percentile bootstraps over
works with seeds `0..1999`. Transition-level pooling is forbidden.

## Registered interpretation

After parity, coverage and positive-control power pass, candidates are examined
in the fixed least-invasive order `L64_latest`, `L8_latest`, `L2_latest`:

1. the first arm satisfying both efficacy and safety is the selected arm;
2. selecting `L64_latest` yields `opening_phase_readout_bottleneck`;
3. selecting `L8_latest` or `L2_latest` yields
   `phase_hysteresis_bottleneck`;
4. if an arm meets efficacy but every efficacious arm fails safety, the result
   is `transition_gain_static_cost`;
5. if no candidate meets efficacy while the positive control passes, the result
   is `registered_decoder_ladder_negative`.

The interpretation is `inconclusive` if baseline parity, synthetic preflight,
positive-control power, complete arm output, exact registered coverage or the
minimum of 30 fully observable transitions fails. A candidate result authorises
only a separate decoder regression/implementation phase on non-oracle data. It
does not authorise neural training, a metrical adapter or a product default.

The arm means across `L64_latest`, `L8_latest`, `L2_latest` and
`L0_latest_control` are additionally reported in decreasing-cost order.
Non-monotonicity is a diagnostic warning, not a post-hoc arm-selection rule.

## Operational contract

The runner reuses M0c's bounded scheduler, atomic outcome files and
pause/resume semantics under new M0d checkpoint/artifact schemas. Output,
checkpoint and pause paths are outside the repository. A fresh run refuses
existing paths. Resume fails closed on changes to commit, source artifact,
binary/model/manifest digest, arm order, costs, thresholds, selected order,
metric schema or code schema. Worker count is operational and may change across
resume sessions.

`--pause-file` stops submissions, drains active workers, checkpoints and exits
75. `--limit` and skipped audio verification are diagnostic-only and force an
`inconclusive` artifact. The full run must start from `tree_clean: true` and only
after this document, implementation and tests are reviewed and committed.

## What would invalidate the run

- any M0d corpus output before this registration is fixed;
- changing neural/frontend output, beat times, oracle impulses or population
  between arms;
- deriving a transition or observability boundary from an M0d arm;
- using the zero-cost control as a candidate;
- selecting an arm by its mean while ignoring its paired interval or safety;
- scoring stable accuracy on a denominator chosen separately by each arm;
- pooled transition confidence intervals that ignore work clustering;
- exclusions after any arm output, an incompatible resumed checkpoint, dirty
  provenance or a digest mismatch;
- reading `summary` instead of independently recomputing from `records` when
  accepting the final artifact.
