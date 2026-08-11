# M0b — meter-diverse causal oracle ladder

Status: **pre-run draft requiring independent review**. No M0b corpus output may
be interpreted before this document and the generated manifest are reviewed.
Diagnostic synthetic and `--limit` runs are implementation tests, not results.

## Question

M0a showed that the causal `BarTracker` reaches near-perfect fixed-meter phase
when it receives reference tactus and reference bar lines, but its development
material is almost entirely four. M0b asks the narrower final question:

> On clean, meter-diverse audio, including supported meter changes, does the
> current causal decoder preserve reference tactus/downbeat evidence well enough
> that neural front-end work remains justified?

M0b does not evaluate room robustness, denominator prediction, or a learned
`meter_family` head. The current decoder emits grouping only, so those would
measure absent functionality.

## Immutable system under test

The binary, BeatNet weights, generated canonical manifest, and every canonical
annotation are digested. A binding run requires `tree_clean: true`, complete
audio SHA-256 values in the manifest, both required coverage conditions below,
and no `--limit` or `--skip-audio-verification` flag.

All four arms use the same `tracking::BarTracker` through `dump_analysis`'s
causal bar-replay seam. The batch `--beats/--salience` resolver is not used.
Lookahead remains 50 ms on top of the front-end's existing centered-frame
delay. The run does not tune `BarTracker`.

## Canonical acoustic contract

One annotation row is one accepted **tactus**, with:

- time in seconds;
- 1-based tactus position;
- `tactus_beats_per_bar` grouping;
- `subdivisions_per_tactus` (`2`, `3`, or `unknown`);
- `meter_family` and notated signature as annotation metadata;
- `notation_basis=annotated` when the score supplied the signature;
- a segment identifier and a supported/not-supported flag.

Notated compound meters with numerator divisible by three and denominator 8 or
16 map to dotted-beat tactus: 6/8 → grouping 2, 9/8 → 3, 12/8 → 4, each with
three subdivisions. Other signatures use the numerator as grouping and two
subdivisions. This is an annotation conversion, not a model prediction.

Only groupings `{2,3,4,6}` enter the primary verdict. A recording containing a
complete bar outside that set is kept exploratory in full; its difficult span
is never silently removed from a primary score. Incomplete first/last bars and
bars whose source beat grain cannot be divided exactly into the annotated
tactus are rejected by the builder and counted.

## Data and unit of inference

The builder reads clean audio and annotations from:

- RWC 2.0 beat-position annotations;
- BPSD beat + measure annotations and unfolded MusicXML;
- Rubato beat + measure annotations and MusicXML;
- KRAISLER dry mixes with beat/downbeat and time-signature annotations;
- Candombe as a four-beat control.

All material is development data. No threshold from M0b becomes a locked-test
product threshold. Dataset rights remain research-only where their source
license requires it.

The shipped BeatNet model 1 held GTZAN out and trained on the other four source
collections in the paper: Ballroom, Beatles, Carnatic and Rock Corpus. None of
RWC 2.0, BPSD, Rubato, KRAISLER or Uruguayan Candombe is one of those named
collections. This is a source-level overlap audit, not an audio-fingerprint
proof against an undocumented duplicate. Source: BeatNet paper, Table 1 and
training split description: <https://archives.ismir.net/ismir2021/paper/000033.pdf>.

The inferential unit is the **musical work**, not the recording. Multiple
performances of one BPSD/Rubato work are averaged first. Bootstrap resampling is
over works, preventing heavily represented compositions from dominating the
interval. Corpus and work IDs are retained in every record.

A binding verdict requires:

1. at least two corpora in the primary work set; and
2. at least five independent works containing each supported grouping
   `{2,3,4,6}`.

Failure of either condition is `inconclusive`, regardless of point estimates.

## Arms

| Arm | Tactus grid | Downbeat evidence |
|---|---|---|
| A1 | reference | impulses at reference downbeats |
| A2 | reference | frozen BeatNet downbeat channel |
| A3 | predicted | reference downbeats projected monotonically to unique nearest predicted tactus beats |
| A4 | predicted | frozen BeatNet downbeat channel; exact ordinary causal replay |

A3 projection resolves equal-distance ties to the earlier predicted beat and
never reuses a beat. This is an oracle diagnostic, not a shippable operation.

Every arm is rescored after the latest first non-zero meter decision among the
four arms for that recording. An arm that never answers remains in the
intention-to-treat denominator with zero score.

## Metrics

Primary, at work level:

- bar-line F1 at 70 ms;
- grouping accuracy and balanced grouping accuracy;
- exact tactus-position accuracy;
- coverage, false-confident share, and unnecessary-unknown share;
- share of supported meter changes acquired within two new-meter bars.

Reference tactus and predicted tactus are matched monotonically one-to-one at
70 ms. Unmatched reference events are failures, not dropped rows.

A change is acquired at the first reference tactus from which one complete new
bar has the correct grouping and exact position sequence. If this does not
happen before the next supported change or recording end, the change is not
acquired. Latency in seconds is reported but the registered gate uses bars to
avoid tempo confounding.

Secondary:

- per-grouping event counts and accuracies;
- A1−A4 paired work-level differences;
- per-corpus and per-family descriptive tables;
- exploratory records containing unsupported grouping;
- builder exclusions and rejected-measure reasons.

No pooled event-level confidence interval is allowed. It would treat thousands
of tactus events from one performance as independent evidence.

## Synthetic preflight

Before corpus work, the unchanged C++ replay receives a planted sequence
`3 → 4 → 6 → 2`, 72 tactus beats per segment, at 120 BPM. In the final complete
bar of every segment it must output the planted grouping on every beat and the
correct position on at least 75% of beats. Failure aborts before corpus output.

This proves seam wiring and reacquisition capacity; it is not a corpus gate.

## Decision

All intervals are paired 2,000-draw percentile bootstraps over work-level
values, with deterministic seeds `0..1999` as in M0a/S0.

### Decoder not falsified

All coverage preconditions pass and A1 simultaneously has:

- mean phase F1 ≥ 0.90 and lower 95% bound ≥ 0.85;
- mean balanced grouping accuracy ≥ 0.90 and lower bound ≥ 0.85;
- mean share of changes acquired within two bars ≥ 0.80.

This permits neural front-end work to continue. It does not prove
`meter_family`, denominator, room, or locked-test performance.

### Decoder bottleneck

All coverage preconditions pass and the upper 95% bound of A1 phase F1 or A1
balanced grouping accuracy is below 0.80. Reference evidence is then being lost
inside the current decoder/contract strongly enough that front-end escalation
is not the next metrical step.

### Inconclusive

Every other outcome, including an interval between the two bands, insufficient
grouping coverage, missing corpora, diagnostic flags, or failed input
verification.

The 0.80/0.90 bands deliberately leave a ten-point indifference region. They
are fixed before model output exists; M0a's fixed-meter A1 near 0.98 is context,
not an M0b threshold fitted from these data.

## Registered limitations

- The corpus is mostly classical/RWC and is not a product-domain prevalence
  estimate.
- Score notation supplies simple/compound labels; M0b does not credit them to
  the model.
- The current `BarTracker` holds its previous answer through ambiguity. M0b
  reports false-confident and unknown shares but does not add the deferred
  product state machine.
- Repeated performances reduce acoustic independence; work-level aggregation
  fixes statistical overweighting but does not make the recordings unrelated.
- A positive M0b opens S2/metrical-adapter investigation; it does not accept an
  adapter that has not yet been trained or evaluated.

## What would void the run

- editing this file, the builder, scorer, thresholds, or source manifest after
  seeing corpus model outputs without recording a dated deviation;
- a dirty repository or missing/changed input digest;
- using room or synthetic-room audio in the primary set;
- treating notation denominator as model output;
- resampling recordings instead of works;
- dropping abstentions, unmatched beats, difficult works, or meter changes
  after arm output is inspected.
