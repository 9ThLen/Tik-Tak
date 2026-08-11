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

### Registered A1 format-sensitivity controls

M0a's amplitude and +/-20 ms sensitivity arms were mathematically inert for
this decoder: `BarTracker` takes the maximum downbeat evidence within +/-70 ms,
so neither arm was capable of changing the selected evidence. M0b does not
reinterpret or repair M0a. It adds two diagnostic replays before any successful
M0b corpus result exists:

- **Profiled oracle.** Select the highest frame in the frozen BeatNet downbeat
  channel of that recording, resolving an exact tie to the earliest frame.
  Copy the channel centred on that frame from one median reference-tactus
  interval before it through one interval after it, rounding the half-width up
  to a whole activation frame.
  Centre the same copied profile on every reference downbeat, mapping an exact
  nearest-frame tie to the earlier frame. Out-of-recording
  source samples are zero-padded; overlapping copies combine by maximum, never
  by addition. This preserves a real, recording-specific amplitude and local
  temporal shape instead of substituting a unit impulse.
- **Positive control: shifted one tactus.** Use the identical copied profile,
  but centre each copy on the reference tactus immediately after its downbeat.
  The reference grid and scoring span remain unchanged. This deliberately
  supplies a one-position-wrong phase and must show that the sensitivity path
  is capable of moving the result.

Both controls use the A1 reference grid, reference publication clock, unchanged
decoder, and the primary arms' common start. Their own first-decision times are
recorded so a late control acquisition cannot be mistaken for a format effect.
They are diagnostics, not fifth and sixth primary arms, and run only for
primary-eligible records. At work level, paired phase-F1 differences are
bootstrapped and reported. The two gates deliberately use different constants
because one is an equivalence check and the other is a power check:

1. if `abs(profiled oracle - A1) > 0.05`, the hard impulse has a material format
   advantage or disadvantage and the binding verdict is withheld;
2. if the shifted-one-tactus control has mean phase F1 **above 0.30**, the
   deliberately wrong phase did not produce a sufficiently absolute failure to
   make a near-zero format difference interpretable, and the verdict is
   withheld. `profiled oracle - shifted one tactus` remains reported but is not
   the power gate.

Either failure forces `inconclusive` regardless of the primary A1 thresholds.
Passing closes only the shape-and-amplitude concern. It does **not** reproduce
the model's unaligned false competing peaks across the whole recording; those
remain an explicit limitation rather than being declared tested.

The A1 impulse, profiled oracle, and shifted control additionally report
matched-tactus bar-line F1 by reference grouping `{2,3,4,6}`. This breakdown is
diagnostic rather than another gate. In particular, it makes visible whether
the two-interval-wide copied profile saturates the channel specifically when
grouping 2 places successive downbeats two tactus intervals apart.

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

A change is acquired at the first **position-1 reference tactus** from which one
complete new bar has the correct grouping and exact position sequence. A
mid-bar window such as `3,4,1,2` is not acquisition. If a complete bar does not
start correctly before the next supported change or recording end, the change
is not acquired. Latency in seconds is reported but the registered gate uses
bars to avoid tempo confounding.

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

Failure of either registered A1 format-sensitivity rule is also inconclusive.

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
- The profiled-oracle control does not retain unaligned false competing model
  evidence outside its copied window. It tests oracle shape and amplitude, not
  the complete error distribution of the frozen model channel.

## Sensitivity-control revision before a successful execution — 2026-08-11

Independent review of the completed M0a artifact established that both of its
registered format perturbations changed A1 by exactly zero on every scored
recording because neither could alter the maximum selected within +/-70 ms.
No M0b record, arm metric, aggregate, interval, or verdict has been persisted
or inspected: the aborted execution described below predated checkpointing and
wrote only completion counts. The profiled-oracle and shifted positive controls
above are therefore fixed before a successful M0b result exists. This revision
adds diagnostics and a withholding condition; it does not alter A1-A4, corpus
selection, primary metrics, bootstrap, or the decoder thresholds.

Independent review before execution then separated the controls' two unlike
questions. The 0.05 equivalence margin remains attached only to profiled A1
versus impulse A1. The shifted positive control instead receives the absolute
phase-F1 ceiling 0.30. The same review required the grouping breakdown and made
the phrase "one complete new bar" executable by requiring acquisition to start
at reference position 1. No corpus output existed when these rules were fixed.

## Operational revision after the aborted first execution — 2026-08-11

The first corpus execution reached 950/1005 completed futures and was then
terminated by its orchestration shell's 7,200-second timeout. The old runner
wrote only at the end, so it produced neither a result artifact nor a partial
record file. No arm metric, outcome record, aggregate, confidence interval, or
verdict from that process was persisted or inspected. Only completion counts
and elapsed times were printed. This revision therefore changes failure
recovery only; it does not change an arm, corpus, score, threshold, exclusion,
bootstrap, or decision rule.

Every subsequent execution uses a checkpoint directory outside the repository:

- immutable `header.json` fixes the commit, binary/model/manifest digests,
  selected-record digest, flags, primary arm list, sensitivity controls and
  thresholds, and record order;
- each completed record is written to its own temporary JSON, flushed with
  `fsync`, and atomically renamed into `outcomes/`;
- `state.json` is an atomic operational status file and is not a result;
- `--resume` fails closed unless the current clean checkout and every run
  identity field exactly match the checkpoint;
- a fresh run refuses an existing checkpoint, and every run refuses an existing
  final output, preventing accidental overwrite;
- the scheduler keeps at most `--workers` futures active. When `--pause-file`
  appears, it submits no more work, drains and checkpoints the active futures,
  marks the checkpoint `paused`, and exits with code 75;
- output, checkpoint, and pause paths inside the repository are rejected, so
  operational writes cannot invalidate `tree_clean` after provenance capture;
- after an abrupt process or host failure, resume recomputes at most the active
  uncheckpointed futures. Already checkpointed outcomes are not rerun;
- worker count is operational rather than part of run identity. It may change
  on resume and is recorded in each `state.sessions` entry; record values and
  deterministic final ordering do not depend on it;
- the final artifact is itself written atomically only after all selected
  outcomes exist. A complete checkpoint can reconstruct it without model
  inference if final aggregation is interrupted.

Checkpoint outcomes are intermediate cache entries, not individually
interpretable experimental results. The binding result remains the one complete
final artifact produced from all selected records. This operational revision
requires independent review and a new clean commit before another binding run.

### Detached Windows run and pause/resume procedure

The binding invocation must not inherit a short-lived shell timeout. From a
clean eval worktree, set `PYTHONPATH=research` and launch the same command with
PowerShell `Start-Process -WindowStyle Hidden -PassThru`, redirecting stdout and
stderr to files outside the repository. The argument list must include:

```text
python -m eval.m0b_oracle
  --manifest <immutable-manifest>
  --music-root <music-root>
  --binary <dump_analysis.exe>
  --model <beatnet_model_1.ttw>
  --output <m0b-final.json>
  --checkpoint <m0b-checkpoint-directory>
  --pause-file <m0b.pause>
  --workers 8
```

Store the returned PID beside the redirected logs. To request a graceful pause,
create `<m0b.pause>` and wait until `state.json` says `paused` and the process
exits with code 75. Then remove only the pause file and launch the identical
argument list with `--resume`. If the process is killed without a graceful
pause, launch that same resume command after confirming no old worker remains.
Never use `--skip-audio-verification` to recover a binding run.

The two sensitivity replays raise the primary-record replay count from four to
six; exploratory records remain at four because their controls do not enter any
registered summary. Budget up to roughly 50% additional replay time relative
to the aborted four-arm execution, although model inference itself is still
performed only once per record.

## What would void the run

- editing this file, the builder, scorer, thresholds, or source manifest after
  seeing corpus model outputs without recording a dated deviation;
- a dirty repository or missing/changed input digest;
- using room or synthetic-room audio in the primary set;
- treating notation denominator as model output;
- resampling recordings instead of works;
- dropping abstentions, unmatched beats, difficult works, or meter changes
  after arm output is inspected.
