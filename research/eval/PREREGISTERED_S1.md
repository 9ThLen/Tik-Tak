# S1 — preregistered stateful block-training ablation

Status: **pre-run registration**. No comparative S1 training output exists at
the time of this revision. Smoke overfit on synthetic or at most two real
records is apparatus validation and cannot enter the S1 verdict.

Registration date: 2026-08-12. The ambiguity audit scores goal 0.95, boundary
0.92, constraints 0.86 and acceptance 0.91, for weighted ambiguity **0.09**.
All dimensions exceed the 0.20 readiness gate inherited from the specification
workflow.

## Question and causal claim

S0 showed that carrying the frozen BeatNet recurrent state raises bar-phase F1
substantially. M0e then showed that lowering decoder hysteresis does not transfer
through the current non-oracle frontend and increases churn. S1 asks the next
narrow question:

> With model parameters, data, labels, block boundaries, loss support,
> optimiser, update count and seeds held fixed, does carrying BeatNet state
> between contiguous TBPTT blocks improve product-level bar phase over resetting
> it at every block?

S1 tests a training procedure, not a new architecture. Both arms remain the
published 3-class BeatNet and must export to unchanged `TTBN v1`.

## Fixed sources and claim boundary

- Source checkpoint: `beatnet_model_1_weights.pt`, SHA-256
  `619091bc317ca3e83b45591d46f6de3d5a41588bcb39fe9fe7be30cffa6aca84`.
- Runtime source model: `beatnet_model_1.ttw`, SHA-256
  `812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1`.
- Upstream BeatNet source/checkpoint revision:
  `81cedd4beeb7235262db80969a0c9ce9a48a0ed4`.
- Canonical data manifest: M0b v2, SHA-256
  `484efd0d699aef2c40b1a1ba4ac651a2baaa388b8f188b1574a1af99671d88fd`.
- Successful population source: M0e artifact, SHA-256
  `b866228e9c115c2c48c43acde1eac5e745bc99f8c8a70482d7d66d2d5502d278`:
  980 records, 414 work IDs, no technical exclusions.

The corpus has already been used for development diagnostics. S1 is therefore
a **development ablation**, not locked evidence or a product-generalisation
claim. RWC 2.0 is CC BY-NC 4.0; every S1 cache, checkpoint and exported model is
`research_only` and cannot ship or be redistributed as product weights without
a separate rights decision.

## Population and deterministic split

Only the 980 records present in the fixed M0e artifact are eligible. Records
are grouped by `(corpus, work_id)`; every performance of one work stays in one
split. For each corpus independently, works are sorted by

```text
SHA256("tiktak-s1-v1\0" + corpus + "\0" + work_id), then work_id
```

and the first `ceil(0.20 * corpus_work_count)` works become development. The
rest become training. This fixes 84 development works and 330 training works:

| corpus | all works | dev | train |
|---|---:|---:|---:|
| BPSD | 31 | 7 | 24 |
| Candombe | 35 | 7 | 28 |
| KRAISLER | 20 | 4 | 16 |
| Rubato | 14 | 3 | 11 |
| RWC2 | 314 | 63 | 251 |

The generated split artifact records every work and recording plus source
hashes. Any overlap, missing M0e record, duplicate identity, digest mismatch or
corpus count mismatch fails closed before feature extraction.

## Fixed model arm

The primary base arm is **A3**, the smallest arm that can teach recurrent state:

- convolution, `linear0` and LSTM layer 0 are frozen;
- LSTM layer 1 and the existing 3-class output layer are trainable;
- class order is `beat`, `downbeat`, `null`;
- no new head, positional encoding, attention, meter loss or decoder parameter
  is added.

A2 is not primary because its LSTM is frozen and therefore cannot answer
whether training teaches state to carry structure. A4 is not run in S1 unless
A3 is later rejected and a separately registered escalation authorises it.

## Arms and isolation contract

| arm | recurrent state at a 400-frame block boundary |
|---|---|
| `A3_reset` | reset both LSTM layers to zero |
| `A3_stateful` | carry both layers from the immediately preceding block of the same recording, then `detach` |

Both arms consume exactly the same 400-frame (8 s) contiguous blocks in the
same order. A recording is never shuffled internally. State resets at recording
start/end and when a batch slot receives another recording; it never crosses a
work, performance or batch slot. Annotation segment boundaries, pauses and
meter changes do not reset state because the runtime has no equivalent causal
reset detector.

The first 100 frames (2 s) of **every block in both arms** are excluded from
loss. Sharing this mask holds supervised support fixed and prevents a warm-up
difference from masquerading as state carry. Incoming state in the stateful arm
is detached at every block boundary, so gradients span at most 400 frames while
the numerical state spans the recording.

## Features and labels

The existing local BeatNet frontend is reused: 22,050 Hz mono, centred 1,411
sample frames, 441 sample hop, 136 unit-area log-frequency bands plus positive
first difference, 272 float32 features at 50 fps. Feature-cache entries bind
audio, annotation, frontend-code and configuration digests.

Canonical tactus events are snapped to the nearest 50-fps frame with an
earlier-frame tie. Position 1 is `downbeat`; other positions are `beat`; all
other labelled-span frames are `null`. A downbeat cannot be overwritten by a
regular beat. Loss is enabled only inside canonical annotation segments and is
disabled outside them, in missing/rejected gaps, in block padding and in the
shared 100-frame block warm-up. Missing labels never become null examples.

## Optimisation and pairing

- seeds: `17`, `29`, `43`;
- batch slots: 8;
- optimiser: Adam, learning rate `5e-4`, default betas and epsilon;
- class weights `[50, 400, 5]` for beat/downbeat/null;
- gradient norm clip: `5.0`;
- maximum 50 epochs;
- product validation every 5 epochs and at the final epoch;
- patience: four product-validation points;
- checkpoint/resume only at epoch boundaries;
- deterministic algorithms and a configuration snapshot are mandatory.

Within one seed, both arms start from byte-identical source weights and use the
same split, recording order, block order, masks and maximum update budget.
Checkpoint selection is lexicographic on development metrics: reject any point
whose beat-F loss exceeds 0.01 from that seed's frozen A0; among the remainder,
maximise bar-phase F1, then downbeat F1, then beat F1. If no point clears beat
non-inferiority, the arm has no eligible checkpoint.

## Registered endpoints

All comparative values are paired by independent development work. Multiple
performances of a work are combined before comparison. For each work, first
average the three seed-specific arm differences; report every seed separately
as a stability diagnostic. Confidence intervals use 2,000 deterministic
work-level bootstrap draws with seeds `0..1999`.

S1 efficacy passes only if all hold:

1. mean paired bar-phase-F1 gain `A3_stateful - A3_reset` is at least **+0.03**;
2. its 95% bootstrap lower bound is greater than zero;
3. no seed's work-mean bar-phase difference is below **-0.01**.

Safety additionally requires:

- beat-F difference lower bound at least **-0.01**;
- downbeat-F difference lower bound at least **-0.01**;
- stable exact-position difference lower bound at least **-0.03**;
- false-switch-rate difference upper bound at most **+1.0 per five minutes**;
- >=1-bar wrong-episode-rate difference upper bound at most
  **+0.25 per five minutes**.

Always report beat/downbeat precision and recall, `usable_strict`, phase F1,
position accuracy, balanced grouping accuracy, coverage, false-confident and
unnecessary-unknown shares, acquisition latency, all wrong episodes, state
changes, per-corpus results, seed spread, train/dev losses and throughput.
These diagnostics cannot replace a failed primary or safety gate.

## Interpretation

After source, split, parity, completeness and determinism checks pass:

- efficacy and every safety gate pass -> `stateful_training_positive`;
- efficacy passes but any safety gate fails -> `stateful_gain_with_regression`;
- efficacy fails and all safety gates pass -> `stateful_training_no_material_gain`;
- efficacy and any safety gate fail -> `stateful_training_negative`.

Any missing seed/work/arm, changed split, failed source/export parity,
non-equivalent resume, non-finite loss/gradient, technical exclusion or
incomplete product evaluation forces `inconclusive`.

## Apparatus gates before comparative output

Before the first binding S1 training run, all must pass:

1. source PyTorch probabilities match the frozen local implementation within
   `2e-6` on a deterministic multi-block fixture;
2. source weights export to byte-identical frozen `TTBN v1`;
3. tiny-set overfit lowers masked loss by at least 50% for both arms;
4. a two-epoch uninterrupted run is tensor-identical to one epoch plus resume;
5. tests prove state carry, boundary detach, no cross-recording/slot leakage,
   equal loss masks and equal block order;
6. a trained synthetic checkpoint exports and loads in C++, with frame
   probabilities within `2e-5`;
7. a 1–5% non-binding throughput probe reports GPU memory, frames/s, cache size
   and projected six-run duration without entering any quality verdict.

## Boundaries

In scope: A3 reset/stateful infrastructure, deterministic split, feature cache,
training/checkpoint/resume, product-metric validation, unchanged TTBN export and
all apparatus tests above.

Out of scope: A4, room A5–A7, S2/metre heads, decoder tuning, click/AEC,
product-default changes, locked testing, mobile benchmarking and redistribution
of trained weights. Each requires a later gate or data-rights decision.

## Operational contract

Binding outputs, checkpoints and caches live outside the repository. The run
starts from a clean commit and records code, source model, data, split, config,
cache and binary SHA-256 values. Worker count and cache location are operational;
arm definitions, masks, block size, seeds, split and thresholds are identity.
Resume fails closed on any identity mismatch.

Any correction before the first comparative A3 training output is a dated
pre-run revision. After either arm emits a real-corpus trained checkpoint, a
changed rule is a deviation and cannot silently replace this registration.

## 2026-08-12 pre-run apparatus revision: product binary

The frozen A0 development baseline exposed that a newer local
`dump_analysis.exe` can provide the published bar fields while omitting the
internal `*_all` traces required by the fixed M0e scorer. It spent the complete
development pass before the missing field surfaced. No A3 checkpoint or
comparative training output existed.

All S1 product evaluations therefore require the exact M0e binary, SHA-256
`49c47437423f0d79c2f30dde3bcba506f1075099b9f3a7c780efcffe2eed647d`, and
verify that digest before submitting corpus work. This does not change a model
arm, endpoint, threshold or population; it fixes the evaluator implementation
to the one that produced the registered M0e baseline and makes a known
apparatus incompatibility fail before a long run.
