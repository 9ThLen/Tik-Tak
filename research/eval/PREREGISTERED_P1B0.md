# P1-B0 protocol pilot — preregistration

Written 2026-08-16, before any performance for this corpus has been recorded.

`LIVE_MIC_PILOT.md` is the collection protocol: why the corpus must exist, the
independence hierarchy, the three splits, the condition matrix, the annotation
fields, the two alignment mechanisms, the metrics and the rights policy. It says
of itself that it is "a collection protocol and a split policy, not a
pre-registration of a hypothesis".

This document is the missing layer and nothing else. It does not restate the
protocol. It fixes, before any recording exists, **what would make the protocol
acceptable and what would make it unusable** — because the pilot's four
questions as written ("does the capture and alignment procedure survive contact
with a rehearsal") have no pass/fail rule, and a feasibility judgement made
after seeing the data is not a feasibility result. That failure class has been
closed repeatedly on this project in the last week; it should not be reopened at
the point where the work becomes expensive in other people's time.

## Question

Can 20–30 independent performances be captured, aligned and annotated to the
standard `P1-B1` requires, at a cost that fits the available resource?

## What this cannot establish

Nothing about model architecture, parameter count, or whether a causal model of
feasible size reaches Beat This!'s level. Nothing about the product contract.
The pilot produces budgets and feasibility findings, and `plan.md` forbids
training and learning-curve work until its analysis is complete.

It also cannot establish the size of the domain shift as a *product* number. It
measures the shift on its own material to size the full collection; the locked
evaluation that carries a product claim is a later, separately registered run.

## Order: the click micro-check runs first

`plan.md:292` permits a cheap directional micro-check on the five existing
compositions before the full pilot: the click physically played through a
speaker and recaptured, at several levels. Software mixing of a click into an
existing capture does not count, because it reproduces neither room feedback nor
AEC nor the self-confirming loop.

**This runs before any performance is scheduled**, because it is the one gate
that can reject the protocol for a few hours of work rather than for twenty-five
people-days. The click-bleed condition has never been tested anywhere in this
repository: every published number comes from a harness that plays no click, so
every one is an upper bound for a shell with audible output.

Registered outcome of the micro-check — it does not decide the product, it
decides whether the pilot proceeds as designed:

| observation | consequence |
|---|---|
| click bleed leaves alignment recoverable at every level tested | pilot proceeds with click-bleed as one matrix condition |
| alignment recoverable only below some level | pilot proceeds, and that level is recorded as a protocol constraint before recording |
| alignment unrecoverable at product-plausible levels | **pilot does not proceed as designed**; the shell's audible-output assumption is re-registered first |

## Acceptance gates, fixed now

Each is a property of the *protocol*, measured on the pilot, and each is
decidable without judgement.

**A1 — alignment is model-independent and works.** Every capture carries head
and tail slate transients, and alignment is derived from them or from a
reference channel mapped through them. Beat This! agreement may not be used as
an alignment check on this material at any point, because the material must be
unseen by every model under evaluation and an alignment test that requires the
model to recognise the content is not available here.

Gate: alignment succeeds on **≥ 90%** of captures at first attempt, and the
head-to-tail clock-drift estimate agrees between the two slates on every capture
that passes. The two-slate agreement is the parameter-free part and is the one
that decides; a peak that merely looks convincing is not evidence, as the slate
work already established on this repository.

**A2 — variance is measurable and sizes the full collection.** The pilot must
yield a per-condition variance estimate for the primary metric with a usable
interval. Gate: after 20 performances, the estimated number of performances
needed for the full collection is **finite and stated with an interval**. If the
interval's upper bound exceeds the available resource, the label or product
contract is narrowed — `plan.md:187` requires that narrowing to be explicit and
forbids silently reducing QC.

**A3 — annotation cost is measured, not estimated.** Timed, on real recordings,
not projected from the corpus work. The following are fixed as *measurements the
pilot must produce* before `P1-B1` opens, per `plan.md:187`:

- annotator minutes per minute of audio, per label type;
- share of independent double annotation actually achieved;
- the adjudication rule as applied, with the disagreement rate it produced;
- reference-channel multiplier — how many matrix cells one annotation served;
- QC and rework share;
- total person-hours, against a ceiling fixed before recording starts.

The ceiling is a resource decision and is **not** set in this document. It must
be written into this file, with a number, before the first performance is
recorded; a pilot that discovers its own budget afterwards cannot fail A3.

**A4 — independence holds.** No composition in the pilot may appear in any of
the sixteen training sets enumerated in `LIVE_MIC_PILOT.md`, and no commercial
recording that could plausibly enter a future training set. Gate: this is
checked per performance and recorded in the manifest; one violation makes that
performance `research_only` and removes it from any locked material, and does
not void the pilot.

## What would void the pilot

- Ground truth derived from any model under evaluation, in any form — including
  as a starting point for hand correction or as a tie-break.
- Alignment established by model agreement on this material.
- Any training run, learning-curve slice or architecture decision taken before
  the pilot analysis is complete.
- Changing the condition matrix, the annotation fields or the split policy after
  recordings exist, without a dated revision in this file.
- A person-hours ceiling written after recording has begun.
- Reporting feasibility without reporting the click micro-check outcome.

## Operational contract

Recordings, alignment artifacts and annotations live outside the repository.
Every published number carries the digest of what produced it, and — following
what the record-bundle retrofit established on 2026-08-16 — the pilot's
**per-recording records are committed to the repository** so that anyone holding
the repository can recompute its budgets and variance estimates. A digest
pointing at a path on one machine is enough to audit a claim and not enough to
recompute one.

Committed record bundles are covered by `research/results/.gitattributes`
(`*RECORDS_*.json -text`) and every published digest is the SHA-256 of the git
blob, never of the working copy.

## Open before this can be executed

1. **The person-hours ceiling** — a resource decision, not a technical one.
2. **Who performs, where, and on what** — the pilot is calendar-bound and this is
   the long pole; everything on the critical path to `P1-B1` waits on it.
3. **Two annotators** — the protocol requires a shared subset with agreement
   reported, which needs a second person identified before recording starts.
