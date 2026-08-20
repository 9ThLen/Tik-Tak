# P1-B0 protocol pilot — preregistration

Written 2026-08-16, revised 2026-08-17 before any capture for this corpus
exists. The revision is not cosmetic: the pilot changed shape when a resource it
depended on turned out not to exist.

`LIVE_MIC_PILOT.md` is the collection protocol — why the corpus must exist, the
independence hierarchy, the three splits, the condition matrix, the annotation
fields, the two alignment mechanisms, the metrics and the rights policy. It says
of itself that it is "a collection protocol and a split policy, not a
pre-registration of a hypothesis".

This document is the decision layer: what makes the protocol acceptable, what
makes it unusable, and — added in the revision — what the programme may no
longer claim.

## Two narrowings, registered rather than absorbed

`plan.md:187` requires that when a budget does not fit, the declared contract is
narrowed **explicitly**, and forbids quietly reducing QC instead. Both of these
are that narrowing.

**There will be no live band, now or later.** The protocol wanted 20–30
independent performances. That resource does not exist. Anything about
spatially separated live sources — a drummer and a bass player in different
parts of a room, each with its own directivity — is therefore permanently
unmeasured, and no result from this corpus may be reported as evidence about it.

**There is one annotator and there will not be two.** The protocol requires two
annotators on a shared subset with agreement reported. With one, inter-annotator
agreement is not measured at all — not measured poorly, not on a small subset,
not measured. Any later claim resting on annotation reliability has to say so.

## What the pilot became

A **replay pilot**. GTZAN excerpts played through a loudspeaker into a phone, in
varying rooms, distances, devices and levels, plus a live vocalist singing over
a backing track played from an external device.

Three properties make this the route rather than a compromise:

- **GTZAN is withheld from both models under evaluation.** BeatNet `model_1` and
  Beat This! `final0` both exclude it, so replaying it carries no train-on-test
  confound — the reason the matched `+0.138` is treated as clean.
- **Annotation is inherited, not produced.** GTZAN is annotated; the slate
  transfers that annotation onto every capture. The two hours go to setup and
  supervision, not to marking beats. This also removes the prohibited path
  entirely: ground truth cannot come from a model under evaluation, because it
  comes from no model at all.
- **The pairing is exact.** Clean-to-room consistency needs the same music clean
  and degraded. With live performance, the clean version of a given performance
  does not exist. Here it is the source file.

The phenomenon already survives this chain: five Harmonix tracks through a
speaker onto a phone cost 0.390 of mean F, against 0.036 for the synthetic room
sweep's worst cell.

## What replay cannot answer

A loudspeaker reproducing a mix is one source with one directivity. It is not a
band. Nothing here establishes that a model trained on replayed captures
transfers to live ensembles, and that gap is now permanent rather than deferred.

The live-vocal condition is the one real acoustic source available. It is a
genuine near-field source in a room, and it is not a substitute for an ensemble.

## Registered questions

The pilot answers questions about the protocol and the domain, never about
architecture. `plan.md` forbids training and learning-curve work until its
analysis is complete.

1. How large is the clean-to-room loss on GTZAN, with an interval, per condition
   cell.
2. How much of it is room, how much device, how much distance, how much level.
3. Does capture and alignment survive an unattended replay session.
4. What per-condition variance sizes the full collection.
5. **Does a live vocalist over external backing degrade tracking, and by how
   much** — the same backing captured with and without a vocal, paired.

Question 5 exists to be answered before anything is built on top of it. A
front-end stage separating the vocalist from the music by direction is a
plausible next step, and this repository has already recorded six post-hoc
repairs of room damage, all of which failed; what worked was training on real
captures rather than cleaning the signal afterwards. A separation stage is a
seventh post-hoc repair. If the vocal costs little, the line is unnecessary; if
it costs much, the cheaper first response is to put the condition into the
training data.

With backing on an external device there is no reference signal, so echo
cancellation is unavailable and only blind separation remains — weakest below
roughly 1–2 kHz, which is where kick and bass carry most beat salience.

## The click micro-check runs first

`plan.md:292` permits a cheap directional micro-check before the full pilot: the
click physically played through a speaker and recaptured at several levels.
Software mixing of a click into an existing capture does not count, because it
reproduces neither room feedback nor AEC nor the self-confirming loop.

It runs first because it is the one gate that can reject the protocol for an
hour of work. The click-bleed condition has never been tested anywhere in this
repository: every published number comes from a harness that plays no click, so
every one is an upper bound for a shell with audible output.

| observation | consequence |
|---|---|
| alignment recoverable at every level tested | pilot proceeds with click bleed as a condition |
| recoverable only below some level | pilot proceeds, that level recorded as a constraint before capture |
| unrecoverable at product-plausible levels | pilot does not proceed as designed; the audible-output assumption is re-registered first |

## The capture preflight, which is irreversible

Decided before the session, because it cannot be recovered afterwards.

**Multichannel or mono.** A mono capture destroys direction-of-arrival
information permanently, and with it any future work separating a near-field
voice from a far-field loudspeaker. The preflight records, per device: whether
raw multichannel capture is available, and which OS-side processing — echo
cancellation, noise suppression, automatic gain — could not be disabled. Where
multichannel is available it is used, whether or not that line is pursued.

**The signal sent to the speaker is recorded** alongside every capture.
Without it, no later work on cancellation can be evaluated at all.

Capturing with unknown OS processing is permitted. Capturing without recording
that it was unknown is not.

## Acceptance gates

**A1 — alignment is model-independent and works.** Head and tail slate
transients on every capture. Beat This! agreement may not be used as an
alignment check on this material at any point.

Gate: alignment succeeds on **at least 90%** of captures at first attempt, and
the head-to-tail clock-drift estimate agrees between the two slates on every
capture that passes. Two-slate agreement is the parameter-free part and is what
decides; a peak that merely looks convincing is not evidence.

**A2 — variance sizes the full collection.** Gate: after the session, the
estimated number of captures needed for the full collection is finite and stated
with an interval. If its upper bound exceeds the resource, the contract is
narrowed explicitly rather than QC reduced.

**A3 — the human cost fits the ceiling.** **The ceiling is two hours** of
annotator and operator time, fixed here before any capture exists. Measured and
reported: setup, supervised capture, alignment verification, and rework.
Inherited annotation makes the marking cost zero by construction, which is why
the ceiling can be this small; if verification alone exceeds it, the session is
cut rather than the verification.

**A4 — independence holds.** Every replayed excerpt must be withheld from every
model under evaluation. GTZAN satisfies this for both current models; any other
source requires the check redone and recorded per item.

## What would void the pilot

- Ground truth derived from any model under evaluation, in any form — including
  as a starting point for correction or as a tie-break.
- Alignment established by model agreement on this material.
- A capture whose OS-side processing was not recorded.
- Any training run, learning-curve slice or architecture decision before the
  pilot analysis is complete.
- Changing the condition matrix or the split policy after captures exist,
  without a dated revision here.
- Reporting feasibility without reporting the click micro-check outcome.
- Reporting any result as evidence about live ensembles, or about annotation
  reliability.

## Operational contract

Captures, alignment artifacts and manifests live outside the repository. The
pilot's **per-capture records are committed**, so anyone holding the repository
can recompute its budgets and variance estimates — following what the record
retrofit established on 2026-08-16, a digest pointing at a path on one machine
is enough to audit a claim and not enough to recompute one.

Committed bundles are covered by `research/results/.gitattributes`
(`*RECORDS_*.json -text`), and every published digest is the SHA-256 of the git
blob, never of the working copy.
