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

## The click micro-check is struck, and why

Revised 2026-08-17. The earlier version ordered a click micro-check first,
because click bleed had never been tested anywhere in this repository and every
published number is an upper bound for a shell with audible output.

**The product will not emit an audible click while tracking.** That is
`plan.md:287`, the second of the two options the plan already set out:

> **Не має** → під час чутного кліку стан явно заморожується, позначається
> `stale`, і повторне захоплення не заявляється. Це чесно і дешево.

The first option — adapt the metre through the click — needed a separate
evidence channel, reference-click subtraction or AEC, and had to answer the
self-confirmation argument at `plan.md:280`: the accent is *derived* from the
bar decision, so feeding ungated frames would let an accented click confirm the
very bar line it placed. `bar.hpp:38` records that the tracker loses most of the
beat's evidence when it hears its own click, and that nothing on the output
would look wrong. That option is not taken.

So the micro-check has nothing left to gate. It existed to size a problem the
product no longer claims to solve, and running it would spend fifty minutes of a
two-hour ceiling measuring a condition that will not occur.

**What this narrows, recorded rather than absorbed.** The product does not track
through its own audible metronome. During audible click it freezes, marks
`stale`, and does not claim reacquisition. Any later claim about tracking with
the metronome on is outside what this programme measures or promises.

**What it does not narrow.** A click from an *external* source — a drummer's
in-ear feed leaking, a band's click through a PA — remains ordinary room
interference, and is neither excluded nor addressed here. It carries none of the
self-confirmation problem, because the tracker did not place that click. If it
ever matters it needs its own registration; it is not this pilot's condition and
its absence from the matrix is deliberate.

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
- Reporting any result as evidence about tracking through an audible
  metronome, which this programme no longer measures.
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

## Revision 2026-09-25: the session as it can actually be run

Written before any capture for this corpus exists. The resources are now known,
and the three gaps they leave are recorded as narrowings under `plan.md:187`
rather than absorbed.

**Resources.** One room. The same loudspeaker as room sessions 1–3. Two capture
devices recording every pass simultaneously: the phone that made the existing
room pairs, and a USB dynamic microphone (the t.bone MB7 Beta USB) on the laptop.
No vocalist.

**Three narrowings.**

- **Question 5 is not answered.** Without a vocalist there is no live-vocal
  condition. Whether a singer over external backing costs anything stays
  unmeasured, and the separation stage it was meant to gate stays unopened.
- **The room share of question 2 is not answered.** One room is one sample
  (`docs/ROOM_PROTOCOL.md`), so variance between rooms is not measured. A2 sizes
  the full collection *within this room*, and has to say so wherever it is
  quoted.
- **The device factor is phone against USB microphone,** not phone against
  phone. The USB microphone is not a product input. It shows how much of the
  loss comes from the capture and how much from the air, which is
  `ROOM_PROTOCOL`'s fourth condition. It says nothing about other phones.

**Material, fixed by rule and by content.** Built by
`research/eval/replay_programme.py build`: per GTZAN genre, the manifest rows
marked `ok`, sorted by track id, at positions `round((j + 0.5) * n / 2)` for
`j = 0, 1`. The rule never looks at the music. That gives twenty excerpts:

    blues.00025 blues.00075 classical.00025 classical.00075 country.00025
    country.00075 disco.00025 disco.00075 hiphop.00025 hiphop.00075
    jazz.00024 jazz.00075 metal.00025 metal.00075 pop.00025 pop.00075
    reggae.00024 reggae.00074 rock.00025 rock.00075

Each excerpt is resampled to 48 kHz and wrapped in head and tail slates. The
wrapped excerpts are concatenated with 8 s gaps into one 14.0-minute programme
per level. The build is deterministic, and the same inputs rebuild the same
bytes:

| programme | SHA-256 |
|---|---|
| `programme_normal.wav` | `f584031007fa075c829583c77a2838a0f4ebeb1c68111ae61c05f98ebf0aaa9d` |
| `programme_quiet.wav` | `3127720a052b768669638bc2aaae76e487daecf7cefd6007eb7b3b77cedadb3b` |

One −4.70 dB factor applies to every excerpt and both levels alike. Four sources
are clipped at full scale (metal.00075, blues.00075, pop.00075, pop.00025), and
resampling reconstructs their inter-sample peaks up to 1.53. Scaling per excerpt
would have altered the relative levels between excerpts.

**Condition matrix.** Eight cells: distance × level × device.

- *distance*: near is 1 m. Far is the furthest practical point of the room,
  aiming for at least 3 m, measured and recorded.
- *level*: normal and quiet. Quiet is the same programme 12 dB down **in the
  file**. The speaker volume is set once, on the normal programme, and not
  touched again, so the level difference is exact rather than a dial position.
- *device*: phone and USB microphone, co-located, recording the same pass.

Four passes in the order near/normal, near/quiet, far/normal, far/quiet, so the
microphones move once. That makes 8 captures of 20 takes each, 160 takes in all.

**The signal sent to the speaker.** It is the programme file, identified by the
digest above, and the slates map it onto every capture. No loopback capture is
taken. What happens after the file is recorded in the session log as the
playback chain: player, connection, and any EQ or enhancement.

**What is scored.** The primary tracker is frozen BeatNet `model_1` through
`dump_analysis`. Any other tracker scored on the same aligned takes is
secondary.

**The clean arm is the digital loopback of the same programme.** It is the
programme file read, split and cut by the same code as a capture, at the same
level. This is a measured necessity. Building the scorer showed that the live
tracker is deterministic but not smooth: on `rock.00025` the same music scores
F 0.622 as float, 0.434 at 24 bits and 0.203 at 16 bits, which differ by at
most one quantisation step. It is also not gain-invariant: 12 dB of digital
attenuation alone moves 17 of 20 excerpts. A clean arm that differs from the
capture path by anything but the room writes both into the room's column. With
the loopback as clean arm, the programme scored as its own capture gives a
paired difference of exactly 0.000 in every cell, and that is the check the
scorer passed before this revision.

Two controls are reported beside every result and contain no room:
`digital_path` is the float excerpt against its normal-level loopback, and
`tracker_level` is the quiet loopback against the normal one. On this material,
before any capture, they are −0.005 [−0.035, +0.017] and −0.061
[−0.132, +0.001]. A room effect smaller than the `digital_path` interval is
not distinguishable from re-quantisation.

Per cell, the scorer reports mean F clean, mean F room, the paired room − clean
difference with a percentile-bootstrap 95% interval over excerpts (10,000
draws, seed 20260925), and the room `usable` rate. A1 is computed over all 160
takes against its 90% gate. A pass that falls below 18 of 20 in the in-session
`check` is re-recorded at once, and the re-recording is marked in the log. The
first attempt still counts toward A1.

**A2, made decidable.** The full collection is sized to a ±0.03 half-width on
a cell's mean paired difference: `n = (1.96 · SD / 0.03)²`, using the cell's
observed SD. Its interval comes from the SD's chi-square 95% interval. The
number is reported for the worst cell, not the mean one.

**Session log.** `session.json`, validated by the scorer, which refuses to
score a capture whose `os_processing`, device model, channel count or file is
missing. "unknown" is accepted when it is written down. The log also records
room description and dimensions, speaker, playback chain, speaker volume,
distances, background noise and the person-minutes that A3 needs.

## Revision 2026-09-29: A2 narrowed to ±0.05, after the session

Written after the session was scored. This is the explicit narrowing that A2
requires when a budget does not fit, not a gate moved to make a result pass.

At the registered ±0.03 half-width, the worst cell (far, quiet) needs 511
[296, 1089] excerpts. The per-excerpt SD of the paired difference is 0.27–0.35,
roughly the size of the effect itself. The contract is narrowed to a **±0.05**
half-width on a cell's mean paired difference. The QC gates are unchanged:
A1's two-slate rule, the loopback clean arm and the session-log refusals all
stay as registered.

What the narrowing costs: a ±0.05 interval resolves the room loss itself, 0.32–0.46,
but not the distance and level contrasts between cells, which are about 0.1. A
claim about those needs its own registration and sizing.

The secondary arm is Beat This! (`final0`, `models/beat_this.onnx`), whose
activation goes through `--live-activation` into the same `LiveTracker` and
is scored on the same aligned takes with the same loopback clean arm. GTZAN is
held out from `final0` as well. It is a whole-excerpt, bidirectional
observation replayed causally, so it is a bound on what a better front end
could give in this room, not a causal model's number.
