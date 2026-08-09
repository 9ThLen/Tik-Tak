# Live-mic pilot: what to record, how to annotate, and what it may decide

Written before any recording exists. This is a collection protocol and a split
policy, not a pre-registration of a hypothesis — the experiments this corpus
will serve get their own registrations, and this document exists so that they
*can* be registered honestly.

## Why this corpus has to exist

Three results make it necessary rather than desirable.

**The room is the dominant loss, and it acts entirely on the observation.**
Five Harmonix tracks through a speaker onto a phone: mean F 0.922 → 0.532,
usable 0.80 → 0.20. `LiveTracker` receives nothing but `(time, activation)`
pairs, so every bit of that damage necessarily arrives through the front end.
No decoder-side work addresses it.

**The simulation does not predict it.** The synthetic room sweep's worst cell
costs 0.036 of mean F; the real captures cost 0.390. A factor of ten, and it
does not identify which recordings are damaged either. Augmentation designed
against the simulation is designed against the wrong distribution.

**No existing corpus can settle the teacher question.** BeatNet `model_1` is
trained on Ballroom, Beatles, Carnatic, GTZAN and Rock Corpus. Beat This!
`final0` is trained on sixteen sets including ASAP, Ballroom, Beatles,
Candombe, Filosax, Groove MIDI, GTZAN, GuitarSet, Hainsworth, Harmonix, HJDB,
JAAH, RWC, SIMAC and SMC. Every corpus in this repository — GTZAN, Ballroom,
SMC, RWC, Harmonix — is inside Beat This!'s training set. The measured +0.138
(GTZAN, matched decoder) and +0.212 (room, five tracks) are therefore upper
bounds carrying a train-on-test advantage at full strength, and no further run
on existing audio can remove it.

Other routes exist in principle — an external corpus outside both training
sets, a leave-one-corpus-out Beat This! checkpoint, independent performances of
known works. This corpus is preferred because it removes the confound *and* is
the product domain, and because it is the only one of those we can execute
ourselves.

## The trap this protocol exists to avoid

**Re-recording a corpus song through a microphone does not make it independent.**
A model that memorised the recording can still recognise the musical content
after room degradation. The five room captures already taken are Harmonix
songs, which is why they are a pilot signal and not an evaluation.

Locked evaluation material must contain no composition present in any of the
sixteen sets above, and no commercial recording that could plausibly appear in
a future training set. In practice: performances played for this corpus.

## The independence hierarchy

Split at the highest level that two recordings share:

    composition → performance → session → room → device → capture

Two captures of the same performance differ only in device and placement; they
are one unit. Two performances of the same piece by the same ensemble are still
one unit, because arrangement, tempo habits and room are shared. **The split
unit is the composition**, and the validator refuses any manifest where one
composition appears on both sides of a split.

Sample size is counted in independent performances, never in files. A session
that yields forty captures of one piece is one unit of evidence about pieces and
forty about placement.

## Three splits, fixed before collection

| split | may be used for | size target |
|---|---|---|
| `train` | training, augmentation | the bulk |
| `development` | architecture, size, thresholds, all tuning | ~⅕ |
| `locked` | one confirmatory read, ever | see below |

`locked` is sealed with a SHA-256 manifest before any number is read from it,
and its material is subject to the independence rule above in its strict form.

**Sizing `locked`.** `eval/downbeat_benchmark.py` already records the arithmetic
this repository uses: with zero observed failures, the 95% Wilson bound needs 73
independent test groups for a 5% wrong-rate budget and 35 for a 10% conditional
one. Those are the anchors. The pilot's job is to measure the variance that
turns them into a number for *this* metric; until it does, 73 independent
compositions in `locked` is the working target and 35 the floor.

## Pilot first

The pilot is ~20–30 independent performances, and it exists to answer questions
about the protocol rather than about the product:

1. how large is the clean→room loss, with an interval, on material no model has
   seen;
2. how much of it is room, how much is device, how much is distance;
3. does the capture and alignment procedure survive contact with a rehearsal;
4. what per-condition variance sizes the full collection.

Nothing about model architecture is decided from the pilot.

## Condition matrix

Every capture records its cell. Conditions vary one at a time where possible,
and each performance is captured in at least three cells.

- **room** — small dead (bedroom, carpet), medium live (rehearsal room), highly
  reverberant (hall, stairwell, tiled space). Record RT60 where measurable.
- **distance** — 0.3 m, 1 m, 3 m, plus one deliberately bad placement.
- **device** — at least two phones of different make, one laptop built-in mic,
  one interface with a dynamic mic. Note the model, OS and any processing that
  cannot be disabled.
- **level** — nominal, and one capture near clipping and one near the noise
  floor.
- **buffer and rate** — the values the product will actually use, recorded, not
  assumed.
- **click bleed** — captures with the app's own metronome audible in the room.
  **This condition has never been tested anywhere in this repository**: every
  published number comes from a harness that plays no click, so every one of
  them is an upper bound for a shell with audible output. The click gate exists
  and is unmeasured.
- **ensemble** — full band, drums only, no drums, single instrument, and at
  least a few captures of a band stopping and restarting mid-piece.

## What gets annotated

Per recording: `beat`, `downbeat`, bar phase, meter per segment, and the exact
beat index of every meter change. The existing `.beats` format plus the
versioned `.meter.json` described in the meter plan.

**Ground truth may not come from any model under evaluation, in any form** —
not as a starting point for hand correction, not as a tie-break. Annotation is
by hand against a waveform, or derived from a separately recorded reference
(see below).

Two annotators on a shared subset, with agreement reported. Disagreement above
threshold marks the recording as ambiguous rather than resolving it by fiat.

## Alignment

Alignment must be model-independent, which rules out the trick used for the
corpus audio checks — Beat This! agreement is an alignment test only when the
model has seen the material, and here it must not have.

Two acceptable mechanisms:

- a **slate transient** (clap or electronic pop) at the head and tail of every
  capture, recorded in the room by every device simultaneously;
- a **reference channel** — DI, close mic, or the players' own click — recorded
  in sync with the room capture, annotated once, and mapped onto every capture
  of that performance through the slate.

The reference-channel route is preferred because it makes one annotation serve
every cell of the matrix for that performance, which is most of the cost.

## Metrics

**Primary, absolute, on `locked`:** F, `usable` / `usable_strict` (in whatever
version is current after the `settled_at` redefinition), metrical lock rate,
wrong-level duration, and the episode counts already defined.

**Secondary, diagnostic:** paired clean→room deltas. These are the numbers the
pilot produces and they are not primary anywhere, because when two arms both
start near the ceiling a difference of deltas hides an order-of-magnitude
difference in level — the mistake already recorded against the AUC-drop
statistic.

**Every comparison at a matched latency budget.** Points at 0, 50, 100, 200 and
400 ms of model delay, chosen to match the product rather than a convenient
step. Delay is simulated by withholding delivery while preserving the frame's
own timestamp — frame `t` reaches the tracker when the clock passes `t + L` and
is observed *as* `t`. Shifting the activation simulates delayed audio, which is
a different experiment.

## Rights

Every recording carries its rights status in the manifest. Performances made
for this corpus with a signed release are commercial-safe; anything else is
`research_only` and may not reach a release bundle. The split policy and the
rights policy are independent: a `research_only` recording may sit in `train`
only if the resulting weights are themselves marked `research_only`.

## What this corpus cannot decide

It sizes the domain shift, it removes the train-on-test confound, and it
supplies training material. It does **not** by itself establish that a causal
model of feasible size reaches Beat This!'s level — that needs the causal sweep
through `LiveTracker` and, eventually, a trained candidate. Nothing here
authorises choosing an architecture or a parameter count.
