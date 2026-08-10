# Which corpora may be trained on, and which must stay able to measure

Decided 2026-08-10, before any weights exist. Recorded here because it is the
one decision in this project that cannot be taken back: a corpus trained on is
a corpus that can no longer evaluate, and no later care undoes it.

## Held out, permanently

**GTZAN and RWC may not enter training, in any form, at any stage.**

They carry almost everything currently known:

* **GTZAN** is BeatNet `model_1`'s withheld fold, which is what makes every
  BeatNet number here an honest one rather than recall of its own training
  material. It is also the corpus the causal teacher gate ran on.
* **RWC** carries the oracle table (`usable` 0.207 → 0.549 under a perfect
  observation), the matched-corpus agility sweep, and the accented-oracle
  control that discharged the octave caveat.

Train on either and all of that becomes train-on-test for the model it was
supposed to judge. The comparison would not merely weaken — there would be
nothing left to compare against.

Harmonix is not formally held out here, but the same caution applies: it carries
the 0.365 → 0.952 oracle result, the largest single figure in the repository.
Anything that trains on it must say so loudly in the same breath as any Harmonix
number it later quotes.

## The independent surface

**`bpsd`, `rubato` and `kraisler`** are the evaluation surface this project did
not have.

Beat This! `final0` is trained on sixteen sets — ASAP, Ballroom, Beatles,
Candombe, Filosax, Groove MIDI, GTZAN, GuitarSet, Hainsworth, Harmonix, HJDB,
JAAH, RWC, SIMAC, SMC, TapCorrect — and **none of these three is among them**.
Until they arrived, every corpus here was inside the teacher's training set, so
the absolute level of any teacher comparison was unquotable and only the shape
survived. On these three it can be quoted.

Two limits travel with them and must not be dropped:

* they are classical and solo repertoire — the hardest domain measured here
  (RWC-Classical scores 0.000 real and 0.033 under a perfect observation) and
  the furthest from a band in a rehearsal room;
* a level quotable against Beat This! is still not a level quotable for the
  product, which is a phone microphone in a room.

## Rights

`bpsd` (Beethoven Piano Sonata Dataset v2) and `kraisler` (KRAISLER) are cleared
for use by permission obtained 2026-08-10.

**`rubato` is in the plan but its permission was not stated in that grant.** It
is used here on the same footing pending confirmation, and any release bundle
must resolve it first. Recorded rather than assumed, because the difference
between `research_only` and commercial-safe decides whether trained weights can
ship at all.

`beat-this-annotations` is MIT and is annotations only.

## Deleted, and why

The seven `beat-this-npz` packs — `beatles`, `groove-midi`, `guitarset`,
`hainsworth`, `hjdb`, `jaah`, `rwc`, 47.48 GB — were removed on 2026-08-10.

They were Beat This!'s own training data in its own precomputed form:
`track.npy` plus pitch-shifted variants per recording, spectrograms rather than
audio. Three things made them wrong for this project rather than merely
redundant:

* **no audio**, so they cannot be played through a room, convolved with the
  measured impulse response, or re-amped — and the room is where the damage was
  measured to arrive;
* **the wrong input representation**, tied to Beat This!'s front end rather than
  the 136-band log filterbank at 50 fps this project reads;
* **the teacher's training set**, so training on them would have shared data
  with the model they were meant to be compared against, and `rwc.npz` alone
  would have burned the RWC results listed above.

They are re-downloadable if a later decision needs them.
