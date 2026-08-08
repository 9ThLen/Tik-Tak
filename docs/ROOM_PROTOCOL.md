# Measuring the live path through an actual room

Every live number in `research/results/README.md` was measured on clean decoded
files handed straight to the tracker. The product's first scenario — a released
track playing through a speaker into a phone microphone — has never been
measured at all.

`research/eval/room_degradation.py` measures the acoustic part of that gap in
simulation: reverberation and pink noise, swept, no hardware. It deliberately
does not model a microphone's frequency response, automatic gain control,
clipping, or the tracker hearing its own click. This document is the protocol
for the part that needs a person, a speaker and a microphone.

It is written to be executable by someone who is not the person who wrote it,
and to produce an artifact comparable with the simulated sweep rather than a
separate island of numbers.

## What it answers, and what it does not

**Answers:** how much of the corpus stays usable when the same recordings reach
the tracker through the air instead of through a decoder, on one particular
setup.

**Does not answer:** anything about a different room, a different phone or a
different speaker. One room is one sample. The value is the *gap* against the
clean baseline on the same recordings, not the level.

## What is needed

- A speaker capable of a flat-ish response — a monitor or a decent Bluetooth
  speaker, not a laptop's built-in driver, whose bass rolloff removes the band
  the low-band onset cue depends on.
- A microphone. The phone is the honest choice, because it is the product's
  input; a USB microphone is a cleaner measurement of a worse-matched question.
- A quiet room and a noisy one, if both are available. Two rooms is the smallest
  number that says anything about variance between rooms.
- `desktop/` built, and `tiktak listen` available.

## Procedure

1. **Pick the subset.** Twenty recordings, chosen by stride from the Harmonix
   list so genres are spanned rather than truncated — the same rule
   `room_degradation.py` uses. Write the list down; it has to be the same
   twenty for every condition.

   If a property of the music is used to pick instead — tempo, say — take it
   from the annotations, `60 / median(diff(reference beats))`, and from nothing
   else. The six recordings this repository has were picked for a tempo spread
   reaching 300 BPM using `live_bpm` from a results file, which is the value the
   tracker held at the end of the file. The set that came back is 87 to 172 BPM
   and the fast end was never recorded at all.

2. **Establish the clean baseline first**, by running the ordinary benchmark on
   those twenty files. Everything below is read as a difference from it, so a
   run without it is not interpretable.

3. **Measure the round trip.** `tiktak measure` plays a metronome and records
   its own output, and reports mean latency and jitter. The mean is what
   `--latency-ms` compensates. Record both numbers in the log: a room with 80 ms
   of jitter is not the same experiment as one with 5.

4. **Set the input level** so that the loudest passage peaks around −6 dBFS.
   Clipping is a different degradation with a different character, and mixing
   it in makes neither readable. Note the level; do not change it between
   conditions.

5. **For each recording**, play the file through the speaker and capture with
   `tiktak listen --input <device>`, saving the emitted beat list. One pass per
   recording per condition.

6. **Score against the same annotations** the clean baseline used. The
   annotations are of the recording, not of the playback, so playback must not
   be resampled or time-stretched, and the capture must be trimmed to the
   playback start. A constant offset is fine and should be fitted and reported;
   a drifting one means the playback and capture clocks disagree and the run is
   void.

## Conditions worth separating

Run them in this order, because each is cheap once the setup exists and the
first two are the ones that decide whether the rest matters.

| condition | why |
|---|---|
| speaker → microphone, click off | the acoustic path alone, comparable with the simulation |
| speaker → microphone, click on through the same speaker | the self-confirmation case the gate exists for; the one condition simulation cannot reach |
| second room, click off | between-room variance, without which one room is an anecdote |
| phone microphone against USB microphone | how much of the loss is the capture and how much is the air |

## What to record beside the numbers

The log has to carry enough that a disagreement a year later can be settled:
speaker model, microphone model, approximate room dimensions and surfaces,
measured round-trip latency and jitter, input level, and the commit and weight
hash of the binary that scored it. Without those the run cannot be repeated,
and a run that cannot be repeated cannot be argued with.

## The gate before any of it

If the clean baseline on those twenty recordings does not reproduce the
published figures, stop. Something about the subset, the build or the scoring
differs, and every room number would inherit it.
