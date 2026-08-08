# Can the room damage be taken back out? — registered 2026-08-08

`PREREGISTERED_room_diagnosis.md` found what a room does: the beats lose half
their height, the floor between them rises fifty to three hundred fold, and on
the three worst captures the loudest thing between two beats is as tall as the
beats. This asks whether any of that can be removed after the fact, and **where**
the removal would have to live.

## The question this decides

An argument has been made that because `LiveTracker` sees only `(time,
activation)` pairs, all room damage necessarily reaches it through the
activation, and therefore the work is in the front end. The first half is true
by construction. **The second half does not follow.** A decoder does not have to
prevent the damage, only to be robust to it, and whether it can be is an
empirical question that the diagnosis did not ask.

So the arms are deliberately split across the two places a repair could live,
and the reading of each outcome is fixed here.

## Corpus

The five aligned captures in `music/room-aligned`, against the corpus files.
Baseline (room, untouched) mean F **0.5323**. Ceiling (clean file) mean F
**0.9216**. Both from `room_recording_phone.json` at commit `aee5465`.

`0875_redbelt` is void and is excluded, as it was there.

## Precondition — replay parity

Before any arm, the **unmodified** room activation is dumped and fed back
through `--live-activation` with its recorded frame-release schedule and its
own timestamps. It must reproduce the baseline beat list exactly.

This is not ceremony. Getting any one of the three -- values at full precision,
release order, timestamps -- merely close made 0 of 20 recordings reproduce the
core on an earlier experiment. If parity fails, every activation-side arm is
measuring the replay and not the repair, and the run is void.

## Arms

Six, all with constants fixed here and **no sweep**. Each constant comes from
the mechanism the diagnosis measured, not from trying values.

**Audio side** — room audio → transform → BeatNet → `LiveTracker`:

* `audio_gate` — per-band spectral subtraction of a noise floor, the floor being
  the 10th percentile of that band's magnitude over the file, over-subtracted
  ×1.5, never below 0.05 of the original. Aimed at the risen floor.
* `audio_dereverb` — per band, subtract an exponentially decaying trace of that
  band's past magnitude, RT60 0.5 s, ×1.0, same 0.05 backstop. Aimed at the tail
  that fills the gaps.
* `audio_both` — gate, then dereverb.

**Activation side** — room audio → BeatNet → transform → `LiveTracker`:

* `act_subtract_floor` — subtract a 2 s running median, clip at zero.
* `act_normalise` — divide by a 2 s running 95th percentile, clip to [0, 1].
* `act_sharpen` — subtract a decayed trace of the activation itself, τ 0.3 s,
  ×1.0, clip at zero. The dereverb idea applied one stage later.

## Reading, decided now

**Primary:** mean beat F over the five, against 0.5323 and 0.9216.

**The trichotomy, which is the answer to "does `LiveTracker` need fixing":**

| what happens | what it means |
|---|---|
| an activation-side arm recovers ≥ half the gap (mean F ≥ 0.727) and no audio-side arm does | the observation model inside `LiveTracker` is where the work is |
| an audio-side arm recovers ≥ half the gap and no activation-side arm does | the front end is, and the structural argument is right for the right reason |
| both recover | either place will do, and the cheaper one wins |
| nothing moves more than ±0.02 | the damage is not removable by cheap post-hoc processing on either side, and the answer is a model that has heard a room |

**Do no harm, and it disqualifies:** any arm that puts `0116_goodies` below
0.95 is disqualified whatever its mean. That recording works in a room today.
A repair that trades away the case that already works is not a repair, and a
mean improved that way is an average hiding a regression.

## What this run cannot do

**No adoption.** Five recordings, one room, one phone, one speaker, and the same
five that the diagnosis was read on. Every constant above was chosen from the
mechanism rather than fitted, but nothing here is held out, so a winning cell is
a **candidate** and its number is not an estimate of anything. Confirming it
needs new recordings, and the protocol for those is `docs/ROOM_PROTOCOL.md`.

This is registered as a probe. If a cell wins and is then swept for better
constants on these same five, the sweep is tuning on the confirmatory set and
its output is not reportable as a result.
