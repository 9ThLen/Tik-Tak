# Is the room a property of the room or of BeatNet? — registered 2026-08-08

> **Provenance correction, 2026-08-09 (after registration):** official Beat
> This! `final*` checkpoints exclude GTZAN. The registered contamination claim
> below is therefore wrong for GTZAN, although it remains true for the Harmonix
> room pairs. This correction does not retroactively change the registered
> decision rule; it changes how the resulting GTZAN level may be interpreted.

Two questions, one run, because they share every mechanism.

## Question 1 — does a stronger front end survive a room?

`PREREGISTERED_room_diagnosis.md` found that in a room BeatNet's beat channel
loses half its height while the floor between beats rises fifty to three hundred
fold, until on the worst captures the loudest thing between two beats is as tall
as the beats. Whether that is a fact about rooms or a fact about BeatNet has
never been asked.

Same five aligned captures, same five clean files, **the same four statistics**
as the diagnosis — AUC of salience against floor, half-height width,
between-beat ratio, cross-correlation — computed on Beat This!'s beat channel
instead.

**This question is immune to the training contamination below.** It compares one
model against itself on clean and room versions of the same recording. If
`final0` memorised the clean file, that inflates the clean arm and makes the
drop *larger*, not smaller — so a small drop cannot be an artifact of
memorisation, only a large one could be.

**Reading, fixed now:**

| observation | reading |
|---|---|
| Beat This!'s AUC drop is ≥ half BeatNet's on the tracks where BeatNet lost ≥0.10 | the room damages any front end; room training data is required and no model choice avoids it |
| it is < half | the model is a lever for the room, which would be the largest result of this line of work |
| Beat This! drops *more* | the advantage on clean files is bought with something a room removes |

## Question 2 — how much of Beat This!'s advantage is the model?

The causal sweep compared Beat This! decoded by its own `beats_and_downbeats`
against BeatNet decoded by `LiveTracker`, so its +0.102 changed the model and
the decoder together and the +0.07 attributed to the model was struck from
`research/results/README.md`. This runs Beat This!'s activation **through the
same `LiveTracker`**, on the same audio, so the decoder is held constant and
only the observation differs.

**Everything here is an upper bound, twice over, and the number is reportable
only as one.**

* As registered, this premise was wrong for GTZAN: official `final*` checkpoints
  exclude GTZAN. Ballroom, RWC and Harmonix are in the training collection.
  GTZAN is a held-out clean benchmark; the Harmonix room arm is contaminated in
  Beat This!'s favour.
* Beat This! is a transformer over the whole file, and fed through
  `--live-activation` without a recorded release schedule it is observed on an
  analytic availability delay. So this also bounds what a causal version could
  give, from above.

**Reading, fixed now:** if the upper bound on the model's share is **under 0.03
of F**, the observation model is not the lever and no training project is
justified by this evidence. Above that it is not settled, because an upper bound
that is large says nothing on its own — it would need a corpus outside the
training set to become a claim.

## What neither question licenses

No adoption. Nothing here changes what ships. Question 1 is five recordings from
one room; question 2 is bounded above and cannot establish a level.
