# What does a room do to the activation? — registered 2026-08-08

Five phone captures are aligned against the files they were played from, and
four of the five lost between 0.26 and 0.80 of beat F
(`research/results/room_recording_phone.json`). That is the largest single cost
measured anywhere in this repository, and there is no diagnosis of it. This
registers one.

It is a **descriptive** measurement, not an A/B with an adoption gate. A
threshold missed here means "not this explanation", never "not approved".

Written before any activation has been dumped. The reason for writing it down
is that four candidate explanations are all plausible after the fact, and the
numbers will be read within an hour of choosing what they mean.

## The audio

The five recordings that aligned, and for each one two files that differ only
in having been through a room:

* **clean** — the corpus file, as the published baseline used it;
* **room** — the capture, trimmed at the fitted offset so sample zero is sample
  zero of the clean file. The same trim the scored run used, written out rather
  than thrown away.

Annotations are the corpus's own and are not moved. `0875_redbelt` is void and
is excluded here as it was there.

## The observation

`dump_analysis --dump-activation`, which runs BeatNet in the tracker's own
512-sample blocks and prints the beat and downbeat channels frame by frame at
50 fps. Both arms go through it identically.

## The four measurements

Per recording, per arm.

**1. Discriminability — is the beat visible at all?**

*Salience* is the maximum of the beat channel in ±70 ms around each annotated
beat, which is what `sample_at_beats` already means by the word. *Floor* is
every frame at least 70 ms from any annotated beat. The headline is **AUC**: the
probability that a randomly chosen beat's salience exceeds a randomly chosen
floor frame. It is scale-free, so a capture that is merely quieter does not
register as a capture whose beats are invisible — which a raw peak height would.

**2. Smear and delay — are the peaks where they were?**

Cross-correlation of the two arms' beat channels over ±0.5 s: the lag of the
peak and its height. Alignment already fixed the gross offset, so a lag here is
the model reacting differently to a reverberant onset rather than a timing
error. Smear is reported as the mean width at half height of the activation
around annotated beats.

**3. Doubling — does the tail create beats that are not there?**

For each inter-beat interval, the maximum of the beat channel in its middle
half, divided by the mean of the two flanking beat saliences. A reverb tail that
reads as an onset raises this; a quieter or blurrier activation does not.

**4. The phone rather than the room.**

Block RMS in 1 s blocks on both arms, in dB, and the slope of room against
clean. Automatic gain control compresses: it pushes that slope below 1 while
leaving the acoustics untouched. Reported with the range and the drift over the
file.

## What each outcome would mean, decided now

| observation | reading |
|---|---|
| AUC falls by ≥0.10, or below 0.75 | the front end cannot see onsets through the room; the work is in preprocessing or the model |
| AUC falls by <0.05 **and** stays ≥0.85 while F fell >0.20 | the observation survives and the decoder does not — a different problem from the one assumed |
| mid-interval ratio rises by ≥0.15 | the tail is being read as onsets: narrow, and the most likely to be fixable |
| dB slope ≤0.7 | the phone's gain control, not the acoustics |

More than one may fire. If none does, the diagnosis has failed and that is the
result — it would mean the damage is in something none of these four describe,
and the next step would be to look at the audio rather than at derived numbers.

## The falsifier that matters

`0116_goodies` went **up** in the room, 0.976 to 0.984. Any explanation of the
other four has to put it on the undamaged side of whichever number carries the
explanation. A statistic that condemns all five equally has not explained
anything, however large its effect: it would be describing the room, not the
failure.

This is registered as a *rejection* rule. A candidate that fires on all five is
recorded as not explaining the collapse, and is not rescued by a per-track
threshold chosen afterwards.

## What this cannot answer

Five recordings, one room, one phone, one speaker. Whether these mechanisms are
the ones that matter in a different room is not in this data, and no rate
computed here transfers. The question is *which mechanism*, on the evidence
available, and the answer is about this set.

Nothing here is tuned, so there is no train/test split to keep — but no
threshold above may be moved after the numbers are seen, and any that is
moved must be reported as moved.
