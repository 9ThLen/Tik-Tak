# Pre-registered: is the octave residue real, or is it the instrument?

Written before the script exists. Nothing below was chosen after seeing a number
it is measured against.

## The question

`oracle_usable_rwc.json` and `oracle_usable_harmonix.json` both end the same
way: when the observation is perfect, what still fails is the metrical level and
almost nothing else. On RWC, `wrong_octave` is 90% of the residual failure
reasons; on Harmonix, 21 of 581 recordings fail on the level and on no other
condition.

Both artifacts already record why that may prove nothing:

> The oracle activation is a pulse of *equal height* on every beat, so it
> removes precisely the amplitude difference that tells a level from its double.

A pulse train at period P supports 2P and P/2 equally well. So the residue may
be the tracker's limit, or it may be an artefact of the instrument used to
measure it, and the two have opposite consequences: one says a front end must
carry metrical information, the other says the measurement was blind to it.

This is the last thing worth knowing before an architecture is chosen.

## The measurement

Same corpus, same scorer, same seam as `oracle_usable.py`. Only the synthesised
activation changes.

- **corpus** — RWC, all five sub-corpora, 328 recordings, where the residue is
  largest and therefore most measurable.
- **`flat`** — the existing equal-height bump, unchanged. It must reproduce
  `oracle_usable_rwc.json`'s 0.549 within run-to-run noise, and if it does not
  the run is void rather than interesting.
- **`accent_0.5`, `accent_0.25`** — the same bump with annotated downbeats at
  full height and every other beat scaled to 0.5 and 0.25. The accent is the
  only difference; positions, widths and timing are identical.
- **`accent_0.5_shuffled`** — the control that makes the result mean something.
  The same accent depth, applied to the **wrong** beats: the bar phase is
  rotated by a fixed non-zero offset per recording, so the amplitude pattern is
  present and its metrical alignment is not.

## Primary readout

`usable_rate` per arm, and the share of failures citing `wrong_octave`.

## The registered prediction, so this can fail

**At the better accent depth, `usable_rate` rises by at least 0.05 over `flat`,
and the shuffled control rises by less than half as much.**

Forgiving the level entirely takes RWC from 0.549 to 0.704, so about 15 points
are available if the octave were solved outright. A third of that is the bar.

* **Both conditions met** — the accent carries recoverable metrical
  information, the residue was partly the instrument, and a front end that
  produces an accented beat channel is worth building.
* **Neither met** — the residue is not an amplitude artefact. A perfect
  observation genuinely leaves the level unresolved, and the level needs
  something other than a cleaner beat channel.
* **The true accent rises and the shuffled control rises as much** — the gain is
  from amplitude variation as such and not from metre. That is the outcome this
  control exists to catch, and it would make a naive reading of the first
  condition wrong.

The third outcome is named because the closest precedent in this repository
failed exactly there: the downbeat audit's octave arm came in *behind* its own
shuffled control, and only the control revealed it.

## What no outcome licenses

**Not a claim about a real front end.** These are synthesised activations. A
model that emitted accents this clean does not exist, and the arm measures
headroom, not achievability.

**Not a Harmonix result.** RWC is chosen because the residue is large there.
Harmonix's residue is 3.6% and a five-point bar could not be resolved on it.

**Not an adoption decision.** Nothing here changes a shipped configuration.

## Bound in advance

Arms, corpus, accent depths, the shuffle rule and the primary readout are fixed
here. The shuffle offset is derived from the recording's own name by SHA-256, so
it is fixed before the run and identical on every rerun — the reason
`activation_recall.py` gives for seeding its baseline that way.
