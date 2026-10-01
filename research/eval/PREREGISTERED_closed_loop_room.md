# Pre-registered: the click gate in a real room, closed loop

Written before any room pass.

## Why

On the bench (`PREREGISTERED_click_gate_and_antialias.md` and its loud-ungated
addendum), the gate cost far more than the click it hid. That bench is a
digital loop with no acoustic delay, no echo and no speaker or microphone.
Both its own rule and an independent review make the loudspeaker default wait
for a closed-loop test in a real room, in which each arm plays its own clicks
and hears them through a real speaker and microphone. A replay of recorded
music, P1-B0 included, cannot answer that. The review also asked whether an
ungated metronome keeps itself going once the music stops.

## Setup

* **The pass.** `tiktak loop`, the desktop harness at the commit that adds this
  file, plays the P1-B0 programme through the laptop's speaker and mixes the
  metronome's own click on top. The laptop's microphone feeds the live
  tracker, BeatNet `model_1`. The pass records the microphone, every beat
  handed out, and the estimate at 50 Hz, all on the tracker's clock.
* **The programme.** `music/replay-p1b0/programme_normal.wav`: 20 held-out
  GTZAN excerpts, 14 minutes, with 8 s of silence between them.
* **One device plays both music and click.** They share one acoustic path, so
  the click-to-music ratio set digitally is, give or take the speaker's
  frequency response, what the microphone hears. The real use, with music from
  another source, is a variant (`--external`) and not this run.
* **Round trip.** The round trip `tiktak measure` reports is passed as
  `--latency-ms`.
* **Dry run.** The same five passes run first with `--simulate-ms 40`, the
  digital loop through the same code. They must score before the room passes,
  and they are reported beside them as the no-room control.

## Arms

One pass each, in this order, every pass ending in 30 s of silence:

| arm | click | gate |
|---|---|---|
| `l0_silent` | silent: every beat taken, nothing played | — |
| `l1_loud_gated` | -8 dB against the music | on |
| `l2_loud_ungated` | -8 dB | off |
| `l3_quiet_gated` | -20 dB | on |
| `l4_quiet_ungated` | -20 dB | off |

"-8 dB" is the click's power over one annotated beat against the take's music
power, as the median over the 20 takes. `eval.closed_loop levels` turns that
into gains on the click's nominal level: -4.4 and -16.4 dB for this programme.
On RWC and GTZAN, -8 dB is where the bench's "0 dB" sat, and -20 dB its "-12 dB"
(`click_gate_and_antialias/click_music_ratio.json`).

## Endpoints

The unit is the take. Each is scored as the bench scores a recording:
`live_corpus_benchmark._score_one`, applied to the pass's beats and 50 Hz
series, cut to that take's music.

* **Primary:** F. Usable and episode-free are also reported.
* **Self-sustain,** in the silent gap of about 11 s after each take:
  * the share of the gap spent locked (confidence at least 0.25);
  * the change in confidence from the gap's first 3 s to its last 3 s.

  Coasting decays, and a metronome feeding on its own click holds. The click
  count is reported too, but it does not separate the two: in the dry run a
  silent click coasted through the gaps.
* **The final 30 s:** whether the metronome is still clicking at the end, and
  the mean confidence of the last 5 s.
* **Validity:**
  * the programme's acoustic lag at the first take and at the last, which
    shows any clock drift;
  * the gate's misalignment, lag minus round trip. Above 25 ms the gated arms
    did not gate their own click, and the session is repeated with a
    corrected `--latency-ms`.

Differences are paired over takes, with a 95% percentile bootstrap over takes
(10,000 draws). The tracker runs straight through the programme, as a live
metronome does, so each take starts from whatever state the one before it
left. That is the same for every arm.

## Decision

* **Propose removing the gate by default on a loudspeaker,** a product change
  for the project's owner, when all of these hold:
  * at both levels, the ungated arm's F is no more than 0.05 worse than the
    gated arm's: the lower bound of ungated minus gated is above -0.05;
  * the ungated arms do not keep themselves going. Their gap locked share
    exceeds the silent arm's by no more than 0.2, or that interval reaches
    zero, and the final tail's confidence has fallen below 0.25.
* **Keep the gate** when either of these holds. A click canceller then becomes
  the next registered design.
  * An ungated arm's F is more than 0.05 worse than its gated arm's, with the
    interval excluding zero.
  * An ungated arm keeps itself going: its gap locked share exceeds the silent
    arm's by more than 0.2 with the interval excluding zero, or the final tail
    is still locked (at least 0.25) after 30 s.
* **Otherwise the result is underpowered.** The shipped behaviour stays, and
  the session is repeated with a second pass per arm before anything is
  decided.

Headphone mode is not under test here. It is already decided: no gate, since
there is no click to hear.

## Amendment 1 (2026-10-01, before any room pass): two arms that subtract the click

**Why.** The dry run of the five arms above found that an ungated metronome
never stops once the music does. The project's owner asked for the click to be
subtracted instead of gated, and building that before the session saves a
second session.

**What was built.** `render::ClickCanceller` estimates the path from speaker to
microphone by least squares from the clicks already played, and subtracts the
click that path predicts before the tracker hears it.

* The path is modelled with a tap on every sample around the expected return,
  then a tap every quarter millisecond out to 150 ms, and the same spacing for
  20 ms ahead of it. If the direct sound keeps turning up away from where it
  was expected, the model is moved onto it.
* The estimate is averaged across clicks, each weighed by how quiet the room
  was just before it.
* The clicks go out upright or inverted at random, so that music repeating on
  every beat cannot average in with them.
* The predicted click is scaled by how much of the estimate is path and how
  much is music not yet averaged out, and nothing is predicted from fewer than
  four clicks.
* With `gate_when_alone`, a click and 350 ms of its tail are also gated when
  the room has emptied. "Emptied" means the level just before the click is
  20 dB under what it has usually been there, and nothing like music on the
  beat was left under the last click.

**The round trip, for all seven arms.** Every pass now measures its own, from
six probe clicks played before the programme in the same stream, by the
matched filter `tiktak measure` uses. A figure from a separate run of the
device is out by however its two streams happened to start, and the gate has
only 5 ms to spare ahead of a click. `tiktak measure` is no longer a step of
the session. The validity check on the gated arms stands.

**Arms added,** at the same click level as `l1` and `l2`:

| arm | click | own click |
|---|---|---|
| `l5_loud_subtracted` | -8 dB | subtracted, never gated |
| `l6_loud_guarded` | -8 dB | subtracted, and gated when alone in the room |

That makes seven passes, about 105 minutes. Each subtracting pass also records
what the tracker was handed (`<arm>.clean.wav`).

**Disclosure.** The canceller's settings were chosen in the digital loop, on
this same programme, before any room pass:

* an update of 0.05 over 0.15;
* 150 ms of path over 28 and 80 ms, against a made-up room;
* the form of the empty-room rule;
* the inverted clicks, the trust factor and the move, each added after a unit
  test showed the failure it answers.

The dry run of `l5` and `l6` is therefore a check that what was built works,
and not a test of it. Only the room passes test it.

**Endpoints added.**

* **Removed:** at each click in the silent gaps and in the final 30 s, what the
  microphone heard against what subtraction left, in dB, as the median over
  clicks. In a room it is a lower bound, because the room's own noise is in
  both.
* The same F, usable and self-sustain measures as the other arms, against
  `l0`, `l1` and `l2`.

**Decision for the two new arms.** Nothing here changes a default. It replaces
stage 1 of `PREREGISTERED_click_suppression.md` and its first look at a room.

* **Proceed** to the bench stage and a confirming session when all of these
  hold:
  * the median removed over the gaps is at least 20 dB;
  * `l6` is not more than 0.05 F worse than `l2`: the lower bound of `l6` minus
    `l2` is above -0.05;
  * `l6` does not keep itself going, by the two measures registered above,
    against `l0`.
* **Rework** when removed is under 20 dB. The path model is then not good
  enough for this room, and the captures say what it missed (later
  reverberation, a non-linear speaker, a wrong round trip) before anything
  else is tried.
* **Stop** when removed is at least 20 dB and `l6` still keeps itself going, or
  loses more than 0.05 F to `l2` with the interval excluding zero. Subtraction
  does not solve it, and the band-limited gate or the shipped gate is next.

`l5` is reported beside `l6`: the difference between them is what the
empty-room gate does. The decision for the original five arms is unchanged.

**Dry run.** All seven arms run through `--simulate-ms 40`. The four loud arms
(`l1`, `l2`, `l5`, `l6`) also run with the click returned through
`eval.closed_loop path`, a made-up room. That checks the mechanics against a
path longer than the model, and it is no stand-in for a room.

**What this session cannot show.** One device plays both the music and the
click, so the click's level against the music is the one set digitally. A
phone clicking through its own speaker, a hand's width from its own
microphone, while the music comes from across a room, may hear its click well
above the music. That geometry is the `--external` variant and is a session of
its own.
