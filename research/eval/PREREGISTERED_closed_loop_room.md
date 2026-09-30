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
