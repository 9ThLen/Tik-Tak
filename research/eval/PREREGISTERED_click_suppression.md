# Pre-registered: suppressing the metronome's own click (design, not built)

Written before any of it is built. **It is built only if the closed-loop room
test (`PREREGISTERED_closed_loop_room.md`) either keeps the gate or finds that
an ungated metronome keeps itself going.** If the room says the gate can simply
go, none of this is needed.

## The problem the gate solves badly

The gate drops every frame around the metronome's own beat. When the metronome
is right, those frames hold the music's beat, so the tracker loses its
evidence. On the bench that cost 0.16 usable on RWC and 0.37 on GTZAN. The
click itself is narrow:

* each voice is a pure sine with a 60 ms exponential decay: the beat at
  1046.5 Hz, the downbeat at 1568 Hz, the subdivision at 784 Hz;
* its -3 dB bandwidth is about 37 Hz;
* BeatNet reads 136 log-spaced filters at 24 per octave, with FFT bins of
  about 15.6 Hz.

A beat click therefore lands in roughly 2-4 of the 136 filters, for the 5-7
frames the gate already covers. The music's beat, meanwhile, is broadband:
kick, snare and hats.

## Two designs, cheapest first

**A. A band-limited gate.** Where the time gate drops whole frames, this one
touches only the click's filters within the gate window. It replaces them with
their value from just before the window, holding each band flat through the
click and its first reflections, and leaves every other filter as heard. What
it needs is what the gate already needs, the round trip, plus the click's
frequency, which the core knows. It works inside `ml::BeatNetFeatures`, at the
point where the gate currently drops the frame.

* **Known limit.** A room lengthens the click in time but not in frequency, so
  the window may need to be longer than the time gate's 55 ms. The room
  captures will measure how much longer.

**B. An adaptive canceller.** A true echo canceller subtracts the click from
the microphone signal before any feature is computed. The reference is the
rendered click, which is known to the sample. The acoustic path is estimated
by NLMS, frequency-domain for a path of 100-200 ms, delayed by the measured
round trip less a margin.

* **The difficulty.** The music never stops. In echo-canceller terms that is
  permanent double-talk, so adaptation has to be slow, restricted to the
  click's band, or gated by a double-talk detector.

B is the general answer and a research project. A is a day's work, and it only
has to beat a gate that throws away every band.

## How each is tested

Each design is built behind a `LiveConfig` switch that is off by default, and
is tested in three stages. A design moves to the next stage only if it passed
the one before.

1. **Offline, on the room captures.** The closed-loop session records what the
   microphone heard, every click time and the rendered click. That is enough to
   measure suppression without any new session:
   * the click's energy in its own filters at click times, against the
     silent-click pass;
   * the music's energy in every other filter, against the same pass;
   * in the final 30 s of silence, BeatNet's beat activation at click times,
     against the silent pass.

   *Pass:* a click-band reduction of at least 20 dB, other filters within
   1 dB, and a tail activation no higher than the silent pass's upper quartile.
2. **The bench.** The G0-G7 protocol, with the suppressor in place of the gate,
   on RWC and GTZAN, run on three paired seeds, since the effects may be small.

   *Pass:* at both click levels, beat F and usable no worse than the ungated
   arm by more than twice the across-seed SD, and at least as good as the gated
   arm.
3. **The room.** One more closed-loop session, adding `l5_loud_suppressed` and
   `l6_quiet_suppressed` to the five registered arms.
   * *Adopted as the loudspeaker default,* a product change for the project's
     owner, when three conditions hold: at both levels its F is no worse than
     the ungated arm's by more than 0.02, which is the bench's precision and
     not the room's; the lower bound of its difference from the gated arm is
     above zero; and it does not keep itself going, with its gap locked share
     within 0.2 of the silent arm's and its final-tail confidence below 0.25.
   * *Otherwise* the result is reported with its captures needed, and the gate
     stays.

## Not in scope

* Headphone mode, which needs no suppression.
* The spectral-flux front end, which has its own gate path and was never
  measured.
* Downbeat and subdivision clicks, until the beat click works.

## Amendment 1 (2026-10-01, before any room pass): subtraction built first

At the owner's request a third design was built ahead of the two above, and
ahead of the room test: **path-tracked subtraction**, `render::ClickCanceller`.
It sits between A and B.

* Like B, it subtracts the click from the microphone signal, before any front
  end, so it serves BeatNet, spectral flux and Beat This! alike.
* Unlike B, it does not adapt continuously. The path is solved by least squares
  once per click, from correlations averaged across clicks.
* Its answer to permanent double-talk is that averaging, plus a weight on each
  click by how quiet the room was just before it. The music enters each click's
  correlation with a different phase and cancels; the path does not.

The order of candidates is now subtraction as built, then the band-limited
gate (A), then a full adaptive canceller (B).

**Staging.** Stage 1 and the first look at a room are merged into the first
room session: see Amendment 1 of `PREREGISTERED_closed_loop_room.md`, which
carries the decision rule. Stage 2, the bench, follows only if that rule says
proceed.

**What the digital loop showed before the room.** These are mechanics, not
evidence about a room.

| the click's way back | removed in silence | subtracting arms, F | no click at all, F |
|---|---:|---:|---:|
| straight back | 48 dB | 0.68-0.74 | 0.764 |
| a made-up room, 0.4 s reverberation | 32 dB | 0.73-0.77 | 0.764 |

On the same programme the gated arm scores 0.50 and 0.44, and the ungated arm
0.62. The two subtracting arms swap places between the two rows, so their
difference is noise at twenty takes.

**Known limits, each to be read off the room captures.**

* Reverberation later than the 150 ms modelled is left in.
* The first click after a start is not predicted at all.
* A changed path, such as a phone picked up, is forgotten at 5% a click unless
  a pause intervenes.
* The round trip has to exceed one buffer plus 2 ms.
* In silence a tracker hears a click 50 dB down. Subtraction alone therefore
  cannot stop a metronome sustaining itself in an empty room, and the
  empty-room gate is what does.
