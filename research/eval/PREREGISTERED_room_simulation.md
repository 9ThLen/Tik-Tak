# Is a measured-response room simulation usable? — registered 2026-08-08

The invented room in `room_degradation.py` costs 0.005 of F from reverberation
where a real one costs 0.390. A second session recorded a swept sine, sixty
seconds of silence, and two of the same tracks, with nothing moved between them.
This registers whether a simulation built from those measurements may be used to
augment training data.

## What is already known, before this runs

`room_ir.py` on the sweep, at commit `48b8a6d`:

* **The chain is not linear time-invariant.** Three identical sweeps, 13 s
  apart, at levels within 0.6 dB, give responses correlating 0.90 to 0.96.
  Three alternative explanations were tested and refuted: noise in the tail
  (the disagreement is already there at 20 ms, 8 dB below the peak), sub-sample
  misalignment (fractional alignment adds at most 0.012), and no signal in the
  varying bands (measurement SNR is 104 dB at 30–60 Hz).
* **The variation is confined to the bottom two octaves.** 14.2 dB of spread at
  30–60 Hz and 9.8 dB at 60–125 Hz, against ≤2 dB above 125 Hz and ≤0.2 dB
  above 2 kHz.
* **RT60 is 0.33 to 0.37 s** — *shorter* than the 0.4 and 0.8 the invented room
  used.
* Measured SNR of the captures is ≈17 dB; the invented room used 10.
* Harmonic distortion is 0.9% of energy, so the speaker was not overdriven.

So a convolution cannot be right, and the question is whether it is close
enough above 125 Hz to be useful anyway.

## The simulation

Corpus audio ⊛ the measured response, plus the recorded silence scaled to the
capture's measured SNR. Nothing fitted: the response, the noise and the level
all come from the session, and the tracks are the ones captured in it.

## Acceptance, decided now

Validated against the real captures from the **same session** — a different
session is a different room position and would not be a test.

1. **Level.** |simulated F − real F| ≤ 0.05 on every validation track.
2. **Ordering.** The simulation must place the collapsing track below the
   surviving one. A simulation that damages both equally repeats the failure it
   is replacing.

Both must hold. If either fails, the measured-response simulation is **not
approved for augmentation**, and the fallback is real captures only.

`0707_halfwaygone`'s session-2 capture is void on alignment, with candidates at
0.476 s and 0.910 s giving room F 0.340 and 0.204. Its acceptance band is
therefore the interval [0.204, 0.340] widened by 0.05, and a simulation landing
inside it is *not* thereby confirmed — it is only not refuted. `0116_goodies`
at 0.938 is the track that can pass or fail criterion 1 properly.

## What passing would and would not license

Passing licenses **augmentation from this response, for this room**, and nothing
about a different room, phone or speaker. One room is one sample, and a
simulator validated on two tracks from it is a simulator for it.

It does not license reporting simulated numbers as measurements, and it does not
retire `docs/ROOM_PROTOCOL.md`: more real rooms remain the only way to know
whether any of this generalises.
