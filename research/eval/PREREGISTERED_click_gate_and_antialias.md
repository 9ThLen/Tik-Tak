# Pre-registered: what speaker mode costs, and whether BeatNet's resampler aliases

Written before any run. Exploratory, on development corpora except where
stated; nothing here moves a product default by itself.

## Why

**Every published live number was measured with the tracker deaf to its own
output.** `dump_analysis` never called `gateClick` and never put a click into
the input. A metronome on a loudspeaker does both: LiveMetronome gates a
window around every beat it plays — at 120 BPM with BeatNet's 64 ms frames,
about a quarter of all frames — and the microphone hears the click anyway.
Gated frames also used to reach `ActivationTempo` as zeros at exactly the
tracker's own period; that commit masks them instead. On one GTZAN file the
gate alone took the tracker's confidence from 0.95 to 0.54.

**BeatNet decimates by plain linear interpolation.** From 44.1 or 48 kHz,
content above 11 kHz folds into bands the network reads. The corpora most
decisions were made on (GTZAN, Harmonix-ready) are 22.05 kHz, where it never
decimates; RWC and SMC are 44.1 kHz and every phone capture is 48 kHz.
`LiveConfig::beatnet_antialias` puts a 63-tap low-pass first (unit-tested: a
16 kHz tone's fold falls more than tenfold, the passband moves under 2%, frame
times are identical, and at 22.05 kHz nothing changes).

## Arms

One binary from the commit that adds this file, clean worktree, BeatNet
`model_1`, `eval.live_corpus_benchmark --mode model --per-track`,
`--live-sample-hz 50`, 8 workers. The closed loop mixes the core's own click
renderer into the input at each beat the tracker hands out, at the given gain
on its nominal sound, with a zero round trip.

| arm | flags |
|---|---|
| G0 baseline | — |
| G1 gate only (headphones, gated anyway) | `--live-click-gate` |
| G2 speaker, click -12 dB, gated | `--live-click-db -12 --live-click-gate` |
| G3 speaker, click -12 dB, not gated | `--live-click-db -12` |
| G4 gate only, gaps as zeros | `--live-click-gate --live-anchor-gap-zeros` |
| G5 speaker, gated, gaps as zeros | `--live-click-db -12 --live-click-gate --live-anchor-gap-zeros` |
| G6 speaker, click 0 dB, gated | `--live-click-db 0 --live-click-gate` |
| A1 anti-aliased | `--live-antialias` |

G0-G6 on **RWC (328)** and **GTZAN (1000, out of `model_1`'s training)**.
A1 against G0 on **RWC** and **SMC (217)**, with an SMC G0.

Endpoints: `usable_rate` and `no_wrong_level_episode_fraction` (macro over
corpora with n >= 30), `usable_rate_strict`, `mean_correct_share_of_eligible`
as a guard, beat F reported. Noise floor: the across-seed SD of macro usable
from `PREREGISTERED_quickfix_diagnostics.md` Q1.

## Decisions

* **G1 against G0** prices the gate by itself. If it loses more than the noise
  floor, headphone mode is a product feature, not a nicety, and the shell must
  know which output it is on.
* **G2 against G0** is what a loudspeaker costs as shipped. **G3 against G2**
  says whether the gate is needed at all at this level: if G3 is no worse, the
  gate's cost buys nothing here.
* **G4 against G1, G5 against G2**: gaps masked against gaps as zeros, under
  real gating. If zeros are no worse than the mask, the mask's argument was
  right in principle and immaterial in practice, and that is reported.
* **A1 against G0**: if the anti-aliased front end gains on usable by more than
  the noise floor on both RWC and SMC, turning it on is registered for a
  confirmatory run on held-out material and the P1-B0 phone captures before
  the default moves. If not, it stays off and the question is closed.
