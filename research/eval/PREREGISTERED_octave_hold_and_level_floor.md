# Pre-registered: confirming the octave hold, and trying the level floor

Written before any run. Two follow-ups that
`PREREGISTERED_quickfix_diagnostics.md` said would be registered if its
exploratory arms cleared their bars, and both did.

## What the diagnostics found (RWC, development)

**Holding the octave** through the research seam: 20 s took macro usable from
0.164 to 0.192 (+0.028) and episode-free from 0.211 to 0.259 (+0.047), with
correct time -0.001. The noise floor from five seeds of the unchanged tracker
is SD 0.005 for usable and 0.016 for episode-free, and the same code with a
different seed can move episode-free by 0.026 with a paired interval clear of
zero — so single-seed arms are not enough, and this run averages seeds.

**The input level** alone: a digital -24 dB cut cost beat F 0.117 and usable
0.032; -12 dB cost 0.032 and 0.025. Both beyond twice the noise floor.

## What is under test

This commit adds `LiveConfig::anchor_octave_hold_sec` (tracking::OctaveHold,
the research seam's semantics in the core; `--live-anchor-hold`) and
`ml::BeatNetInput::level_floor_dbfs` (lift captures quieter than the floor up
to it, never cut, 5 s symmetric level, +30 dB at most; `--live-level-floor`).
Both off by default. Harness as before: `eval.live_corpus_benchmark --mode
model --per-track`, BeatNet `model_1`, `--live-sample-hz 50`, 8 workers,
outputs outside the tree, one binary from this commit.

## C — the hold, confirmed where it was never tuned

**Harmonix (581 aligned)**, which has not seen any octave-holding arm. Two
configurations on three seeds each (core default, 1, 2): hold off, and
`--live-anchor-hold 20`. Six runs.

Primary: seed-averaged macro `no_wrong_level_episode_fraction`, hold minus off.
Secondary: seed-averaged macro `usable_rate`. Guard:
`mean_correct_share_of_eligible` must not fall by 0.03 or more.

*Decision:* if the seed-averaged episode-free gain exceeds twice the larger of
the two configurations' across-seed SDs, usable does not fall, and the guard
holds, the hold is proposed as the default at 20 s. Otherwise it stays off, and
the result is reported as RWC-only.

## L — the level floor

**RWC**: `--live-level-floor -20` at input gains -24, -12 and 0 dB, and the
floor off at -24 dB (which must reproduce the diagnostics' -24 dB arm: the
code path is unchanged when both options are off, so this checks the build,
not the tracker). **GTZAN (held out from `model_1`)**: floor off and floor -20
at 0 dB, the check that clean released music is not harmed. **SMC**: floor -20
at -24 and 0 dB against the diagnostics' SMC arms.

Endpoints: macro usable and beat F, against the matching unfloored arm.

*Decision:* if at -24 dB the floor recovers at least half of the loss in both
beat F and usable on RWC, and GTZAN at 0 dB loses no more than the noise floor,
the next step is the P1-B0 phone captures, registered separately before they
are scored. If it recovers less, the level is not where BeatNet's room loss
lives and the floor stays off.
