# Sparse peaks as a front-end representation: implementation and work plan

## Why

The room costs the shipped front end more than anything else measured: five
Harmonix tracks through a speaker onto a phone go from mean F 0.922 to 0.532,
and `LiveTracker` reads nothing but `(time, activation)` pairs, so all of that
damage necessarily arrives through the observation.

The mechanism is gap-filling. On the worst capture BeatNet's activation floor
read 0.2049 where Beat This! read 0.0133 — the space between beats, which is
what the tracker steers by, stopped being empty.

Recognition systems of the Shazam family survive far worse conditions by
keeping only **local maxima in the time–frequency plane** and discarding
magnitude. A local maximum stays a local maximum however high the floor rises.
This asks whether that property, applied to the network's own input, holds the
gaps open in a room.

## What has already been ruled out

Adaptive whitening — per-band division by a running peak, the cheap relative of
the idea, already present in `dsp::OdfConfig` — was measured on the same five
recordings and **fails**. Room `floor/peak` ratio: 0.3405 with whitening off,
0.3627 at the shipped 0.5, 0.3994 at full. Off beats shipped on all five
recordings; more whitening is worse than less on four of five.

The reason is specific and it is why the negative does not carry over. Room,
off → full: the peak loses 49% of its height and the floor only 37%. A running
peak is an automatic gain control over time, and in a room the reverberant tail
holds that denominator high *between* beats — so the next beat is divided by an
inflated number and compressed harder than the smear it was meant to suppress.

Any per-band level-following denominator fails for that reason. A 2-D local
maximum has no denominator. That is the distinction this plan tests.

## Implementation

### A. Per-band dump — `tools/eval/dump_analysis.cpp`, no core changes

`ml::BeatNetFeatures::process()` hands a callback the complete feature vector
for each frame: 136 log-filterbank bands at 24 per octave from 30 Hz to 17 kHz,
plus 136 positive differences, 272 values, 50 frames a second. That is the
per-band spectrum, and it is **the network's actual input** rather than a
neighbouring signal.

`tools/parity/dump_beatnet.cpp` already dumps exactly this, and its loop is the
one to copy. It is not the tool to use, for one reason: it reads raw `f32`,
while every room number so far came through `dump_analysis`'s decoder. Decoding
the captures a second way would put the experiment on a different signal from
the results it is being compared against.

New flag `--dump-features <path>`. No build guard is needed: `src/ml/beatnet.cpp`
is compiled into `tiktak_core`, not into the optional `tiktak_ml`, and
`core/CMakeLists.txt` notes the asymmetry itself. An earlier draft of this plan
said otherwise and was wrong.

- construct `ml::BeatNetFeatures(rate)`;
- feed the same odd-sized `kBlock` chunks everything else uses, so framing
  invariance is exercised rather than assumed;
- write little-endian `float32`, preceded by a self-describing header: magic,
  version, dtype and endianness, frame count, `136 + 136` with the channel
  order named, and the frame rate.

Both halves are written. They are **two channels, not one frequency axis** —
the filterbank is a spectrum, the difference is its positive rate of change —
and the header says so because the peak map treats them separately (below).

About 7 MB per track per condition. Roughly 30 lines.

Two tests, because a silently wrong dump would be unfalsifiable downstream: a
round-trip test on the file format, and a determinism test on the peak map
built from it.

### B. Peak map — Python, and causal

Local maxima over a `(bands × frames)` neighbourhood, **built separately on
each half**, then merged by a rule fixed in advance rather than chosen after.

**The window trails.** A symmetric `maximum_filter` reads future frames, and
whatever it produces is not an input `LiveTracker` could ever have. Two arms:

| arm | window | status |
|---|---|---|
| `causal` | trailing only | the one that may pass |
| `symmetric` | centred | upper bound, reported, **cannot pass** |

If a fixed look-ahead is used instead of a purely trailing window, it is
declared in milliseconds and added to the latency budget, where it competes
with everything else already spending that budget.

Fixed before the run, because each is a place a result could be manufactured:

- **plateaus** — one deterministic peak per plateau, the earliest frame and
  then the lowest band index, never every equal cell;
- **edges** — how the window is truncated at the start of the signal and at
  band 0 and band 135;
- **ties** — the same deterministic rule as plateaus;
- **novelty horizon** — how far back "no peak recently" looks.

Swept: neighbourhood size in bands and frames, density cap, novelty horizon,
and the merge rule. Shazam's working density is roughly 30 peaks a second over
the whole spectrum while a beat arrives about twice a second, so the useful
density here is likely far above theirs — the sweep must reach up, not only
down.

**The density cap is online too.** Taking the strongest K peaks over a whole
track is a global operation, and a global operation is future context wearing a
different hat — it would quietly restore to the `causal` arm exactly what the
trailing window was there to remove. The cap is a sliding past window, a
refractory rule, or a token budget spent forward in time. Never a top-K over
the file.

### B1. Normalisation, and the trap this experiment is built inside

**No arm may be normalised by statistics of its own whole track.** Dividing a
capture by its own mean, peak, or percentile removes a raised floor by
definition — it would erase precisely the quantity under measurement, and could
as easily manufacture a pass as destroy one. Normalisation is either a fixed
constant shared by every arm and both conditions, or it is causal: computed
from past frames only, and then it is part of the representation being tested
and is swept as such.

This is the third appearance of one pattern in this line of work, and it is
worth naming so it is recognised the fourth time. Adaptive whitening failed
because its denominator followed the reverberant tail. A global density cap
would fail because its threshold follows the whole file. Per-track
normalisation would fail because its divisor follows the floor being measured.
**Any statistic computed over material the live path has not yet heard is
either a leak or a cancellation, and in a room it is usually both.**

### C. Collapsing the map back to a signal

Shazam never needs this: it hashes peak pairs and looks for one consistent time
offset, and can afford almost every hash to die. A tracker needs a continuous
value for every frame. Three rules, all measured, none assumed:

1. **count** — peaks per frame across bands;
2. **weighted** — the same, weighted by peak height, which is the quantity
   Shazam deliberately discards;
3. **novelty** — peaks in bands with no peak inside the horizon, which scores
   the arrival of new structure rather than its presence.

If all three fail, that is a much stronger null than one failing. A single
collapse rule could hide a good representation behind a bad readout.

### D. The `dense` control, defined as a signal

`dense` is the arm that separates "peaks helped" from "the feature set
changed", and it is useless until it is a signal. `count` in particular is
degenerate without a peak mask — every frame has the same number of bands.

Fixed definition: **the mean of the positive-difference half per frame, and
separately the mean of the filterbank half per frame**, under the same temporal
normalisation and the same warm-up as every peak arm. Both are reported; the
difference half is the one the peak arms are compared against, because it is
the half that carries onsets.

Three arms in total:

| arm | features | picking |
|---|---|---|
| `odf` | ODF mel flux | none — the number already measured, for orientation only |
| `dense` | BeatNet filterbank + difference | none, collapsed as above |
| `peaks` | BeatNet filterbank + difference | 2-D local maxima, causal |

`odf` is not a control. It is on a different scale and a different feature set,
and it appears only so the new numbers can be placed beside the old ones.

### E. Scoring

**Primary — paired degradation, not a level.** For each arm and each track:

    degradation(arm, track) = ratio(room, arm, track) - ratio(clean, arm, track)

where `ratio = floor / peak`: the median of the signal's maximum within ±70 ms
of each annotated beat over the median across the middle third of each
inter-beat gap. Five seconds of warm-up.

The paired form is required because `dense` and `peaks` are on different
scales, and a fixed absolute bar taken from the ODF's history would measure the
change of feature set rather than robustness to a room.

*This does not contradict the rule recorded against the AUC-drop statistic.*
There, both arms shared a scale and both sat near the ceiling, so the drop hid
a fifteen-fold difference in level and the level was the right endpoint. Here
the arms have different scales and neither is near a boundary, so the paired
delta is the right endpoint. The distinguishing question is whether the two
arms' levels are comparable at all.

**Second metric — top-N, reusing what exists.** `eval/activation_recall.py`
already takes exactly `len(reference)` strongest peaks, matches one-to-one, and
carries a chance baseline seeded from the file's own name by SHA-256 —
deliberately, because Python salts `hash()` per process and an unseeded
baseline moved in the third digit between runs. Reuse that function rather than
writing a second one. `N` is the count of valid reference beats after warm-up.

**Playback delay.** The measured `speaker → room → phone` delay is applied
identically to every arm and recorded in the artifact.

## Pre-registration, written before any number

**Parameters may not be chosen on the recordings the verdict is read from.**
Selection is leave-one-track-out: each track's room number is computed with
parameters chosen without it. Where a parameter can be settled on the clean
pairs alone, it is settled there. Choosing the best of a sweep on the same five
room captures and calling the winner a result is fitting, not measuring, and it
is the single most likely way this experiment produces a false positive.

**Success**, all four, on the `causal` arm only:

1. `peaks` shows smaller degradation than `dense` on at least four of five
   recordings;
2. mean degradation of `peaks` is at most two thirds of `dense`'s;
3. top-N is not worse than `dense`, and clears its own chance baseline;
4. the parameters producing 1–3 were selected without the track they are
   scored on.

**Null:** anything less, under every one of the three collapse rules and across
the sweep. Recorded as a negative with the sweep printed, not retried with a
fourth rule invented afterwards.

`symmetric` is reported beside `causal` as an upper bound. It cannot satisfy
the criterion however large it is.

### What the artifact must carry, per held-out track

The verdict is a fold-wise result, so the artifact has to make every fold
reconstructable on its own rather than only in aggregate:

    held-out track
    parameters selected on the other four, in full
    dense   clean ratio, room ratio
    peaks   clean ratio, room ratio
    paired degradation, both arms
    top-N and its chance baseline, both arms

A mean with no folds under it cannot be audited, and a fold whose parameters
are not written down cannot be repeated.

**Parameters chosen afterwards on all five are a post-evaluation
configuration.** They may be carried into the next stage — that is what they
are for — but they are not part of the confirmed result and must be labelled so
in the artifact. Nothing selected on all five recordings may be quoted with the
verdict's numbers.

## Work plan

| # | step | cost | output |
|---|---|---|---|
| 0 | commit `--odf-whitening-strength`, re-run the whitening A/B on a clean tree | ~30 min | the negative recorded with provenance |
| 1 | `--dump-features`, format round-trip test, rebuild | ~1.5 h | binary dumps, 5 tracks × clean/room |
| 2 | register the bars above | done, above | this section, unchanged afterwards |
| 3 | peak map with fixed tie/plateau/edge rules, determinism test, three collapse rules, `dense` control, leave-one-track-out sweep | 3–5 h | measured table |
| 4 | verdict recorded either way | ~30 min | result artifact, clean tree |

Roughly one working day. Step 0 exists because today's whitening numbers were
produced from a dirty tree, and the standard this repository already applies to
two other artifacts is that a number from an uncommitted implementation is
provisional.

## What a pass does and does not authorise

**A pass does not decide the front end's input representation.** It measures a
representation, not a model. BeatNet is trained on dense features, so comparing
representations properly means training on each, which is the project rather
than an experiment.

What a pass authorises is narrower and is the whole point: that a sparse
representation has **earned a training stage and a live check**, before any
weights exist and while changing the input is still cheap.

**If it fails** — dense features stay, and the model's room robustness has to
come from training data instead of representation: the re-amping engine, and
the simulator calibrated against measured impulse responses rather than a
guessed direct-to-reverberant ratio.

**Either way** the re-amping and impulse-response work is needed, is not
blocked by this experiment, and should run in parallel.

## Limits that no result here removes

Five recordings, one room, one phone, one playback chain, all five of them
Harmonix songs. The direction is readable; the size is not, and none of it is
an independent test set for anything.
