# tiktak-core

Portable C++17 analysis core, shared by the iOS and Android apps.
See [`docs/adr/0001-portable-cpp-core.md`](../docs/adr/0001-portable-cpp-core.md)
for why this is not Swift.

`tiktak_core` has no platform SDK dependencies and no third-party dependencies —
including for the neural network it runs. BeatNet is 0.40 M parameters and
20 MMAC/s, so `src/ml/beatnet` carries the forward pass rather than linking an
inference runtime several times the model's own size; the weights are a blob the
shell hands in, because the core does no I/O. Beat This! keeps its ONNX export,
which is a different job under different constraints — see
[`models/export_beatnet.py`](../models/export_beatnet.py).
It is consumed through the flat C API in
[`include/tiktak/tiktak.h`](include/tiktak/tiktak.h).

`tiktak_decode` is a second, optional library for reading WAV, FLAC and MP3
files, with its own header
[`include/tiktak/tiktak_decode.h`](include/tiktak/tiktak_decode.h). It is
separate precisely so the paragraph above stays true: decoding needs codec
implementations, and a platform that would rather use `AVAssetReader` or
`MediaCodec` links the core alone.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

Options: `TIKTAK_BUILD_TESTS` (on when top-level), `TIKTAK_BUILD_DECODE` (on when
top-level), `TIKTAK_WERROR` (off).

Two things are fetched at configure time: googletest for the tests, and
[dr_libs](https://github.com/mackron/dr_libs) for the decoder, pinned to a
commit because it publishes no tags. Configure with `-DTIKTAK_BUILD_DECODE=OFF`
and `-DTIKTAK_BUILD_TESTS=OFF` to build the analysis core with no downloads at
all.

## What is here

| Path | |
|---|---|
| `src/dsp/fft` | radix-2 FFT behind an interface a SIMD backend can replace |
| `src/dsp/window` | periodic Hann |
| `src/dsp/mel` | triangular mel filterbank, sparse |
| `src/dsp/chroma` | twelve pitch classes from a spectrum — the harmony cue |
| `src/dsp/stft` | streaming STFT, allocation-free, block-size agnostic |
| `src/dsp/dft` | magnitude spectrum at any length, including the ones radix-2 cannot do |
| `src/dsp/logfilt` | logarithmic filterbank, area-normalised — the network's front end |
| `src/ml/beatnet` | BeatNet: a causal beat/downbeat network, weights and forward pass |
| `src/dsp/odf` | onset detection function — full / low / high bands |
| `src/analysis/tempo` | tempo from the ODF: autocorrelation + log-normal prior |
| `src/analysis/tracker` | offline beat tracking by dynamic programming (Ellis 2007) |
| `src/analysis/downbeat` | bar lines and metre from beat-synchronous cues |
| `src/analysis/offline` | whole-file pipeline: audio in blocks → beat grid |
| `src/analysis/grid_cache` | serialised beat grids, keyed by content hash — bytes, not files |
| `src/schedule/scheduler` | beat grid: schedules events ahead, per-channel latency |
| `src/render/click` | the click itself: sample-accurate placement, fixed voice pool |
| `src/render/metronome` | grid + click wired together — one audio callback for every shell |
| `src/render/player` | track playback riding the analysed grid: count-in, bar loops, cues |
| `src/tracking/particle` | online beat tracking: a particle filter over (period, phase) |
| `src/tracking/sync` | manual mode: where the beat sits, when the tempo is known |
| `src/tracking/live` | the microphone path: audio in, beat predictions out |
| `src/render/live_metronome` | tracker + click, and the round-trip arithmetic between them |
| `src/api.cpp` | C API |
| `src/decode/` | WAV / FLAC / MP3 to mono float, over dr_libs — separate library |
| `src/decode_api.cpp` | C API for the decoder |

## Real-time rules

The `dsp/`, `schedule/`, `tracking/` and `render/` components run in an audio callback. The `analysis/`
components deliberately do not — they allocate, size transforms to their input,
and are driven from a file-reading thread. The rules below apply to the former.

Everything downstream of `tt_odf_create` runs in an audio callback, so:

- **No allocation after construction.** Every buffer is sized in a constructor.
- **No locks, no I/O, no exceptions** on the processing path.
- **Block-size agnostic.** A device hands over whatever block size it likes;
  `Stft.BlockSizeDoesNotChangeTheResult` and `Odf.BlockSizeDoesNotChangeTheResult`
  pin this down.

## Status

The ODF front-end, the offline analysis path (tempo, beat tracking, the
`tt_offline_*` API), the beat grid cache, file decoding, the beat scheduler,
the metronome itself (click synthesis plus the callback that drives it) and the
track player (`tt_player_*` — playback on the analysed grid, with count-in and
bar loops) and the online tracker for the microphone path (`tt_live_*`, plus
`render::LiveMetronome`, which is the tracker and the click wired together),
and offline downbeat and metre detection (`tt_offline_beats_per_bar`,
`tt_offline_downbeats`). Missing: the ML models — see below.

The player places the click on the very sample of its beat rather than
compensating for output latency: track and click leave through the same device
buffer, so they arrive together whatever the latency is. Only haptic and
visual cues carry latency arithmetic, compensated against the moment the beat
is *heard*. `downbeat_offset` names which grid beat is "the one"; the offline
analysis now says which one that should be, and the harness feeds it in.

The online tracker is a particle filter over `(period, phase)`, and four
decisions in it are worth knowing before touching the numbers.

*The observation is zero-mean.* An onset moves a particle's weight by
`onset * (window(distance to its beat) - mean window)`, so a frame with no
onset moves no weight at all. Silence therefore costs only diffusion — the
cloud coasts at its last tempo instead of collapsing onto the first noise it
hears — and dropping frames is safe, which is what the microphone path relies
on when it gates out its own click.

*A beat is charged for.* With the reward alone nothing opposes double tempo: a
particle beating twice as fast is right on every real onset and is never
charged for the beats it puts in the gaps. The charge is the point-process half
of the likelihood, scaled by what a beat is currently worth so that silence
stays free.

*Evidence is the square of the onset, in units where a beat is 1.* Nearly all
music has hits between the beats, and with evidence proportional to amplitude a
hi-hat at a third the level buys a third of a beat's belief — which leaves the
beat and its own subdivision within noise of each other, and the tracker
flickering between them. Squaring makes that hi-hat worth a ninth.
`LiveTracker` normalises against a running *peak* rather than a standard
deviation to keep those units true whatever the room's level: a z-score moves
with the material, so dense music would arrive as weaker evidence purely
because there is more of it.

*Confidence is agreement times coincidence.* Resampling makes a cloud agree
with itself within seconds of white noise, so a filter reporting its own
concentration reports near-certainty on material with no beat in it. The share
of onset energy that actually keeps landing on the prediction is the other
half, and both have to hold.

Known and accepted: on material well outside 100-140 BPM the tracker sometimes
settles an octave away — a slow piece tracked on its eighth-notes, a fast one
in half time. That is the tempo prior doing its job, it is the most common
failure of every beat tracker, and it is why the UI has x2 and /2 buttons.

Manual mode (`LiveTracker::setManualTempo`) removes the question the tracker is
worst at. The user's tempo pins the period, and `tracking::PhaseSync` answers
the only thing left.

*The comb correlation is one complex number.* Correlating the onset function
against a comb of impulses at a known period is asking how its energy is
distributed *modulo* that period, and the first Fourier coefficient at that
period is that distribution's centre of mass on the circle. Two decayed
accumulators, no buffer to keep, no grid to search, and a phase that is not
quantised to the frame rate.

*Concentration alone cannot say whether it means anything.* Onsets at random
phases still add up to a resultant of about one over the square root of their
count, so what counts as convincing depends on how many there have been. The
acquire test is therefore the Rayleigh statistic — resultant squared times the
effective number of onsets — and it is what lets manual mode refuse a room whose
beat is not the one asked for, instead of clicking somewhere and calling it
synchronised. A synchroniser that always synchronises has said nothing.

*Correlate to acquire, filter to hold.* The correlation is a mean, so a
syncopated bar drags it; the filter's window is local, so a stray hit merely
lowers the particles that missed it. The phase is handed over once and the
pinned cloud carries it, nudged by at most 2% of a beat at a time — fast enough
for a player drifting, far too slow to chase a tempo that is not the one that
was asked for.

Two of the mode's properties fall out of the filter rather than being written:
with the period pinned, the tempo prior and the per-beat charge can no longer
separate any two hypotheses, so both are dropped; and because the observation is
zero-mean, a room that falls silent moves no weights and the grid simply
continues — which is exactly what a metronome holding the user's tempo should
do.

The grid cache serialises to bytes and leaves storage to the shell, for the
same reason decoding accepts bytes rather than a path — every shell can persist
bytes, while a portable "cache directory" does not exist. The key is the
SHA-256 of the encoded file's bytes, so a renamed file still hits and a
re-encoded one misses; the analysis config is part of the blob's identity, so a
grid computed under a manual tempo hint is never served to an automatic run;
and a truncated or corrupted blob reads as a miss, never as a shorter grid.

`render::Metronome` is the whole audio callback, and it is here rather than in
each shell because it is the same in all of them — the shells differ in how they
get a buffer and a clock, and must not differ in what happens between them.
`Metronome.DoesNotDriftOverAMinute` and
`Metronome.DriftsNoMoreAtATempoThatDoesNotDivideTheSampleRate` are the two that
matter: the second is the one that caught a click rounding into the wrong
buffer, worth a whole sample and invisible at 120 BPM.

The file path works end to end — `DecodeAndAnalyse.FindsTheBeatsOfAnEncodedClickTrack`
takes a real MP3 and produces a beat grid, which is the only test that would
notice two correct pieces being wired together wrongly.

`tools/parity` checks this against the Python reference in `research/` on
identical audio. Both the ODF and the whole offline pipeline are compared; as of
the Phase 3 port every beat time agrees exactly.

Two design points worth knowing before changing anything here:

The scheduler never reads a clock itself — `now_sec` is always passed in. That
is what lets the whole grid, including tempo changes and phase alignment, be
tested exhaustively without waiting in real time.

`whiteningStrength` trades the balance between the low/high bands against
invariance to input level, and the two cannot both be maximised — see
`Odf.WhiteningStrengthTradesBandBalanceForLevelInvariance`. The default of 0.5
is a placeholder, not a measured optimum.

And one that has already caught us out: `comb_harmonics` defaults to 1, meaning
comb scoring over metrical multiples is **off**. It was on, on theoretical
grounds, until it was measured over 140 clips and turned out to be worse on
every metric. Read the comment on `TempoConfig::comb_harmonics` before turning
it back on.

### Bar lines

Three things about the downbeat stage are worth knowing before touching it.

**There is no dynamic programming, and that is a finding.** The plan called for
Markov smoothing over a per-beat downbeat probability, so a single loud beat in
the wrong place could not move the bar line. But smoothing exists only because
the model underneath is allowed to answer per beat, and a metre that does not
change has exactly one degree of freedom beyond itself — the phase. Fix the
metre and the phase and every bar line in the piece is determined, so the whole
hypothesis space is a dozen or so (metre, phase) pairs that can each be scored
exactly. The smoothing a Viterbi pass would approximate is already total. The
price is stated in the header: a piece that *changes* metre partway cannot be
represented, and bringing the DP back is what that would need — for that
reason, not for smoothing.

**Confidence is two numbers, not one — and that was a bug before it was a
design.** The result carries a `phase_margin` (how far ahead of the next best
phase *of the same metre*) and a `meter_margin` (how far ahead of the best other
metre). Only the first existed at first, and it cannot see a metre error at all:
every rival it weighs has already accepted the bar length, so a 4/4 track read as
three can come back with a large phase margin — confidently wrong.
`confident()` requires both, and a caller that accents downbeats should gate on
it rather than on either margin.

Margins are in the salience backend's units. `resolveMeter()` removes a
constant offset but deliberately does not standardise or rescale the incoming
vector: a periodic ripple of a few millionths is still a few millionths of
evidence. Each backend therefore calibrates `min_salience_range`,
`min_phase_margin` and `min_meter_margin` together; numeric ranges measured for
one scorer are not evidence about another.

Those calibrated thresholds also bound the usable dynamic range. Before
scoring, the resolver checks that the peak-to-peak span still leaves eight
floating-point guard bits after accounting for the number of beats. A backend
that mixes `DBL_MAX`-scale values with unit-scale evidence gets no answer rather
than a meter manufactured by rounding. An internal power-of-two shift keeps
ordinary arithmetic finite, but its exponent is restored in every public
margin, so this safety step does not promote weak evidence.
For the same reason, a zero phase threshold is not a usable automatic
configuration; a zero meter threshold is accepted only when the caller has
already reduced the candidates to one eligible meter.

A consequence worth knowing: a metre that divides another is inherently less
separable, because the longer bar fits the shorter pattern exactly. Two-beat
bars score near zero against four and are usually withheld. The measure is being
honest — 4/4 really does fit a 2/4 accent pattern — but it means 2/4 mostly
needs `--beats` to get an accent.

**The score is a contrast, not a sum.** A sum over the chosen beats hands the
win to the shortest bar automatically, because two-four claims half the beats
where four-four claims a quarter. The difference between the mean at the chosen
beats and the mean at the rest does not.

**`accent_weight` defaults to 0, and that one was decided by measurement.** How
hard a beat is struck sounds like a downbeat cue and is very nearly its
opposite: in the ordinary kick-on-one, snare-on-the-backbeat pattern, the snare
is broadband and the kick is not, so the full-band onset function peaks on two
and four while the low band peaks on one. On the reference clip the accent
averages 1.2 on the backbeats against 0.68 on the downbeat. At a weight of 0.5
it won, and the analysis confidently accented beat four. The cue is real in
music where the downbeat genuinely is the loudest event, so it survives as a
parameter — it is simply not a safe default.

A fourth, smaller one: the harmony cue is **not** standardised before being
weighted, unlike the onset cues. A cosine distance between pitch class profiles
already means something absolute; an onset value means nothing until compared
with the rest of the piece. Standardising harmony would take the 0.01-scale
noise of a drums-only track and promote it to an equal vote with the kick.

### What the chroma front end can and cannot see

`ChromaFilterbank` raises the bottom of its range to `resolvedMinHz()` — the
frequency below which a semitone is narrower than a couple of FFT bins and
neighbouring pitch classes share bins. At 48 kHz with the ODF's 2048-sample
window that is about 800 Hz, so the harmony cue is built from the upper partials
of a chord rather than its roots. Measured to be enough:
`Offline.TheHarmonyAloneFindsTheBarWithNoDrumsToHelp` finds the bar line with
both rhythm cues switched off. A dedicated longer window, or a constant-Q
transform, is the real fix and is not here.

Normalising the profile is also a trap worth knowing about, and it is the same
trap as the ODF's whitening floor: divide by a vanishing length and spectral
leakage arrives at full scale, so a bass note below the resolvable range comes
back as a confident chord assembled from nothing. The guard is relative to the
whole spectrum, so it holds at any input gain.

### The ML models are not here yet

Phase 7 in the plan is ONNX Runtime with BeatNet online and Beat This! offline.
What is implemented is the portable half — the cues, the decision, and the
whole path from audio to bar lines to an accented click — which has to exist
whatever computes the per-beat probability, and which now gives any model a
measured baseline to beat rather than an empty slot to fill.
