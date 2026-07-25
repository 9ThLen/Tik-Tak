# tiktak-core

Portable C++17 analysis core, shared by the iOS and Android apps.
See [`docs/adr/0001-portable-cpp-core.md`](../docs/adr/0001-portable-cpp-core.md)
for why this is not Swift.

`tiktak_core` has no platform SDK dependencies and no third-party dependencies.
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
| `src/dsp/stft` | streaming STFT, allocation-free, block-size agnostic |
| `src/dsp/odf` | onset detection function — full / low / high bands |
| `src/analysis/tempo` | tempo from the ODF: autocorrelation + log-normal prior |
| `src/analysis/tracker` | offline beat tracking by dynamic programming (Ellis 2007) |
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
`render::LiveMetronome`, which is the tracker and the click wired together).
Missing: downbeat detection.

The player places the click on the very sample of its beat rather than
compensating for output latency: track and click leave through the same device
buffer, so they arrive together whatever the latency is. Only haptic and
visual cues carry latency arithmetic, compensated against the moment the beat
is *heard*. Bars are bookkeeping until Phase 7 — `downbeat_offset` names which
grid beat is "the one", it does not detect it.

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
