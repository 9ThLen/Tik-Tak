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
| `src/api.cpp` | C API |
| `src/decode/` | WAV / FLAC / MP3 to mono float, over dr_libs — separate library |
| `src/decode_api.cpp` | C API for the decoder |

## Real-time rules

The `dsp/`, `schedule/` and `render/` components run in an audio callback. The `analysis/`
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
`tt_offline_*` API), the beat grid cache, file decoding, the beat scheduler and
the metronome itself (click synthesis plus the callback that drives it).
Missing: the online tracker for the microphone path, and downbeat detection.

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
