# tiktak-core

Portable C++17 analysis core, shared by the iOS and Android apps.
See [`docs/adr/0001-portable-cpp-core.md`](../docs/adr/0001-portable-cpp-core.md)
for why this is not Swift.

The core has no platform SDK dependencies and no third-party dependencies. It is
consumed through the flat C API in [`include/tiktak/tiktak.h`](include/tiktak/tiktak.h).

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

Options: `TIKTAK_BUILD_TESTS` (on when top-level), `TIKTAK_WERROR` (off).
googletest is fetched at configure time; nothing else is downloaded.

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
| `src/schedule/scheduler` | beat grid: schedules events ahead, per-channel latency |
| `src/api.cpp` | C API |

## Real-time rules

The `dsp/` and `schedule/` components run in an audio callback. The `analysis/`
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
`tt_offline_*` API) and the beat scheduler. Missing: file decoding, the online
tracker for the microphone path, and downbeat detection.

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
