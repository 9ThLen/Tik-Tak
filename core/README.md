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
| `src/api.cpp` | C API |

## Real-time rules

Everything downstream of `tt_odf_create` runs in an audio callback, so:

- **No allocation after construction.** Every buffer is sized in a constructor.
- **No locks, no I/O, no exceptions** on the processing path.
- **Block-size agnostic.** A device hands over whatever block size it likes;
  `Stft.BlockSizeDoesNotChangeTheResult` and `Odf.BlockSizeDoesNotChangeTheResult`
  pin this down.

## Status

Phase 0 of [`docs/PLAN.md`](../docs/PLAN.md): the ODF front-end, shared by every
analysis mode. Tempo estimation, beat tracking and downbeat detection land in
Phases 2–7.

Note on `whiteningStrength`: it trades the balance between the low/high bands
against invariance to input level, and the two cannot both be maximised — see
`Odf.WhiteningStrengthTradesBandBalanceForLevelInvariance`. The default of 0.5 is
a placeholder to be tuned against `mir_eval` in Phase 2, not a measured optimum.
