# tiktak-core

Portable C++17 audio-analysis and metronome core shared by platform hosts.

The core has no platform SDK dependency and exposes a flat C API through
[`include/tiktak/tiktak.h`](include/tiktak/tiktak.h). It accepts audio,
configuration, model weights, and time values from its host; it does not own
files, devices, UI, or persistent storage.

## Targets

- `tiktak_core` — DSP, offline and live analysis, scheduling, playback, and
  metronome rendering.
- `tiktak_decode` — optional WAV, FLAC, and MP3 decoding through
  [`include/tiktak/tiktak_decode.h`](include/tiktak/tiktak_decode.h).
- `tiktak_tests` — unit and integration tests, including synthetic audio
  fixtures.

The decoder is separate so hosts can use native media APIs while linking the
analysis core alone.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

Options:

- `TIKTAK_BUILD_TESTS` — build tests; enabled when `core` is configured as the
  top-level project.
- `TIKTAK_BUILD_DECODE` — build the optional decoder; enabled when `core` is
  configured as the top-level project.
- `TIKTAK_WERROR` — treat compiler warnings as errors; disabled by default.

Configuring tests downloads GoogleTest. Building the decoder downloads a
pinned revision of [dr_libs](https://github.com/mackron/dr_libs). A
dependency-free core build is:

```sh
cmake -S . -B build-minimal -G Ninja \
  -DTIKTAK_BUILD_TESTS=OFF \
  -DTIKTAK_BUILD_DECODE=OFF
cmake --build build-minimal
```

## Components

| Path | Responsibility |
|---|---|
| `src/dsp/` | spectral transforms, filterbanks, onset and chroma features |
| `src/analysis/` | offline tempo, beat, downbeat, and metre analysis |
| `src/ml/` | model adapters and portable inference code |
| `src/tracking/` | live beat tracking and manual-tempo phase sync |
| `src/schedule/` | beat-grid event scheduling |
| `src/render/` | click synthesis, metronome, and grid-aware playback |
| `src/decode/` | optional WAV, FLAC, and MP3 decoding |
| `src/api.cpp` | public analysis and rendering C API |
| `src/decode_api.cpp` | public decoder C API |

## Real-time contract

The `dsp`, `tracking`, `schedule`, and `render` processing paths are designed
for an audio callback:

- no allocation after object construction;
- no locks, file I/O, or exceptions on the processing path;
- results do not depend on the host audio block size;
- the host supplies time explicitly, keeping scheduling deterministic and
  testable.

Offline analysis may allocate and is expected to run outside the audio
callback.
