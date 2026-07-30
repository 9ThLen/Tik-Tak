# Tik-Tak Core

Tik-Tak Core is a portable C++17 library for beat, tempo and metre analysis,
beat scheduling, and sample-accurate metronome rendering.

This public repository intentionally contains only the portable core, its
tests, the core CI workflow, and the notices required to use it. Research
workspaces, corpora, model tooling, internal documentation, and development
hosts are kept outside Git.

## Build and test

Requirements: CMake 3.20 or newer and a C++17 compiler. Ninja is optional.

```sh
cmake -S core -B core/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build core/build
ctest --test-dir core/build --output-on-failure
```

To build only the dependency-free analysis core, without tests or the optional
audio decoder:

```sh
cmake -S core -B core/build-minimal -G Ninja \
  -DTIKTAK_BUILD_TESTS=OFF \
  -DTIKTAK_BUILD_DECODE=OFF
cmake --build core/build-minimal
```

## Public API

The flat C API is exposed by:

- [`core/include/tiktak/tiktak.h`](core/include/tiktak/tiktak.h)
- [`core/include/tiktak/tiktak_decode.h`](core/include/tiktak/tiktak_decode.h)

See [`core/README.md`](core/README.md) for targets, build options, architecture,
and real-time guarantees.

Third-party attribution is recorded in [`NOTICE.md`](NOTICE.md).
