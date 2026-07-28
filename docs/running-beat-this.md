# Running Beat This! locally

The offline path can now be driven by Beat This! instead of the dynamic
programming tracker. This is how to build and run it, on Windows or anywhere
else.

## What you need

* **CMake ≥ 3.20** and a C++17 compiler. On Windows, Visual Studio 2022 with
  the *Desktop development with C++* workload is enough.
* **Network access on first configure.** ONNX Runtime 1.22.0 is downloaded from
  the project's GitHub release and checked against a pinned SHA-256; the
  archives for Windows, macOS and Linux are all pinned. Nothing else is fetched.
* **The model.** `models/beat_this.onnx` is not in git — a large ONNX in
  history is forever. Fetch it and check it in one step:

  ```
  python models/fetch.py pin beat_this_cpp_onnx <path-or-url>
  python models/fetch.py verify
  ```

  `verify` is the part that matters: it holds the file to the checksum the
  repository already vouches for, so a download that quietly returned an error
  page fails here rather than three measurements later.

## Build

```
cmake -S tools/eval -B tools/eval/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTIKTAK_BUILD_ML=ON
cmake --build tools/eval/build --config RelWithDebInfo
```

`-DTIKTAK_BUILD_ML=ON` is what pulls in ONNX Runtime. Without it everything
still builds and `--beat-this` is simply absent — the inference library is a
separate target precisely so the core keeps building with no third-party
dependencies at all.

On Windows the runtime's DLLs are copied next to the executable as a
post-build step, so the tool runs from its build directory without anything
being put on `PATH`.

## Run

```
tools/eval/build/dump_analysis <audio-file> --beat-this models/beat_this.onnx
```

Output is JSON on stdout: `beats`, `downbeats`, the tempo, the metre and the
per-beat cues. `"beat_source"` says which path produced the grid —
`beat_this`, `tracker` for the built-in one, or `file` when beats were supplied
with `--beats`.

Drop the flag to get the built-in tracker over the same file, which is the
comparison worth making:

```
tools/eval/build/dump_analysis <audio-file>
```

WAV, FLAC and MP3 all decode. Any sample rate: the audio is resampled to the
model's 22050 Hz with a proper polyphase filter, not by interpolating between
neighbouring samples.

## What to expect, and what not to conclude

On five produced recordings here the two paths agreed about 71% of the time,
and on two of them Beat This! chose a metrical level an octave below the
built-in tracker — 68 BPM against 133 — while still placing 97% of its beats on
beats the other found. That is the classic octave ambiguity rather than a
tracker losing the grid, and under a strict CMLt it is expensive.

**Which of the two is right cannot be answered from these numbers.** Every
comparison in this repository so far measures agreement between implementations,
not correctness, because the corpus has no human annotations paired with audio.
The annotation profile is recorded in `research/eval/corpora.py`; what is
missing is the audio to go with it. Until that lands, treat a disagreement as a
disagreement.
