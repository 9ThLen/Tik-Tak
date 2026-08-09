# Beat This! source-checkpoint to runtime-ONNX parity — registered 2026-08-09

## Question

Does the pinned `models/beat_this.onnx` compute the same frame-level beat and
downbeat logits as the pinned official Beat This! `final0.ckpt` from which the
third-party repository says it was exported?

This is an identity check, not a model-quality experiment. Passing licenses the
runtime graph as an implementation of that checkpoint. It does not validate the
audio frontend, peak picking, decoder, causality, or product performance.

## Inputs fixed before the run

- Source artifact: `beat_this_final0` in `models/manifest.json`.
- Runtime artifact: `beat_this_cpp_onnx` in `models/manifest.json`.
- Both files must match their pinned SHA-256 values, and the conversion record's
  `source_weights_sha256` must match the source artifact.
- Input: eight seconds at 22,050 Hz generated deterministically by
  `models/export_beat_this.py`: 110 Hz and 440 Hz tones, seeded white noise, and
  impulses every 0.5 seconds; seed `20260809`.
- The input is transformed by the project's fixed Beat This! log-mel frontend.
  The report records hashes of both generated audio and resulting spectrogram.
- Outputs: every beat and downbeat logit at every emitted frame. No peak picking,
  thresholding, or decoder is allowed before comparison.

## Registered decision rule

For each head, compute maximum and mean absolute logit difference. The graph
passes only if the maximum absolute difference over both heads and all frames is
at most `1e-4`. Shape disagreement is an automatic failure. The tolerance is
the existing export guard in `models/export_beat_this.py`; it is not selected
after inspecting this runtime graph.

The run is accepted only when shared provenance reports `tree_clean: true`.
Unknown provenance and a dirty tree both fail closed. A dirty diagnostic may be
run explicitly while implementing the harness, but its report must say
`accepted: false` and cannot close P0. The JSON report must be written outside
the repository and must contain source/runtime/manifest digests, the fixed input
recipe and hashes, per-head differences, tolerance, result, and shared run
provenance.

## Interpretation

- Pass: the pinned runtime ONNX is numerically equivalent to the pinned official
  `final0` checkpoint for the registered frame-level probe. Model-quality
  experiments may identify it as that checkpoint, subject to their own gates.
- Fail: no result produced with this ONNX may be attributed to official
  `final0`; rebuild the graph from the pinned source checkpoint and repeat this
  preregistered comparison.
- Infrastructure or provenance failure: inconclusive, not pass or fail.

No tolerance widening, alternate input, output remapping, or selective frame
exclusion is permitted after seeing the result. Any such change requires a new
versioned preregistration and both old and new reports remain part of the audit
trail.
