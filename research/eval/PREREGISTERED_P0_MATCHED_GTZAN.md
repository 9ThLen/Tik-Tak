# P0 clean matched GTZAN rerun — registered 2026-08-09

## Purpose

Repeat the existing BeatNet-versus-Beat-This! matched-decoder comparison with
reproducible provenance. This is a provenance repair and uncertainty estimate,
not a new model-selection gate. The old `+0.138` was produced from a dirty tree
and 100 recordings selected by stride; it remains historical context only.

Both compared checkpoints withhold GTZAN. The result is therefore held-out for
both, but it remains a system-level difference: models also differ in training
corpora, capacity and recipe. No result from this run may be called an
architectural gap or a live-room estimate.

## Locked execution

- Use `beatnet_model_1` and `beat_this_cpp_onnx`, both verified against the
  pinned SHA-256 values in `models/manifest.json`.
- Route both activation streams through the same shipped `LiveTracker` path.
- Run every GTZAN recording resolved by the supplied corpus manifest with
  `--full-corpus`. Do not select by genre, duration, prior score or error mode.
- The harness must write outside the evaluation worktree and shared provenance
  must report `tree_clean: true`. Unknown or dirty provenance aborts the run.
- Report every failed recording by name and error. Failures are excluded from
  numerical means but never from the artifact or denominator accounting.
- Primary estimate: paired per-recording `Beat This! F - BeatNet F`.
- Uncertainty: 10,000 paired percentile-bootstrap resamples of recordings,
  seed `20260809`, with the 2.5th and 97.5th percentiles reported.
- Also report each model's mean F, usable rate, and all per-recording rows.

If a full run is impossible for a documented compute or input-availability
reason, `--corpus-limit N` may produce a diagnostic stride sample. Its artifact
must include the stride and the complete list of excluded corpus/name pairs. A
limited run cannot replace the P0 acceptance run and must not overwrite it.

## Interpretation

The estimate and interval are descriptive. Their sign and magnitude do not by
themselves select fine-tuning, a new model, or a decoder change. The result may
be cited as the clean held-out GTZAN matched-decoder benchmark only if the run
is full, all artifact digests are present, and `tree_clean` is exactly `true`.
