# S1 execution

This package prepares and runs the preregistered A3 reset/stateful training
ablation in `research/eval/PREREGISTERED_S1.md`. It is research-only because
the fixed development corpus includes RWC 2.0 material under CC BY-NC 4.0.
Nothing produced here is a product weight.

The order is fixed:

1. From a clean committed tree, build the complete resumable feature cache
   with `python -m training.beatnet.cache`. The output directory and optional
   pause file must be outside the repository. Creating the pause file drains
   the current recording and exits 75; remove the pause file and rerun the
   same command to resume.
2. Export/evaluate the frozen `beatnet_model_1.ttw` once with
   `python -m training.beatnet.evaluate`. This creates the A0 development
   baseline used only for checkpoint eligibility.
3. Run `python -m training.beatnet.probe` once. It trains both arms on a
   deterministic 5% cache prefix and reports GPU memory, throughput and a
   six-run training-only projection. It is explicitly non-binding.
4. Run all six `(A3_reset, A3_stateful) x (17, 29, 43)` jobs with
   `python -m training.beatnet.run`. One GPU training job at a time is the safe
   default on the 6 GiB GTX 1660 Ti. Product evaluation is CPU work and accepts
   several `--eval-workers`.
5. Run `python -m training.beatnet.summarise` with six repeated `--run`
   arguments. It reloads the selected per-work records and recomputes the
   registered work-level bootstrap; it does not trust per-run means.

The training pause file is checked at epoch boundaries. Remove it and add
`--resume` to continue. Resume binds the clean commit, source checkpoint,
complete cache manifest, frozen baseline, arm, seed and full JSON config. A
mismatch fails closed. Candidate checkpoints are exported and evaluated every
five epochs; patience is four product-validation points.

The cache is intentionally about 14 GiB. A diagnostic cache made with
`--limit` is marked `diagnostic_only` and the binding runner refuses it.
Likewise, a baseline without clean provenance, exactly 84 development works,
or the pinned frozen TTBN digest is rejected.
