# Spike Conventions

## Stack

- C++17 shipping core and evaluation binary.
- Python + NumPy research orchestration and statistics.
- No new dependency for a spike when the repository already has the needed
  metric and corpus harness.

## Structure

- Registered protocols, experiment code and artifacts stay in `research/`.
- `.planning/spikes/` is an index and investigation trail, not a second source
  of formulas or acceptance thresholds.

## Patterns

- Experimental live behavior is disabled by default.
- A research policy may control a narrow live seam, but the live filter itself
  is never reimplemented in Python.
- Corpus order is enforced by executable checks, not by comments alone.

## Tools & Libraries

- Reuse `live_corpus_benchmark.py` for product metrics.
- Reuse cached BeatNet activations only after byte-level live-series parity.
- Keep Beat This! as an optional offline comparison backend, outside live
  acceptance decisions.
