# Pre-registered: the learned file path, measured where it ships

Written before the run. The arms, corpus and endpoints are fixed here, and so is
what each answer is used for.

## What changed, and what has not been measured

The measured gain of Beat This! over the onset path (+0.102 beat F, +0.138
CMLt on all of GTZAN) came from `dump_analysis --beat-this`, a research seam
that swaps the model's beats and its own downbeat picks in after the onset
analysis has run and reports the onset path's tempo and metre beside them.
`tt_offline` could not reach it.

Branch `claude/learned-front-ends` puts the model inside `OfflineAnalyzer`
(`--learned` in the tool): the same features, chunking and peak picker, so the
beats are the seam's to the bit, but the bar lines now come from `resolveMeter`
fed the model's downbeat probability at each beat — one metre, one phase, one
pair of margins, switch cost 20. **Whether that bar decision is as good as the
model head's own picks, on human annotation, has never been measured.** The
0.772 -> 0.933 in `downbeat.hpp` was scored against the head's own picks and is
circular.

## Arms

One binary, built from this commit in a clean tree, three runs per recording:

| arm | flags | beats | bar lines | metre |
|---|---|---|---|---|
| `onsets` | none | onset DP | cue resolver | cue resolver |
| `seam` | `--beat-this small0.onnx` | model | model head's picks | cue resolver |
| `learned` | `--learned small0.onnx` | model | resolver on model probability | resolver on model probability |

Corpus: **GTZAN, all recordings with a reference**, the only corpus held out
of `small0`'s training (`tiktak-gtzan-heldout-bias-signs`). Scored with
`eval.harness` (mir_eval, 5 s trim, ±70 ms). Recordings without annotated
downbeats are excluded from downbeat endpoints only.

## Endpoints and decisions

1. **Replication.** `learned` beats must equal `seam` beats on every recording,
   and `learned - onsets` beat F must reproduce +0.102 within its interval.
   *If not, the port is wrong and nothing below is read.*

2. **The bar decision.** Downbeat F, `learned` (resolver) against `seam` (head
   picks), paired bootstrap over recordings, 10,000 resamples. Also the
   *rendered* downbeat F — what a player given one offset plays, with the phase
   that holds the most bar lines — for `learned`.
   *If `learned` is no worse than `seam` by more than 0.01 on the lower bound,
   the resolver stays the product's bar decision. If it is worse, the next step
   is handing the player every bar line (per-beat positions) and keeping the
   head's picks, not tuning the resolver.*

3. **Metre**, descriptively only: `learned` and `onsets` beats-per-bar against
   the annotated modal metre, beside the constant "always 4". GTZAN is about
   95% four-four, so this cannot rank decoders (`tiktak-metre-corpora-cannot-answer`).

4. **The accent gate**, descriptively only: coverage and conditional error of
   `downbeat_confident` on `learned` at the provisional thresholds. Calibrating
   them needs a split and is its own run.

## Not asked here

Full-length songs (Harmonix, RWC) need the fold-matched full checkpoints,
because `small0` trained on them; and the room is the live path's question.
