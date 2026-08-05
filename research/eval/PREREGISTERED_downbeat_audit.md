# Pre-registered: is a bar period recoverable from BeatNet's outputs at all?

Written before any of it exists. Nothing below was chosen after seeing a number
it is measured against.

## The question, and the gap it closes

`eval/PREREGISTERED_downbeat_channel.md` measured three arms and left one thing
conflated. Its oracle takes the bar length **from the annotation**, so it never
reads BeatNet's downbeat output. What that run established is that an accurate
bar period has strong leverage — 19.1 points of episode-freeness on Harmonix —
and that one generic autocorrelation over the downbeat channel failed to supply
one. It says nothing about whether the channel *could* supply one.

Two possibilities are still indistinguishable:

- the downbeat output carries no usable bar information;
- `ActivationTempo` is the wrong instrument for reading it.

This audit separates them, offline, with **no change to the live core**. It is
the smallest experiment that closes the largest remaining uncertainty, and no
further live policy is written until it reports.

## Why beat-synchronous, and not another autocorrelator

The downbeat head is not a slow tempo detector. It answers *which of these beats
begins a bar* — a question that only makes sense at beat positions. Reading it
by autocorrelating a 50 fps probability track asks it a question it was not
trained to answer, and spends the beat grid, which is the one thing already
known accurately.

So the decoder under test is:

1. take the predicted beat positions;
2. read the downbeat probability at each one;
3. for each metre in {2, 3, 4, 6} and each bar phase within it, accumulate a
   score over the last 2–4 bars;
4. report the winning metre, the bar phase, and the margin to the runner-up;
5. only then ask whether that evidence separates `P` from `P/2`.

Step 5 is the point. A decoder that recovers the metre but cannot tell a beat
grid from its own double has not answered this question.

## The arms

Four, and the last two are the controls that decide whether any of it means
anything:

| arm | what it reads |
|---|---|
| **beat-sync** | the downbeat probability at predicted beat positions, decoded as above |
| **autocorr** | the current `ActivationTempo` over the downbeat channel — the arm that already failed, carried forward so the comparison is like for like |
| **shuffled** | the same beat-sync decoder over downbeat probabilities **randomly permuted across beat positions**, within each recording |
| **beat-as-downbeat** | the same decoder fed the **beat** channel where the downbeat channel should be |

`shuffled` destroys the alignment between the downbeat channel and the beat grid
while preserving its marginal distribution, so any decoder that scores above it
is reading structure rather than level. `beat-as-downbeat` is the sharper
control: the beat channel is high at *every* beat, so a decoder that scores well
on it is finding periodicity in the beat grid it was handed, not downbeat
evidence.

**A result that does not clear both controls is not a result.**

## What is measured

Not the end-to-end BPM. That is what the previous experiment measured, and it
cannot say which stage failed. Directly:

- **bar period accuracy** — the share of recordings whose recovered bar period
  is within 8% of the annotated one, the same octave tolerance the live
  benchmark uses;
- **metre accuracy** — the share whose recovered metre matches the annotated
  `beats_per_bar`;
- **octave separation** — given a beat grid and its own double, the share where
  the decoder's score prefers the true one. This is the quantity the whole line
  of work is about and it has never been measured directly;
- **coverage** — the share of recordings where the decoder answers at all;
- **false corrections** — among recordings whose baseline octave is already
  right, the share the decoder would move;
- **bars to a stable decision** — how many bars before the winning metre stops
  changing;
- **share of the oracle gap recovered** — the previous run's 19.1 points is the
  denominator.

## Acceptance conditions

Fixed here, before the run:

| | condition |
|---|---|
| **A1** | at least **one third** of the oracle gap recovered |
| **A2** | a **substantial** margin over `shuffled` on octave separation — at least 15 points, and non-overlapping 95% intervals |
| **A3** | **no more than 5%** false corrections on baseline-correct recordings |
| **A4** | a threshold chosen on one corpus transfers **as a number** to the other, holding A1–A3 |

**Failing any of A1–A4 closes the downbeat head for octave correction with a
documented negative.** Not "try another filter" — the point of naming the
controls and the conditions in advance is that a failure here is informative
enough to stop on.

## Corpora

Harmonix (581) and GTZAN (999), the same two the previous experiment used, with
the threshold chosen on GTZAN and transferred to Harmonix. GTZAN chooses because
fold 1 holds it out of training and because Harmonix has now been spent three
times over.

Both are offline reads of activations already computable with the shipped
binary; nothing here needs a corpus that has not already been used.

## Predictions

- **P1.** `beat-sync` beats `autocorr` on bar period accuracy. If it does not,
  the instrument was never the problem and the channel is the problem.
- **P2.** `beat-sync` clears `shuffled` on metre accuracy comfortably — the
  metre is the easier half, and a downbeat head that could not do this well
  would be broken rather than weak.
- **P3.** Octave separation is the failure, if there is one: the metre comes
  back and the `P` against `P/2` decision does not. A bar phase repeats at both
  grids, so the evidence for the octave is weaker than the evidence for the
  metre, which is the same shape `analysis/downbeat.hpp` already records for the
  half-bar.
- **P4.** `beat-as-downbeat` scores well above chance on metre and near chance
  on octave separation. That is what "finding periodicity in the grid it was
  handed" looks like, and it is the control most likely to expose a decoder
  that is fooling itself.

## What would sink this

- Any of A1–A4 missed.
- `beat-as-downbeat` matching `beat-sync`. The decoder would then be measuring
  the beat grid, and every number above would be about the wrong channel.
- Coverage so low that the accuracies are computed on a self-selected minority.
  Reported as a share of all recordings, always, for exactly this reason.

## Not in scope

No live decoder, no core change, no new policy. If the audit passes, the causal
implementation is a separate pre-registration with the six live metrics; if it
fails, there is nothing to implement.
