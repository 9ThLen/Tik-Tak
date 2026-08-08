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

---

## Deviations, found after the run, 2026-08-05

Appended, not edited into the text above. These were **not** recorded before the
result was published, which is itself the defect: the run was reported as an
answer to this document, and it is an answer to a different decoder.

| registered | run |
|---|---|
| "take the **predicted** beat positions" (§ decoder, step 1; arms table) | `audit_one` calls `load_reference_beats` — the **annotated** grid |
| "accumulate a score over the **last 2–4 bars**" (step 3) | the mean over the **whole recording** |
| four arms, including `autocorr` | three; `autocorr` never implemented, so **P1 is unmeasured** |
| seven measured quantities | three reported; bar period accuracy, coverage, false corrections, bars to a stable decision and oracle-gap share were not |

**Which direction each deviation pushes.** The annotated grid *favours*
`beat-sync` — it is strictly better than what live has — so A2's failure is
robust to it. The whole-recording window does not: a global mean over a
four-minute song is diluted by arrangement change in a way a two-bar window is
not, so "offline sees more, therefore it is a ceiling" holds for an optimal
decoder and not for this one. That defence was written after the run and is
withdrawn.

**A further confound, in the unfavourable direction.** `shuffled` is permuted
independently on the plain grid and on the doubled one, so the null is not
matched across the comparison it adjudicates: the doubled grid has twice the
points, the maximum of a noise contrast over (metre, phase) is systematically
smaller there, and the octave comparison therefore mixes the channel's
information with the decoder's geometry. A correct null shares one time-shift
across both grids, preserving their nesting.

**What A2's failure therefore licenses**, stated to the width of the evidence:

> An unnormalised global contrast score, maximised over (metre, phase), gets no
> octave advantage from the downbeat channel.

Not "the downbeat head contains no octave information". Three of the four
acceptance conditions were never measured, so this document was not executed and
cannot close the direction on its own terms. It rejects the decoder it tested.

The metre result is unaffected: it is a comparison between arms reading the
*same* grid with the *same* decoder, where the geometry cancels.

---

## The causal arm, registered 2026-08-08, before it was run

"Not in scope" above says a causal decoder would be a separate pre-registration.
This is it, and it is deliberately small, because it is this protocol with **one
factor changed** rather than a new question.

The audit measured the metre over a whole recording and called that a ceiling:
"a causal decoder seeing two to four bars cannot beat what an offline one seeing
all of them extracts". It found the metre carried decisively — 82.9% on Harmonix
and 60.8% on GTZAN against a shuffled null of 30.1% and 23.5% — and the octave
not carried at all. Commit `44c8c56` then built the causal reader. The question
registered here is the one that commit's own message says is unmeasured:
**how much of the metre survives the trailing window?**

### What changes, and what does not

Changed: the decision is made over the last **32 beats the live tracker actually
handed out**, re-made on every beat, by `analysis::resolveMeter` inside the
shipping core — not over the whole recording by the audit's Python decoder.

Unchanged, and taken from above without amendment: the corpora (GTZAN and
Harmonix), the metre-accuracy definition, and both controls. `shuffled` permutes
the downbeat channel across frames; `beat_as_downbeat` feeds the beat channel
where the downbeat channel belongs. A result that does not clear both is not a
result.

### Three differences that are not the factor under test, and are confounds

Named now so they cannot be offered afterwards as explanations for a bad number.

1. **The grid is the tracker's, not the annotation's.** The audit scored the
   annotated beat grid. The live path scores the beats it found, which on this
   material is right about two thirds of the time. A causal metre read off a
   wrong grid is wrong for a reason that has nothing to do with causality.
   Handled by reporting the causal accuracy **restricted to recordings the
   tracker tracked at the annotated level**, beside the unrestricted figure, and
   treating the restricted one as the answer to the registered question.
2. **The decoder is not the audit's.** `resolveMeter` has metre priors, a
   minimum salience range and two margin gates; the audit's decoder had none of
   these and always answered. So a lower number could be the window or could be
   the gates. Separated by reporting coverage — the share of recordings that
   answer at all — and by scoring **only among those that answer**, alongside
   the share of all.
3. **Gating.** Not a confound here: the harness plays no click, so no frame is
   withheld. Recorded because it will be one in the product, and the figure
   below is therefore an upper bound on what a shell with an audible click sees.

### What is measured

- **M1 metre accuracy, causal** — share of recordings whose held
  `beats_per_bar` at the end of the recording matches the annotated one.
- **M2 metre accuracy, causal, per decision point** — share of handed-out beats
  whose current held metre is correct. M1 is a verdict, M2 is what a listener
  watching a display experiences, and the two can disagree.
- **M3 coverage** — share of recordings that answer at all.
- **M4 bars to a stable answer** — beats before the held metre stops changing.
- **M5 switches** — how many times the held metre changes per recording. A
  decoder that reaches the right answer by flickering through every candidate
  is not usable however good M1 is.

### Acceptance

There is nothing to adopt: the mechanism already ships behind a default-off
flag, and this decides whether it is worth turning on and measuring further.

- **C1.** On both corpora, M1 restricted to correctly-tracked recordings clears
  `shuffled` by at least **15 points** with non-overlapping 95% intervals.
- **C2.** It clears `beat_as_downbeat` by at least **10 points** on both.
- **C3.** M1 restricted retains at least **two thirds** of the audit's
  whole-recording figure on the same corpus — 55.3% of Harmonix's 82.9%, 40.5%
  of GTZAN's 60.8%.

Failing C1 or C2 means the causal window is reading the grid rather than the
channel, and the flag should stay off. Failing C3 alone means the evidence is
there and the window is too short, which is a parameter sweep and not a dead
end — and the sweep would then need its own registration, because
`window_beats` would have been chosen after seeing the result.

### Predictions

- **P5.** M1 restricted lands **below** the audit's whole-recording figure on
  both corpora, because a 32-beat window is strictly less evidence. Named
  because the opposite result would mean something is wrong with one of the two
  measurements rather than that causality is free.
- **P6.** M2 is below M1: early windows are short and wrong, and they are
  counted in M2 and not in M1.
- **P7.** The unrestricted M1 is well below the restricted one, by roughly the
  share of recordings tracked at the wrong level.

### The metre arm answered, and the corpora cannot carry it, 2026-08-08

Run on GTZAN (991 scored, 727 restricted) and Harmonix (579 scored, 491
restricted), commit `dbdd619`, clean tree, all arms byte-identical on beats.

|  | GTZAN | Harmonix |
|---|---:|---:|
| always-4 | **0.949** | **0.976** |
| `beat_sync` | 0.867 | 0.894 |
| `beat_as_downbeat` | 0.791 | 0.729 |
| `shuffled` | 0.492 | 0.499 |

C1 passes on both, by 37.5 and 39.5 points. C3 passes on both, and the causal
figure is *above* the whole-recording audit's — falsifying P5. C2 fails on GTZAN
at 7.6 points against a registered 10 and passes on Harmonix at 16.5.

**None of that is the result.** The restricted sets are 690 of 727 and 479 of 491
in four, so answering "4" and nothing else scores 0.949 and 0.976 — eight points
above the decoder on both. Off the majority metre there are **49 recordings in
1218**, and on them `beat_sync` scores 0.189 and 0.250 against `shuffled`'s 0.108
and 0.333. At those counts nothing is distinguishable from anything.

Two things follow, and the second is why this section exists rather than a
verdict line.

**The constant baseline was missing from C1–C3, and that is a fault in this
registration.** It was written the same day and compares only against shuffles
and substitutions, both of which a metre prior clears without deciding anything.
Same shape as the octave veto's "A3 passing is not a defence". It now appears in
the harness summary so no later run can omit it, and the reading applies
backwards: the audit's own 0.608 on GTZAN was compared against a shuffled 0.235
and never against the 0.949.

**The metre was the wrong endpoint for this material.** If 96% of the corpus is
in four, then the bar question on this corpus is almost entirely *which beat
starts the bar*, and this arm did not measure it. `analysis/downbeat.hpp` has
said all along that phase is the failure a listener notices — "a metronome
accenting beat 3 is worse than one accenting nothing" — and both the original
audit and this arm inherited an endpoint that cannot see it.

### Bar phase, added 2026-08-08 after seeing the above

Registered as an addition and not a substitution: the metre numbers above stand
and are reported whatever this says. It is being added **after** seeing the
metre result, for a reason stated plainly — the corpus composition that makes
metre unanswerable was discovered in the run, not before it. That ordering is a
weakness and it is recorded rather than hidden.

**F1 bar-line agreement.** Of the beats the tracker handed out at bar position 0,
the share that fall within 70 ms of an annotated downbeat (precision), and of the
annotated downbeats inside the tracked span, the share matched by one (recall).
Reported as F-measure. Restricted, as above, to recordings tracked at the
annotated level, and computed only over beats after the metre first settled.

**F2 phase-correct share.** Per beat, whether its position agrees with the
annotated bar phase; averaged over the recording. What a listener watching a
display experiences, where F1 is a verdict.

**The null is not a shuffle this time.** With the metre almost always four, a
decoder that picks a bar phase uniformly at random is right one time in four, so
`random_phase` — the tracker's own metre and settled grid with the phase drawn
from a seeded generator — is the baseline that has to be cleared. `shuffled` and
`beat_as_downbeat` are still run and still reported.

**Conditions.** F1 clears `random_phase` by at least **20 points** on both
corpora with non-overlapping 95% intervals, and clears `beat_as_downbeat` by at
least **10**. Failing either leaves the flag off and closes the causal bar with
a documented negative, since metre has already failed to separate from a
constant.

**P8.** F1 lands well below the metre figures — around 0.4 to 0.6 — because
phase is the harder half and because the offline resolver's own phase margin was
measured at AUC 0.713 for predicting agreement, which is weak.
