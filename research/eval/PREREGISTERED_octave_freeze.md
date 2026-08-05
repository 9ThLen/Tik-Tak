# Pre-registered: holding the octave when the estimator stops being sure

Written before any of it exists. Nothing below was chosen after seeing a number
it is measured against.

## The question

`research/results/phase_instability_*.json` established that
`live_anchor_margin` — how far the activation-tempo estimator's winning octave
leads its best rival at another metrical level — warns of **85.9% of every
wrong-level episode over four seconds, one to four seconds before it starts**,
with 16.9% of correct locked frames above the threshold. That is a ranking
result on a signal the live path already computes and uses for nothing.

This asks whether a policy built on it makes the tracker better, which is a
different question and cannot be answered by an ROC.

## Why this is not the experiment that already failed

**`LiveConfig::anchor_octave_margin` already exists and already lost.** It gates
the anchor on exactly this quantity, and `core/src/tracking/live.hpp` records
the result: on 120 ballroom recordings, F 0.752 with no gate, 0.738 at 0.15,
0.714 at 0.30. It ships at 0.0, meaning off.

The same file says why, and the reason is the whole basis for trying again:

> a tie in the estimator is not a coin toss downstream. Both rivals are
> metrical relatives of each other, so anchoring the wrong one still puts the
> filter on a grid the right beats fall on, whereas refusing to anchor leaves
> it with the fixed prior, which is worse than either.

So the measured failure is not of *distrusting a weak margin*. It is of the
**response**: `live.cpp` calls `filter_.clearAnchor()`, and the filter falls
back to a fixed log-normal prior over all tempi. This proposal keeps the
trigger and changes the response — hold the last octave that *was* confidently
chosen, which is a specific recent claim rather than a prior over everything.

`live.cpp` also states the objection to holding:

> An anchor is a claim that the metrical level is known, and when the estimator
> stops saying so the claim has to go with it — otherwise a tempo measured in
> the first chorus outlives the evidence for it.

That objection is correct and is what the timeout below exists to bound. It is
not answered by argument; if the freeze arm fails, this is the most likely
reason, and the timeout sweep is where it would show.

## The system under test

**Shipped fold 1, not the ensemble.** `EnsembleMean` was measured and its
adoption was **not approved** — three acceptance gates failed. Building on top
of a configuration that is not adopted would make this experiment
uninterpretable whichever way it came out.

One consequence is worth stating because it is unusually favourable: fold 1
holds GTZAN out of training, so GTZAN is honest ground here in a way it is not
for any averaged tracker. Ballroom is not — fold 1 trained on it — and is
excluded from every number quoted.

## The four arms

| arm | trigger | response |
|---|---|---|
| **baseline** | — | shipped fold 1, `anchor_octave_margin` 0.0 |
| **clear** | `margin < τ` | `clearAnchor()` — the existing stateless mechanism, at the new τ |
| **freeze** | `margin < τ` | hold the last confidently chosen octave |
| **abstain** | `margin < τ` | publish nothing while the margin is weak |

`clear` and `abstain` are **mechanistic controls, not candidates.** `clear`
isolates the response: same trigger, same τ, the old reaction, so a difference
between `clear` and `freeze` is attributable to holding rather than to
triggering. `abstain` is a diagnostic upper bound on how much of the episode
metric is reachable by saying less, and is not proposed for adoption under any
outcome.

Note that τ below is far above the 0.15 and 0.30 that were measured and
rejected, so `clear` is expected to be *worse* than the old measurements, not
better. That is the point of including it.

## Exact semantics of `freeze`

Written out because "hold the octave" has several plausible readings and only
one of them is being tested.

State: `held_octave_bpm`, initially unset; `held_since_sec`, initially unset.

1. **`margin ≥ τ`** — accept the anchor exactly as the baseline does, then set
   `held_octave_bpm` to the accepted BPM and `held_since_sec` to now. A
   confident anchor always refreshes the hold, including when it lands on a
   different octave from the one held: the freeze never overrides confident
   evidence.

2. **`margin < τ` and `held_octave_bpm` is set and
   `now - held_since_sec < freeze_timeout_sec`** — take the estimator's BPM `b`
   and anchor at the octave equivalent nearest the hold:

       k     = round(log2(held_octave_bpm / b))
       b'    = b * 2^k
       anchor at b', with the shipped `anchor_width_octaves`

   `b'` is the estimator's own tempo moved by whole octaves only. **The tempo
   inside the octave is not frozen** — a recording that accelerates from 128 to
   132 anchors at 132, not at 128 — and the beat phase is not touched at all.
   `held_since_sec` is **not** refreshed here: the timeout measures time since
   the last *confident* anchor, not since the last frame.

3. **`margin < τ` and no confident anchor has been seen yet** — behave exactly
   as the baseline: accept the anchor. There is nothing to hold, and refusing
   to anchor at the start of a recording is the arm that already lost.

4. **`margin < τ` and the timeout has expired** — behave exactly as the
   baseline: accept the anchor, and clear `held_octave_bpm`. The next confident
   anchor starts a fresh hold. This is the concession to the `live.cpp`
   objection: a hold cannot outlive its evidence indefinitely.

`freeze_timeout_sec` is **4.0**, fixed here and not swept on any corpus that
scores the confirmatory comparison. It is chosen to match
`MAX_WRONG_OCTAVE_SEC`: a hold that could last longer than the episode
definition could convert a short slip into a long one.

If a timeout sweep is wanted afterwards it is a separate pre-registration, run
on RWC and GTZAN, and its result may not be quoted as a Harmonix number.

## τ, as a number

**τ = 0.5916.**

Taken from `research/results/phase_instability_rwcpop.json`: the threshold at
which 20% of clean negative windows trigger on RWC-Pop. It is written here as a
constant so that it is not re-derived per corpus — recomputing a quantile on
each corpus would make every arm a different policy and the comparison
meaningless.

No parameter of any arm is tuned on Harmonix.

## Corpora

Harmonix (581) carries the confirmatory comparison and the gates, because the
gates below are the ensemble pre-registration's and were measured there.
Harmonix has been spent repeatedly — the seam experiment, the ensemble A/B, and
the threshold transfer in the phase spike — which is precisely why nothing here
is fitted on it and why the gates are copied rather than re-derived.

GTZAN (998), RWC (328) and SMC (217) are reported beside it. RWC-Pop chose τ and
so cannot confirm it; the rest of RWC can.

## Acceptance gates

Unchanged from `PREREGISTERED_ensemble_in_core.md`, deliberately, so that the
goalposts are the same ones a different proposal was already judged against.
Baselines are fold 1's own, from `results/fold1_in_core_harmonix.json`.

| Harmonix | baseline | to accept |
|---|---:|---:|
| no wrong-level episode >4 s | 41.5% | ≥ 46.5%, p < .05 |
| usable, strictly | 26.2% | ≥ 30% |
| correct time (eligible, mean) | 77.5% | ≥ 75% |
| switches / eligible 5 min | 4.21 | ≤ 4.21 |
| settle P90 | 36.61 s | ≤ 36.61 s |
| beat F | 0.7953 | ≥ 0.785 |

**Failing any acceptance gate means "adoption not approved".** Not "mixed", not
"promising", not "the sinking list was clean". The gates decide.

## How `abstain` is scored

Silence counts as **lost output time, not as a correct state**. Specifically:
a frame on which the arm publishes nothing is counted in the denominator of
correct time and not in its numerator, and its metrical state is `silent`
rather than being skipped.

Without that, `abstain` wins the episode endpoint by construction — a tracker
that says nothing spends no time at the wrong level — and would look like the
best arm while being useless. This is the same trap the endpoint's own
definition already has, and it is why `abstain` is a bound and not a candidate.

## Primary comparison

**`freeze` against `baseline`, on Harmonix, paired per recording, on
`no_wrong_level_episode_fraction`.** Exact two-sided binomial sign test on the
discordant pairs, α 0.05, Holm-corrected over the family reported.

`clear` and `abstain` are compared descriptively and are **not** in the
correction family. Putting four arms into one family would triple the
correction on the one comparison that is actually being made, to buy
significance tests on two arms that cannot be adopted whatever they show.

## Predictions

Recorded so a wrong one cannot be reread as a right one afterwards.

- **P1.** `freeze` clears the episode gate on Harmonix: ≥ 46.5%, p < .05.
- **P2.** `freeze` beats `clear` on the episode endpoint. If it does not, the
  gain is in distrusting a weak margin and not in holding, and
  `anchor_octave_margin` should simply be raised — which is a one-constant
  change and does not need any of this.
- **P3.** `clear` at τ = 0.5916 is *worse* than baseline on beat F, continuing
  the trend already measured at 0.15 and 0.30.
- **P4.** `abstain` reaches the highest episode-free fraction of the four and
  the lowest correct time, and fails the correct-time gate.
- **P5.** `freeze` does not move beat F by more than a point in either
  direction. It changes which octave is anchored, not where the beats sit
  inside it, so a large move means the octave mapping is disturbing phase and
  is a bug rather than a result.
- **P6.** The switch rate falls under `freeze`. Holding an octave is
  mechanically a reduction in level changes, and if it does not fall, the
  policy is not doing what it is described as doing.

## What would sink this

- The episode gate missed, or met without significance.
- `freeze` indistinguishable from `clear`: nothing here is worth a state
  machine if a threshold change does the same job.
- Beat F down by more than a point, or the switch rate up.
- Any acceptance gate failed — which is the same sentence as above, restated
  because the previous experiment showed how easily this list gets read as a
  softer alternative to the gates. It is not. It is a list of ways to fail
  *early*, not a lower bar.

## Before any corpus is touched

Synthetic tests, on a `LiveTracker` driven by constructed activations, all
passing first:

1. **No confident anchor yet.** With `margin < τ` from the first frame, every
   anchor decision is identical to baseline.
2. **Octave mapping.** A held octave of 120 with the estimator reporting 60,
   240 and 61 anchors at 120, 120 and 122 respectively — the nearest octave
   equivalent, not the held value itself.
3. **Tempo is not frozen.** A held octave of 120 with the estimator drifting
   128 → 132 at weak margin anchors at 132, not at 120 or 128.
4. **Recovery.** A confident anchor at a different octave replaces the hold
   immediately, on the same frame.
5. **Timeout.** After `freeze_timeout_sec` of continuously weak margin the arm
   is byte-identical to baseline, and a later confident anchor starts a new
   hold.
6. **Phase is untouched.** Beat times under `freeze` and under baseline are
   identical on a recording where the margin never falls below τ.

Test 6 is the one worth writing first: it fails loudly if the octave mapping is
implemented anywhere near the phase.
