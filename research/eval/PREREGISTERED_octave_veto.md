# Pre-registered: can beat-synchronous metre evidence judge the octave switches the live tracker already proposes?

Written before any of it exists. Nothing below was chosen after seeing a number
it is measured against.

## Why this document exists at all

`eval/PREREGISTERED_downbeat_audit.md` was **not executed as written**. Its own
deviations appendix records three unregistered changes — annotated beat grids
where predicted ones were registered, a whole-recording mean where two to four
bars were registered, and a missing `autocorr` arm — and three of its four
acceptance conditions were never measured. Its A2 was measured on a null that is
permuted independently on each grid, so it does not adjudicate the comparison it
was built for.

**That, and only that, is the licence for this run.** Not the disappointing
result. A completed protocol that failed would close the direction; an
incomplete one has to be completed before it can close anything. The distinction
is narrow and it is named here so it cannot be widened later: this document
finishes a protocol, and if it fails there is no third attempt.

Everything the previous run established about the **metre** stands. That
comparison held the grid and the decoder fixed across arms, so the geometry
cancels, and 82.9% against a 30.1% null on Harmonix is not in question here. It
is a different use of the channel and it is out of scope.

## The change that matters most: the unit of the experiment

The previous three experiments in this line asked *does a feature predict a
future failure*. `eval/PREREGISTERED_octave_freeze.md` is the cautionary case:
the anchor margin predicts wrong-level episodes 1–4 s ahead in 85% of cases, and
the policy built on that prediction moved episodes by nothing, p = 0.86, while
demonstrably acting.

The gap between those two results is the gap between a **predictor** and a
**control action**. So the unit here is not the frame and not the recording. It
is the **decision point**:

> When the live tracker actually proposes moving to another octave, does
> beat-synchronous metre evidence correctly allow or veto **that** switch?

A decoder is scored as a binary classifier over real proposals, against
matched-cost policies that spend the same amount of the product's behaviour.

## The system under test

**Shipped fold 1**, `anchor_width_octaves = 0.02`, `anchor_octave_freeze = false`
— the shipped configuration, not `EnsembleMean`, whose adoption was not
approved.

Replay uses the **causal** activation stream the live core sees, not an offline
re-analysis. Before any measurement, replay must reproduce the live core
**byte-identically** on the same audio, on all five of:

1. beat times;
2. the published BPM sequence;
3. the published confidence sequence;
4. the `measured` BPM sequence from `activation_tempo_.estimate()`;
5. the extracted proposal-event list, by onset time and sign.

Beat times alone are not enough. Items 2–5 are the actual input to this
experiment, and a replay that reproduces beats while diverging on the estimator
would evaluate a decoder on events the product never had.

---

## 1. What counts as an octave-switch proposal

The anchor is rewritten at every frame from `activation_tempo_.estimate()`
(`core/src/tracking/live.cpp:214`), fifty times a second. Counting frames would
turn one sustained disagreement into hundreds of near-identical events and
inflate every count and every interval in this document. So:

**Committed level.** The octave of the published BPM from `estimate()` — what
the product displays, not what the filter is internally entertaining.

**Instantaneous proposal.** At frame `t`, with `measured` the estimator's answer
and `committed` the published BPM:

    k(t) = round(log2(measured.bpm / committed.bpm))

A proposal exists at `t` when `measured.answered()` and `k(t) != 0`.

**One event, from a run of frames.** A maximal run of consecutive frames with
the same **sign** of `k` is **one event**, timestamped at its first frame. The
event closes when `k` returns to 0 for at least **1.0 s** or the sign flips.

**Minimum separation: 2.0 s** between event onsets. An onset within 2.0 s of the
previous onset is merged into it rather than counted again.

Both constants are fixed here and swept on nothing. They exist to make events
countable, not to tune a result; the primary statistic is clustered by recording
precisely so that a bad choice here cannot manufacture significance.

**Labels**, from the annotation, at the 8% octave tolerance the live benchmark
already uses against the annotated median beat period:

| committed | proposed | correct action | in primary |
|---|---|---|---|
| correct | wrong | **veto** | yes |
| wrong | correct | **allow** | yes |
| correct | correct | — | no, ambiguous |
| wrong | wrong | — | no, ambiguous |

Ambiguous events are **reported with counts**, never folded into the primary.
If they outnumber the labelled events the experiment is uninterpretable and that
is reported as such rather than worked around.

**Proposals before the first lock are excluded from the sample.** Acquisition is
a separate, already-measured problem, the committed level is not yet a claim
about anything, and including them would score the decoder on a state it has no
business ruling on.

## 2. The window both candidates see

"The last four bars" is ambiguous the moment the two candidates disagree about
the bar, because four bars at `P` and four bars at `2P` are different amounts of
audio and the longer one would win on quantity.

**The window is a fixed audio interval: the 16 most recent committed beats
completed at or before the proposal frame.** Both candidate grids are read from
that same interval. **No frame after the proposal is visible to anything.**

This is 4 bars of the committed state in 4/4, and the asymmetry is stated rather
than hidden: if the committed state is itself doubled, the window is 2 true
bars. That shortens the evidence in exactly the case the decoder most needs to
get right, and it is the honest window, because at the moment of the decision
the true period is the unknown.

**Only two grids per event** — the committed level and the level actually
proposed. Scoring `P/2`, `P` and `2P` together would put three grids of three
different lengths into one maximum, which is the geometry that broke the last
run, tripled.

## 3. The score, in full

Let a grid `G` restricted to the window carry downbeat probabilities
`d_1 … d_N`, read at the nearest activation frame to each beat.

For metre `m` in {2, 3, 4, 6} and bar phase `p` in `0 … m-1`:

    on      = { d_i : i mod m == p }          n_on  = |on|
    off     = { d_i : i mod m != p }          n_off = |off|
    raw     = mean(on) - mean(off)
    se      = sd(d) * sqrt(1/n_on + 1/n_off)
    z(m,p)  = raw / se

`sd(d)` is the standard deviation over the whole grid. **The standardisation is
the point.** The previous run maximised `raw` over (m, p), and the null
distribution of that maximum shrinks as `N` grows, which tilted every octave
comparison toward the shorter grid before any evidence was read. `z` is a
two-sample statistic whose null does not depend on `N` to leading order.

**A residual multiplicity remains**, because a grid admits a metre only when
`N >= 2m`, so the longer grid can offer more (m, p) pairs to maximise over.
Removed by construction: **both grids are scored over the same set of (m, p)
pairs, those admissible on the shorter of the two.**

    score(G) = max over the common (m,p) set of z(m,p)

**Degenerate cases, fixed here so they are not decided at a keyboard later.**
`sd(d)` is the **population** standard deviation over the grid. With
`ε = 1e-9`:

- `sd(d) < ε` → **every `z(m,p)` on that grid is 0**. A channel with no variance
  carries no contrast, and I2 constructs exactly this case, so the formula has
  to answer it rather than divide by zero.
- `n_on == 0` or `n_off == 0` → that (m, p) is skipped. It cannot arise inside
  the common set, and the guard is there because a silent `nan` is worse than a
  redundant branch.
- **Fewer than 16 committed beats available** at the proposal, or fewer than 4
  points on the shorter grid, or an empty common (m, p) set → the event is
  **unanswered**. An unanswered event is **allowed**, is byte-identical to
  baseline, and is **excluded from A2 and A3** while being **counted in
  coverage** and **included in A1**, which measures the policy end to end and
  must pay for its own abstentions.

**Metre 6 is not always reachable, and this is a resolution limit rather than a
bias.** Sixteen committed beats give the halved grid 8 points, and `N >= 2m`
then admits {2, 3, 4} but not 6. Because the common set is defined by the
*shorter* grid, metre 6 drops out of **both** sides of the comparison whenever
the proposal is a halving — so it costs sensitivity on 6/8 material and cannot
tilt the octave decision either way. **Coverage is reported broken down by
annotated metre**, and I7 below constructs the case where the grid does admit 6.
The window is not widened to rescue it: 24 committed beats is twelve seconds of
history before a live decision, and buying one metre with that is the wrong
trade for the thing being built.

When the proposal is `2 * P_c`, the proposed grid is every other committed beat
and there are two ways to choose which — both are evaluated and the better is
taken. That maximum over 2 applies to the proposed grid only, and it is
therefore the one asymmetry the standardisation does **not** remove. It is
reported, and invariant I4 below is the test that it does not decide anything.

## 4. The null: one shift, both grids

The previous run permuted the two grids independently. That destroys the nesting
that makes the comparison meaningful and gives each grid its own null.

Here the **raw 50 fps activation track is circularly shifted once**, by `δ`, and
**both** grids are then resampled from the shifted track. The grids keep their
nesting, the channel keeps its marginal distribution and its autocorrelation,
and only the alignment to the beat grid is destroyed.

**Deterministic shifts, no RNG:** `δ ∈ {W/5, 2W/5, 3W/5, 4W/5}` where `W` is the
window length. Four per event, fixed here.

    null(G) = mean over δ of score(G_δ)

## 5. The decision statistic and the policy

    Δ = [score(G_committed) - null(G_committed)]
      - [score(G_proposed)  - null(G_proposed)]

`Δ > τ` → **veto** the switch. Otherwise → **allow**, which is byte-identical to
baseline.

### What `veto` does, exactly

The freeze experiment's whole lesson is that the *response* to a signal decides
the result more than the signal does, so "veto" is not left to mean whichever of
three plausible things is convenient later. In `core/src/tracking/live.cpp`, at
the branch that currently reads `filter_.anchorTempo(measured.bpm,
config_.anchor_width_octaves)`:

    allow:  filter_.anchorTempo(measured.bpm, width);

    veto:   filter_.anchorTempo(
                octaveNearest(measured.bpm, committed_bpm), width);

`octaveNearest` already exists (`live.cpp:19`) and already carries this meaning
for the freeze arm. **The veto blocks the metrical level and nothing else:**
tempo continues to move inside the committed octave — a band drifting 128 → 132
anchors at 132 under both arms — the phase is never touched, and the anchor is
never dropped. It is neither `clearAnchor()` nor a freeze. Those are separate
policies, both already measured, both already rejected.

### When the decision is taken

**Once, at the event onset frame, and held until the event closes.** Not
re-evaluated per frame: at 50 fps a per-frame decision would let a statistic
oscillate across `τ` mid-event and produce a policy nobody registered.

Between onset and close, every frame anchors through the branch the onset
decision selected. When the event closes, the held decision is discarded.

**Fewer than 16 committed beats available at the onset → allow**, i.e. baseline.
No answer is not evidence, which is the same rule the bar-rate arm used.

The asymmetry is deliberate: the decoder may only ever *block*. It never
proposes a switch of its own, never touches phase, and never moves tempo inside
an octave. A decoder that cannot beat baseline while only subtracting behaviour
has no case for being allowed to add any.

**Deployability, stated now so it is not discovered later as a surprise.** `Δ`
costs four extra passes over a window of at most ~20 s of activations, at
proposal events only — not per frame. That is affordable. `Δ_raw = score(G_c) -
score(G_p)`, without the null subtraction, is **reported alongside and is not
gated**: if it tracks `Δ`, the shipped version can be cheaper. That is an
engineering note. The registered decision statistic is `Δ`, because the biased
cheaper one is what produced the last result.

## 6. Synthetic invariants, all passing before any corpus is touched

On constructed activation tracks, no audio:

- **I1 — iid noise decides nothing.** Over 1000 draws of an iid channel, the
  mean of `Δ` is within 0.05 of zero and `Δ > 0` on 50% of draws ± 3 points.
  Stated **at `τ = 0`**, because the invariants run before `τ` exists and a
  condition phrased in terms of `τ` would be uncheckable when it is needed. Once
  `τ` is fixed on RWC, its realised veto rate on the same noise is reported as a
  separate line — after the fact, and not as an acceptance condition.
- **I2 — a beat-only channel creates no downbeat preference.** A channel high at
  every committed beat and low between yields `|Δ|` below 0.05 — in particular it
  does not prefer the doubled grid, which is exactly what `beat-as-downbeat`
  did at 23.6% and 7.8% in the previous run.
- **I3 — clean 4/4 is decided correctly.** A downbeat pulse every 4th beat at
  the true period gives `Δ > 0` against both a halved and a doubled proposal.
- **I4 — length alone does not flip the sign.** The same signal scored over 16,
  24 and 32 committed beats keeps the sign of `Δ`.
- **I5 — a waltz is not pulled towards 4.** A 3-metre signal returns metre 3 and
  the correct sign of `Δ`.
- **I6 — the decoder can say allow.** A committed state that is genuinely
  doubled, with a correct proposal, gives `Δ < 0`. A decoder that only ever
  vetoes is a rate limiter with extra steps.
- **I7 — six is decided when six is reachable.** A 6/8 signal, on a window long
  enough that both grids admit metre 6, returns metre 6 and the correct sign of
  `Δ`; and on a 16-beat window against a halving proposal, metre 6 is absent
  from the common set on **both** grids, so the exclusion is symmetric.

**I1 and I4 are written first.** They are the specific defect the last run had,
and a decoder that passes I3 and I5 while failing them is the last run again
wearing a new formula.

## 7. Choosing every free parameter, before any of them is free

### How `τ` is chosen

"`τ` is fixed on RWC" names the corpus and leaves the procedure open, which
leaves the single largest researcher degree of freedom in this document to be
settled after the numbers are on screen. The whole procedure, fixed here:

- **Candidate set:** `τ ∈ {0.0, 0.25, 0.5, … , 5.0}`, 21 values on a fixed grid.
  Nothing outside it, and no refinement pass.
- **Objective:** maximise `no_wrong_level_episode_fraction` on **RWC**.
- **Constraints:** A3 holds, and every standing cost gate holds, on RWC. A
  candidate that violates either is not eligible whatever its objective value.
- **Tie-break, in order:** greater retained correct locked-time; then lower veto
  rate; then smaller `τ`.

### How the simple policies' parameters are chosen

Same discipline, same corpus, fixed grids:

| policy | candidate set |
|---|---|
| debounce | `D ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` s |
| wider margin | `anchor_octave_margin ∈ {0.0, 0.1, … , 0.9}` |
| rate limit | `N ∈ {5, 10, 20, 30, 60, 120}` s |
| total ban | no parameter |

Each is chosen to **match the decoder's retained correct locked-time on RWC to
within 0.5 points**, and among candidates that do, the one with the best
episode-freeness — the strongest form of each policy, not a convenient one.

### If the match does not survive the transfer

A policy matched on RWC can cost something different on Harmonix, and the word
"matched" would then be doing work it has not earned.

**If any comparison policy's retained correct locked-time differs from the
decoder's by more than 0.5 points on Harmonix, the result is "adoption not
approved".** Not re-tuned, not re-matched, not reported with an asterisk. The
comparison the primary endpoint rests on would not exist, and manufacturing it
on the transfer corpus is exactly the move this document is built to prevent.

### The matched-cost comparison itself

A veto policy improves episodes by refusing to move. Against baseline alone that
reads as a win. So the primary comparison is against simple policies **spending
the same amount of behaviour**.

The policies:

| policy | parameter |
|---|---|
| **debounce** | require `k != 0` sustained for `D` seconds before the octave may move |
| **wider margin** | raise `anchor_octave_margin` |
| **rate limit** | at most one octave change per `N` seconds |
| **total ban** | no octave change at all after first lock |
| **oracle bar** | the annotated bar period — the ceiling, not a candidate |

**Primary comparison: the decoder against the single best simple policy at
matched cost.** The rest are diagnostic. This does not create new acceptance
gates; it replaces "better than doing nothing" with "better than doing less".

Every result reports all three of these together, never the first alone:

1. how many wrong-level episodes were removed;
2. how much correct locked time was lost;
3. **how many correct escapes were blocked** — `wrong → correct` events the
   policy vetoed, which is the only way a block-only policy can spoil a state,
   and is A3's numerator.

## 8. Corpora, and an honest statement about holdout

| role | corpus | why |
|---|---|---|
| **development** | **RWC (328)** | full-length, and the one corpus where the octave is the *dominant* failure rather than recall; already spent on the freeze threshold, so nothing fresh is burned choosing here |
| **transfer** | **Harmonix (581)** | opened once, after the formula, `τ`, and every matched-cost parameter are committed to the repo at a named hash |
| diagnostic | GTZAN (999) | reported without gates |

Formula, normalisation, window, `τ`, and the matched-cost parameters are **all**
fixed on RWC and committed before Harmonix is read. The commit hash goes in the
results file.

**The order of work, so that "before" is a fact in the git history and not a
recollection:**

1. replay, event extraction, and I1–I7 — no corpus touched;
2. full baseline parity, all five sequences;
3. RWC; `τ` and every matched-cost parameter chosen by §7's procedure and
   committed **as numbers, in their own commit**;
4. Harmonix, opened once, read under those numbers;
5. GTZAN, diagnostic, reported without gates.

**There is no untouched corpus and this document will not pretend otherwise.**
RWC, Harmonix and GTZAN have all influenced past decisions. A pass here
therefore licenses **implementation and a further confirmatory run on real live
recordings**, which do not exist yet. It does not license a product claim.

## 9. Acceptance conditions

All four, restated in full, because the previous document's A2 is also invalid —
it was measured on an annotated grid, a whole-recording window and an unmatched
null, and does not carry over.

| | condition | where |
|---|---|---|
| **A1** | at least **one third of the 19.1-point oracle gap** recovered — Harmonix episode-freeness ≥ **47.9%** (baseline 41.5%, oracle-bar 60.59%) — **and** ahead of the best matched-cost policy | transfer |
| **A2** | **balanced accuracy** on labelled events at least **15 points** over the same policy driven by the shifted null, and the lower bound of the paired cluster-bootstrap 95% interval on that difference above **0** | both |
| **A3** | at most **5%** of `wrong → correct` events vetoed | both |
| **A4** | `τ` fixed on RWC transfers **as a number**, with A1–A3 holding on Harmonix under it | transfer |

**A2 is balanced accuracy — the mean of the veto rate on `correct → wrong`
events and the allow rate on `wrong → correct` events — and not plain accuracy.**
The two classes will not be balanced, and plain accuracy would let a decoder win
by agreeing with whichever is commoner. A decoder that vetoes everything and a
decoder that allows everything both score 50% here, which is what a chance level
is supposed to do.

**A3 counts `wrong → correct` events, not committed-correct ones**, and an
earlier draft of this table had the wrong denominator. A block-only policy
cannot damage a committed-correct state at all: vetoing `correct → wrong` is the
right action, and both ambiguous classes are level-neutral by construction. The
one thing this policy can break is an escape from a wrong level, so that is what
the 5% bounds.

**Plus the standing cost gates**, unchanged from the three previous
pre-registrations so a fourth proposal is judged against the same goalposts, all
re-measured for fold 1 in the same run:

| Harmonix | baseline | to accept |
|---|---:|---:|
| usable, strictly | 26.2% | ≥ 30% |
| correct time (eligible, mean) | 77.5% | ≥ 75% |
| switches / eligible 5 min | 4.21 | ≤ 4.21 |
| settle P90 | 36.61 s | ≤ 36.61 s |
| beat F | 0.7953 | ≥ 0.785 |

**Failing any acceptance gate means "adoption not approved".** Not "mixed", not
"promising".

**Failing any of A1–A4 closes the downbeat head for octave correction
permanently and without reservation.** The protocol will then have been executed
as written, on predicted grids, with a matched null, over real decision points,
against matched-cost alternatives — every objection raised against the previous
run answered. There is no fifth document.

## 10. Statistics

The primary endpoint is **paired per recording**, exact two-sided binomial sign
test on discordant pairs, α 0.05.

**The Holm family is these three tests and no others**, enumerated here so its
membership cannot be adjusted after the run:

1. decoder vs **baseline**, Harmonix, `no_wrong_level_episode_fraction`, paired
   sign test;
2. decoder vs the **best matched-cost policy**, Harmonix, same endpoint, same
   test;
3. decoder vs the **shift-driven control**, Harmonix, balanced accuracy, paired
   cluster bootstrap.

All three are on the transfer corpus. RWC is development and its tests are
descriptive. A3 is a bound, not a test. A4 is a conditional restatement of
A1–A3 and adds none. `oracle bar` is excluded: it cannot be adopted whatever it
shows, and buying it a correction would only inflate the three that matter.

**Events are clustered by recording and are not treated as independent.** A2 and
A3 are computed as per-recording rates and aggregated across recordings; the
per-event counts are reported descriptively and carry no interval. Intervals on
A2 come from a **cluster bootstrap resampling recordings, not events**, 10 000
resamples, paired against the control on the same resampled recordings. One
pathological recording contributing forty proposals must not be able to decide
the experiment.

`oracle bar` is descriptive and is **not** in the correction family: it cannot
be adopted whatever it shows.

## 11. Predictions

- **P1.** The decoder clears A2 on RWC. If the previous run's negative was
  wholly a geometry artefact, removing the geometry has to show here first.
- **P2.** `Δ_raw` and `Δ` agree in sign on **more than 90%** of events. The null
  subtraction should be correcting a bias, not carrying the signal; if it is
  carrying the signal, `Δ` is measuring the shift and not the channel.
- **P3.** A3 is the condition most at risk, and the tension is internal to the
  decoder rather than between two populations: the same `Δ > τ` that usefully
  blocks `correct → wrong` also blocks some `wrong → correct`, because in both
  the evidence favours the committed grid and in the second it is wrong to. 5%
  is tight against that.
- **P4.** The best matched-cost simple policy is **debounce**, and it is close.
  Most of what a veto policy buys is available from waiting.
- **P5.** Ambiguous events outnumber labelled ones on GTZAN and not on RWC —
  30-second excerpts rarely contain a settled wrong-level state to switch out of.
- **P6.** If A1 passes, the gain is concentrated in the `wrong → correct`
  direction being *allowed* rather than the `correct → wrong` direction being
  *vetoed*, because the freeze result already showed that blocking alone moves
  nothing.

## 12. What would sink this

- Any of A1–A4 missed, on the corpus named for it.
- Any standing cost gate failed. Restated for the fourth time because the
  ensemble experiment showed how readily a shorter list of ways to fail gets
  read as a softer alternative to the gates.
- I1 or I4 failing — the decoder is then the previous decoder.
- `Δ` and `Δ_raw` disagreeing on more than 10% of events with `Δ` winning: the
  statistic would be reading its own null rather than the downbeat channel.
- Ambiguous events dominating both scored corpora.
- The replay failing byte-identity against the live core on any of the five
  sequences named in **The system under test**.
- A comparison policy's cost drifting more than 0.5 points from the decoder's on
  Harmonix — the matched comparison would not exist, and it is not rebuilt there.

## 13. What follows either way

**If it passes:** a small `allow`/`veto` resolver in the live core, with its own
`confidence` and `reason` fields so a decision is attributable in a log, and a
confirmatory run on real live recordings before any product claim.

**If it fails:** the downbeat head is closed for the octave. It is kept for the
metre and for bar-line placement, where it reads 82.9% against a 30.1% null and
where the offline resolver currently scores 0.417 F on GTZAN — a separate
experiment that must not touch BPM.

And the octave then needs a different live front end: not another accumulator of
periodicity, but something conditioned on the candidate that can see the
asymmetry between adjacent beats — spectral shape, kick against snare, the
alternation of accent strength. Periodic and phase-based methods lose exactly
that asymmetry, which is why three of them in a row have now failed on the same
question.

---

## Deviations found during implementation, 2026-08-06

Appended, not edited into the text above. Everything before this line is what
was registered; everything below is what implementing the decoder revealed,
recorded **before any corpus was touched**. `eval/octave_veto.py` and
`tests/test_octave_veto.py` are the code; 39 tests pass.

### I2 is false as written, and the mechanism matters more than the failure

Registered: *a beat-only channel yields `|Δ|` below 0.05.*
Measured: **`Δ = −2.02`** at `k = +1`. Not a near miss.

The arithmetic, which is checkable and is in the test file:

- a channel high at every committed beat is **constant** when sampled on the
  committed grid, so `sd = 0`, every `z` is 0 by §3's own degenerate rule, and
  `score_committed = 0`;
- the same channel on the doubled grid alternates high, low, high, low — a
  perfect metre-2 contrast at `z = 5.57`;
- so the decoder prefers the doubled grid and **allows** the doubling.

Metre 2 on a grid of half-beats implies a bar every committed beat: 0.50 s, or
**120 bars a minute** at the 120 BPM this was constructed at. That is not a bar
in any music this product is for. §3 defines the score as a maximum over
(metre, phase) with no admissibility constraint, so I2 contradicted §3, and **I2
is the half that was wrong.**

**The obvious repair is refused, deliberately.** Constraining the implied bar
period to something musical would decide the octave by arithmetic on the
committed beat period — the quantity in dispute — and
`PREREGISTERED_downbeat_channel.md` refused to break its ties with the tempo
prior for exactly this reason: it converts independent evidence into a
re-reading of the belief that sank the octave freeze. A constraint that would
become the decoder's main mechanism, derived from the disputed quantity, is not
a cleanup.

**What replaces I2**, keeping what it was for:

- **the flat-channel test**, which is the geometry question I2 was actually
  built to ask, and which holds *exactly*: `Δ == 0.0`, not merely small, in both
  directions. Geometry alone manufactures nothing.
- **the beat-only case as a named limitation with a direction**, asserted at its
  measured value rather than at the predicted one.

### What that limitation predicts, registered now

The failure produces false **allows**, never false vetoes. **A3 cannot see it** —
A3 bounds vetoes of `wrong → correct` events — and A1 will pay for it silently.
So two things are added, before the corpus:

- **D1, a diagnostic:** the share of `correct → wrong` doubling events where
  `score_committed == 0`, i.e. where the committed grid was near-constant and
  the decoder allowed a wrong doubling for the reason above. Reported per
  corpus. Not a gate.
- **P7, a prediction:** D1 is non-trivial — above 5% — on both corpora, because
  a downbeat channel roughly uniform across beats is what weak-downbeat music
  produces, and that is common.

If A1 fails and D1 is large, the failure is attributable to this and not to the
channel being empty. That distinction is worth having in advance, and it is
exactly the distinction the previous audit could not make.

### Three smaller specification gaps, all closed the same way

1. **Ties between metres were unspecified.** {2, 4} and {3, 6} are each closed
   under doubling, so a bar pattern at 2 scores identically at 4 and ties are
   routine rather than exotic. **Ties go to the smaller metre**, because
   reporting the larger would name a metre the evidence never distinguished.
2. **`window_beats` is a parameter, not the module constant.** I4 varies the
   window, and a test that mutates global state to do so is testing the harness.
3. **The `k = -1` maximum over two alignments is confirmed as the one real
   asymmetry**, and the null is confirmed as what cancels it: on iid noise
   `mean(Δ_raw) = −0.19` while `mean(Δ) = −0.02`. That is now its own test.

### Two magnitudes worth having on record before the run

- **The null subtraction is not a small correction.** On the clean 4/4 case
  `Δ_raw = +0.37` while `Δ = +1.97` — same sign, five times the magnitude. P2
  predicts sign agreement above 90% and says nothing about magnitude; that was
  the right thing to predict, and this is why a magnitude prediction would have
  been wrong.
- **The allow signal is an order of magnitude weaker than the veto signal.**
  I6 gives `Δ = −0.18` where I3 gives `+1.97`. This bounds where `τ` can
  usefully sit and it sharpens P6: if the decoder helps, it will be by allowing,
  and the allows are quiet.
