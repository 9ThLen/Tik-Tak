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
re-analysis. Before any measurement, one check: beat times produced under replay
must be **byte-identical** to beat times produced by the live core on the same
audio. A replay that cannot reproduce the baseline cannot evaluate a change to
it.

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
  mean of `Δ` is within 0.05 of zero and the veto rate is within 2 points of the
  rate `τ` implies under symmetry. Neither grid wins systematically.
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

**I1 and I4 are written first.** They are the specific defect the last run had,
and a decoder that passes I3 and I5 while failing them is the last run again
wearing a new formula.

## 7. Matched-cost baselines

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

**Matched on: retained correct locked-time.** Each simple policy's parameter is
chosen on the development corpus so that its retained correct locked-time equals
the decoder's within 0.5 points. Then frozen. Nothing is re-tuned on the
transfer corpus.

**Primary comparison: the decoder against the single best simple policy at
matched cost.** The rest are diagnostic. This does not create new acceptance
gates; it replaces "better than doing nothing" with "better than doing less".

Every result reports all three of these together, never the first alone:

1. how many wrong-level episodes were removed;
2. how much correct locked time was lost;
3. how many already-correct states were spoiled.

## 8. Corpora, and an honest statement about holdout

| role | corpus | why |
|---|---|---|
| **development** | **RWC (328)** | full-length, and the one corpus where the octave is the *dominant* failure rather than recall; already spent on the freeze threshold, so nothing fresh is burned choosing here |
| **transfer** | **Harmonix (581)** | opened once, after the formula, `τ`, and every matched-cost parameter are committed to the repo at a named hash |
| diagnostic | GTZAN (999) | reported without gates |

Formula, normalisation, window, `τ`, and the matched-cost parameters are **all**
fixed on RWC and committed before Harmonix is read. The commit hash goes in the
results file.

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
| **A2** | veto accuracy on labelled events at least **15 points** over the same policy driven by the shifted null, non-overlapping 95% intervals, clustered by recording | both |
| **A3** | at most **5%** false veto on committed-correct events | both |
| **A4** | `τ` fixed on RWC transfers **as a number**, with A1–A3 holding on Harmonix under it | transfer |

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
test on discordant pairs, α 0.05, Holm-corrected over the family reported.

**Events are clustered by recording and are not treated as independent.** A2 and
A3 are computed as per-recording rates and aggregated across recordings; the
per-event counts are reported descriptively and carry no interval. One
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
- **P3.** A3 is the condition most at risk. A decoder that vetoes usefully on
  wrong-committed states will also veto some correct switches, and 5% is tight.
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
- The replay failing byte-identity against the live core.

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
