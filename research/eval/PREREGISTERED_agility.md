# Pre-registered: does filter agility help or hurt, and does the observation decide?

Written before the script exists. Nothing below was chosen after seeing a number
it is measured against.

## The contradiction this exists to resolve

Two measurements in this repository disagree about the same knob, and they were
taken on different corpora, so neither can correct the other.

**On a perfect observation, more agility helps** — `oracle_activation.json`,
RWC, recall under the oracle bump:

| corpus | bump | bump_no_anchor | bump_r0.08 |
|---|---:|---:|---:|
| RWC-Classical | 0.560 | 0.648 | **0.688** |
| RWC-Genre | 0.838 | **0.879** | 0.878 |
| RWC-Jazz | 0.853 | 0.861 | **0.900** |

**On the real observation, more agility hurts, monotonically** —
`live_usable_rough*.json` and `live_usable_no_anchor.json`, ballroom, gtzan,
root and smc:

| setting | usable | F | coverage |
|---|---:|---:|---:|
| roughening 0.02 | 0.333 | 0.659 | 0.992 |
| roughening 0.08 | **0.243** | 0.615 | 0.960 |
| no anchor | 0.251 | 0.608 | **0.872** |

The core ships 0.01. **RWC-Classical, where the help was largest, does not
appear in the real sweep at all.** So the apparent sign flip may be an
interaction between a noisy observation and an agile filter, or it may be that
the two sets of corpora are simply different music.

The comment in `oracle_activation.py` says why the sweep was added at all: *"if
a knob we already have recovers the loss, no new decoder is needed, and if it
does not, the limit is structural"*. That question is still open because it was
never asked on one corpus.

## The measurement

Both arms, one set of corpora, one grid, one scorer.

- **corpora** — RWC, all five sub-corpora, 328 recordings. Chosen because it is
  the only annotated full-length set here that spans pop, genre, jazz and
  classical, and because classical is the corpus the real sweep was missing.
- **arms** — `real`, the shipped BeatNet activation; `oracle`, the
  equal-height bump from `oracle_activation.synthesise`, written once per
  recording and reused across every setting so the observation is byte-identical
  and only the filter differs.
- **settings** — `shipped` (core defaults), `--live-roughening` at 0.02, 0.04
  and 0.08, and `--live-no-anchor`. Five, matching the two runs above.
- **scorer** — `live_corpus_benchmark._score_one`, the verdict every published
  live number came from, sampled at 50 Hz as `oracle_usable.py` samples it.

That is 328 x 2 x 5 dumps.

## Primary readout

Change in `usable_rate` against `shipped`, computed **separately for each arm**,
reported for RWC as a whole and per sub-corpus.

## The registered prediction, so this can fail

**The sign flip survives matched corpora.** Concretely:

* on the `real` arm, every raised setting is at or below `shipped`;
* on the `oracle` arm, at least one raised setting is above `shipped`.

If both arms move the same way, the cross-corpus comparison that produced the
flip was an artefact, and "a noisy front end plus an agile filter is unstable"
loses its only evidence. If the flip survives, that reading is supported on one
corpus and the knob is confirmed as unavailable until the observation improves.

A third outcome is possible and would be the most useful: the flip survives
overall but reverses on a sub-corpus. That would say agility is available on
some material and not others, which is a different and better answer than
either.

## Secondary, reported and not gated

`p70`, `r70`, F, coverage, worst wrong-octave seconds, acquisition, and the
failure-reason decomposition. These are where the cost of agility on a noisy
observation should show if it is there — extra beats, a collapsing coverage, a
level that wanders — and reading them as endpoints after the fact would be
choosing a result.

## What no outcome licenses

**Not a default change.** This measures a knob on research corpora. Moving the
shipped configuration needs its own registration against the live usable
criteria, the way the anchor width and the octave button were each measured
before adoption was considered. Nothing here is an adoption decision.

**Not a claim about rooms.** Every recording is a clean file. The room result
says the observation is where the damage arrives, and this run cannot speak to
whether agility behaves differently on a degraded one.

## Bound in advance

The grid, the corpora and the primary readout are fixed here. If the run has to
be cut short, sub-corpora are dropped whole and the artifact says which — never
recordings sampled out of one.
