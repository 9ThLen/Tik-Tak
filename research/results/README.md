# What is in here, and what each number is allowed to say

Artifacts written by the scripts in `research/eval/`. Every one carries a
`provenance` block: the commit, whether the tree was clean, SHA-256 of the
binary and of each model, and the per-corpus counts. **`tree_clean: false` means
the commit does not identify the binary that produced the number.** Committing
afterwards does not repair that — the run has to be repeated.

Which corpus a number came from decides what it can be used for, and this is not
a formality here:

| corpus | status | what it can answer |
|---|---|---|
| ballroom | in `beatnet_model_1`'s training set | nothing about performance |
| GTZAN | out of fold 1's training, in folds 2 and 3's | fold 1 only |
| SMC | out of all three | anything, but at 3.2% usable it has no resolution |
| RWC | out of all three — **development corpus since 2026-08-03** | debugging, factorization, regression; not confirmation |
| Harmonix | out of all three — **spent 2026-08-04** on the pre-registered ensemble test | that one hypothesis, honestly; a development corpus from now on |

RWC became a development corpus the moment `anchor_width_octaves` was chosen by
looking at it and the averaged activation was taken seriously after seeing its
scores. It is still the most useful corpus here for finding out *why* something
fails. It can no longer say that a chosen configuration is good.

**An averaged activation narrows that table further, and permanently.** The row
for GTZAN says "fold 1 only" for a reason: folds 2 and 3 were trained on GTZAN,
and folds 1 and 3 on Ballroom. So a mean of the three is train-on-test on both,
and the shipped single fold may be quoted on GTZAN where the ensemble may not —
they are not comparable there at all. Adopting `EnsembleMean` retires 1,697 of
the 2,760 annotated recordings here as evaluation ground, leaving Harmonix,
RWC and SMC. That is a cost of the ensemble, not merely of testing it, and it is
the strongest argument for recording new material.

## The octave residue is real: accenting a perfect observation does not fix it

`accented_oracle_rwc.json`, answering `eval/PREREGISTERED_accented_oracle.md`.
All 328 RWC recordings, four arms on the same synthesised activation, no
failures, clean tree at `785b95b`.

Both oracle runs ended with `wrong_octave` as almost the only surviving failure,
and both recorded the reason that might prove nothing: the bump is the same
height on every beat, so it removes the amplitude difference that tells a level
from its double. This tested that excuse.

| corpus | n | `flat` | `accent_0.5` | `accent_0.25` | `accent_0.5_shuffled` |
|---|---:|---:|---:|---:|---:|
| RWC-Classical | 61 | 0.033 | 0.033 | **0.000** | 0.016 |
| RWC-Genre | 102 | 0.569 | 0.559 | 0.549 | 0.578 |
| RWC-Jazz | 50 | 0.520 | 0.480 | **0.380** | 0.500 |
| RWC-Pop | 100 | 0.810 | 0.830 | 0.800 | **0.840** |
| RWC royalty-free | 15 | 0.867 | 0.867 | 0.867 | 0.867 |
| **all** | 328 | **0.549** | **0.546** | **0.512** | **0.555** |

`flat` reproduces 0.549 exactly, which is the validity check the registration
demanded before any other column could be read.

**The registered condition fails on both halves.** The better true accent gains
**−0.003** against a bar of +0.05, and the deeper accent loses 0.037. Accenting
a perfect observation does not help; it hurts, and more accent hurts more.

### The control is the finding

`accent_0.5_shuffled` — the same amplitude pattern applied to the **wrong**
beats, bar phase rotated by an offset from the recording's own name — scores
**+0.006**, above the correctly aligned accent's −0.003.

The tracker does not use where the accents are. Misaligned accents do
marginally better than aligned ones, so what little moves is amplitude variation
and not metre. This is the outcome the registration named in advance as the one
the control existed to catch, and it is the second time in this repository that
an octave arm has come in behind its own shuffled control.

### Why the octave share falls while the result gets worse

| arm | usable | failures | `wrong_octave` share | absolute octave failures | mean F |
|---|---:|---:|---:|---:|---:|
| `flat` | 0.549 | 148 | 0.899 | ~133 | 0.846 |
| `accent_0.5` | 0.546 | 149 | 0.832 | ~124 | 0.808 |
| `accent_0.25` | 0.512 | **160** | 0.744 | ~119 | **0.786** |

The share of failures blamed on the level falls, and reading that alone would
suggest accents help. They do not: total failures rise from 148 to 160 and mean
F falls by 0.060. Accenting buys about fourteen fewer octave failures and pays
about twenty-six more of everything else, because lowering three beats in four
takes signal out of the beat channel. A share is a ratio, and this one improved
by growing its denominator.

### What this settles, and what it costs

**The residue is not an instrument artefact.** A perfect observation — even one
carrying clean bar-level accents — leaves the metrical level unresolved. The
caveat both oracle sections carried is now discharged: their octave residues
stand.

That makes five independent negatives on the octave: the anchor-margin gate, the
freeze and abstain policies, the downbeat head, the octave button, and now the
accented oracle. Nothing tried on either the observation side or the decoder
side has moved it.

**So a front end must not be expected to fix the level by producing better
beats.** Train it for what the oracle says is there — Harmonix `usable` 0.365 to
0.952, precision failures gone entirely — and treat the octave as a separate
unsolved problem, which the ×2/÷2 control already addresses from the user's
side.

### The limit of this test, named rather than left implicit

The accent measured is **bar-level**: downbeats at full height, every other beat
scaled. That is the accent the annotations support and the one registered. A
**beat-level** strong-weak alternation — the cue that would speak directly to P
against 2P rather than to the bar — was not tested and is not answered here.
It would need its own registration, and after five negatives it needs a reason
to expect a different answer before it earns one.

## The causal teacher gate: half the advantage survives, and it does not reach the product yet

`causal_teacher_gtzan.json`, answering `eval/PREREGISTERED_causal_teacher.md`.
Forty GTZAN clips on a fixed stride, 297 prefix passes each, no failures, clean
tree at `c77294f`. Every arm — BeatNet included, via `--dump-activation` and
replay — enters the same `LiveTracker` through `--live-activation`, so nothing
here is a difference of delivery.

| arm | mean F | usable | share of the advantage |
|---|---:|---:|---:|
| `beatnet` | 0.7112 | 0.525 | 0.000 |
| `at_most_0.1s` | 0.7841 | **0.525** | **0.533** |
| `at_most_0.2s` | 0.8095 | 0.600 | 0.718 |
| `at_most_0.3s` | 0.7949 | 0.600 | 0.611 |
| `at_most_0.5s` | 0.8060 | 0.600 | 0.693 |
| `offline` | 0.8481 | 0.725 | 1.000 |

**The registered condition is met.** At the tightest bound, 53.3% of the
teacher's advantage survives, against a registered bar of 50%. The advantage
itself measures +0.1369, which reproduces the +0.138 obtained independently by
`beat_this_front_end` — a consistency check across two scripts and two runs.

So most of Beat This!'s edge is the model rather than the lookahead, and a
causal student has something real to be aimed at. That was the last open gate
before training.

### The pass is narrower than the headline

`usable` does not move at the tightest bound: 0.525 for BeatNet and **0.525**
for `at_most_0.1s`, on the same forty clips. F rises by 0.073 and the product
verdict does not change at all. Usability only appears at `at_most_0.2s` and
even then reaches 0.600 against the unbounded 0.725.

The live metronome's own lookahead is 50 ms before buffer and round trip, so
`at_most_0.2s` is not obviously affordable. What the gate licenses is training;
it does not promise that matching a bounded teacher would be felt by a user.

Why the F gain does not convert is visible in what still fails:

| arm | failing | `too_few_beats` | `wrong_beats` | `wrong_octave` |
|---|---:|---:|---:|---:|
| `beatnet` | 19/40 | 1.00 | 0.95 | 0.53 |
| `at_most_0.1s` | 19/40 | 0.89 | 0.79 | 0.37 |
| `offline` | 11/40 | 0.91 | 0.73 | 0.45 |

Precision and recall both improve — p70 0.732 → 0.822, r70 0.706 → 0.769 — and
the same nineteen clips still fail, because they were failing by margins wider
than the gain. This is the same shape the Harmonix oracle showed from the other
end: a better observation moves the beat metrics first and leaves the level.

### What must not be read from this

**The curve is not monotone and n is 40.** `at_most_0.2s` scores above
`at_most_0.3s`, and `at_most_0.5s` sits below `at_most_0.2s`. At this sample
size the ordering between the three loosest arms is noise; only the gap between
the tightest arm and the ends is large enough to read.

**The level is not quotable.** `beat_this.onnx` is `final0`, trained on sixteen
sets including GTZAN, and GTZAN is BeatNet `model_1`'s held-out fold — the
comparison is maximally unfavourable to BeatNet in both directions at once. The
shape survives that because it is a comparison within one model; the height does
not.

**This is an upper bound on the architecture question.** It measures what a
non-causal model *retains* under a causal constraint, not what a causal one
would achieve.

## Agility: the sign flip is real, the knob is not a lever, and no-anchor was misread

`agility_sweep_rwc.json`, answering `eval/PREREGISTERED_agility.md`. All 328 RWC
recordings, both arms, five settings, one scorer, no failures, clean tree at
`2c790e0`. The oracle bump is written once per recording and reused across every
setting, so only the filter differs.

### The registered prediction holds

| arm | shipped | r0.02 | r0.04 | r0.08 | no_anchor |
|---|---:|---:|---:|---:|---:|
| `real` | 0.207 | −0.009 | −0.012 | −0.012 | −0.061 |
| `oracle` | 0.549 | +0.000 | +0.012 | **+0.027** | −0.113 |

The real arm never rises above its own baseline and the oracle arm rises twice,
**on the same corpora**. The sign flip was not an artefact of comparing ballroom
and gtzan against RWC: raising agility helps when the observation is perfect and
does not when it is real.

### And the third registered outcome fired

| corpus | n | real, best raised | oracle, best raised |
|---|---:|---:|---:|
| RWC-Classical | 61 | +0.000 | +0.033 |
| **RWC-Genre** | 102 | **+0.020** | +0.020 |
| RWC-Jazz | 50 | +0.000 | +0.020 |
| RWC-Pop | 100 | **−0.040** | +0.050 |
| RWC royalty-free | 15 | +0.000 | +0.000 |

**On RWC-Genre the real arm rises too**, which the pooled row cannot show. So
agility is not simply unavailable on a noisy observation — it is available on
some material and not on others, and pop is where it is most clearly not. The
pooled statement and the sub-corpus statement are both true and they are not in
conflict.

### The mechanism: agility buys beats and spends the metrical level

| arm | setting | usable | F | coverage | worst wrong octave |
|---|---|---:|---:|---:|---:|
| `real` | shipped | 0.207 | 0.601 | 1.005 | 19.5 s |
| `real` | r0.08 | **0.195** | **0.614** | 1.048 | **20.6 s** |
| `oracle` | shipped | 0.549 | 0.846 | 0.955 | 6.1 s |
| `oracle` | r0.08 | **0.576** | **0.882** | 0.995 | **5.5 s** |

On the real observation a more agile filter measurably **improves the beat
metrics** — F rises 0.601 to 0.614 — and still loses `usable`, because it also
spends a second longer at the wrong level. On a perfect observation both move
the right way at once. That is the trade, and it explains the flip without
appealing to instability in general.

### `bump_no_anchor` was a recall result and it does not survive the verdict

`oracle_activation.json` reported that switching the anchor off *raised* recall
under a perfect observation — 0.560 to 0.648 on classical, 0.838 to 0.879 on
genre — and that reading is what put `no_anchor` in this grid. Through the full
four-condition verdict it is the worst setting tested, on every corpus, and by a
wide margin on jazz (**−0.260**).

The reason is visible in the same row. Oracle arm, shipped → no_anchor:

* recall **rises**, 0.837 → 0.868;
* coverage overshoots to **1.117** — 12% more beats emitted than exist;
* worst wrong-octave time nearly **triples**, 6.1 s → 17.1 s;
* `wrong_octave` climbs from 89.9% to 97.3% of failure reasons.

Removing the anchor buys beats by giving up the level. This is the same lesson
the oracle-recall table needed and did not have, now demonstrated on a knob
rather than argued: **recall is one of four conditions, and a change that
improves it can make the product worse.**

### The answer to the question the sweep was added for

`oracle_activation.py` states it plainly: *"if a knob we already have recovers
the loss, no new decoder is needed, and if it does not, the limit is
structural"*.

It does not recover the loss. The best any raised setting does on the real
observation is **+0.020 on RWC-Genre**, two recordings in 102, and it costs
0.040 to 0.060 on pop. Nothing here is worth adopting and nothing here removes
part of the task. The limit is structural, which is the alternative that
sentence named.

## On full-length pop the front end is nearly the whole problem

`oracle_usable_harmonix.json`, the other half of the table `oracle_usable_rwc.json`
started. All 581 aligned Harmonix recordings, no failures, clean tree at
`9b20dd5`. The script is byte-identical to the one that produced the RWC run —
`git diff 82705c6 9b20dd5` over `oracle_usable.py`, `oracle_activation.py` and
`live_corpus_benchmark.py` is empty — so the two are directly comparable.

| corpus | n | real | oracle | oracle, level forgiven |
|---|---:|---:|---:|---:|
| **Harmonix** | 581 | 0.365 | **0.952** | 0.988 |
| RWC-Pop | 100 | 0.440 | 0.810 | 0.970 |
| RWC royalty-free | 15 | 0.333 | 0.867 | 0.933 |
| RWC-Genre | 102 | 0.137 | 0.569 | 0.735 |
| RWC-Jazz | 50 | 0.100 | 0.520 | 0.700 |
| RWC-Classical | 61 | 0.000 | 0.033 | 0.164 |
| RWC, all | 328 | 0.207 | 0.549 | 0.704 |

**A perfect observation takes Harmonix from 36.5% usable to 95.2%.** That is the
largest figure anywhere in this repository, and it is on the corpus that looks
most like the product's likely material: full-length popular music rather than
thirty-second excerpts.

### What the observation fixes, and what it does not

| | real | oracle |
|---|---:|---:|
| p70 | 0.798 | 0.973 |
| r70 | 0.807 | 0.993 |
| recordings failing | 369 | **28** |
| of those, `wrong_beats` | 52.6% | **absent** |
| of those, `too_few_beats` | 55.8% | 17.9% |
| of those, `wrong_octave` | 85.6% | 96.4% |

Precision failures do not merely shrink, they **disappear**: not one recording
fails on `wrong_beats` under the oracle. Recall failures fall from 55.8% of
failures to 17.9%. What is left is the metrical level, and almost nothing else:

    21  wrong_octave alone
     5  too_few_beats + wrong_octave
     1  slow_acquisition + wrong_octave
     1  slow_acquisition

**21 of 581 recordings — 3.6% — fail on the level and on nothing else.**

### This does not contradict "recall is the dominant survivor"

The octave-ceiling section found `too_few_beats` in 84.8% of Harmonix's
surviving failures and concluded recall was the binding constraint. Both hold,
because they answer different questions. Forgiving the *level* while keeping the
*real* observation leaves recall broken. Fixing the *observation* fixes recall
and leaves the level. The lever is the front end; the residue is the octave.

### Two things a reader must carry with these numbers

**They are sampled at 50 Hz.** `oracle_usable.py` passes `--live-sample-hz 50`,
so the real arm reads 0.365 — which is the 36.49% the acquisition section
measured at 50 Hz, not the 30.98% the 1 Hz baselines report. The octave-ceiling
table above (31.0% → 51.3%) is a 1 Hz table and must not be differenced against
this one.

**The oracle bump is equal-height pulses on every beat**, so it removes the
amplitude difference that tells a level from its double. Some of that 3.6%
residual is the instrument rather than the tracker, exactly as the RWC section
records. The accented-oracle control it asked for is now worth more, not less:
it is the only way to learn whether the last 3.6% is real.

## A second documented negative: repaired sparse peak front end

`peak_front_end.json`, produced by `eval.peak_front_end` at clean commit
`c67466d`, supersedes the withdrawn run in `3b2fb9c`. The repaired run regenerates
all ten feature dumps, hashes the binary, audio, annotations, dumps and alignment
artifact, and scores each clean/room pair on one shared time interval and one
shared set of reference beats.

The peak map is built separately on the 136 filterbank bands and 136 positive
differences. Channels merge only after picking, spend one refractory budget,
and resolve all equal maxima by earliest frame then lowest band. Parameters are
selected leave-one-track-out.

**Under the registered ratio calculation, conditions 1 and 2 formally pass.
Condition 3 fails under both its strict and mean readings.**

| held out | peaks degradation | dense degradation | peaks top-N | dense top-N | peaks chance |
|---|---:|---:|---:|---:|---:|
| `0116_goodies` | **-0.0417** | 0.1409 | 0.178 | 0.188 | **0.277** |
| `0132_iceicebaby` | 0.0833 | **0.0624** | 0.386 | 0.614 | 0.221 |
| `0466_onthedarkside` | **0.0333** | 0.0928 | 0.492 | 0.531 | 0.302 |
| `0707_halfwaygone` | **-0.0250** | 0.1408 | 0.348 | 0.442 | 0.191 |
| `0837_nottonight` | **-0.0143** | 0.0834 | 0.315 | 0.563 | 0.169 |
| **mean** | **0.0071** | 0.1040 | **0.344** | 0.468 | |

Conditions 1 and 2 pass only under the registered ratio calculation; that pass
must not be cited as evidence that the peak signal is robust. The median
on-beat novelty falls in the room on all five tracks (`4 -> 3`, `6 -> 4`,
`5 -> 3`, `5 -> 4`, `7 -> 5`). Three folds show negative degradation only
because their already-quantised floor falls faster than the peak. Consequently,
the mean `0.0071` is a readout artifact, not an estimate of near-perfect room
robustness, and is not a quotable result. The selected peak signal also has
worse beat top-N than dense on every track, and `0116_goodies` is below its own
signal-specific shuffled baseline. Emptying the gaps by removing evidence of
the beats is still not progress.

### The null holds under all three collapse families

Each family was selected and judged independently, as the registered null
requires:

| family | tracks improved | degradation | top-N vs dense 0.468 | result |
|---|---:|---:|---:|---|
| `count` | 2/5 | 0.1038 | 0.475 | fails conditions 1, 2 and strict top-N |
| `weighted` | 5/5 | 0.0404 | 0.369 | fails top-N |
| `novelty` | 4/5 | 0.0071 | 0.344 | fails top-N |

The count mean is slightly above dense, but condition 3 binds per track and it
also fails its own chance gate on at least one track. The symmetric upper bound
also fails: it improves only three tracks and scores top-N 0.412 against 0.468.
No causal pooled or family fold now has an identical clean/room ratio, so the
negative no longer rests on the quantised-median defect seen in the withdrawn
run.

The historical caveat remains: the strict reading of condition 3 was adopted
after a smoke run exposed an ambiguity in the wording, so it is not itself
preregistered. That ambiguity does not decide this result because the mean
reading fails too.

## A documented negative: adaptive whitening in the room

`whitening_room.json`, produced by `eval/whitening_room.py`. Five matched
Harmonix clean/room pairs, 30 runs in total, commit `3dba708`, clean tree. The
artifact carries SHA-256 for the binary, ten audio files and five annotation
files.

The endpoint is the ratio between the median ODF floor in the middle third of
inter-beat gaps and the median peak within 70 ms of an annotated beat. Lower is
better.

| condition | whitening off | shipped 0.5 | full 1.0 |
|---|---:|---:|---:|
| clean | 0.1886 | 0.1946 | 0.1931 |
| room | **0.3405** | 0.3627 | 0.3994 |

Disabling whitening beats the shipped exponent on all five room captures;
full whitening is worse than shipped on four of five. From off to full, the
room peak loses 48.9% while the floor loses only 37.3%. The running per-band
denominator follows the reverberant tail and compresses the next beat more than
the smear it was meant to remove. This retires adaptive whitening as the fix
for room gap-filling; it does not test sparse time-frequency peaks, which use
no level-following denominator.

## The baseline every later arm is measured against

`live_baseline_gtzan_family.json`, `live_baseline_rwc.json`,
`live_baseline_harmonix.json`. Commit `4422afc`, clean tree, nothing dropped
(1914 of 1915, 328 of 328, 581 of 581). The shipped configuration: fold 1,
`anchor_width_octaves` 0.02, the core's own front end.

```bash
research/.venv/Scripts/python.exe -m eval.live_corpus_benchmark --mode model --model models/beatnet_model_1.ttw --workers 8 --output results/live_baseline_gtzan_family.json
```

with `--manifest music/rwc2/manifest.csv --music music/rwc2` for RWC and
`--corpora harmonix` for Harmonix.

| corpus | n | no episode >4 s | usable | strict | correct time | longest run | settle P50/P90 | sw / 5 min | never settled | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ballroom¹ | 698 | 78.1% | 60.9% | 58.2% | 80.8% | 24 s | 7.0 / 13.6 | 9.14 | 10.5% | 0.820 |
| GTZAN | 999 | 67.9% | 44.5% | 43.6% | 67.7% | 23 s | 5.0 / 14.0 | 5.34 | 24.0% | 0.685 |
| SMC | 217 | 28.1% | 3.2% | 3.2% | 13.9% | 0 s | 12.0 / 31.0 | 3.82 | 80.2% | 0.228 |
| RWC-Pop | 100 | 47.0% | 39.0% | 38.0% | 81.0% | 147 s | 7.0 / 32.5 | 5.74 | 4.0% | 0.799 |
| RWC-Genre | 102 | 19.6% | 12.7% | 12.7% | 55.1% | 65 s | 8.0 / 66.8 | 4.62 | 22.5% | 0.583 |
| RWC-Jazz | 50 | 16.0% | 8.0% | 8.0% | 49.5% | 30 s | 15.0 / 63.4 | 6.27 | 14.0% | 0.539 |
| RWC-Classical | 61 | 1.6% | 0.0% | 0.0% | 20.2% | 7 s | 31.0 / 204.8 | 6.24 | 39.3% | 0.352 |
| **Harmonix** | 581 | **41.5%** | **31.0%** | **26.2%** | **77.5%** | 114 s | 8.0 / 36.6 | 4.21 | 4.5% | 0.795 |

¹ in fold 1's training set — not quotable as performance, present only because
the GTZAN-family run produces it.

**Average correctness is not the binding constraint, and has not been for some
time.** On Harmonix the tracker is right for 77.5% of the time after warm-up and
usable on 31.0% of recordings; on RWC-Pop, 81.0% against 39.0%. Those describe
the same runs. The reconciliation is in the same table: the median longest
correct run is 114 seconds, and 58.5% of Harmonix has at least one slip to the
wrong level lasting more than four seconds. A recording that is right for 95% of
its length fails on the other 5% if that 5% is contiguous. So a target of
"raise correct time above 70%" would have been asking for a number that already
passes by seven points.

`no_wrong_level_episode_fraction` is therefore the primary endpoint from here,
and the target table is written around episode-freeness rather than averages.

Three denominators are reported for correct time and they are not
interchangeable — the column above is the mean over recordings. On Harmonix the
three read 77.5% (mean over recordings), 76.8% (pooled over seconds) and 82.1%
(over *locked* time only, which is the old `active_state_shares.same`). The last
is the flattering one, because silence leaves its own denominator; a plan was
recently built around a `64.6%` whose denominator nobody could name.

SMC is not a hard corpus so much as one the tracker never starts on: 80.2% never
settle and the median longest correct run is zero seconds. Its oracle ceiling is
4.1%, so it has no resolution to lend any comparison.

## The pre-registered test: does averaging the three folds hold up out of sample

`beatnet_ensemble_harmonix.json`. 581 full-length recordings, `attempted 581,
dropped 0`, commit `e6cf8bd`, clean tree. The protocol and the four predictions
were fixed in `eval/PREREGISTERED_harmonix_ensemble.md` before the corpus was
looked at; primary endpoint `usable_strict`, primary comparison mean against
fold 1, α 0.05 uncorrected because the hypothesis was fixed in advance.

| arm | usable | strict | any level | F | CMLt |
|---|---|---|---|---|---|
| fold 1 — ships today | 31.7% | 27.5% | 52.7% | 0.8027 | 0.6911 |
| fold 2 | 31.2% | 25.8% | 51.5% | 0.7970 | 0.7008 |
| fold 3 | 32.5% | 25.6% | 52.7% | 0.7872 | 0.6809 |
| **mean** | **38.7%** | **33.0%** | **60.2%** | **0.8445** | **0.7404** |
| max | 28.6% | 22.2% | 41.1% | 0.7564 | 0.6239 |

Every comparison the mean makes is significant after Holm correction over all
eight:

| the mean against | criterion | won | lost | p |
|---|---|---|---|---|
| max | strictly | 72 | 9 | <1e-6 |
| max | usable | 79 | 20 | <1e-6 |
| fold 2 | strictly | 59 | 17 | 1e-6 |
| fold 3 | strictly | 62 | 19 | 2e-6 |
| fold 2 | usable | 70 | 26 | 8e-6 |
| fold 1 | usable | 69 | 28 | 3.8e-5 |
| **fold 1** | **strictly (primary)** | **54** | **22** | **3.1e-4** |
| fold 3 | usable | 67 | 31 | 3.6e-4 |

**Two of the four predictions were wrong**, both because the ensemble did better
than expected:

| | prediction | result | |
|---|---|---|---|
| P1 (primary) | mean beats fold 1 on strict, p<0.05 | p = 3.1e-4 | ✅ |
| P2 | margin smaller than RWC's 5.3 pts | 5.5 pts | ❌ |
| P3 | mean strict in 30–50% | 33.0% | ✅ |
| P4 | mean does not beat fold 3 | p = 2e-6 | ❌ |

P4's failure retires a claim: **"fold 1 is the weakest" does not replicate.** On
RWC the folds spread 14.7 / 18.2 / 18.7; here they sit inside 1.3 points with
fold 1 in the middle. That ranking was corpus-specific noise. What replicates,
and more strongly out of sample, is that the mean beats all of them — and since
there is no best fold to pick, no corpus need be spent picking one.

P2's failure retires a worry rather than a claim. The margin was expected to
shrink because RWC had chosen the width; it grew from 5.3 to 5.5 points. The
premise was wrong: a width chosen on RWC moves every arm together, so it biases
the absolute level and not a fold-against-mean contrast.

### Where the remaining distance is

Same run, same best configuration, why the 581 recordings fail — shares of the
whole corpus, so they overlap:

| | mean | fold 1 |
|---|---|---|
| wrong metrical level over 4 s | **49.4%** | 59.2% |
| wrong beats (precision) | 24.1% | 32.9% |
| too few beats (recall) | 24.1% | 31.7% |
| slow to acquire | 17.4% | 15.0% |

The level is twice the next failure, and forgiving it outright is worth
**21.5 points** (38.7% → 60.2%). See `core/src/tracking/live.hpp` for why the
next thing to try is giving `LiveTracker` the downbeat channel it currently
discards.

## The three BeatNet folds and their average, on RWC

`beatnet_ensemble_rwc.json`, produced by `eval/beatnet_ensemble.py`. 328
full-length recordings, `anchor_width_octaves` 0.02.

BeatNet publishes three checkpoints, each withholding a different corpus. Only
fold 1 had ever been measured here, for no better reason than that it was
fetched first.

Run natively, through `--live-model`, which is what would ship:

| fold | usable | strict | any level | F |
|---|---|---|---|---|
| 1 — ships today | 14.9% | 14.7% | 24.6% | 0.6015 |
| 2 | 18.1% | 16.9% | 28.8% | 0.6140 |
| 3 | 18.7% | 16.2% | 27.4% | 0.6033 |

Every arm through one seam, `--live-activation`, so the mean is compared against
the folds on the same code path rather than across two. Each fold lands within
0.4 points of its native run above, which is the control that makes the last two
rows readable:

| arm | usable | strict | any level | F | CMLt |
|---|---|---|---|---|---|
| fold 1 | 14.7% | 13.9% | 24.6% | 0.6127 | 0.4786 |
| fold 2 | 18.2% | 16.2% | 29.3% | 0.6238 | 0.4776 |
| fold 3 | 18.7% | 16.4% | 28.9% | 0.6120 | 0.4737 |
| **mean** | **20.6%** | **19.2%** | **33.7%** | **0.6499** | **0.5100** |
| max | 15.2% | 14.4% | 21.9% | 0.5899 | 0.4286 |

Per corpus, the mean is ahead on all three that reach the macro minimum —
genre 20.6% against the best fold's 19.6%, jazz 18.0 against 16.0, pop 44.0
against 41.0 — so it is not one corpus carrying the average. Classical is 0.0%
for every arm.

### What the rate table cannot say

Two rates two points apart over 328 recordings can be six tracks moving or forty
moving both ways. Paired over recordings, exact two-sided sign test, Holm-
corrected over the whole family of eight the harness runs:

| the mean against | criterion | won | lost | p | corrected |
|---|---|---|---|---|---|
| fold 1 | usable | 25 | 7 | .0021 | **.0168** |
| fold 1 | strictly | 25 | 9 | .0090 | .0633 |
| max | usable | 25 | 9 | .0090 | .0633 |
| max | strictly | 23 | 9 | .0201 | .1003 |
| fold 3 | strictly | 18 | 9 | .1221 | .4883 |
| fold 2 | strictly | 19 | 10 | .1360 | .4883 |
| fold 2 | usable | 19 | 12 | .2810 | .5621 |
| fold 3 | usable | 18 | 12 | .3616 | .5621 |

One row is established. **The mean beats fold 1 on the headline criterion**, and
that is what shipping fold 1 costs. The same comparison read strictly does not
survive the correction, at .0633; an earlier revision of `live.hpp` claimed it
did by quoting the loose column's correction for both. That the mean beats the
*best* single fold is not established either — 18 against 12 is churn with a
favourable sign. Folds 2 and 3 are indistinguishable from each other (+17 −16,
p 1.00 uncorrected). `max` is worse than the mean, so this is not "any pooling
helps": what a mean suppresses and a max keeps is one fold being confident and
wrong.

Recompute any of this without re-measuring:

```bash
research/.venv/Scripts/python.exe -m eval.beatnet_ensemble --from research/results/beatnet_ensemble_rwc.json
```

### Why not simply ship the best fold

Each fold withholds a *different* one of BeatNet's five training corpora — 1 →
GTZAN, 2 → Ballroom, 3 → Rock Corpus — so on any of those five the folds are not
comparable, because each has a different subset memorised. That leaves RWC and
SMC as the only corpora equally unseen by all three, and SMC has no resolution.
Choosing a fold on RWC would spend RWC on that choice.

The honest position is that RWC has *already* been spent, on the width and on
taking the mean seriously, and neither the mean nor any fold can be confirmed
there now. That is what `eval/PREREGISTERED_harmonix_ensemble.md` exists to fix.

## The pre-registered test: does the core reproduce it, and at what cost

`ensemble_in_core_{harmonix,rwc,smc}.json` against
`fold1_in_core_{harmonix,rwc,smc}.json`, verdict in
`ensemble_in_core_verdict.json`, produced by `eval/ensemble_in_core.py`. Six
arms, commit `fa781bc`, `tree_clean` true on all six, nothing dropped (581 of
581 Harmonix, 328 of 328 RWC, 217 of 217 SMC). Protocol and predictions were
fixed in `eval/PREREGISTERED_ensemble_in_core.md` before `EnsembleMean` existed.

The seam experiment above handed the tracker a pre-computed activation. This
asks the different question: **does the core, running three networks itself over
one shared front end, reproduce that gain** — and is what it costs worth what it
buys.

Read the reproduction first, because it is what makes the rest mean anything.
Every fold-1 arm reproduced the `4422afc` baselines **exactly**, to the last
printed digit, on all three corpora. The harness is deterministic and averaging
did not disturb the single-checkpoint path it shares code with.

### The six gates

| Harmonix | fold 1 | ensemble | required | |
|---|---:|---:|---:|---|
| no wrong-level episode >4 s | 41.5% | **48.2%** | ≥ 46.5%, p<.05 | ✅ |
| usable, strictly | 26.2% | 28.9% | ≥ 30% | ❌ |
| correct time (eligible, mean) | 77.5% | 79.0% | ≥ 75% | ✅ |
| switches / eligible 5 min | 4.21 | 4.46 | not above baseline | ❌ |
| settle P90 | 36.61 s | 36.81 s | not above baseline | ❌ |
| beat F | 0.7953 | **0.8300** | ≥ 0.785 | ✅ |

### Paired over recordings, Holm-corrected over all six comparisons

| corpus | endpoint | n | fold 1 | ensemble | won | lost | p | p Holm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| harmonix | no wrong-level episode >4 s | 581 | 41.5% | 48.2% | 70 | 31 | 0.0001 | **0.0008** |
| harmonix | usable, strictly | 581 | 26.2% | 28.9% | 45 | 29 | 0.0805 | 0.2415 |
| rwc | no wrong-level episode >4 s | 328 | 25.3% | 31.1% | 28 | 9 | 0.0026 | **0.0128** |
| rwc | usable, strictly | 328 | 18.0% | 22.3% | 23 | 9 | 0.0201 | 0.0802 |
| smc | no wrong-level episode >4 s | 217 | 28.1% | 25.8% | 26 | 31 | 0.5966 | 1.0000 |
| smc | usable, strictly | 217 | 3.2% | 2.8% | 0 | 1 | 1.0000 | 1.0000 |

The correction is not a formality here. RWC's strict-usability row is p = 0.020
raw and 0.080 corrected, so it is reported as having moved and **not** as having
been shown.

### The five predictions

| | prediction | outcome | |
|---|---|---|---|
| P1 | episode gate cleared on Harmonix, ≥46.5%, p<.05 | 48.2%, p_holm 0.0008 | ✅ |
| P2 | episode gain larger than the strict-usability gain | +6.7 against +2.7 pts | ✅ |
| P3 | RWC-Pop the same direction, by less than Harmonix | +5.0 against +6.7 pts | ✅ |
| P4 | SMC does not improve | −2.3 pts, p_holm 1.0 | ✅ |
| P5 | the core within 2 points of the seam | 48.2% against ~51% | ❌ |

**P5's miss is not the bug it was written to catch.** It said a discrepancy over
two points means the shared front end or the per-frame averaging is not doing
what the offline average did. The averaging is not at fault: the core's averaged
activation agrees with the mean of three separately dumped activations to 8e-6,
while the folds themselves differ by up to 0.99 on the same file. What is left
is the front end underneath, and this file already records that the two paths
differ by about a point. The core reproduces roughly half the seam's gain on
strict usability, +2.7 against +5.5, and that is a property of the ensemble on
the core's front end rather than of the arithmetic.

### The verdict: effect confirmed, adoption not approved

**The effect is real.** Episodes fall on Harmonix and again on RWC, both
surviving correction, and beat F rises 3.5 points. Averaging the folds does
what it was adopted to do.

**Adoption is not approved, because three acceptance gates failed** — strict
usability 28.9% against a 30% bound, switches 4.46 against 4.21, settle P90
36.81 s against 36.61 s. The table those come from is headed "to accept", and a
failed acceptance gate means the thing is not accepted. The separate "what
would sink this" list was hit by nothing, and an earlier version of this section
presented the two readings as an open disagreement; that was wrong. A shorter
list of ways to fail outright cannot retire the gates that were written to
decide the question, and reading it as if it could is exactly the move
pre-registration exists to prevent.

What the "what would sink this" list being clean does mean is narrower and
still worth stating: nothing here disqualifies the approach, so the gates are
worth another attempt rather than the idea being finished.

**The two failed cost gates change sign by corpus.** Switches per five minutes
*fell* on RWC-Pop (5.74 → 5.07), RWC-Classical (6.24 → 3.97) and
RWC-Royalty-Free (5.48 → 4.51), and rose only on Harmonix, RWC-Genre and
RWC-Jazz. A cost that changes sign with the material is not the cost that gate
was written to catch, and "episodes bought with churn" is not supported here as
a general claim.

**None of this is a decision to ship, and the pre-registration says so.** Real-
time factor with three networks on the target phone is out of its scope, and if
three networks do not fit the CPU budget the corpus verdict is moot. That is the
cheapest measurement that can close the question, and it is worth taking before
paying the adoption price — GTZAN and Ballroom retired permanently as evaluation
corpora, 1,697 of the 2,760 annotated recordings here.

The whole protocol re-runs as one command:

```bash
python -m eval.ensemble_in_core --family
```

## The anchor width sweep

`live_usable_width*.json` and `live_usable_rwc_width*.json`. The table and the
reasoning live beside the constant, in `core/src/tracking/live.hpp` at
`anchor_width_octaves`, because that is where someone changing it will look.

## Can a wrong level be seen coming: a documented negative, and an accident

`phase_instability_{rwcpop,harmonix}.json`, from `eval/phase_instability.py`,
where the tables and the reasoning live. The question was whether the settled
phase relationship between the low and high ODF bands comes apart *before* the
tracker slips to the wrong metrical level. RWC-Pop chose the threshold, Harmonix
never chose anything.

Harmonix is the **threshold-transfer corpus, not a held-out one** — it was
already spent on the seam experiment above, and calling it held out would claim
more than it carries.

**It does not.** With the threshold carried across, the phase feature warns of
16.3% of every episode over four seconds — the diagonal. It is not better than
a plain fall in coherence, which leads it on both corpora, and it is one of
only two signals here that cannot even see every episode: no settled phase ever
forms on 10% of them. The mechanism is in the same file — a single band's phase
mostly fails the Rayleigh threshold the core already uses to decide whether a
phase means anything, so there is little reliable relationship there to come
apart.

**What the controls turned up instead is worth more than the hypothesis was.**
`live_anchor_margin`, which the live path computes every frame and uses for
nothing else, warns of **85.9% of every wrong-level episode one to four seconds
before it starts**, at a threshold chosen on RWC-Pop and carried to Harmonix as
a number, with 16.9% of correct locked frames above it. It sees all 1,063
episodes, so that rate has no hidden denominator. Anticipatory, not concurrent:
0.895 AUC on the window ending a second before the onset against 0.932 on the
transition itself.

**That 16.9% is not a cost and must not be quoted as one.** It is the share of
correct locked frames above the threshold. What a gate costs depends on what
the gate does, and freezing the octave while leaving tempo and phase free is
nearly free on a frame where the octave was not going to move. Only replaying
the tracker under the policy turns that column into a cost — and a tracker that
abstains spends no time at the wrong level, so the episode endpoint can always
be improved by saying less. Anything built on this is scored on locked time
kept as well as on episodes avoided.

An earlier version of this section quoted 85.3% / 18.3% from an evaluator that
scored every signal on the *phase* feature's availability, which silently made
the denominator 996 of 1,063 episodes. The numbers above are from the corrected
one; `tests/test_phase_instability.py` pins the difference.

## Acting on that warning: the octave freeze, measured

`arm_{baseline,clear,freeze,abstain}.json` with their per-track files. Four arms
of `eval/PREREGISTERED_octave_freeze.md`, Harmonix, 581 of 581 on each, commit
`a2c18eb`, `tree_clean` true on all four. Shipped fold 1 throughout, τ = 0.5916
carried from RWC-Pop as a number.

| Harmonix | episode-free | strict | correct time | sw / 5 min | settle P90 | F |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 41.48% | 26.16% | 77.54% | 4.21 | 36.61 s | 0.7953 |
| clear | 30.64% | 18.07% | 74.61% | 7.12 | 51.71 s | 0.7798 |
| **freeze** | **41.82%** | **26.85%** | **77.87%** | **3.81** | **38.61 s** | **0.7965** |
| abstain | 84.17% | 18.93% | 64.74% | 0.71 | 50.91 s | 0.6979 |

Paired against baseline, per recording:

| | won | lost | p |
|---|---:|---:|---:|
| freeze, episode-free (**primary**) | 17 | 15 | **0.86** |
| freeze, usable strictly | 7 | 3 | 0.34 |
| clear, episode-free | 17 | 80 | <1e-4 |
| abstain, episode-free | 248 | 0 | <1e-4 |

**Adoption not approved. The freeze is inert on the endpoint it was built for.**
41.82% against a 46.5% bound, and a sign test that could hardly be more null.

**Five of the six predictions held; the primary did not.** The freeze beats
`clear` by eleven points (P2), `clear` at this τ is worse than baseline on F as
its older measurements predicted (P3), `abstain` takes the highest episode-free
and the lowest correct time and fails the correct-time gate (P4), F moves by
0.0012 (P5), and the switch rate falls, 4.21 → 3.81 (P6).

**So the policy did act, and the episodes did not care.** P6 is the important
one: switches fell by a tenth, F did not move, and `clear` at the same trigger
was catastrophic — the arm is demonstrably doing what it is described as doing,
to the right recordings, at the right moments. The episode rate still did not
move. Whatever makes a wrong-level episode on this corpus, it is not an anchor
switching octave while the estimator is unsure.

**A signal that predicts is not a policy that helps, and this is the cleanest
demonstration of it here.** The same `live_anchor_margin` separates episodes at
0.895 AUC one to four seconds ahead. Acting on exactly that warning, at exactly
that threshold, changes nothing. The prediction is real and the lever was the
wrong one.

**`abstain` is why the endpoint needs its guard.** 248 recordings better and
none worse, to 84.17% episode-free — by saying nothing on more than half the
polls, at a cost of thirteen points of correct time and ten of F. It was never a
candidate, and it is the measured size of what silence buys on this metric.

## An accurate bar period has strong leverage; reading one off BeatNet is unproven

`{hx,gz}_{baseline,barrate,oracle}.json` with their per-track files. Three arms
of `eval/PREREGISTERED_downbeat_channel.md`, commit `dce26bb`, `tree_clean` true
on all six, 581 of 581 Harmonix and 999 of 999 GTZAN. Shipped fold 1 throughout.

`bar-rate` estimates the bar period from BeatNet's downbeat head and uses it to
pick the beat octave. `oracle-bar` is the same decision rule handed the
annotated bar length — a bound, never a mode, since nothing in a room knows the
bar in advance.

| Harmonix | episode-free | strict | correct time | sw / 5 min | settle P90 | F |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 41.48% | 26.16% | 77.54% | 4.21 | 36.61 s | 0.7953 |
| **bar-rate** | **41.48%** | 25.82% | 78.35% | 4.69 | 35.02 s | 0.7929 |
| oracle-bar | **60.59%** | 33.56% | 84.83% | 1.19 | 23.01 s | 0.7833 |

| GTZAN | episode-free | strict | correct time | sw / 5 min | settle P90 | F |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 67.87% | 43.64% | 67.74% | 5.34 | 14.0 s | 0.6854 |
| bar-rate | 68.37% | 43.54% | 67.87% | 5.55 | 13.0 s | 0.6845 |
| oracle-bar | 76.18% | 44.94% | 72.93% | 2.98 | 12.0 s | 0.6759 |

**Adoption not approved: `bar-rate` is exactly baseline on the primary
endpoint.** 41.48% against 41.48%, and a paired sign test of 10 won to 10 lost,
p = 1.0000. Not "no significant difference" — the same number.

### The diagnosis, which is why the oracle arm was worth its runtime

Both arms fire on almost exactly the same recordings:

| Harmonix | recordings whose tempo it changed | episode-free won | lost | p |
|---|---:|---:|---:|---:|
| bar-rate | 247 of 581 | 10 | 10 | 1.0000 |
| oracle-bar | 243 of 581 | **114** | **3** | <1e-4 |

The decision rule is right, the firing rate is right, and **the estimated bar
period is no better than a coin about which octave to take.** Handed the true
bar length instead, the identical rule wins 114 recordings against 3.

### What this does and does not license, stated carefully

**`oracle-bar` never touches BeatNet's downbeat output.** It reads the bar
length from the annotation. So what these six runs establish is:

> An accurate bar period has strong leverage on the octave — 19.1 points of
> episode-freeness on Harmonix, 8.3 on GTZAN — and one generic autocorrelation
> estimator failed to supply one.

They do **not** establish that the downbeat channel carries recoverable bar
information. Two possibilities remain conflated and this design cannot separate
them: BeatNet's downbeat output may be uninformative, or `ActivationTempo` may
be the wrong way to read it. The matching firing rates do not help — two arms
can fire on the same recordings while disagreeing about the period on every one
of them, which is exactly what 10-against-10 versus 114-against-3 looks like.

An earlier version of this section was headed "the evidence works, the estimator
over it does not", and the commit that added it claimed "the downbeat channel
decides the octave". Both overclaimed: the oracle is silent about the channel.

The question is now its own experiment — see
`eval/PREREGISTERED_downbeat_audit.md`, which asks whether a bar period can be
recovered from the raw beat and downbeat activations at all, before any further
live policy is written.

### The predictions

| | prediction | outcome | |
|---|---|---|---|
| P1 | `bar-rate` clears 46.5%, p<.05 | 41.48%, p = 1.0 | ❌ |
| P2 | — | retired before the run; true by construction, see the pre-registration's deviations | — |
| P3 | `oracle-bar` clears 55% | 60.59% | ✅ |
| P4 | `bar-rate` recovers a third of the way to the oracle | recovers none of it | ❌ |
| P5 | beat F moves under a point | bar-rate −0.24; oracle −1.20 | ⚠ |
| P6 | GTZAN moves the same way | +0.50 against +0.00 | ✅ |

P5 is flagged rather than passed: the oracle gives up 1.2 points of F for its
19.1 of episodes. That is the shape this project has taken before — a recording
that crosses the usable threshold is worth more than one that goes from 0.79 to
0.82 — but it is a cost and not a rounding error, and any successor to the
estimator inherits it.

P6 passes on direction and means little on size, and the per-track counts say
why: `bar-rate` fires on 95 of 999 GTZAN excerpts against 247 of 581 Harmonix
songs. The bar estimator needs a twelve-second window before it answers at all,
which is most of a thirty-second excerpt.

### What this leaves

Not "replace the estimator" — that assumes the answer to the question above.
What is open is whether the bar period is recoverable from the model's own
outputs at all, and the next experiment is the smallest one that closes it: an
offline audit reading the downbeat probability **at the predicted beat
positions** rather than autocorrelating it, since the downbeat head answers
"which of these beats starts a bar", not "how slow is the bar".

No further live policy until that audit reports.

## The downbeat audit: the metre is there, the octave is not

`audit_{gtzan,harmonix}.json`, from `eval/downbeat_audit.py`, answering
`eval/PREREGISTERED_downbeat_audit.md`. Offline, no live core involved: read
BeatNet's downbeat probability at each beat of a grid and score, over every
metre in {2, 3, 4, 6} and every bar phase, the contrast between the beats that
would be downbeats and the beats that would not. Then score the same grid
**doubled** and ask which one the evidence prefers.

**What was run is not what was registered.** Three deviations, none recorded
before publishing, all found on re-reading the pre-registration against the
code:

1. **Annotated beat positions, not predicted ones.** The pre-registration says
   "take the predicted beat positions" twice; `audit_one` calls
   `load_reference_beats`. This one favours `beat-sync` — a perfect grid is
   better than the live tracker's — so it does not rescue a failure.
2. **Whole-recording accumulation, not "over the last 2–4 bars".** The README
   originally defended this as a ceiling: a causal decoder seeing two to four
   bars cannot extract more than an offline one seeing all of them. That is
   true of an *optimal* offline decoder and false of this one, which takes a
   global mean and so is diluted by arrangement change, dropouts and a chorus
   whose downbeat is strong against a verse whose downbeat is not. It was a
   deviation rationalised after the fact, not a registered choice.
3. **The `autocorr` arm was never implemented**, so P1 is unmeasured.

Four of the seven named measurements — bar period accuracy, coverage, false
corrections, bars to a stable decision, share of the oracle gap — were not
reported either.

| GTZAN, n = 991 | metre | octave separation |
|---|---:|---:|
| **beat-sync** | **60.8%** [57.7, 63.9] | 76.2% [73.4, 78.8] |
| shuffled | 23.5% [20.9, 26.3] | **84.2%** [81.7, 86.4] |
| beat-as-downbeat | 38.1% [35.1, 41.2] | 23.6% [21.0, 26.4] |

| Harmonix, n = 579 | metre | octave separation |
|---|---:|---:|
| **beat-sync** | **82.9%** [79.6, 85.9] | 79.6% [76.1, 82.8] |
| shuffled | 30.1% [26.3, 34.0] | **84.1%** [80.9, 87.0] |
| beat-as-downbeat | 60.1% [56.0, 64.1] | 7.8% [5.7, 10.3] |

### The metre is carried, decisively

`beat-sync` clears both controls on both corpora — 37 and 53 points over
shuffled, and 23 points over `beat-as-downbeat` on each. That last comparison is
the one that matters, because the beat channel alone reaches 38% and 60%: some
metre accuracy is available from the grid's own periodicity, and the downbeat
channel adds a large amount on top of it.

### The octave is not, and the control is the only reason we know

**`beat-sync` scores 76.2% and 79.6% on octave separation — and shuffled noise
scores 84.2% and 84.1%.** The intervals do not overlap on GTZAN. The signal is
*behind* its own null on both corpora.

Without the control this would have read as a strong result. It is not one, and
the reason the null sits so high is structural: the doubled grid carries twice
as many points, so the maximum over (metre, phase) of a noise contrast is
systematically smaller there, and the comparison tilts toward the shorter grid
before any evidence is consulted. 84% is what that tilt is worth. Nothing in
the downbeat channel beats it.

`beat-as-downbeat` confirms the instrument from the other side: at 23.6% and
7.8% it prefers the *doubled* grid outright, which is exactly what a decoder
finding periodicity in the grid it was handed looks like — the beat channel is
high at every beat, so doubling it manufactures a clean alternation.

### Verdict

**A2 fails: the pre-registration asked for at least 15 points over shuffled and
the result is 8.0 and 4.5 points behind it.** That rejects **this decoder**, and
it is the load-bearing sentence of the whole run.

It does not, on its own, close the head, and an earlier version of this section
said it did. The claim has to be sized to what was measured:

> An unnormalised global contrast score, maximised over (metre, phase), gets no
> octave advantage from the downbeat channel.

rather than "the downbeat channel contains no octave information". The gap
between those two is the tilt described above — the null is not a clean null,
because the two grids are permuted independently and have different lengths, so
the comparison mixes the channel's information with the decoder's geometry. The
tilt does not save the result (A2 fails by a wide margin either way) but it does
bound what the result is about.

A1, A3 and A4 are unmeasured. A1 and A3 need the live path; A4 needs a threshold
this decoder never earned. So **three of four acceptance conditions were never
taken, and the fourth was taken with an unregistered grid and an unregistered
window.** The protocol was not completed, and a verdict on an incomplete
protocol is a verdict on the decoder, not on the direction.

**P3 predicted exactly this** — "the metre comes back and the `P` against `P/2`
decision does not" — for the reason `analysis/downbeat.hpp` already records
about the half bar: a bar phase repeats at both grids, so a wrong octave that
repeats on the period the evidence is accumulated over cannot be broken by
accumulating more of it.

### What survives

Not the octave. But an 82.9% metre read on full-length songs, against a 30.1%
null, is a large amount of unused information about something else the product
gets wrong: the offline downbeat resolver scores 0.417 F on GTZAN with the
built-in cues. That is a lead for bar-line placement, not a proposal, and it is
a different experiment from this one.

## Everything else

`oracle_activation*.json`, `activation_recall.json`, `octave_blame_*.json`,
`timing_irregularity.json`, `tempo_stress.json`, `live_usable_rough*.json`,
`live_usable_no_anchor.json`, `live_usable_split*.json` — arm-versus-arm
experiments, all measured at `anchor_width_octaves` 0.10, which shipped before
2026-08-03. Each is a contrast between two arms at one width, which is what they
are cited for and what they are still worth. None is an absolute level any more.

## The octave veto: the decoder is below chance wherever it acts

`octave_veto_rwc.json`, from `eval/octave_veto_experiment.py rwc`, answering
`eval/PREREGISTERED_octave_veto.md`. RWC, all 328 recordings, shipped fold 1,
commit `b102324`, tree clean. **Harmonix was never opened.**

The unit is the decision point, not the frame and not the recording: when the
live tracker actually proposes moving to another octave, does beat-synchronous
metre evidence correctly allow or veto *that* switch. 1029 proposals on the
baseline arm, 97-98% of them scoreable at every metre, 678 labelled against 346
ambiguous — so the experiment is interpretable and the negative is not an
artefact of unlabelable events.

### A2 fails, and the shape of the failure is the finding

| τ | switches | balanced accuracy | false veto | episode-free |
|---:|---:|---:|---:|---:|
| baseline | 1913 | — | — | 0.2744 |
| 0 | 1367 | **0.4859** | 52.8% | 0.2927 |
| 0.5 | 1627 | **0.4951** | 30.1% | 0.2835 |
| 1 | 1844 | **0.4902** | 9.2% | 0.2805 |
| 1.5 (selected) | 1906 | **0.4998** | 0.6% | 0.2744 |
| shift control | 1913 | 0.5000 | 0.0% | 0.2744 |

Read the first two columns together. **Balanced accuracy is below chance at
every threshold where the decoder acts, and reaches exactly 0.5 only as its
action goes to zero.** The more it does, the worse than chance it is. That is
not a decoder with a weak signal; it is a decoder with none, whose apparent
neutrality at the selected threshold is the neutrality of doing nothing.

A2 required a **15-point** margin over the shift-driven control with the
interval's lower bound above zero. Measured: **−0.0002, 95% CI [−0.0060,
+0.0050]**, p = 1.0, 10 000 cluster-bootstrap resamples over recordings. The
interval is tight enough to exclude anything above half a point.

**A3 passes, and that is not a defence.** §7 selects τ by episode-freeness
subject to A3 and the cost gates, and A3's 5% bound on blocked correct escapes
eliminates every threshold below 1.5 — 52.8%, 30.1%, 9.2% all violate it. What
survives selection is the threshold at which the decoder vetoes 7 switches out
of 1913 and leaves every published number equal to baseline to four decimals.
The constraint that protects correct escapes rules out every setting at which
this decoder does anything at all.

It is also **behind the best matched-cost policy on the endpoint**: `margin_0.3`
reaches 0.2927 episode-freeness against the decoder's 0.2744.

### What the comparison policies show on their own

`total_ban` has the best episode rate of any arm, 0.3293, and buys it by cutting
switches from 1913 to 730 while losing correct locked time, 0.5623 → 0.5286.
That is exactly the trade the matched-cost design exists to expose, and it is
the reason "better than baseline" was never allowed to be the endpoint.

### Verdict

**A2 fails on the development corpus.** By the terms fixed before the run, that
closes the downbeat head for octave correction — permanently and without
reservation. Every objection raised against the previous audit was answered
first: predicted grids, a matched null that shifts one track and resamples both
nested grids from it, real decision points, matched-cost alternatives, a
standardised score whose null does not depend on grid length. The protocol was
executed as written. There is no further document.

Two registered predictions also failed, and both are recorded rather than
dropped. **P2** asked for over 90% sign agreement between `Δ` and `Δ_raw`;
measured 80.7% on 996 events, which says the null subtraction is carrying
weight rather than correcting a bias. **P7** predicted D1 above 5% — the share
of doubling proposals on a committed-correct state where the committed grid is
constant, the mechanism I2's failure exposed. Measured **0 of 170**. That
limitation, registered as the thing most likely to sink A1, never occurred on
real music at all.

### What this leaves

Not the octave. The line that ran from the anchor margin through the octave
freeze, the bar-rate arm, the downbeat audit and now this one is closed: **the
metrical level is not recoverable from BeatNet's own outputs**, by any reading
of them that has been tried, and what remains is a different front end.

The metre survives untouched — 82.9% on full-length songs against a 30.1% null —
and it is evidence about bar-line placement, where the offline resolver scores
0.417 F on GTZAN. A different experiment, and one that must not touch BPM.

## What a perfect octave would actually buy

`octave_ceiling_per_track_{rwc,harmonix}.json`, from
`eval.live_corpus_benchmark --per-track` on the shipped configuration. No new
mechanism, no hypothesis: a re-cut of what the benchmark already computes.

`usable_any_octave` scores each recording at **whichever octave reading came
closest**. It is therefore the exact ceiling on everything the closed octave line
was reaching for — anchor margin, octave freeze, bar rate, downbeat audit,
octave veto. Usable means precision and recall both at or above 0.80 with
acquisition inside the limit.

| corpus | n | usable | at best octave | the octave buys | still fails |
|---|---:|---:|---:|---:|---:|
| RWC-Pop | 100 | 39.0% | 60.0% | +21.0 | **40.0%** |
| Harmonix | 581 | 31.0% | 51.3% | +20.3 | **48.7%** |
| GTZAN | 999 | 44.5% | 49.2% | +4.7 | 50.8% |
| RWC-Genre | 102 | 12.7% | 24.5% | +11.8 | 75.5% |
| RWC-Jazz | 50 | 8.0% | 14.0% | +6.0 | 86.0% |
| **RWC-Classical** | 61 | 0.0% | 0.0% | **+0.0** | **100%** |
| SMC | 217 | 3.2% | 4.1% | +0.9 | 95.9% |

**The 23 points quoted from RWC is the best case, not the typical one.** It is
roughly RWC-Pop's 21. On GTZAN a perfect octave is worth 4.7 points, and on
RWC-Classical it is worth **nothing at all**: not one of those 61 recordings
becomes usable at any reading of the level.

### What survives it

Among recordings that fail at their own best octave — 283 of 581 on Harmonix,
230 of 328 on RWC:

| reason set | Harmonix | RWC |
|---|---:|---:|
| too few beats **and** wrong beats | 44.2% | 63.9% |
| both, plus slow acquisition | 14.8% | 17.8% |
| too few beats alone | 18.7% | 8.7% |
| slow acquisition alone | 13.8% | 3.9% |
| wrong beats alone | 1.1% | 2.6% |

**Recall is the dominant survivor: `too_few_beats` appears in 84.8% of Harmonix's
and 93.5% of RWC's.** Precision follows at 60.4% and 84.3%, and precision alone
is almost nonexistent — 1.1% and 2.6%. The two fail together far more often than
either fails apart.

### What this settles

The metrical level was never the binding constraint on most material. Solving it
perfectly takes Harmonix from 31.0% to 51.3% and leaves half the corpus failing
because the beat grid is simultaneously too sparse and in the wrong places.

So the closing sentence of the octave-veto section above — "what remains is a
different front end" — is right in form and wrong in aim. A front end better at
the **octave** has a 51.3% ceiling on Harmonix and a 0.0% ceiling on classical.
Any successor should be pre-registered against **beat-grid recall**, which is a
different question with different acceptance conditions.

One tractable piece is separable: **13.8% of Harmonix's surviving failures are
slow acquisition alone**, with precision and recall both already good. That is
39 recordings failing on a stopwatch rather than on the tracking, and it is the
one part of this picture that does not need a new observation.

## Acquisition was measured on a grid too coarse to see it

`acquisition_50hz_per_track_harmonix.json` beside
`octave_ceiling_per_track_harmonix.json`. Same binary, same model, same
recordings; the only difference is `--live-sample-hz`, which changes how often
the harness reads the tracker and nothing about the tracker. **Beat counts are
identical on all 581 recordings**, which is what makes the comparison a
measurement question rather than a behavioural one.

`acquired_at` is reconstructed in Python from the polled confidence series, and
every live number in this repository was polled **once a second**. The bar for
`slow_acquisition` is eight seconds. Re-read at 50 Hz:

| | 1 Hz | 50 Hz |
|---|---:|---:|
| Harmonix usable | 30.98% | **36.49%** |
| of the 39 slow-acquisition-only failures, now under 8 s | — | **29** |
| median shift in `acquired_at` on those | — | **−4.74 s** |

**This is not quantisation, and the first diagnosis of it was wrong.** A
one-second grid can misplace a threshold crossing by at most one second; these
move by nearly five, and one by fifteen. The mechanism is aliasing: confidence
fluctuates across the 0.25 threshold, and a once-a-second sampler keeps landing
in the gaps. On `0925_sweetdisposition`, **zero of the six polls between 4 s and
10 s** catch it at 1 Hz and the first catch is at 16.02 s; at 50 Hz, 33 of 258
polls in the same window are over threshold and the first is at 4.18 s.

### Verified against something that needs no sampling at all

The tracker's beat list is exact. On ten of the moved recordings, when the first
beat was actually handed out:

| track | 1 Hz | 50 Hz | first beat |
|---|---:|---:|---:|
| 0099_forgetyou | 9.01 | 4.272 | **4.281** |
| 0132_iceicebaby | 10.01 | 4.133 | **4.443** |
| 0418_inthedark | 12.00 | 4.133 | **4.401** |
| 0344_beautifullife | 8.01 | 4.087 | **4.111** |
| 0925_sweetdisposition | 16.02 | 4.180 | **5.832** |
| 0324_yeah3x | 12.00 | 4.830 | 11.583 |
| 0434_lights | 8.01 | 7.825 | 8.276 |

Eight of ten start playing under the bar. So the 1 Hz figure was wrong by
seconds, and the 50 Hz figure is close but still an approximation — it reports a
confidence crossing, and two of the ten crossed without publishing.

### What this costs and what it implies

Every live result here inherits a `usable_rate` that is too low, by 5.5 points
on Harmonix, for a reason that has nothing to do with the tracker. Comparisons
*between* arms measured the same way are unaffected — the error is common to all
of them — but absolute rates are not, and neither is any claim of the form "the
tracker acquires slowly".

**`acquired_at` should be derived from the beat list rather than from a sampled
confidence series.** The beat list is what a listener hears, it is exact, and it
is independent of how often anything is polled. That changes published numbers
and is a decision rather than a fix, so it is recorded here and not applied.

## The causal bar: phase carries, metre cannot be measured here, and the gate fails

`causal_metre_gtzan.json` and `causal_metre_harmonix.json`, from
`eval.causal_metre`, answering the causal arm registered in
`eval/PREREGISTERED_downbeat_audit.md` on 2026-08-08. Commit `676059d`, clean
tree. GTZAN 991 scored of 999 (one corpus defect, `jazz.00054` is not a WAV),
Harmonix 579 of 581, no failures. **All arms published byte-identical beat lists
on every recording**, which is the invariant that makes them comparable: the bar
decision reads a channel nothing else reads and writes nothing back.

The mechanism under test is `44c8c56` — `analysis::resolveMeter` over the last
32 beats the live tracker handed out, re-resolved every beat, inside the
shipping core. The arms differ in one file: the downbeat channel.

### The metre, and why these corpora cannot answer it

| | GTZAN | Harmonix |
|---|---:|---:|
| **always answer 4** | **0.949** | **0.976** |
| `beat_sync` | 0.867 | 0.894 |
| `beat_as_downbeat` | 0.791 | 0.729 |
| `shuffled` | 0.492 | 0.499 |

Restricted to recordings the tracker tracked at the annotated level, which is
what the registration made the answer.

C1 passes on both by 37.5 and 39.5 points, so the decoder is reading structure
and not level. C3 passes, and the causal figure is *above* the whole-recording
audit's 0.608 and 0.829 — which falsifies prediction P5 outright.

**None of that survives the constant.** 690 of 727 restricted GTZAN recordings
and 479 of 491 Harmonix ones are in four, so answering "4" and nothing else
beats the decoder by eight points on both. Off the majority metre there are
**49 recordings in 1218**, and there `beat_sync` scores 0.189 and 0.250 against
`shuffled`'s 0.108 and 0.333. Nothing is distinguishable from anything at those
counts.

That baseline was missing from C1–C3, which were written the same day and
compare only against shuffles and substitutions — both of which a metre prior
clears without deciding anything. The same reading applies backwards: the
original audit's 0.608 was compared against a shuffled 0.235 and never against
the 0.949.

### The phase, which is what the material actually varies

Registered as an addition after the metre result, for a reason stated in the
protocol: the corpus composition that makes metre unanswerable was discovered
in the run. F1 is bar-line agreement at 70 ms over the beats after the metre
settled; the null is the mean over all rotations of the same grid, which is the
exact expectation of a uniformly random bar line.

| | GTZAN F1 | Harmonix F1 |
|---|---:|---:|
| `beat_sync` | **0.522** [0.492, 0.552] | **0.606** [0.581, 0.631] |
| `beat_as_downbeat` | 0.329 [0.305, 0.353] | 0.516 [0.491, 0.541] |
| `random_phase` | 0.209 [0.203, 0.214] | 0.217 [0.212, 0.221] |
| `shuffled` | 0.193 [0.178, 0.208] | 0.207 [0.200, 0.214] |

**The contrast inside a single run is the finding.** Same recordings, same
decoder, same channel: the metre cannot separate from a constant because the
corpus has almost no metre variation, and the phase separates from its own null
by 31.3 and 38.9 points because a bar line has four places to be and the corpus
does not decide which. Both the original audit and the causal metre arm measured
the half of the problem this material holds fixed.

P8 predicted 0.4 to 0.6 and it came in at 0.522 and 0.606.

### Verdict: the flag stays off

The registered condition was "clears `random_phase` by at least 20 points on
both corpora *and* clears `beat_as_downbeat` by at least 10. Failing either
leaves the flag off." The first holds by 31.3 and 38.9. The second holds on
GTZAN at 19.3 and **fails on Harmonix at 9.0**. So the condition fails, and
`bar_tracking` stays off with a documented negative.

**A fact about that control, recorded and not used to overturn the result.**
`ml/beatnet.hpp` computes `beat = p[0] + p[1]` and `downbeat = p[1]`, so the
beat channel *contains* the downbeat channel additively. `beat_as_downbeat` is
therefore not the wrong evidence with the right shape; it is the right evidence
with the rest of the beat channel added to it, and a large gap was never
available. That was knowable from the code before the run and it is a fault in
the 10-point bar rather than a reason to move it now.

**What a clean control would be**, if this is picked up again: `p[0]` alone —
beat-but-not-downbeat — which is `beat − downbeat` and computable from the two
channels already dumped, with no new model pass. It would need its own
registration, precisely because it is being named after a failure.

### What this leaves

The bar mechanism ships, off, tested, and costing nothing. What is now known:

- a causal 32-beat window is **not** the limitation — it beats whole-recording
  reads of the same channel on both corpora;
- the phase signal is real and roughly half of what a perfect bar line would be;
- **GTZAN and Harmonix cannot evaluate a metre decision at all**, and any future
  claim about metre needs a corpus with metre in it;
- the click gate remains untested here, because the harness plays no click, so
  every figure above is an upper bound for a shell with audible output.

## The octave button: measured, not approved, and the range guard is why

`octave_press_rwc.json`, from `eval.octave_press_experiment`, answering
`eval/PREREGISTERED_octave_press.md`. RWC, all 328 recordings, all five
collections, `beatnet_model_1`, commit `4e7902d`, clean tree, no failures,
`NOTICE_SEC = 2.0`, `MAX_PRESSES = 3`, every arm sampled at 50 Hz. **Harmonix was
not opened**: §7 makes transfer conditional on RWC, and RWC did not pass.

| arm | usable | correct share | mean F | presses | refused |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2073 | 0.5623 | 0.6015 | 0 | 0 |
| `press` | 0.2287 | 0.5749 | 0.5402 | 341 | 467 |
| `press_random` | 0.2195 | 0.5328 | 0.5566 | 451 | 357 |
| `press_delayed` | 0.2195 | 0.5860 | 0.5552 | 315 | 487 |

### Every primary fails

| | measured | required | Holm p |
|---|---:|---:|---:|
| **A1** press vs baseline, usable | **+2.1** pts, CI [+0.3, +4.3] | +5.0 | 0.122 |
| **A2** press vs random, usable | **+0.9** pts, CI [+0.0, +2.1] | +4.0 | 0.211 |
| **A3** press vs baseline, share | **+0.013**, CI [−0.017, +0.042] | +0.05 | 0.401 |

**A2 is the one that matters and it is at zero.** Its lower bound is +0.0000.
The registered reason for that arm was that `setOctaveOffset` re-seeds the cloud
and moves the anchor, so any such disturbance perturbs a stuck tracker, and
without a same-times random-direction control the result cannot separate the
judgement from the kick. Measured, the judgement is worth under a point over the
kick.

**Cost gate C1 fails outright.** Mean F falls 0.0615, against a bound of 0.010.
And it falls in *every* press arm — 0.5566 random, 0.5552 delayed — so pressing
damages beat tracking whatever direction it is in.

**C4 passes, and is the reason to trust the rest.** Of 19 recordings the
baseline never had at the wrong level, the listener fired on **zero**. The
arming condition does what §3 says.

### The binding constraint is the BPM range, not the listener

**467 of 808 attempted presses were refused — 57.8%.** By direction: **342 of
them were ×2 and 125 were ÷2.** The filter's range is 40..220 BPM, so ×2 is
unavailable above 110, and a tracker sitting on the eighths of a 120 BPM song is
at 240 and out of reach.

§11 named this in advance: a refusal rate "over 40% of attempted presses" means
"the range guard, not the listener, is the binding constraint. That would not
sink the idea but would redirect the work to the BPM range, which is a different
experiment with different costs." That is where this lands.

It also distorts the control. The arms are matched on press *times*, not on
presses that landed, so `press_random` got 451 accepted presses to `press`'s 341
— a random direction is refused less often, because ÷2 is usually available and
×2 usually is not. A2's margin is measured between arms that pressed a different
number of times, and that is a weakness of the registered design rather than of
the result.

### Predictions

- **P1** wanted over 60% of improving recordings to need one press. 145 of 328
  never pressed; of those that did, 85 pressed once, 38 twice and **60 hit the
  cap of three**. A listener spending its whole budget on a fifth of the corpus
  is not the "one press and it holds" picture the mechanism was built for.
- **P2** predicted refusals would be asymmetric and predominantly ×2. **342
  against 125.** Correct.
- **P3** predicted `press_random` would land *below* baseline rather than level
  with it, because a wrong press is held. Correct share 0.5328 against 0.5623.
  Correct — and it confirms the hold in `602f9ad` works as designed.
- **P5** predicted the realised gain would be under +5.6 points. It is +2.1.
  Correct.

### Verdict

**Adoption not approved.** Three primaries missed, one cost gate failed. The
mechanism stays in the core, off nobody's path, because a person is entitled to
overrule a tracker whether or not it helps on average — but the +21.0 point
ceiling on RWC-Pop is **not** reachable by a button as specified, and must not be
quoted as though it were.

What this redirects to, and what it does not license: the 40..220 BPM range is
now the measured constraint, and widening it is a config change that moves the
tempo prior and the resample clamp under every published number here. That is
its own pre-registration with its own cost gates, and nothing above says it would
work — only that the present result cannot be read as a verdict on the button
until the button is allowed to press.

## The octave button again, with the range moving: better everywhere, still not approved

`octave_press_rwc_shifted.json`, answering the re-run registered in
`eval/PREREGISTERED_octave_press.md` on 2026-08-08. Same 328 RWC recordings,
same gates, same `NOTICE_SEC = 2.0` and `MAX_PRESSES = 3`, 50 Hz. Code at
`9babda2`; the tree reads dirty only because the void run below had left its
output file behind. **Harmonix still not opened.**

`77e7bae` stops the configured 40..220 BPM range outranking a press: the range
and the prior centre move by the user's octave, shifted rather than widened so
that no width anywhere changes.

| arm | usable | correct share | mean F | accepted | refused |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2073 | 0.5624 | 0.6015 | 0 | 0 |
| `press` | 0.2409 | 0.6166 | 0.5717 | 469 | 0 |
| `press_random` | 0.2287 | 0.5082 | 0.5325 | 444 | 25 |
| `press_delayed` | 0.2165 | 0.6085 | 0.5696 | 464 | 2 |

### The gates

| | first run | now | required | verdict |
|---|---:|---:|---:|---|
| **A1** press vs baseline, usable | +2.1 | **+3.35** CI [+1.5, +5.5] | +5.0 | fails |
| **A2** press vs random, usable | +0.9 | **+1.22** CI [+0.00, +2.74] | +4.0 | fails |
| **A3** press vs baseline, share | +0.013 | **+0.054** CI [+0.024, +0.085] | +0.05 | **passes** |
| **C1** mean F | −0.0615 | **−0.0298** | ≤0.010 | fails |
| **C4** presses on clean recordings | 0 | **0** | 0 | passes |

Holm over the registered family: A1 0.000, A3 0.0016, A2 0.129.

**The change did exactly what it was supposed to.** Refusals went from 57.8% to
**0.0%** — 469 attempts, none refused, P9 confirmed outright. The accepted-press
gap between `press` and `press_random` fell to 5.3%, under the 10% bar, so A2 is
no longer confounded by the arms doing different amounts of work. And the damage
per press dropped: mean F fell 0.0298 where it fell 0.0615 before, on *more*
accepted presses — because a press now lands where the cloud can live instead of
being scaled into a clamp.

**A2 is still the finding, and it is still at zero.** Its lower bound is exactly
+0.0000. With the guard gone, matched press counts and a control that can now do
real damage, pressing in the *right* direction is worth **1.2 points of usable
recordings** over pressing in a random one.

But read A2 beside A3, because they disagree in a way that means something.
On time at the correct level the direction matters enormously — `press` 0.6166
against `press_random` 0.5082, over ten points — while on the pass/fail verdict
it is worth one. The octave is one clause of four in `usable`, and fixing it
leaves precision, recall and acquisition where they were. That is the same thing
the ceiling measurement said: solving the octave perfectly still leaves half the
corpus failing on the beat grid.

`press_delayed` is 2.4 points below `press` (p = 0.009), so acting early is
worth something real — which is the one place the listener's timing showed up.

### Predictions

P9 confirmed (0.0% refusals). P10 confirmed — `press_random` got *worse*, 0.5082
against the first run's 0.5328, because it can now reach octaves it used to be
refused. P11 confirmed — C1 fails again, as named in advance. **P1 is decisively
wrong**: 124 of 328 recordings hit the three-press cap, against 60 before. Freed
from the guard, the listener thrashes.

### Verdict

**Adoption not approved.** A1 and A2 fail, C1 fails. A3 passing is the first
primary this line has ever met, and it is not enough on its own.

The honest summary of two runs: the range guard was a real constraint and
removing it bought about a point and a half of usable recordings and five points
of correct-level time — but the button was never going to reach the +21.0
ceiling, because that ceiling is an oracle applied to whole recordings and a
person acts at a moment, three times at most, and only on the one clause of four
that the octave touches.

### Two process failures, both recorded

**The first re-run was void, and not for the reason the precondition was written
to catch.** `tools/eval/build` is a separate CMake tree, so building
`tiktak_core_tests` in `core/build` left `dump_analysis.exe` 1 h 39 min older
than the change under test. That run measured the old mechanism. It was caught
by the registered baseline precondition — written for a different purpose
entirely — and without it the old mechanism's numbers would have been published
as the new one's.

**One recording will not reproduce, and it is not this change.** `RWC_P065`'s
baseline differs between the first press run and every measurement since:
F 0.6266 there against 0.6462 in the void run, 0.6462 in this one, and 369 beats
in seven isolated reruns. Two runs on the old binary and one on the new agree
exactly; the first run is the outlier, and it predates `77e7bae`, so the change
cannot be its cause. The precondition therefore fails against a reference value
that has never reproduced, while its purpose — detect a leak from `77e7bae` — is
met. The mechanism of the original discrepancy is **unknown**, its effect on the
first run's published figures is one recording of 328 with the verdict unchanged
and mean F moved by 0.00006, and it stays on the record unexplained.

## Where the recall is lost, on the corpora that look like the product

`oracle_activation_rwc.json` and `oracle_activation_harmonix.json`, from
`eval.oracle_activation`. No new mechanism and no hypothesis: the existing
oracle harness, pointed at the two full-length corpora it had never been run
on. Recall at 70 ms after a five-second warm-up; the oracle arm feeds the filter
a pulse at every annotated beat and nothing else, through the same
`LiveTracker.observe` seam.

| corpus | n | real | oracle | front end costs | **decoder costs** |
|---|---:|---:|---:|---:|---:|
| Harmonix | 581 | 80.7 | 99.3 | 18.6 | **0.7** |
| RWC-Pop | 100 | 80.2 | 98.0 | 17.8 | **2.0** |
| RWC royalty-free | 15 | 62.4 | 95.8 | 33.4 | 4.2 |
| GTZAN *(published)* | 998 | 66.7 | 92.7 | 26.0 | 7.3 |
| RWC-Genre | 102 | 59.0 | 83.8 | 24.8 | 16.2 |
| RWC-Jazz | 50 | 51.1 | 85.3 | 34.2 | 14.7 |
| RWC-Classical | 61 | 35.5 | 56.0 | 20.5 | **44.0** |
| SMC *(published)* | 217 | 26.1 | 54.6 | 29.0 | 45.4 |

### The cascade A/B is answered before it was run

`tracking/live.hpp` has carried this for a long time: BeatNet's paper reports
0.754 beat F on GTZAN from their two-level cascade against 0.666 from the same
activation through this tracker, and "the A/B that would make it a measurement
is the next piece of work". The measurement above makes it unnecessary for the
material the product is for. **On 681 full-length pop and rock recordings the
decoder loses between 0.7 and 2.0 points.** A better decoder cannot recover
what is not being lost, and porting a cascade would have been building for two
points on pop.

The 45-point decoder loss that made it look urgent is an SMC number, and
RWC-Classical reproduces it at 44.0. That is not a different corpus of songs, it
is a different kind of material: the decoder's loss tracks tempo irregularity
and essentially nothing else — rho −0.66, −0.82 and −0.65 on pop, genre and
jazz, and −0.01 on classical only because its median spread is 0.197 and the
axis has stopped discriminating. Harmonix's median spread is **0.000**.

### What that last number means, and its limit

Harmonix is produced music, most of it cut to a click, so a decoder is being
asked the easiest question there is. That is not a flaw in the measurement — it
is the situation the product's first scenario is in, a user playing a released
track. It does **not** transfer to a band rehearsing, and nothing here measures
that case.

### Two knobs already in the core, unswept on real audio

With a perfect observation, dropping the anchor and loosening the filter both
recover a large share of the decoder's remaining loss on irregular material:

| | shipped | no anchor | roughening 0.08 |
|---|---:|---:|---:|
| RWC-Classical | 56.0 | 64.8 | **68.8** |
| RWC-Jazz | 85.3 | 86.1 | 90.0 |
| RWC-Genre | 83.8 | 87.9 | 87.8 |
| Harmonix | 99.3 | 99.6 | 99.3 |

Both are oracle-fed, so they say the filter is under-agile for irregular tempo
and not that loosening it survives a noisy observation — a more agile cloud also
chases noise, and that has not been measured here. It is worth a sweep on the
real activation before any new decoder is considered, and it is worth nothing
at all for the product's main case, where the numbers are already 99.3.

### What this settles

The remaining recall on full-length songs is **in front of the decoder**, and
`too_few_beats` was already the dominant surviving failure — 84.8% on Harmonix
and 93.5% on RWC. Both now point at the same place: the causal front end. The
one measured candidate for a better observation, Beat This!, is not causal, so
it does not apply to this path, and no causal replacement has been measured.

## What a perfect front end would buy, what causality costs, and what a room does

Three measurements, run together because each was about to be used to justify
building something.

### 1. A perfect observation nearly triples the usable rate — and leaves the octave

`oracle_usable_rwc.json`. The oracle bump from `oracle_activation`, scored
through `live_corpus_benchmark._score_one`. All 328 RWC recordings, no failures.

| corpus | n | usable | recall | what still fails |
|---|---:|---|---|---|
| RWC-Pop | 100 | 0.440 → **0.810** | 0.802 → 0.980 | `wrong_octave` **100%** |
| RWC royalty-free | 15 | 0.333 → 0.867 | 0.624 → 0.958 | `wrong_octave` 100% |
| RWC-Genre | 102 | 0.137 → 0.569 | 0.590 → 0.838 | `wrong_octave` 93% |
| RWC-Jazz | 50 | 0.100 → 0.520 | 0.511 → 0.853 | `wrong_octave` 92% |
| RWC-Classical | 61 | 0.000 → 0.033 | 0.355 → 0.560 | `wrong_octave` 83% |
| **all** | 328 | **0.207 → 0.549** | 0.600 → 0.837 | `wrong_octave` **90%** |

**A caveat that belongs beside the octave column and not below it.** The oracle
activation is a pulse of *equal height* on every beat, so it removes precisely
the amplitude difference that tells a level from its double. Some of that 90% is
the instrument, not the tracker. The recall and precision columns are the
trustworthy ones; the octave residual needs an accented-oracle control before it
is read as a finding.

### 2. Beat This! keeps its advantage without seeing the future

`beat_this_causal_gtzan.json`, 100 GTZAN recordings.

| the model may hear past a beat | F |
|---|---:|
| the whole file | 0.8767 |
| at most 5 s | 0.8689 |
| at most 3 s | 0.8660 |
| at most 2 s | 0.8633 |
| at most 1 s | 0.8459 |

**One second of lookahead costs 3.1 points of F.** That is the result, and it is
sound because all five arms share one postprocessor and differ only in how much
audio the model had heard.

~~The +0.102 advantage over BeatNet-through-this-tracker is the model, not the
bidirectionality; a causal version would still be roughly +0.07 ahead.~~ **That
sentence was wrong and is struck rather than deleted.** These arms decode with
Beat This!'s own `beats_and_downbeats` (`beat_this_causal.py:94`), while the
BeatNet figure it is compared against is an activation driven through
`LiveTracker`. The comparison therefore changes the model *and* the decoder at
once, and neither share is separable from it. Attributing +0.07 to the model is
not supported by this run. Separating it needs Beat This!'s activation through
the same `LiveTracker`, on a corpus `final0` did not train on — this one is
GTZAN, which it did, so the level was never quotable either.

At a one-second step the 1.5 s and 1 s arms read the same prefix, so there are
five distinct points, not six.

### 3. A real room, and a simulation that predicts neither the size nor the shape

Six Harmonix tracks played through a speaker and captured on a phone at 48 kHz
with the room noise deliberately kept. Five aligned; one is void.

| track | annotated bpm | F clean → room | room fails by |
|---|---:|---|---|
| `0837_nottonight` | 87.0 | 0.974 → 0.555 | recall, octave |
| `0116_goodies` | 102.0 | 0.976 → **0.984** | — |
| `0132_iceicebaby` | 115.7 | 0.896 → 0.635 | recall |
| `0707_halfwaygone` | 125.0 | 0.951 → **0.151** | recall |
| `0466_onthedarkside` | 172.0 | 0.810 → 0.337 | recall, octave |

Across the five, mean F 0.922 → 0.532 and usable 0.80 → 0.20.

`0116_goodies` is the control that makes the rest readable: 0.984 in the room,
through the same alignment, decode and scorer. A broken pipeline cannot score
0.98 against untouched annotations, so the others' collapse is the room.

**The simulation understates the damage by a factor of ten.** Its worst cell —
RT60 0.8 s at 10 dB SNR — costs 0.036 of mean F and takes usable from 0.402 to
0.274. The real captures cost **0.390** of mean F. That comparison uses no
tempo column and is the finding. The likely gap is the direct-to-reverberant
ratio: the synthetic tail carries 0.7 of the direct path's energy, and a phone
across a room from a speaker hears far more than that, before any microphone
response or phone-side processing. The five are not a random sample and are
easier than the sweep's subsample (clean F 0.922 against 0.804), which makes
the gap larger rather than smaller.

**It does not predict which recordings are damaged either, and an earlier
version of this section said it did.** That claim came from a tempo column
taken from `live_bpm` in `octave_ceiling_per_track_harmonix.json`, which is the
value the live tracker *held at the end of the file* — on `0707_halfwaygone` it
reads 217 for a song that is 125 BPM and that the same run tracked at 125 for
390 beats. Against the annotated tempo the two columns agree within 2% on only
51% of the subsample, and both orderings dissolve:

| annotated tempo | simulated drop | published as |
|---|---:|---:|
| 0–100 (n=37) | +0.048 | +0.029 |
| 100–140 (n=63) | +0.025 | +0.034 |
| 140–190 (n=15) | +0.057 | +0.046 |
| 190–300 (n=2) | +0.023 | **+0.081** |

The real five are not ordered by tempo either — the slowest loses 0.42, the next
loses nothing, and the largest loss is at 125 BPM. Spearman is −0.60 on n=5,
where nothing short of ±1.00 would mean anything. The same bad column also chose
the six: they were picked for a spread reaching 300 BPM and the set that came
back is 87 to 172, so **fast material was never recorded**, and the two arms
disagree hardest in exactly the band that has two tracks in simulation and none
on tape. **So the mechanism story —
that a reverb tail covers the next beat when beats are short — has no support
here, and neither does "the simulation gets the shape".** What is left is one
number: a real room costs about ten times what the simulation says, and what
distinguishes a `0116_goodies` from a `0707_halfwaygone` is not known.

The void recording is worth reading rather than skipping. `0875_redbelt`'s
windows split into two camps 1.21 beats apart — 2.741 s from the coherent sum,
3.300 s from what the recordist recalls. Scored at both, outside the accepted
set, it gives room F **0.490** and **0.100**. The ambiguity reaches the
conclusion whole, so the recording stays void; the sensitivity check is in the
artifact under `sensitivity`.

`0837_nottonight` was void in the first run and is not any more. It contains two
takes — playback was restarted about 15 s in — and no constant offset can
describe a file holding the same music twice. Discarding the abandoned take
puts the restart at 14.714 s and the alignment then passes its own unchanged
test. The four recordings that already scored carry no session notes and
reproduce to the digit, which is the control on that change.

### 4. What the room does to the activation

Registered in `research/eval/PREREGISTERED_room_diagnosis.md` before any
activation was dumped, with the reading of each candidate fixed there.
`room_activation.py`, on the five aligned captures, against the clean files.

| track | F clean → room | AUC clean → room | salience | floor | width | between beats | level slope | cross peak |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `0116_goodies` | 0.976 → **0.984** | 1.000 → **0.997** | 0.950 → 0.884 | 0.0005 → 0.025 | 0.055 → 0.096 | 0.05 → **0.49** | 0.36 | **0.688** |
| `0132_iceicebaby` | 0.896 → 0.635 | 0.989 → 0.940 | 0.857 → 0.678 | 0.0007 → 0.095 | 0.057 → 0.118 | 0.73 → 0.88 | 0.41 | 0.492 |
| `0837_nottonight` | 0.974 → 0.555 | 1.000 → 0.957 | 0.941 → 0.745 | 0.0003 → 0.090 | 0.050 → 0.100 | 0.02 → 0.97 | 0.78 | 0.447 |
| `0466_onthedarkside` | 0.810 → 0.337 | 0.983 → 0.868 | 0.816 → 0.492 | 0.0030 → 0.174 | 0.085 → 0.167 | 0.28 → 0.87 | 0.36 | 0.321 |
| `0707_halfwaygone` | 0.951 → **0.151** | 0.999 → **0.842** | 0.922 → 0.480 | 0.0007 → **0.205** | 0.060 → 0.188 | 0.05 → **1.09** | 0.70 | 0.307 |

**The damage is in the observation.** The registered discriminability test is
the only candidate that survives its falsifier: AUC falls by 0.115 and 0.157 on
the two worst recordings and by 0.003 on the one that survived. Underneath it,
every part of the beat channel degrades at once — the beats lose half their
height, the floor between them rises **fifty to three hundred fold**, the peaks
roughly double in width, and on the three worst the loudest thing between two
beats is as tall as the beats themselves (0.87, 0.97, 1.09; 1.09 means it is
taller). None of this is a timing error: the cross-correlation lag is 0.000 on
four of the five.

**The phone's gain control is ruled out, and cleanly.** It is real — every
capture compresses, slopes 0.36 to 0.78 — but it runs *backwards* against the
damage: the recording that survived is the most compressed of the five, and the
worst-hit is among the least. Spearman +0.20. This is what the falsifier was
registered for.

Two registered rules did not work, and are recorded rather than adjusted:

* The **decoder** candidate's falsifier is ill-posed. It is defined as "the
  observation survived", and the control recording is the one whose observation
  survived, so it fires there by construction and the rule rejects it. The
  rejection carries no information. It should have been conditioned on the F
  loss.
* The **doubling** threshold was written as a rise of ≥0.15 where a level was
  needed. Every capture rises by more than that, including the survivor, so it
  is rejected — while the level separates the survivor (0.49) from the four
  collapsed (0.87 to 1.09) perfectly. The threshold stands as registered.

**Every observation statistic orders the damage; the one non-observation
statistic does not.**

| statistic | Spearman against F lost |
|---|---:|
| cross-correlation peak against the clean activation | **−1.00** |
| AUC drop / room AUC / room salience / room floor / peak width | ±0.90 |
| between-beat height | +0.70 |
| level slope (gain control) | +0.20 |

How much the room activation still *resembles* the clean one ranks the five
exactly. Eight statistics were examined, so that −1.00 is p = 0.017 alone and
0.13 after correcting for the eight, and none of them was the registered
primary: it is suggestive and it is not established. The consistency across
five statistics that measure different things is worth more here than any one
coefficient.

**What this means for the live path.** The tracker is not making bad decisions
about a good observation in a room; it is being handed an activation in which
the beats and the gaps have nearly the same height. That points the work at the
front end for microphone input — preprocessing, or a model that has heard a
room — and it says the decoder-side ideas that keep being proposed will run on
noise again, exactly as in the earlier salience-versus-decoder finding.

### 5. The room damage does not come back out — and `LiveTracker` is not the place

Registered in `research/eval/PREREGISTERED_room_repair.md`, constants fixed
there and none swept. Six arms on the five aligned captures: three transforming
the audio before BeatNet, three transforming the activation before
`LiveTracker`. Baseline 0.5323, clean ceiling 0.9216, half-gap target 0.7270.

**Replay parity is exact** — 6623 beats against 6623, maximum difference 0.0,
on all five — so the activation arms measured the repair and not the replay.

| arm | mean F | Δ | usable | survivor | |
|---|---:|---:|---:|---:|---|
| baseline (room) | 0.5323 | — | 0.20 | 0.984 | |
| replay, untouched | 0.5323 | +0.0000 | 0.20 | 0.984 | parity |
| `act_subtract_floor` | 0.5204 | −0.0119 | 0.20 | 0.988 | |
| `audio_gate` | 0.4982 | −0.0341 | 0.00 | 0.715 | disqualified |
| `act_normalise` | 0.4828 | −0.0495 | 0.00 | 0.794 | disqualified |
| `act_sharpen` | 0.4465 | −0.0858 | 0.00 | 0.867 | disqualified |
| `audio_both` | 0.4376 | −0.0947 | 0.20 | 0.811 | disqualified |
| `audio_dereverb` | 0.3481 | −0.1842 | 0.00 | 0.405 | disqualified |

**Not one arm helps, and five of six are disqualified for wrecking the
recording that already worked.** By the registered trichotomy this is the last
row: the damage is not removable by cheap post-hoc processing on either side,
and the answer is a model that has heard a room. **So `LiveTracker` does not
need fixing for this** — an observation model that subtracts the risen floor is
the most direct decoder-side answer to the diagnosis, it is `act_subtract_floor`,
and it moves the mean by −0.012.

That also settles the argument the split was built to settle. The claim that all
room damage reaches the tracker through the activation is true and does not
imply the front end is where the repair goes; here neither side takes it out,
which no amount of reasoning about what `LiveTracker` can see would have told
us.

**What the per-track numbers add, and what they do not.**

| track | baseline | best arm | worst arm |
|---|---:|---:|---:|
| `0116_goodies` | 0.984 | 0.988 `subtract_floor` | 0.405 `dereverb` |
| `0132_iceicebaby` | 0.635 | **0.781** `audio_gate` | 0.443 `dereverb` |
| `0837_nottonight` | 0.555 | 0.562 `subtract_floor` | 0.329 `sharpen` |
| `0466_onthedarkside` | 0.337 | 0.343 `normalise` | 0.162 `subtract_floor` |
| `0707_halfwaygone` | 0.151 | **0.427** `act_sharpen` | 0.151 baseline |

The transforms move individual recordings a great deal — the worst capture
nearly triples under `act_sharpen`, and `audio_gate` is worth +0.146 on
`iceicebaby` — but each helps a *different* recording and each pays for it
somewhere else. `act_sharpen` takes `0707` from 0.151 to 0.427 while taking
`0837` from 0.555 to 0.329.

That is the signature of a transform whose right strength depends on the
recording, and it is a **hypothesis for new data, not a result**. Choosing per
recording, or sweeping strength, on these same five would be fitting the set the
answer is read on; the registration says so and it is worth repeating here,
because the temptation is exactly proportional to how good 0.151 → 0.427 looks.

**Where this points.** A model that has heard a room. The five captures are the
only real room data in the repository, and
[section 3](#3-a-real-room-and-a-simulation-that-predicts-neither-the-size-nor-the-shape)
found the synthetic room understates the damage tenfold and does not predict
which recordings suffer — so augmenting training with `room_degradation.py`'s
impulse responses would be training on the wrong distribution. Real captures are
needed, and they are needed as *training* data now and not only as a
measurement.

The artifact records `clean: false`: the script under test was untracked when it
ran, which is the change itself, and the commit that follows it is that script.

### 6. The room, measured — and a simulation that still fails, differently

A second session recorded a swept sine, sixty seconds of silence and two of the
same tracks, with nothing moved between them. Registered in
`PREREGISTERED_room_simulation.md` before the simulation was built.

**The chain is not linear time-invariant.** Three identical sweeps 13 s apart,
at levels within 0.6 dB, give responses correlating 0.90 to 0.96. Three
alternative explanations were tested and refuted:

| explanation | test | result |
|---|---|---|
| noise in the tail | correlate over growing windows | already 0.944 at 20 ms, 8 dB below peak; flat thereafter |
| sub-sample misalignment | align to 0.02 of a sample | adds at most 0.012 |
| no signal in the varying bands | measurement SNR per band | 104 dB at 30–60 Hz |

**The variation is all in the bottom two octaves** — 14.2 dB of spread at
30–60 Hz, 9.8 dB at 60–125 Hz, against ≤2 dB above 125 Hz and ≤0.2 dB above
2 kHz. The top of the chain is highly repeatable; whatever the phone does below
125 Hz, it does differently each time.

**Two measurements settle an older question on their own.** RT60 is 0.33–0.37 s,
*shorter* than the 0.4 and 0.8 the invented room used, and the captures' SNR is
17 dB where it used 10. **The invented room was harsher than the real one on
both knobs and cost 0.036 where the real one costs 0.390.** The tail was never
too short and the noise was never too quiet — the character of the response was
wrong, not its size.

**The measured simulation fails its registered acceptance:**

| track | clean | real room | simulated | |
|---|---:|---:|---:|---|
| `0116_goodies` | 0.976 | 0.938 | **0.809** | error −0.129, tolerance ±0.05 |
| `0707_halfwaygone` | 0.951 | 0.204–0.340 (void) | **0.581** | outside the interval |

Criterion 1 (level) fails. Criterion 2 (ordering) **passes** — and that is new,
because the invented room damaged everything equally.

The failure has one shape: **the simulation damages uniformly where the room
does not.** Real drops are 0.038 and ≈0.68; simulated drops are 0.167 and 0.37.
It over-damages the track that survives by four times and under-damages the one
that collapses by half. A convolution applies one filter to all material, so it
must damage all material similarly — and the room's defining property, the one
nothing has yet predicted, is that it destroys some recordings and leaves others
alone.

**Not approved for augmentation.** The fallback is real captures.

One caveat on the registered bar, which is worth stating because it is not a
reason to move it. "Reproduce the room" is the right criterion for a simulator
used to *predict* a rate. For a simulator used to *augment training* the
criterion is different — training does not need the exact room, it needs a
distribution containing it — and that criterion is end-to-end: train with it,
evaluate on real captures. That test is not this one and was not run.

### 7. The front end is worth more in a room than anywhere else

Registered in `PREREGISTERED_beat_this_front_end.md`. Beat This!'s activation
driven through the **same** `LiveTracker` BeatNet is driven through, so the
decoder is constant and only the observation differs.

**The delivery control first.** The Beat This! arm reaches the tracker through
`--live-activation` on an analytic availability delay, while the shipped path
uses BeatNet's recorded release schedule. BeatNet's own activation through the
same replay seam scores 0.5413 against 0.5323 — **the delivery path is worth
+0.009**, so the comparison below is between models and not between seams.

| track | BeatNet room | Beat This! room | clean, Beat This! |
|---|---:|---:|---:|
| `0116_goodies` | 0.984 | 0.960 | 0.998 |
| `0132_iceicebaby` | 0.635 | **0.778** | 0.986 |
| `0466_onthedarkside` | 0.337 | **0.571** | 0.940 |
| `0707_halfwaygone` | 0.151 | **0.548** | 0.981 |
| `0837_nottonight` | 0.555 | **0.911** | 0.991 |
| **mean** | 0.5413 | **0.7536** | 0.9792 |

**+0.212 of F in a room, with the decoder held constant.** That is the largest
single improvement measured anywhere in this repository, and it is on real
captures rather than on a corpus. The worst recording goes from 0.151 to 0.548.

**And the room still costs the better model 0.226.** Beat This! falls from 0.979
to 0.754. The registered reading of question 1 holds as written: on the two
tracks where BeatNet's AUC fell by ≥0.10, Beat This!'s fell by 0.137 and 0.097,
both more than half — **the room damages any front end, and no model choice
avoids needing room data.** Both statements are true at once and neither
replaces the other.

The activation tells the same story more directly. On `0707_halfwaygone` the
room floor is 0.0133 under Beat This! against **0.2049** under BeatNet — fifteen
times lower — and salience 0.751 against 0.480. The gaps still fill, but far
less.

**On clean files, decoder held constant, 100 GTZAN recordings by stride:**

| front end | mean F | usable |
|---|---:|---:|
| BeatNet | 0.6813 | 0.48 |
| Beat This! | **0.8196** | **0.68** |

**+0.138 of F and +20 points of usable.** Registered threshold was 0.03. The
registration's contamination premise was later corrected: official `final*`
checkpoints exclude GTZAN, and BeatNet `model_1` also withholds it. This is
therefore a held-out, matched-decoder system gap, not a train-on-test upper
bound. It still differs in training corpora, recipe and capacity, was produced
from a dirty tree, and used 100 recordings selected by stride, so P0 requires a
clean full-split repeat before quoting it as the benchmark level. What it does
already establish is that the +0.102 struck from
[section 2](#2-what-a-perfect-front-end-would-buy-what-causality-costs-and-what-a-room-does)
was not inflated by the decoder swap — held constant, the gap is larger, not
smaller.

Two faults in the registration, recorded rather than adjusted:

* **AUC drop was the wrong statistic to hang question 1 on.** It is a difference
  between two numbers both near 1.000, and it says "damaged similarly" while the
  absolute floor differs fifteenfold and the beats recovered differ by 0.212.
  The registered reading is reported above as registered; it is answering a
  narrower question than the one that mattered.
* **`between_beat_ratio` divides by the flanking beats** and returns 14.79 on
  one Beat This! room arm, where a few intervals have near-zero flanks. As a
  level it is unreadable in that regime; only its clean-versus-room direction is.

### What these three change together

The oracle budget said the decoder costs 0.7 to 2.0 points on full-length songs
and the work is all in the front end. **That is true of clean files and false in
a room.** Four of five real captures lost between 0.26 and 0.80 of F — more than
the front end, the decoder and the octave combined. Any claim about where the
live path loses its beats now has to say whether it is talking about a decoded
file or a microphone, because on this evidence they are different problems.

The corollary is about method, not about rooms: **the tempo of a recording comes
from its annotations.** A tracker's own reading is an outcome, and using it as a
covariate lets the thing being measured choose the axis it is measured on. The
first version of this section did that and produced a monotone table in both
arms that is not there.
