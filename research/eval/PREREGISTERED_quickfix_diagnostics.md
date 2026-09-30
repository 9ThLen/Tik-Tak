# Pre-registered: four diagnostics after the activation-tempo fix

Written before any run. The arms, corpora and endpoints below were fixed before
a single number was produced, and the decision rules say in advance what each
answer will be used for.

**This is exploratory, on development corpora, except where stated.** Nothing
here changes a product default by itself. A result can make a later
confirmatory run worth registering; the rules below say which.

## What is under test

| | commit | binary |
|---|---|---|
| base | `75fcde7` (main) | `tools/eval/build/RelWithDebInfo/dump_analysis.exe`, built in a clean worktree |
| quickfix | branch `claude/core-quickfix` at the commit that adds this file | the same, from its own clean worktree |

The quickfix commits change the live path in one place: `ActivationTempo` bins
frames with a representation slack instead of a bare `floor()`, refreshes on
the same slack, answers nothing for a numerically flat window, and masks gaps
instead of writing zeros. The last is inert on the bench, where nothing is
gated or dropped. The offline tracker change (the trimmed-start objective) does
not reach the live beats, which never read the offline grid unless
`--live-seeded` is passed, and it is not.

Model: `beatnet_model_1.ttw`, SHA-256 `812ed11a…78f1`. Harness:
`eval.live_corpus_benchmark --mode model --per-track`, 8 workers, every arm with
`--live-sample-hz 50` — one second aliases acquisition by about five seconds
(results README, "Acquisition was measured on a grid too coarse to see it").
Per-track outputs are written outside the repository so that `tree_clean`
holds for every arm.

Corpus status, from `research/results/README.md`: RWC and Harmonix are
development corpora; GTZAN is out of `model_1`'s training (fold 1) and may be
quoted for it; SMC is out of all three but has little resolution.

Endpoints, as the live-path memory fixes them: **`no_wrong_level_episode_fraction`**
and **`usable_rate`** (macro over corpora with n >= 30), `usable_rate_strict`
beside them, `mean_correct_share_of_eligible` as a guard rail, beat F reported
and not decided on. Paired per recording wherever two arms share a corpus.

## Q0 — what the activation-tempo fix does on the bench

Base against quickfix, default flags, on **RWC (328)**, **Harmonix (581)** and
**GTZAN (1000)**.

At stream origin 0 the binning defect touched 1-2% of frames, and the flat
guard acts only where BeatNet's output is exactly constant (digital silence).
The expected mean effect is therefore small and of unknown sign; the tracker is
chaotic, so individual recordings will move either way.

*Decision:* none about shipping — the fix corrects a translation dependence
that is proven by test, and it stands whatever this says. What Q0 does decide
is the baseline: every later comparison uses the quickfix binary, and the
published live figures are re-read against it, not against `75fcde7`.

## Q1 — how much of a verdict is the particle draw

Quickfix, RWC, RNG seed in {core default, 1, 2, 3, 4} crossed with particles in
{512, 2048}: ten arms, one of which is Q0's quickfix RWC run.

Reported: the share of recordings whose usable verdict flips across the five
512-particle seeds; the spread (min, max, SD) of macro usable and episode-free
across those seeds; and 512 against 2048, paired within each seed.

*Decision:* the across-seed SD of macro usable at 512 becomes the noise floor
for Q2 and Q3. If 2048 particles beat 512 on usable by more than that SD on at
least four of five seeds, a confirmatory run on Harmonix is registered before
the default moves.

## Q2 — holding the octave for longer than six seconds

Quickfix, RWC, the research resolver seam: `--live-octave-debounce D` for D in
{10, 20, 30, 60} s, and `--live-octave-ban`. Five arms against Q0's baseline.

On RWC the earlier sweep found debounce up to 6 s flat and the total ban
ahead on usable (+3.4) and episode-free (+5.5) at a cost of 3.4 points of
correct time (`octave_veto_rwc.json`). The debounce timer resets on any
estimate inside the octave (`dump_analysis.cpp`, `OnlineOctavePolicy::resolve`),
which may be why the short debounces did nothing; this is the first test of the
long ones.

*Decision:* an arm is a candidate if it beats baseline on episode-free and on
usable by more than the Q1 noise floor while `mean_correct_share_of_eligible`
falls by less than 0.03. A candidate earns a confirmatory registration on
Harmonix, which has never seen these arms. No arm, candidate or not, moves
into the core from this run.

## Q3 — how much the level alone costs BeatNet

Quickfix, `--live-input-gain-db G` for G in {-24, -12, -6, +6, +12} on
**RWC** and **SMC**: ten arms against the 0 dB baseline (Q0's RWC run, and one
0 dB SMC run).

BeatNet's features are `log10(1 + |X|)` with nothing in front, so a quieter
input moves them toward the linear regime; phone captures arrive 14.5-27 dB
below their sources (P1-B0). This is a digital dose-response only: it prices
the level, not the room.

*Decision:* if -12 dB or -24 dB loses more than twice the Q1 noise floor in
macro usable on RWC, a boost-only level normaliser inside `BeatNetFeatures` is
worth building, and its test on the P1-B0 captures is registered before it is
run. If neither does, the level is not where BeatNet's room loss lives and the
normaliser is not built.

## What is not asked here

The click gate (a speaker-mode metronome blinding itself around its own
clicks) needs a closed-loop harness that mixes the click into the input, and
anti-aliased resampling into BeatNet needs a core change behind a flag. Both
are next, and both will be registered on their own.
