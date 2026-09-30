# Pre-registered: 2048 particles, paired by seed, off RWC

Written before any run.

## Why

In `PREREGISTERED_quickfix_diagnostics.md` Q1 (RWC, five seeds), 2048 particles
against the shipped 512 failed the registered usable rule: +0.003 usable, which
is inside the noise. It did gain **+0.020 beat F** at an across-seed SD of
0.001, and it cut the seed noise on the verdict metrics by a half to two
thirds. RWC is a development corpus, and that run did not measure the cost.
Q1's 2048 arms also took about 50% longer through the eight-worker benchmark.

## Arms

One binary: `dump_analysis.exe` at SHA-256 `8494fd25622e…`, the
octave-and-level build, in a clean worktree at the commit that adds this file.
BeatNet `model_1`, `eval.live_corpus_benchmark --mode model --per-track`,
`--live-sample-hz 50`, and 8 workers.

| arm | flags |
|---|---|
| 512 (ships) | — |
| 2048 | `--live-particles 2048` |

Each arm runs at seeds default, 1 and 2, on **GTZAN** (1000 recordings, held
out from `model_1`) and **Harmonix** (581, never used for a particle
decision). Where batch three already ran the 512 arm with this same binary
and the same flags, that run is reused rather than repeated:

* Harmonix: `c_off_seed{0,1,2}_harmonix`;
* GTZAN, default seed: `l_off_gain0_gtzan`.

The tracker is deterministic for a fixed binary and flags, and runs of
unchanged paths have reproduced exactly (`octave_hold_and_level_floor`).

**Cost.** After the benchmark arms, with nothing else scheduled, each
configuration times `dump_analysis --live` single-process on the same five
GTZAN and five Harmonix recordings. The runs are interleaved 512, 2048, 512,
2048, 512, 2048, so that background load drifts equally over both. The
statistic is the median, over the ten recordings, of the ratio of wall time
per second of audio, 2048 over 512. `dump_analysis` is single-threaded, so for
one process on an otherwise idle machine, wall time stands in for CPU time
(`eval/particle_cost.py`).

## Endpoints

* **Primary:** beat F, 2048 minus 512, macro over corpora with n >= 30, paired
  seed by seed. Both the three per-seed differences and their mean are
  reported, per corpus.
* **Also reported:** usable, episode-free, correct time and the cost ratio.
  Each gets its per-seed paired differences and the across-seed SD of each
  arm.

## Decision

* **Proposed as a default.** On both corpora, 2048 raises beat F by at least
  0.010 averaged over seeds, and every one of the three seed-paired
  differences is positive. Usable and episode-free must also not fall by more
  than twice the larger across-seed SD. The proposal covers the desktop and
  offline-capable paths. The mobile default waits for the cost to be measured
  on a phone, which is a product decision for the project's owner.
* **Otherwise** 512 stays, and the RWC gain is reported as RWC-only.
