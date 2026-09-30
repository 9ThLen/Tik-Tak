# Pre-registered: the level floor on real room captures (P1-B0)

Written before any run. This is the next step that
`PREREGISTERED_octave_hold_and_level_floor.md` named. There, at -24 dB on RWC,
`--live-level-floor -20` recovered 89% of the beat F lost and 62% of the usable
rate, and it cost GTZAN nothing on usable at the recording's own level.

## Why a room

The P1-B0 replay (`research/results/P1B0_RECORDS_20260929.json`, branch
`claude/p1b0-replay-programme`) found that a room costs frozen BeatNet 0.32-0.46
of F on held-out GTZAN, and quiet playback costs about 0.1 more. No input-level
repair was ever tried on it.

The captures are quiet. Their levels were measured before this was written;
no F was looked at. Median RMS:

| takes | median RMS |
|---|---|
| phone, normal level | -40 to -42 dBFS |
| phone, quiet level | -48 to -52 dBFS |
| loopback (the programme itself), normal | -21.5 dBFS |
| loopback, quiet | -33.5 dBFS |

So the floor lifts every phone take, by up to its +30 dB limit. That makes
this a test of the level's share of the room loss, and of nothing subtler.

## Material

The P1-B0 aligned takes in `music/replay-p1b0/aligned/`, cut by that branch's
registered slate alignment, used here as data. Each take's SHA-256 goes into
the provenance.

* 79 phone room takes. `near/normal` has 19, because one take failed alignment
  (A1 79/80); the other three cells have 20 each.
* 40 loopback takes, 20 at each level.

The annotations are GTZAN's, and GTZAN is held out from `model_1`.

## Arms

One binary from the commit that adds this file, in a clean worktree, BeatNet
`model_1`, scored per take with `live_corpus_benchmark._score_one`, exactly as
P1-B0 scored them, and `--live-sample-hz 50`.

| arm | flags |
|---|---|
| off | — |
| floor | `--live-level-floor -20` |

Each arm runs at seeds default, 1 and 2 (`--live-rng-seed`), and the arms are
paired seed by seed.

## Endpoints

The unit is the excerpt. For each excerpt, each cell and each seed, take
`F(floor) - F(off)`, then average over the three seeds. That gives one paired
difference per excerpt and cell.

* **Primary.** The mean over excerpts of the two quiet cells, averaged per
  excerpt so that near and far of one excerpt are not counted as independent.
  It is reported with a 95% percentile bootstrap over excerpts (10,000 draws).
* **Harm check.** The same for the two normal cells.
* **Reported beside them:**
  * each cell separately;
  * the across-seed SD of each cell's mean F in both arms;
  * usable;
  * episode-free and the share of time active, since at SMC's own level the
    floor raised wrong-level episodes;
  * the room loss against the loopback within each arm;
  * the floor's effect on the loopback alone.

## Decision

* **Proceed.** The primary gain is at least +0.05 (P1-B0's A2 half-width) with
  its interval excluding zero, and the harm check's lower bound is above -0.05.
  The floor is then proposed as the live path's default. That is a product
  change for the project's owner, and the next capture session repeats this
  comparison as its confirmation.
* **Underpowered.** The primary gain is positive but below +0.05, or its
  interval reaches zero. The floor stays off, and the number of captures
  needed for ±0.05 is reported with the formula A2 used.
* **Harm.** The primary change is negative with its interval excluding zero,
  or the harm check's whole interval is below zero. The floor is closed for
  rooms.

Twenty excerpts are few. The rule asks for a large, consistent effect before
anything moves, and says in advance that a smaller one will be reported as
underpowered, not as absent.
