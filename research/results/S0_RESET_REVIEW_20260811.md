# S0 reset-horizon result — preregistered review

## Overall assessment: ready to use, with two recorded output caveats

S0 is **positive** under the acceptance rule fixed in
`research/eval/PREREGISTERED_M0a_S0.md`. Carrying BeatNet's recurrent state
improves bar-phase F1 by substantially more than the registered five-point
margin on both required development corpora, and every adjacent confidence
interval satisfies the registered monotonicity rule.

This result promotes stateful block training (S1) to an ablation over A2–A4. It
does not show what a statefully trained model will achieve, does not establish a
meter-family result, and is not a locked-test claim.

## Artifact and population

- Raw artifact: `s0_reset_gtzan_harmonix_20260811.raw.json`
- Raw SHA-256: `b643d2a310fa490eee3f420cee02361dd4a375bdd94f5117f95e4b082417a3ec`
- Run commit: `5d645b20b40d5b0873e5d08075f12274274f8c20`
- Clean-tree provenance: `true`
- Model: `beatnet_model_1.ttw`, SHA-256
  `812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1`
- Binary SHA-256:
  `e04881ec4344e451cbdbb44c56ffb7c4b98408ba0d1eff2fc129d1ded620b426`
- Manifest SHA-256:
  `81eceb2edcc0b9915a5bc60992cb2142ad30c88339427faaf4de38ab50e6e5cd`
- Started: `2026-08-11T03:06:53Z`; elapsed: 16,176.8 s (4 h 29 min 37 s)
- Selected: 1,581; scored: 1,580; technical exclusions: 1
- Resampling: 2,000 deterministic paired recording-level bootstrap draws

All 1,580 scored records have unique `(corpus, name)` keys, all six registered
arms, exact reset schedules, and R∞ replay parity. Accounting closes exactly:
`1,580 scored + 1 excluded = 1,581 selected`.

## Primary result

Macro means are recording-level bar-phase F1 after the common 2.0 s initial
cut. Confidence intervals are paired at recording level.

| corpus | n | R2 | R4 | R8 | R16 | R32 | R∞ | R∞−R2 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GTZAN | 999 | 0.262 | 0.324 | 0.363 | 0.403 | 0.420 | 0.420 | +0.158 [0.136, 0.180] |
| Harmonix | 581 | 0.308 | 0.381 | 0.441 | 0.481 | 0.514 | 0.545 | +0.237 [0.217, 0.258] |

The registered positive-margin condition requires the lower bound to be at
least `0.05` on both corpora. Both clear it by a wide margin.

| adjacent step | GTZAN mean [95% CI] | Harmonix mean [95% CI] |
|---|---:|---:|
| R4−R2 | +0.062 [0.046, 0.078] | +0.073 [0.062, 0.082] |
| R8−R4 | +0.039 [0.023, 0.054] | +0.060 [0.051, 0.069] |
| R16−R8 | +0.041 [0.026, 0.055] | +0.040 [0.033, 0.047] |
| R32−R16 | +0.016 [0.006, 0.027] | +0.033 [0.027, 0.040] |
| R∞−R32 | 0.000 [0.000, 0.000] | +0.031 [0.024, 0.039] |

No adjacent lower bound falls below `−0.01`; monotonicity therefore passes
without using the one-exception allowance. GTZAN's exact R32/R∞ equality is
structural: all 999 scored clips are shorter than the first 32 s reset. It is
not evidence that memory saturates at 32 s. Harmonix still gains beyond R32.

## Secondary metrics

The raw artifact's GTZAN downbeat aggregate is `NaN` because six records have
no scorable downbeat reference after the common cut and the first writer used
plain `mean`. Finite-only recomputation gives:

| corpus | support | beat F, R2 → R∞ | downbeat F, R2 → R∞ |
|---|---:|---:|---:|
| GTZAN | beat 999; downbeat 993 | 0.555 → 0.656 | 0.211 → 0.343 |
| Harmonix | beat/downbeat 581 | 0.664 → 0.790 | 0.300 → 0.532 |

This secondary serialization defect does not enter bar-phase F1, its paired
intervals, monotonicity, or the S0 verdict. The harness is corrected after this
run to emit `null`, report `n_scored`, and reject non-standard JSON on write.
The raw artifact is retained unchanged so its recorded SHA-256 remains useful.

## Exclusion and sensitivity

`gtzan/jazz.00054` was excluded because the `.wav` bytes have no RIFF/WAVE,
FLAC, or MP3 header; the file begins with apparent raw PCM. This is a technical
input defect, not a model-dependent exclusion. The annotation remains digested
in the artifact.

As a deliberately hostile sensitivity check, assigning this one excluded track
the minimum possible paired delta (`R∞−R2 = −1`) leaves the GTZAN result at
`+0.157 [0.135, 0.179]`, still well above the registered `0.05` lower-bound
threshold. The exclusion therefore cannot change the S0 decision.

## Independent calculation checks

- Recomputed all 12 primary arm means from per-record values: match.
- Recomputed both R∞−R2 intervals from the registered deterministic bootstrap:
  exact match.
- Recomputed all ten adjacent intervals: exact match.
- Reapplied the registered positive/negative/inconclusive decision logic:
  `positive`.
- Checked metric domains and finiteness: all primary phase F1 and beat F values
  are finite and in `[0, 1]`.

## Required caveats

1. GTZAN and Harmonix are development corpora here; this is a procedural gate,
   not a final generalisation claim.
2. S0 isolates the inference-time value of state in the frozen published
   BeatNet. It supports trying S1 but cannot estimate S1's eventual gain.
3. The raw file is Python's historical JSON dialect because it contains six
   literal `NaN` secondary values. Use the raw file for provenance and per-record
   evidence; use the finite-only values above for the affected aggregate.
