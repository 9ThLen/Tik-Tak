# Spike Manifest

## Idea

Test whether BeatNet's causal downbeat head can veto live octave-switch
proposals at the actual anchor decision, improving long wrong-level episodes
without buying the result by suppressing correct locked time.

## Requirements

- Live tracking is the primary product path.
- Beat This! `small0` remains an optional offline comparison backend.
- The registered A1--A4 gates decide adoption; a mixed result is not approval.
- RWC selects every free number. Harmonix is opened once after that selection is
  committed, and is never used to retune.
- The decoder formula has one implementation. Policy effects run through the
  real C++ particle filter, not a Python reimplementation.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|---|---|---|---|---|
| 001 | octave-veto-a1-a4 | standard | Given real live octave proposals, when a causal downbeat veto is replayed at matched cost, then A1--A4 and all standing gates hold | PENDING | live, tempo, downbeat, replay |
