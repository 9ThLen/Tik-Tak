# Pre-registered: calibrating the learned path's accent gate

Written before the run.

## Why

On the learned file path, `downbeat_confident` decides whether the user hears
an accent at all, and its thresholds (`min_phase_margin` 0.25,
`min_meter_margin` 0.40, `min_salience_range` 0.05) are the cue backend's,
carried over by argument; a probability is not in the cues' units. The first
GTZAN run found them already usable — 65.7% of recordings accented, 7.2% of
those mostly wrong — but that is a reading of provisional numbers on the
material they would be tuned on, not a calibration.

## Procedure

The one `research/eval/README.md` already fixes for the cue backend, applied to
the learned path without change of budgets:

* **Material:** GTZAN, held out from `small0`, through `dump_analysis --learned`
  (the product path), one run per recording at the commit that adds this file.
* **Split:** by the first byte of SHA-256 of the recording name — even to
  validation, odd to held-out — fixed before any result exists. Each excerpt is
  its own group: GTZAN repeats some artists and recordings, so the Wilson bounds
  below are, if anything, optimistic, and that is stated beside the result.
* **Choice:** `eval.downbeat.sweep` over the observed margins on the validation
  half, then `choose_margins` with `max_wrong_rate` 0.05 and
  `max_conditional_error` 0.10, both applied to the upper 95% Wilson bound: the
  widest coverage inside both budgets.
* **Check:** the chosen pair once on the held-out half — coverage, wrong rate
  and conditional error with their Wilson bounds — beside the provisional pair
  on both halves.

## Decision

If the held-out wrong rate and conditional error sit inside their budgets at
the upper bound, the chosen pair replaces the provisional one as
`OfflineConfig::learned_downbeat`'s defaults. If they do not, the provisional
pair stays, marked as provisional, and the shortfall in groups is reported
(`evidence_gap`).
