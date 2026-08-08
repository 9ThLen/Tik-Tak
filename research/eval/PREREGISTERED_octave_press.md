# Pre-registered: how much of the octave ceiling does a person pressing a button actually recover?

Written 2026-08-08, before the harness that answers it exists and before any
number has been produced. Commit `602f9ad` is the state of the core it is
written against: `tt_live_set_octave_offset` exists, is tested, and has never
been run on a corpus.

## Why this document exists at all

Five experiments have now asked whether the live tracker can decide its own
metrical level. The last of them closed the question: the beat-synchronous
decoder judging real switch proposals scored **below chance wherever it acted**
and exactly 0.500 only where it was inert
(`PREREGISTERED_octave_veto.md`, verdict in `results/README.md`).

A sixth experiment then asked what the answer would be worth if it existed, and
found the ceiling is corpus-dependent and lower than the line had assumed:
`usable_any_octave` buys **+21.0 points on RWC-Pop, +20.3 on Harmonix, +4.7 on
GTZAN and +0.0 on RWC-Classical**. Pooled over all 328 RWC recordings it is
**+11.3 points**.

`docs/PLAN.md` has named ×2 / ÷2 as the mitigation for this risk since its first
revision. The core could not hold such a press until `602f9ad`. So the question
this document registers is the one the whole line has been circling without
asking: **not whether the octave can be decided automatically — it cannot — but
how much of that ceiling a person is actually able to take.**

This must be registered before it is measured for a specific reason. The
simulated listener has free parameters — how long it takes to notice, how often
it may press — and those parameters move the answer. Choosing them after seeing
the answer would produce a number that means nothing.

## What is different about this experiment, and what is not

**Not different.** The unit is still the decision point, the corpora are still
RWC for development and Harmonix for transfer, the confirmatory corpus is still
untouched until the development corpus has fixed every free parameter, and a
failed acceptance gate still means "adoption not approved" and not "try again
with a different threshold".

**Different in one way that matters.** The previous five arms were automatic
policies, and their controls were shuffles and shifts of their own evidence.
This one has a human in it. The listener cannot be shuffled, so the control has
to be built differently — see §5 — and the listener has to be *specified* rather
than assumed, because an unspecified listener is a free parameter wearing a
costume.

## 1. What the simulated listener knows, and what it does not

**It knows the level.** At any instant it can tell whether the click is at the
annotated metrical level or an octave off it. This is not cheating and it is not
an oracle over the recording: a person singing to a track hears immediately that
the click is going twice as fast as the music. It is exactly the judgement a
listener is good at.

**It does not know anything else.** It cannot see the confidence, the anchor,
the margin, or the future. It cannot tell a 3:2 error from a correct grid — only
octave errors are visible to it, because only octave errors are what a ×2 button
addresses.

**It is not instant.** It takes `NOTICE_SEC` of continuous wrongness before it
acts, and it will only press `MAX_PRESSES` times in a recording.

This makes the arm an **upper bound on a realistic control and a lower bound on
nothing**. It is a tighter bound than `usable_any_octave`, which corrects a whole
recording retroactively and in one move; and it is looser than a real person,
who will sometimes be wrong, distracted, or holding an instrument. Both
directions are stated here so that neither can be claimed later.

## 2. What counts as a wrong level, exactly

At time `t`, let `ref(t)` be the annotated local tempo: `60 / m`, where `m` is
the median inter-beat interval over the annotated beats within ±4 beats of `t`.
Let `bpm(t)` be the tracker's currently published tempo.

Define `r = log2(bpm(t) / ref(t))` and `k = -round(r)`.

The state at `t` is **wrong-level** when both hold:

- `k != 0` — the tracker is not at the annotated level; and
- `|r + k| <= log2(1.08)` — what it *is* at is an octave multiple of the
  annotated level, within the same 8% tolerance the rest of this work uses.

The second clause is the one that keeps the experiment honest. A tracker at 1.5×
the annotated tempo is wrong, but it is not wrong in a way a ×2 button can fix,
and counting it would credit the button for errors it cannot address. `|k| >= 2`
is in scope and is pressed for in one move.

Both clauses use `bpm(t)` as published through the C API, not the filter's
internal mean, because that is what a listener hears.

## 3. When the listener presses

A press fires at the first `t` such that:

1. the state has been continuously wrong-level throughout `[t - NOTICE_SEC, t]`;
2. `t >= WARMUP_SEC` (5.0 s, as everywhere else here) — before that there is
   nothing to be wrong about;
3. fewer than `MAX_PRESSES` presses have fired in this recording;
4. `t >= last_press_sec + NOTICE_SEC`.

Condition 4 exists so that a correction the cloud takes a second to complete does
not read as a second error and draw a second press. Without it the listener
would press repeatedly during its own correction and the press cap would be spent
on one event.

The press is `setOctaveOffset(current_offset + k)`, with `k` from §2 evaluated at
`t`. If the core **refuses** it (§4), the press is recorded as refused, no offset
changes, no press is consumed against `MAX_PRESSES`, and condition 4 still
applies — so a refused press costs `NOTICE_SEC` before the listener tries again.
That is deliberate: a person whose button does nothing does not press it fifty
times a second.

### The parameters, and why these values

| parameter | primary | also run | why |
|---|---:|---|---|
| `NOTICE_SEC` | **2.0** | 4.0, 8.0 | At 100 BPM two seconds is three to four beats — about a bar, which is what it takes to be sure a click is on the wrong multiple, plus the tap. |
| `MAX_PRESSES` | **3** | 1, unlimited | The tracker changes level about 5.8 times per RWC recording. Three is a real constraint and not a formality. |

`NOTICE_SEC = 2.0` is chosen on the physical argument above and **not** because
of what it does to any endpoint. It has to be said plainly that it interacts
sharply with one of them: the product's wrong-level criterion fails a recording
that spends **more than 4 s** at the wrong level, so a listener who needs 4 s to
notice can never rescue that clause and a listener who needs 2 s sometimes can.
That threshold is a property of the endpoint, is known now, is why 4.0 and 8.0
are run beside 2.0, and is registered as prediction P4 rather than discovered
afterwards.

## 4. Where the press is refused, and why that is a result

`tt_live_set_octave_offset` refuses when the shifted tempo would leave the
filter's configured range, which ships as 40..220 BPM. So ×2 is unavailable above
110 and ÷2 below 80.

This is not a detail to be worked around. A tracker sitting on the eighths of a
120 BPM song is at 240 and outside the range, which means the error a shell will
most often be able to fix is "tracker too fast, press ÷2", and the opposite may
simply be unavailable. **The refusal rate is a registered secondary endpoint**
(§6, S3), broken down by direction and by annotated tempo.

The range is **not** changed for this experiment. Widening it moves the tempo
prior and the resample clamp underneath every published number in this
repository, so it is a separate decision needing its own measurement, and
changing it here would confound the two.

## 5. The arms, and the control that carries the weight

All five arms run on identical audio, identical weights, identical binary, and
are sampled at **50 Hz** (see §8).

| arm | what it does |
|---|---|
| `baseline` | shipped configuration, no press. |
| `press` | the listener of §1–§3. |
| `press_random` | **the control.** Presses at the same times as `press`, with `k` drawn from `{+1, -1}` by a seeded generator instead of from §2. |
| `press_delayed` | presses at the same times with the correct `k`, but shifted later by one `NOTICE_SEC`. |
| `oracle_level` | `usable_any_octave`, already measured. The ceiling. |

**`press_random` is the arm this experiment stands on.** Every previous negative
in this line was caught by a control, and the failure mode here is specific:
`setOctaveOffset` re-seeds the cloud and shifts the anchor, and *any* such
disturbance perturbs a stuck tracker. Without `press_random` a result would not
distinguish "the user's judgement helps" from "kicking the filter helps". The
press count and press times are taken from the `press` arm's run so that the two
differ in direction and in nothing else.

`press_delayed` separates "the correction helped" from "it helped because it was
early", which is the mechanism the `NOTICE_SEC` sweep is about.

## 6. Endpoints

**Primary, on RWC:** `usable_rate` — precision ≥ 0.80, recall ≥ 0.80, acquisition
≤ 8.0 s, and no wrong-level episode longer than 4.0 s. Unmodified. This is the
product criterion and it is not being adjusted to suit the arm.

**Co-primary, on RWC:** `mean_correct_share_of_eligible` — the share of tracked
time spent at the annotated level. This is reported as a co-primary and not as a
secondary because it is the quantity a control that *ends* episodes can move,
whereas the wrong-level clause of `usable_rate` asks it to *prevent* them, which
it physically cannot at every `NOTICE_SEC`. Both are gated in §7. Neither
replaces the other and both are reported whatever they say.

**Secondaries, all reported regardless of outcome:**

- S1 `no_wrong_level_episode_fraction`
- S2 presses actually fired per recording, and the share of recordings that fired
  zero, one, two, three
- S3 refusal rate, split by direction and by annotated tempo decile
- S4 `switches_per_five_minutes` and `p90_settle_sec`
- S5 `f_measure`, precision and recall separately
- S6 the failure-reason decomposition (`too_few_beats`, `wrong_beats`,
  `slow_acquisition`, `wrong_level`) on the recordings that change verdict

## 7. Acceptance conditions

Adoption of a ×2 / ÷2 control as a **measured** product feature — as opposed to
the mechanism, which already ships — requires **all** of A1 to A4.

- **A1 (primary).** On RWC, `press` beats `baseline` on `usable_rate` by at least
  **+5.0 points**, with the 95% cluster-bootstrap CI over recordings excluding
  zero. Five points is 44% of RWC's +11.3 pooled ceiling; a control that cannot
  reach half of a ceiling it has oracle knowledge of is not worth a button.
- **A2 (the control).** On RWC, `press` beats `press_random` on `usable_rate` by
  at least **+4.0 points**, CI excluding zero. This is the gate that says the
  user's judgement and not the disturbance is doing the work.
- **A3 (co-primary).** On RWC, `press` beats `baseline` on
  `mean_correct_share_of_eligible` by at least **+0.05**, CI excluding zero.
- **A4 (transfer).** With every parameter fixed by RWC and changed in no way,
  Harmonix reproduces A1 and A2 at **at least 60%** of the RWC effect size, each
  with a CI excluding zero.

**Cost gates. Any one of these failing means adoption is not approved even if
A1–A4 all pass:**

- C1 `f_measure` on RWC does not fall by more than 0.010.
- C2 `switches_per_five_minutes` does not rise by more than 15% over baseline.
- C3 `p90_settle_sec` does not rise by more than 1.0 s.
- C4 On recordings that were **already at the correct level for their whole
  duration**, `usable_rate` does not fall at all beyond bootstrap noise. A
  control that damages the recordings it should never touch is worse than no
  control, and this is the arm's equivalent of the veto's A3.

C4 is the one to watch. The listener of §1 presses only when the level is wrong,
so in principle it cannot fire on a clean recording — C4 therefore also functions
as an implementation check, and a failure of it is a bug before it is a result.

## 8. How it is run, and the two mistakes already paid for

**Online, in the core, not as a schedule.** The press policy is implemented as an
`AnchorBpmResolver`-style online policy inside `tools/eval/dump_analysis`,
reached by a flag, in exactly the way the veto's comparison policies had to be
after the schedule form was found to have no fixed point. The reason is the same
and is structural: when the tracker presses, its own future level changes, so the
set of wrong-level intervals depends on the presses that a schedule would have to
have computed in advance. `debounce_1.5` never converged for this reason and cost
a day. It is not being rediscovered.

**Sampled at 50 Hz.** Every arm, including the baseline it is compared against.
`acquired_at` reconstructed from a 1 Hz confidence poll is wrong by seconds
through aliasing, and Harmonix's `usable_rate` is 5.5 points too low because of
it. The arms here are all measured the same way so the comparison is unaffected
either way, but the absolute rates would not be quotable at 1 Hz and one of the
four `usable_rate` clauses is the acquisition bar.

**The parity gate applies unchanged.** If cached-activation replay is used, the
20-recording byte-identity gate from `PREREGISTERED_octave_veto.md` §8 must pass
before any number is produced. That gate has already caught one run that would
have produced a clean-looking table from a tracker emitting 74 beats where the
product emits 116.

## 9. Statistics

Cluster bootstrap over recordings, 10 000 resamples, the recording as the cluster
because two excerpts of one piece are not two observations. Paired within
recording where the arms are paired, which is all of them.

Holm correction over a family of **exactly three** tests: A1, A2, A3. A4 is a
transfer replication and is not in the family; the cost gates are bounds and not
hypotheses. Naming the family here is what stops it growing to fit the result.

## 10. Predictions

Recorded so that being wrong is visible.

- **P1.** Most recordings that improve will need **one** press. The offset is
  held, so one correct press fixes the remainder of the song. Testable against
  S2: I predict over 60% of improving recordings fire exactly one press.
- **P2.** Refusals will be strongly asymmetric — predominantly ×2 attempts, on
  material whose annotated tempo is above 110 BPM. Testable against S3.
- **P3.** `press_random` will come out **below baseline**, not level with it. A
  wrong press is held for the rest of the recording, so the control is not a
  neutral disturbance; it is an injected error. If `press_random` lands level
  with baseline instead, the mechanism is not holding as designed and that is a
  bug in `602f9ad`, not a result.
- **P4.** A1 will pass at `NOTICE_SEC = 2.0` and fail at 4.0 and 8.0, because the
  wrong-level clause fails a recording at more than 4 s and a listener cannot act
  before it notices. The co-primary A3 will pass at all three, because time at
  the correct level is recoverable at any latency.
- **P5.** The realised gain will be **well under half the oracle ceiling** on
  every corpus. Named now: under +5.6 points on RWC. Note this sits directly
  against A1's +5.0 bar, which is deliberate — the gate and the prediction are
  close enough that the experiment can actually fail.
- **P6.** RWC-Classical will show no gain from any arm, because its ceiling is
  +0.0. If it shows one, something is wrong with the harness.

## 11. What would sink this

- A2 failing while A1 passes. That is the `press_random` control saying the
  button is a kick, not a judgement, and it would mean the whole framing is
  wrong.
- C4 failing. Presses firing on recordings that were never at the wrong level is
  an implementation fault in §3's arming condition.
- Refusal rates high enough (say over 40% of attempted presses) that the range
  guard, not the listener, is the binding constraint. That would not sink the
  idea but would redirect the work to the BPM range, which is a different
  experiment with different costs.
- The transfer failing at A4 after RWC passes, which on this project's record is
  the most likely single outcome and has happened before.

## 12. What follows either way

**If it passes:** the control is measured, and the shell work in Phase 8 has a
number to build against — including how to present a refusal, which S3 sizes.
The next question becomes the BPM range, since S3 will have measured what the
guard costs.

**If it fails:** the mechanism stays, because it costs nothing and a person is
entitled to overrule a tracker whether or not it helps on average. What does not
happen is quoting the +21 ceiling as though a button reaches it. The ceiling
would then be known to be unreachable from both ends — not automatically, and
not by a person either — and the remaining work is the one the ceiling
measurement already pointed at: **beat-grid recall**, which is a listed failure
on 84.8% of Harmonix's and 93.5% of RWC's surviving failures and is not an octave
problem at all.

---

## Re-run with the range guard removed, registered 2026-08-08

The first run failed and is recorded in `results/README.md`: A1 +2.1 against
+5.0, A2 +0.9 against +4.0 with a lower bound of zero, A3 +0.013 against +0.05,
and cost gate C1 failing with mean F down 0.0615. It also measured why —
**57.8% of presses refused, 342 of them ×2** — which §11 had named in advance as
the condition under which "the range guard, not the listener, is the binding
constraint".

`77e7bae` removes that guard for the user and only for the user: a press moves
the filter's tempo range and prior centre by the same octave, so the range still
says what tempo music is likely to be and simply stops outranking a person about
which octave it is in. **Shifted, not widened**, because `estimate()` bins
log-period into a fixed 48 bins across the range, so widening would coarsen the
bins and raise confidence for a cloud that had not moved — contaminating every
comparison against an arm that had not pressed.

**This is a new experiment, not a repeat.** The mechanism under test changed, so
the earlier verdict stands as a verdict on the earlier mechanism.

### What does not change

**The gates.** A1 +5.0, A2 +4.0, A3 +0.05, C1 ≤ 0.010 of mean F, C2, C3, C4, and
the Holm family of exactly A1/A2/A3. Moving a bar after failing it is the thing
this whole practice exists to prevent, and the bar was set against a measured
ceiling that has not moved either.

Also unchanged: `NOTICE_SEC = 2.0`, `MAX_PRESSES = 3`, RWC as development,
Harmonix untouched unless RWC passes, 50 Hz everywhere.

### A precondition, checked before anything is read

**The `baseline` arm must reproduce the previous run's `baseline` exactly** — the
same `usable` verdict on all 328 recordings and the same mean F to
floating-point equality. `77e7bae` claims to change nothing when no press
happens, and the unit tests assert it on a synthetic cloud; this is the same
claim on 328 real recordings, and it costs nothing because the arm is run
anyway. **If it fails, the run is void** and the leak is found before any
endpoint is looked at.

### What is expected to be different, registered now

- **P9.** The refusal rate collapses from 57.8% to under 5%. What remains should
  be the physical floor alone — a beat period shorter than two evidence windows,
  about 703 BPM at 48 kHz — which needs a press from above 350 BPM and should be
  rare. If refusals stay high, something other than the guard is refusing and
  the diagnosis in the first run was wrong.
- **P10.** `press_random` gets *worse* than it was, not better. It can now
  reach octaves that were previously refused, so a wrong press does more damage.
  A2 is therefore an easier test than it was, and if A2 still fails the result
  is stronger rather than weaker.
- **P11.** C1 still fails. Mean F fell 0.0615 in the first run and fell in
  *every* press arm including the random one, so the damage is from perturbing
  the filter rather than from being refused. Removing the guard lets more
  presses land, which should make this worse. Named now because C1 failing again
  would otherwise look like a discovery.

### The design fault from the first run, and what is done about it

The arms were matched on press *times* and not on presses that landed, so
`press_random` got 451 accepted to `press`'s 341. With the guard gone that gap
should close on its own, but it is not assumed: **accepted-press counts per arm
are reported beside every endpoint**, and if they differ by more than 10% the A2
margin is reported as confounded rather than as a result.

### What would make this uninterpretable

- The baseline precondition failing.
- Refusals staying above 20%, which would mean the mechanism change did not do
  what `77e7bae` says it does.
- Accepted presses differing between `press` and `press_random` by more than
  10%, which would leave A2 comparing arms that did different amounts of work.
