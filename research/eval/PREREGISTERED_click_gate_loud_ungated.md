# Pre-registered: a loud click without the gate

Written after `PREREGISTERED_click_gate_and_antialias.md` was scored and before
this run.

## Why

The registered arms fill three cells of the level × gate table and leave the
fourth empty. The gate by itself cost 0.157 usable on RWC and 0.369 on GTZAN
(G1); an ungated click at -12 dB cost 0.012 and 0.023 (G3); a click at 0 dB
with the gate cost as much as the gate alone (G6). Nothing measured a loud
click without the gate.

That cell is what decides between removing the gate and redesigning it. The C
API tells every shell to gate because, ungated, "the tracker locks onto its own
output". At -12 dB the lock shows in the confidence (median 0.84 against 0.28 on
RWC) and in beats emitted too densely (22% of RWC recordings above 1.5 times the
annotated count, against 8%), and hardly at all in the endpoints. Whether that
stays true when the click is as loud as the music is the open question.

## Arm

The parent run's binary (`dump_analysis.exe`, SHA-256 `9cb7961c4d02…`, built at
`5bc76b7`), a clean worktree at the commit that adds this file, and the parent
protocol unchanged: BeatNet `model_1`, `eval.live_corpus_benchmark --mode model
--per-track`, `--live-sample-hz 50`, 8 workers, default seed.

| arm | flags |
|---|---|
| G7 speaker, click 0 dB, not gated | `--live-click-db 0` |

On **RWC (328)** and **GTZAN (1000)**, compared with the parent run's G0, G3 and
G6. Same endpoints, and the same noise floor: the across-seed SD of macro usable
from `PREREGISTERED_quickfix_diagnostics.md` Q1, 0.005.

## Decisions

* **G7 against G6** asks whether the gate earns its cost at 0 dB.
  * If G7 beats G6 on usable by more than the noise floor on both corpora, the
    gate buys nothing at either measured level. The recommendation is then no
    gate by default. The C API's instruction to gate would be withdrawn, and the
    confidence reported in speaker mode flagged as inflated. The product change
    itself waits for the project's owner and a speaker-mode check on real
    captures.
  * If G7 loses to G6 by more than the floor on either corpus, a loud click
    needs suppressing. The gate's cost then has to be designed down rather than
    removed. A narrower window and subtracting the known click are the
    candidates, each registered before it is built.
  * If neither holds, a loud click costs as much ungated as the gate does, and
    neither is acceptable. That is also a redesign, with the same candidates.
* **G7 against G0** is reported: what a loud ungated click costs. Beside it go
  the median confidence and the share of recordings above 1.5 times the
  annotated beat count, the signature of a self-lock.
