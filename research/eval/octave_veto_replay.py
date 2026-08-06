"""Replay: the five sequences the octave-veto experiment is an experiment on.

`eval/PREREGISTERED_octave_veto.md`, "The system under test", requires that
replay reproduce the live core byte-identically on beat times, published BPM,
published confidence, the estimator's own BPM, and the extracted event list —
because items 2 to 5 are the actual input here, and a replay that reproduces
beats while diverging on the estimator would evaluate a decoder on events the
product never had.

**Nothing is re-implemented to achieve that.** The replay *is* the live core:
`dump_analysis --live` runs `LiveTracker` over the file exactly as the shell
does, and this module only reads what it prints. The one thing that had to
change is how often those series are sampled — every experiment before this was
measured at 1 Hz, and a proposal is defined as a run of frames that closes after
a second at `k = 0`, which a series sampled at exactly that period cannot
resolve. `--live-sample-hz` raises it; the default stays 1 so nothing already
published moves.

That flag is observational by construction and the parity check below is what
proves it: `estimate()` and `tempoFromActivation()` are both const, `takeBeat`
sits outside the sampling guard, so the beat list at 50 Hz must equal the beat
list at 1 Hz. If it does not, the flag perturbs the tracker and every number
downstream of it is measuring the harness.

The downbeat channel needs no core change either. `--dump-activation` already
prints `activation_downbeat` from a fresh pass with fresh state, and the
pre-registration's causality requirement is met by the window rather than by the
pass: no frame after the proposal is read.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess

import numpy as np

# The tracker's own publishing hysteresis, already reproduced at these values in
# `eval/live_corpus_benchmark.py`. Taken from there rather than re-derived, so
# "before the first lock" means here what it means in every other live result.
LOCK_CONFIDENCE = 0.25
RELEASE_CONFIDENCE = 0.02

# Fine enough to resolve the event definition, and no finer: the activation
# frame rate is 50 fps and both polled quantities only move once a frame.
SAMPLE_HZ = 50.0


@dataclasses.dataclass(frozen=True)
class Replay:
    """One recording, as the live core reported itself."""

    beats: np.ndarray
    times: np.ndarray
    bpm: np.ndarray
    confidence: np.ndarray
    measured_bpm: np.ndarray
    measured_margin: np.ndarray
    activation_times: np.ndarray
    downbeat: np.ndarray

    @property
    def answered(self) -> np.ndarray:
        """`ActivationTempoEstimate::answered()` is `bpm > 0`, and nothing else."""
        return self.measured_bpm > 0.0

    @property
    def locked(self) -> np.ndarray:
        """The published lock, latched exactly as the tracker latches it."""
        out = np.zeros(len(self.confidence), dtype=bool)
        held = False
        for i, c in enumerate(self.confidence):
            if not held and c >= LOCK_CONFIDENCE:
                held = True
            elif held and c < RELEASE_CONFIDENCE:
                held = False
            out[i] = held
        return out


def run(binary: pathlib.Path, audio: pathlib.Path, weights: pathlib.Path,
        sample_hz: float = SAMPLE_HZ, activation: bool = True,
        extra: list[str] | None = None) -> dict:
    """One `dump_analysis --live` invocation, as parsed JSON."""
    args = [str(binary), str(audio), "--live", "--live-model", str(weights),
            "--live-sample-hz", repr(float(sample_hz))]
    if activation:
        # `--dump-activation` reads the same `--live-model` list the tracker was
        # given, so the dumped channel is of whatever the tracker was run on.
        args += ["--dump-activation"]
    if extra:
        args += list(extra)
    done = subprocess.run(args, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"dump_analysis failed on {audio.name}: "
                           f"{done.stderr.strip()[:400]}")
    return json.loads(done.stdout)


def run_activation(binary: pathlib.Path, audio: pathlib.Path,
                   activation_path: pathlib.Path,
                   sample_hz: float = SAMPLE_HZ,
                   extra: list[str] | None = None) -> dict:
    """Replay cached beat activations through the same live core.

    The downbeat head is not needed by C++: Python already used it to build the
    fixed veto schedule.  This is the registered "model once, policies many"
    path; callers must establish parity with ``run`` before trusting it.
    """
    args = [str(binary), str(audio), "--live-activation", str(activation_path),
            "--activation-fps", repr(SAMPLE_HZ),
            "--live-sample-hz", repr(float(sample_hz)),
            "--activation-model-timing"]
    if extra:
        args += list(extra)
    done = subprocess.run(args, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"dump_analysis activation replay failed on {audio.name}: "
                           f"{done.stderr.strip()[:400]}")
    return json.loads(done.stdout)


def replay(binary: pathlib.Path, audio: pathlib.Path, weights: pathlib.Path,
           sample_hz: float = SAMPLE_HZ) -> Replay:
    payload = run(binary, audio, weights, sample_hz)
    return from_payload(payload)


def from_payload(payload: dict) -> Replay:
    array = lambda key: np.asarray(payload.get(key, []), dtype=np.float64)
    return Replay(
        # In live mode `dump_analysis` publishes the tracker's beat list under
        # `beats` (it assigns `beats = live_beats` before printing), so there is
        # no separate live key to read.
        beats=array("beats"),
        times=array("live_times"),
        bpm=array("live_bpms"),
        confidence=array("live_confidences"),
        measured_bpm=array("live_anchor_bpm"),
        measured_margin=array("live_anchor_margin"),
        activation_times=array("activation_times"),
        downbeat=array("activation_downbeat"),
    )


# --- Parity ------------------------------------------------------------------


def parity(binary: pathlib.Path, audio: pathlib.Path,
           weights: pathlib.Path) -> dict:
    """The pre-registration's five sequences, checked on one recording.

    Two questions, and both have to hold:

    * **determinism** — the same invocation twice gives the same bytes. Without
      it nothing below means anything.
    * **rate independence** — raising the sampling rate does not move the
      tracker. Beats are the load-bearing comparison because they are produced
      outside the sampling guard, so if they move, the guard is not a guard.
    """
    slow_a = run(binary, audio, weights, sample_hz=1.0, activation=False)
    slow_b = run(binary, audio, weights, sample_hz=1.0, activation=False)
    fast = run(binary, audio, weights, sample_hz=SAMPLE_HZ, activation=True)

    beats_slow = np.asarray(slow_a.get("beats", []), dtype=np.float64)
    beats_fast = np.asarray(fast.get("beats", []), dtype=np.float64)

    checks = {
        "deterministic": _same_series(slow_a, slow_b),
        "beats_identical": (len(beats_slow) == len(beats_fast)
                            and bool(np.array_equal(beats_slow, beats_fast))),
        # Read once at the end of the run, so it cannot depend on how often the
        # series were sampled unless the sampling perturbed the tracker.
        "final_bpm_identical": slow_a.get("live_bpm") == fast.get("live_bpm"),
        "final_confidence_identical": (slow_a.get("live_confidence")
                                       == fast.get("live_confidence")),
        "n_beats": int(len(beats_fast)),
        "n_slow_samples": int(len(slow_a.get("live_times", []))),
        "n_fast_samples": int(len(fast.get("live_times", []))),
        "n_activation_frames": int(len(fast.get("activation_downbeat", []))),
    }
    # A finer rate that produced no more samples would mean the flag did
    # nothing, which passes every identity check above for the wrong reason.
    checks["rate_took_effect"] = (checks["n_fast_samples"]
                                  > 4 * max(checks["n_slow_samples"], 1))
    checks["activation_present"] = checks["n_activation_frames"] > 0
    checks["passed"] = all(bool(checks[key]) for key in
                           ("deterministic", "beats_identical",
                            "final_bpm_identical", "final_confidence_identical",
                            "rate_took_effect", "activation_present"))
    return checks


def _same_series(a: dict, b: dict) -> bool:
    """Every array and scalar the live path prints, compared exactly."""
    keys = ({k for k in a if k.startswith("live_")}
            | {k for k in b if k.startswith("live_")}
            # In live mode this is the tracker's beat list, and it is the one
            # sequence produced outside the sampling guard.
            | {"beats"})
    for key in sorted(keys):
        left, right = a.get(key), b.get(key)
        if isinstance(left, list) or isinstance(right, list):
            if not np.array_equal(np.asarray(left, dtype=np.float64),
                                  np.asarray(right, dtype=np.float64)):
                return False
        elif left != right:
            return False
    return True


def same_live_series(a: dict, b: dict) -> bool:
    """Public parity predicate for cached-activation policy replay."""
    return _same_series(a, b)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("audio", type=pathlib.Path, nargs="+")
    args = parser.parse_args()

    failures = 0
    for path in args.audio:
        got = parity(args.binary, path, args.weights)
        mark = "ok " if got["passed"] else "FAIL"
        print(f"{mark} {path.name:24s} beats={got['n_beats']:4d} "
              f"samples {got['n_slow_samples']:4d} -> {got['n_fast_samples']:5d} "
              f"frames={got['n_activation_frames']:5d}")
        if not got["passed"]:
            failures += 1
            print("     ", {k: v for k, v in got.items() if v is False})
    print(f"\n{len(args.audio) - failures}/{len(args.audio)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
