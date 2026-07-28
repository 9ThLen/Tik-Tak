"""Downbeat salience backends, and what it takes to compare two of them.

A backend answers one question — how much does each beat look like a bar line
— and nothing else. The bar length and phase are then decided by the shipping
C++ resolver, identically for every backend. That is the whole point of the
seam in ``core/src/analysis/downbeat.hpp``: two scorers compared this way
differ only in the scoring, and any difference in the verdicts is a difference
between the models rather than between two ways of counting bars.

**What may and may not be compared.** The resolver no longer rescales incoming
salience, deliberately — turning a model's 0.500001 and 0.500003 into unit
variance would make an ignorant model look certain. The consequence is that
``phase_margin`` and ``meter_margin`` are in each backend's own units and
**cannot be compared between backends at all**. A table putting the cue
backend's 0.87 next to a model's 0.87 would be comparing nothing.

What survives the change of units is the verdict: correct, wrong metre, wrong
phase, withheld, no answer — each backend judged by *its own* calibration. So
that is what the comparison compares, and it is why a Calibration is one object
of three numbers rather than three loose arguments. Half a calibration is the
mistake this module exists to make impossible.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

__all__ = [
    "Calibration",
    "CUE_CALIBRATION",
    "Backend",
    "cue_backend",
    "sample_at_beats",
    "BEAT_THIS_CHECKPOINT",
    "beat_this_backend",
]


@dataclass(frozen=True)
class Calibration:
    """A backend's three thresholds, in that backend's own units.

    Frozen and passed whole because they are meaningless apart. The C++ tool
    enforces the same rule from the other side: all three flags or none.
    """

    min_salience_range: float
    min_phase_margin: float
    min_meter_margin: float

    def flags(self) -> list[str]:
        return [
            "--salience-min-range", repr(float(self.min_salience_range)),
            "--salience-min-phase-margin", repr(float(self.min_phase_margin)),
            "--salience-min-meter-margin", repr(float(self.min_meter_margin)),
        ]


# Mirrors the defaults in DownbeatConfig. Provisional, on synthetic material —
# see the comment there and research/eval/README.md.
CUE_CALIBRATION = Calibration(
    min_salience_range=0.05,
    min_phase_margin=0.25,
    min_meter_margin=0.40,
)


class SalienceSource(Protocol):
    """Produces one salience value per beat of a grid it is given."""

    def __call__(self, audio: np.ndarray, sample_rate: float,
                 beat_times: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Backend:
    """A named scorer plus the calibration its numbers are quoted in.

    ``salience`` is None for the built-in cues: they live inside the C++ tool
    and need no second pass, which is also why they are the only backend that
    works without the model artifacts in models/.
    """

    name: str
    calibration: Calibration
    salience: SalienceSource | None = None

    @property
    def is_builtin(self) -> bool:
        return self.salience is None


def cue_backend() -> Backend:
    return Backend(name="cues", calibration=CUE_CALIBRATION, salience=None)


def sample_at_beats(activation: np.ndarray, frame_times: np.ndarray,
                    beat_times: np.ndarray, window_sec: float = 0.07) -> np.ndarray:
    """Reduce a frame-wise activation to one value per beat.

    The peak inside a window centred on the beat, not the nearest frame and not
    the mean. Three reasons, in order of how much they cost when ignored:

    * A model's downbeat activation is a *spike* a few frames wide. The nearest
      frame samples a spike at whatever phase the frame grid happens to have,
      so the same model on the same music scores differently depending on an
      offset nobody chose.
    * A tracked beat time and a model's idea of the same beat disagree by a few
      tens of milliseconds routinely. ±70 ms is the tolerance the beat-tracking
      literature settled on for exactly this reason, and this window is that
      tolerance.
    * The mean over a window mostly measures the window: widen it and every
      beat converges on the piece's average, which is the one number carrying
      no information about where the bar line is.

    A beat whose window contains no frame at all gets 0.0 rather than being
    dropped, because the count must stay equal to the beat count — the resolver
    refuses a mismatch, and it is right to.
    """
    activation = np.asarray(activation, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    beat_times = np.asarray(beat_times, dtype=np.float64)

    if activation.shape != frame_times.shape:
        raise ValueError(
            f"{activation.size} activation value(s) against "
            f"{frame_times.size} frame time(s)")
    if not np.all(np.isfinite(activation)):
        raise ValueError("the activation contains non-finite values")
    if window_sec <= 0.0:
        raise ValueError("window_sec must be positive")

    out = np.zeros(beat_times.size, dtype=np.float64)
    if activation.size == 0:
        return out

    starts = np.searchsorted(frame_times, beat_times - window_sec, side="left")
    stops = np.searchsorted(frame_times, beat_times + window_sec, side="right")
    for i, (start, stop) in enumerate(zip(starts, stops)):
        if stop > start:
            out[i] = activation[start:stop].max()
    return out


# ------------------------------------------------------------- Beat This! ---
#
# MIT, weights included, and it emits exactly what this seam wants: a per-frame
# downbeat activation with no opinion about metre, which the C++ resolver then
# turns into a bar length and a phase. See models/README.md for the pinned
# checkpoint and docs/ml-models.md for the licence check.

# The exported graph, not the checkpoint: see _load_beat_this for why this route
# and not torch. This is the `final0` variant — which matters when reading any
# number it produces, because Beat This!'s `final*` checkpoints are trained on
# the full corpus, and that corpus includes Ballroom and GTZAN. Scoring either
# with this is measuring recall of the training set as much as accuracy. The
# fold checkpoints exist for the clean comparison; annotations/ballroom already
# carries the matching 8-folds.split.
BEAT_THIS_CHECKPOINT = (
    pathlib.Path(__file__).resolve().parents[2] / "models" / "beat_this.onnx"
)

# Placeholder, and labelled as one. The cue numbers are the wrong units for a
# model's activations and are here only so the plumbing has something to carry;
# the real values come from a calibration run on annotated recordings, which is
# the step that cannot be skipped and cannot be done on synthetic clips.
BEAT_THIS_CALIBRATION = CUE_CALIBRATION


def beat_this_backend(
        checkpoint: pathlib.Path | None = None,
        calibration: Calibration = BEAT_THIS_CALIBRATION,
        loader: Callable[[pathlib.Path], SalienceSource] | None = None,
) -> Backend:
    """The Beat This! small0 scorer, if its checkpoint and torch are present.

    Raises rather than degrading to the cues: a comparison that silently scored
    the built-in backend twice and labelled one column with a model's name is
    the worst outcome available here, and it would look like a result.

    ``loader`` is the injection point used by the tests, which have neither
    torch nor the weights — and neither does the development environment this
    was written in, so the model call itself is the one part of this file that
    has never been executed. It is marked accordingly in _load_beat_this.
    """
    path = pathlib.Path(checkpoint) if checkpoint else BEAT_THIS_CHECKPOINT
    build = loader or _load_beat_this
    return Backend(name="beat_this_final0_onnx",
                   calibration=calibration,
                   salience=build(path))


def _load_beat_this(model: pathlib.Path) -> SalienceSource:
    """Load the exported model and return a per-beat salience function.

    Goes through eval/beat_this_onnx.py rather than torch and the checkpoint.
    That path is the one tools/parity/check_parity.py holds the C++ core to, and
    on the run that first exercised it end to end the two agreed exactly — same
    beats, same downbeats, on every clip. A second, torch-shaped route to the
    same activation would be a second thing to trust for no gain.

    The three details the previous note asked whoever had the artifact to check,
    now that it is here: the frame rate is FPS in that module and comes back as
    ``Activations.frame_times``; the downbeat head is its own output, not a
    channel to be guessed at; and no state dict is involved, because the graph
    is exported. What is fed to sample_at_beats is the probability rather than
    the logit, so the calibration's units are bounded and a threshold on them
    means the same thing from one recording to the next.
    """
    if not model.is_file():
        raise FileNotFoundError(
            f"{model} is missing. It is deliberately not in git — pin and "
            f"fetch it with:\n"
            f"  python models/fetch.py pin beat_this_cpp_onnx <url-or-path>\n"
            f"See models/README.md."
        )

    from eval.beat_this_onnx import BeatThisOnnx

    # Once, not per recording: the session build dominates a 30 second clip.
    session = BeatThisOnnx(model)

    def salience(audio: np.ndarray, sample_rate: float,
                 beat_times: np.ndarray) -> np.ndarray:
        activations = session.activations(audio, sample_rate)
        return sample_at_beats(activations.downbeat_probability(),
                               activations.frame_times, beat_times)

    return salience
