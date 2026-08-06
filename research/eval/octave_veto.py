"""The beat-synchronous metre decoder, as a judge of octave-switch proposals.

Implements `eval/PREREGISTERED_octave_veto.md`. Nothing here is tuned; every
constant is quoted from that document, which was committed before this file
existed.

The thing being answered is deliberately narrow. Not "does the downbeat channel
know the metre" — `eval/downbeat_audit.py` measured that at 82.9% against a
30.1% null and it is not in question. This asks whether, at the moment the live
tracker proposes moving to another octave, the same channel can say *allow* or
*veto*.

Three shapes of the previous run are corrected here, and each correction has a
name in the pre-registration:

* the score is **standardised** rather than a raw contrast, because the null
  distribution of a raw contrast maximised over (metre, phase) shrinks as the
  grid lengthens, which tilted every octave comparison toward the shorter grid
  before any evidence was read;
* the two grids are scored over the **same** (metre, phase) set, so the residual
  multiplicity from the `N >= 2m` admissibility rule cannot differ between them;
* the null **shifts the raw activation track once and resamples both grids from
  it**, so the grids keep their nesting, instead of permuting each grid on its
  own and giving each its own null.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

# --- Constants, all from the pre-registration -------------------------------

METRES = (2, 3, 4, 6)

# The window both candidates see: the 16 most recent committed beats completed
# at or before the proposal. A fixed interval of audio, so neither candidate can
# win on quantity. Four bars of the committed state in 4/4 — and two true bars
# when the committed state is itself doubled, which is the honest asymmetry: at
# the moment of the decision the true period is the unknown.
WINDOW_BEATS = 16

# Below this the grid carries no contrast worth reading. Distinct from the
# `N >= 2m` rule, which is about whether a metre is representable at all.
MIN_GRID_POINTS = 4

# Deterministic, not sampled: four fractions of the window length. An RNG here
# would make the null a property of a seed.
SHIFT_FRACTIONS = (0.2, 0.4, 0.6, 0.8)

# A grid whose values are constant carries no contrast, and `sd` is then zero.
# I2 constructs exactly that case, so the formula answers it instead of dividing
# by zero.
EPS = 1e-9

# Event extraction. Both fixed in the pre-registration and swept on nothing;
# they exist to make events countable, and the primary statistic is clustered by
# recording precisely so that a bad choice here cannot manufacture significance.
CLOSE_AFTER_SEC = 1.0
MIN_SEPARATION_SEC = 2.0


# --- The score ---------------------------------------------------------------


def admissible_metres(n_points: int) -> tuple[int, ...]:
    """The metres a grid of `n_points` can represent at all."""
    return tuple(m for m in METRES if n_points >= 2 * m)


def common_metres(n_a: int, n_b: int) -> tuple[int, ...]:
    """The metre set both grids are scored over: the shorter grid's.

    Taking the shorter is what makes the exclusion symmetric. Sixteen committed
    beats give a halved grid eight points, which admits {2, 3, 4} and not 6 — so
    against a halving proposal metre 6 leaves *both* sides of the comparison. It
    costs sensitivity on 6/8 material and cannot tilt the octave either way.
    """
    return admissible_metres(min(n_a, n_b))


def zscores(values: np.ndarray, metres: tuple[int, ...]) -> dict[tuple[int, int], float]:
    """The standardised bar-phase contrast for every (metre, phase) on one grid.

        raw    = mean(on) - mean(off)
        se     = sd * sqrt(1/n_on + 1/n_off)
        z      = raw / se

    `sd` is the population standard deviation over the whole grid. The
    standardisation is the entire point: `raw` alone has a null whose spread
    falls as the grid lengthens, and the previous audit maximised `raw`.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    sd = float(values.std())
    index = np.arange(n) % np.array(metres)[:, None] if metres else None
    out: dict[tuple[int, int], float] = {}
    for row, metre in enumerate(metres):
        idx = index[row]
        for phase in range(metre):
            on = values[idx == phase]
            off = values[idx != phase]
            # Cannot arise inside the common set. The guard is here because a
            # silent nan is worse than a redundant branch.
            if on.size == 0 or off.size == 0:
                continue
            if sd < EPS:
                out[(metre, phase)] = 0.0
                continue
            se = sd * math.sqrt(1.0 / on.size + 1.0 / off.size)
            out[(metre, phase)] = float(on.mean() - off.mean()) / se
    return out


def grid_score(values: np.ndarray, metres: tuple[int, ...]) -> tuple[float, int, int] | None:
    """The best (metre, phase) on one grid, or None when nothing is scoreable."""
    scores = zscores(values, metres)
    if not scores:
        return None
    # Ties go to the smaller metre. {2, 4} and {3, 6} are each closed under
    # doubling, so a bar pattern at 2 scores identically at 4 and the tie is
    # routine rather than exotic; picking the larger would report a metre the
    # evidence never distinguished.
    (metre, phase), best = max(scores.items(), key=lambda kv: (kv[1], -kv[0][0]))
    return best, metre, phase


# --- The grids ---------------------------------------------------------------


def doubled_grid(beats: np.ndarray) -> np.ndarray:
    """Every committed beat plus every gap's centre — the `k = +1` proposal.

    `k = +1` means the estimator reports twice the committed BPM, so the period
    it proposes is half the committed one and its grid is the finer of the two.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 2:
        return beats
    return np.sort(np.concatenate([beats, 0.5 * (beats[:-1] + beats[1:])]))


def halved_grids(beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every other committed beat, both ways — the `k = -1` proposal.

    Two of them, because a grid at twice the period has two possible alignments
    to the one it came from and the decoder is not told which. Both are scored
    and the better is taken; that maximum over 2 applies to the proposed grid
    only and is the one asymmetry the standardisation does not remove, which is
    what I4 exists to check.
    """
    beats = np.asarray(beats, dtype=np.float64)
    return beats[0::2], beats[1::2]


# --- Sampling and the shared null -------------------------------------------


def _nearest(frame_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the activation frame nearest each time."""
    index = np.clip(np.searchsorted(frame_times, times), 1, len(frame_times) - 1)
    left = np.abs(times - frame_times[index - 1])
    right = np.abs(frame_times[index] - times)
    return np.where(left <= right, index - 1, index)


@dataclasses.dataclass(frozen=True)
class WindowTrack:
    """The activation inside one decision window, and how to read it shifted.

    The shift is applied to the **raw track**, once, and both grids are then
    resampled from it — so the grids keep their nesting and the channel keeps
    its marginal distribution and its autocorrelation. Only the alignment to the
    beat grid is destroyed. The previous audit permuted each grid separately,
    which gives each grid a null of its own and cannot adjudicate a comparison
    between them.
    """

    frame_times: np.ndarray
    downbeat: np.ndarray

    def sample(self, times: np.ndarray, shift_frames: int = 0) -> np.ndarray:
        """The downbeat probability at `times`, off a circularly shifted track."""
        if len(self.frame_times) == 0 or len(times) == 0:
            return np.empty(0, dtype=np.float64)
        index = _nearest(self.frame_times, np.asarray(times, dtype=np.float64))
        return self.downbeat[(index + shift_frames) % len(self.downbeat)]

    def shifts(self) -> tuple[int, ...]:
        """The four deterministic shifts, in frames."""
        n = len(self.downbeat)
        if n == 0:
            return ()
        return tuple(int(round(f * n)) % n for f in SHIFT_FRACTIONS)


# --- The decision ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Decision:
    """One proposal, judged. `answered` false means the policy falls to baseline."""

    answered: bool
    delta: float = float("nan")
    delta_raw: float = float("nan")
    score_committed: float = float("nan")
    score_proposed: float = float("nan")
    null_committed: float = float("nan")
    null_proposed: float = float("nan")
    metre: int = 0
    metres_scored: tuple[int, ...] = ()
    reason: str = ""

    def veto(self, tau: float) -> bool:
        """Delta above tau blocks the switch. Unanswered always allows."""
        return bool(self.answered and self.delta > tau)


def _best_over(track: WindowTrack, grids: list[np.ndarray], metres: tuple[int, ...],
               shift: int) -> float | None:
    """The best score over one or more alignments of the same grid."""
    best: float | None = None
    for grid in grids:
        got = grid_score(track.sample(grid, shift), metres)
        if got is not None and (best is None or got[0] > best):
            best = got[0]
    return best


def judge(track: WindowTrack, committed_beats: np.ndarray, k: int,
          window_beats: int = WINDOW_BEATS) -> Decision:
    """Score the committed level against the level actually proposed.

    Only those two. Scoring `P/2`, `P` and `2P` together would put three grids of
    three different lengths into one maximum, which is the geometry that broke
    the previous audit, tripled.

    `window_beats` is a parameter rather than the module constant so that I4 can
    vary the window without mutating global state — a test that has to reach
    into the module to set a constant is testing the test harness.
    """
    committed_beats = np.asarray(committed_beats, dtype=np.float64)
    if len(committed_beats) < window_beats:
        return Decision(False, reason="short window")
    if k == 1:
        proposed = [doubled_grid(committed_beats)]
    elif k == -1:
        proposed = list(halved_grids(committed_beats))
    else:
        return Decision(False, reason="not an octave proposal")

    shortest = min(len(committed_beats), min(len(g) for g in proposed))
    if shortest < MIN_GRID_POINTS:
        return Decision(False, reason="grid too short")
    metres = common_metres(len(committed_beats), min(len(g) for g in proposed))
    if not metres:
        return Decision(False, reason="no common metre")

    committed_read = grid_score(track.sample(committed_beats), metres)
    proposed_best = _best_over(track, proposed, metres, 0)
    if committed_read is None or proposed_best is None:
        return Decision(False, reason="unscoreable")

    shifts = track.shifts()
    if not shifts:
        return Decision(False, reason="empty window")
    null_c = [grid_score(track.sample(committed_beats, s), metres) for s in shifts]
    null_p = [_best_over(track, proposed, metres, s) for s in shifts]
    if any(v is None for v in null_c) or any(v is None for v in null_p):
        return Decision(False, reason="unscoreable null")

    nc = float(np.mean([v[0] for v in null_c]))
    npd = float(np.mean(null_p))
    delta = (committed_read[0] - nc) - (proposed_best - npd)
    return Decision(
        answered=True,
        delta=delta,
        # Reported alongside and never gated. If it tracks `delta`, the shipped
        # version can skip the four shifted passes; if it carries the signal
        # instead of correcting a bias, `delta` is measuring the shift and not
        # the channel. P2 is the check.
        delta_raw=committed_read[0] - proposed_best,
        score_committed=committed_read[0],
        score_proposed=proposed_best,
        null_committed=nc,
        null_proposed=npd,
        metre=committed_read[1],
        metres_scored=metres,
    )


# --- Events ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Proposal:
    """One octave-switch proposal: a run of frames, not a frame."""

    onset_sec: float
    close_sec: float
    k: int
    committed_bpm: float
    measured_bpm: float


def octave_k(measured_bpm: float, committed_bpm: float) -> int:
    """round(log2(measured / committed)) — which octave apart the two are."""
    if not (measured_bpm > 0.0) or not (committed_bpm > 0.0):
        return 0
    return int(round(math.log2(measured_bpm / committed_bpm)))


def extract_proposals(times: np.ndarray, committed_bpm: np.ndarray,
                      measured_bpm: np.ndarray, answered: np.ndarray,
                      locked: np.ndarray) -> list[Proposal]:
    """Every octave-switch proposal in one replayed recording.

    The anchor is rewritten from the estimator at every frame. Counting frames
    would turn one sustained disagreement into hundreds of near-identical events
    and inflate every count and every interval in the pre-registration, so a
    maximal run of consecutive frames with the same **sign** of `k` is one
    event, timestamped at its first frame.

    Proposals before the first lock are excluded: acquisition is a separate and
    already-measured problem, and until then the committed level is not a claim
    about anything.
    """
    times = np.asarray(times, dtype=np.float64)
    ks = np.array([octave_k(m, c) if a else 0
                   for m, c, a in zip(measured_bpm, committed_bpm, answered)], dtype=int)
    signs = np.sign(ks)

    first_lock = None
    locked = np.asarray(locked, dtype=bool)
    if locked.any():
        first_lock = float(times[int(np.argmax(locked))])

    out: list[Proposal] = []
    open_at: int | None = None
    zero_since: float | None = None
    for i, t in enumerate(times):
        sign = signs[i]
        if open_at is None:
            if sign != 0:
                open_at = i
                zero_since = None
            continue
        # An event closes on a sign flip, or after CLOSE_AFTER_SEC at k == 0.
        # Not on the first zero frame: the estimator drops in and out of an
        # answer far faster than the disagreement it is reporting resolves.
        flipped = sign != 0 and sign != signs[open_at]
        if sign == signs[open_at]:
            zero_since = None
        elif sign == 0:
            if zero_since is None:
                zero_since = t
        if flipped or (zero_since is not None and t - zero_since >= CLOSE_AFTER_SEC):
            _emit(out, times, committed_bpm, measured_bpm, ks, open_at,
                  zero_since if zero_since is not None else t, first_lock)
            open_at = i if flipped else None
            zero_since = None
    if open_at is not None:
        _emit(out, times, committed_bpm, measured_bpm, ks, open_at,
              float(times[-1]), first_lock)
    return out


def _emit(out: list[Proposal], times: np.ndarray, committed_bpm: np.ndarray,
          measured_bpm: np.ndarray, ks: np.ndarray, start: int, close: float,
          first_lock: float | None) -> None:
    """Append one event, unless it is pre-lock or too close to the last one."""
    onset = float(times[start])
    if first_lock is None or onset < first_lock:
        return
    # Merged into the previous event rather than counted again.
    if out and onset - out[-1].onset_sec < MIN_SEPARATION_SEC:
        return
    out.append(Proposal(onset_sec=onset, close_sec=float(close), k=int(ks[start]),
                        committed_bpm=float(committed_bpm[start]),
                        measured_bpm=float(measured_bpm[start])))


def window_beats(beats: np.ndarray, onset_sec: float,
                 count: int = WINDOW_BEATS) -> np.ndarray:
    """The `count` most recent beats completed at or before `onset_sec`.

    No frame after the proposal is visible to anything, which is what makes the
    window causal. An empty result means the decision falls to baseline.
    """
    beats = np.asarray(beats, dtype=np.float64)
    before = beats[beats <= onset_sec]
    if len(before) < count:
        return np.empty(0, dtype=np.float64)
    return before[-count:]
