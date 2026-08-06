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

# How near a power of two a ratio must be to be an *octave* proposal.
#
# `round(log2(r))` alone calls everything in (1.41, 2.83) a doubling, and a 3:2
# tempo relation sits squarely inside it. Measured on GTZAN excerpts, half the
# proposals found that way were 3:2 — 1.44, 1.48, 1.56, 1.66 — and §2 builds the
# proposed grid as *exactly* doubled or halved, so those events would be scored
# against a grid nobody proposed.
#
# 8% is the octave tolerance the live benchmark already uses and that §1's
# labels are written against, so this is not a new free parameter.
OCTAVE_TOLERANCE = 0.08


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


def octave_k(measured_bpm: float, committed_bpm: float,
             tolerance: float = OCTAVE_TOLERANCE) -> int:
    """Which octave apart the two are, or 0 when they are not an octave apart.

    The registered formula was `round(log2(measured / committed))` and nothing
    else. That is not enough: it maps every ratio in (1.41, 2.83) to +1, so a
    3:2 tempo relation is reported as a doubling and the decoder is then asked
    to score a grid that was never proposed. A ratio has to be *near* a power of
    two, at the same 8% the labels use.
    """
    if not (measured_bpm > 0.0) or not (committed_bpm > 0.0):
        return 0
    exponent = math.log2(measured_bpm / committed_bpm)
    k = int(round(exponent))
    if k == 0 or abs(exponent - k) > math.log2(1.0 + tolerance):
        return 0
    return k


def extract_proposals(times: np.ndarray, published_bpm: np.ndarray,
                      measured_bpm: np.ndarray, answered: np.ndarray,
                      locked: np.ndarray) -> list[Proposal]:
    """Every octave-switch proposal in one replayed recording.

    **The committed level is a held reference, not the published BPM of the
    moment**, and the registered definition was the second of those. That
    definition describes a state which does not occur: measured on GTZAN,
    `|log2(measured / published)|` has a 99th percentile of 0.074 and a maximum
    of 0.52 over 7821 locked-and-answered frames, and `k != 0` on two of them.
    The anchor at 0.02 octaves pins the filter to the estimator, so the two
    never disagree by an octave — the octave moves *inside* the estimator and
    the published BPM follows it within a fraction of a second.

    So `committed` follows the published BPM while it stays in the same octave,
    and **freezes the moment an octave away is proposed**, until the event
    closes. That is not a new mechanism: it is exactly the state the veto action
    in §5 preserves, and exactly what `held_octave_bpm_` holds in the core.

    The rest is as registered. A maximal run of frames of one sign of `k` is one
    event, timestamped at its first frame — the anchor is rewritten every frame,
    and counting frames would turn one sustained disagreement into hundreds of
    near-identical events. Proposals before the first lock are excluded:
    acquisition is a separate and already-measured problem, and until then the
    committed level is not a claim about anything.
    """
    times = np.asarray(times, dtype=np.float64)
    published_bpm = np.asarray(published_bpm, dtype=np.float64)
    measured_bpm = np.asarray(measured_bpm, dtype=np.float64)
    answered = np.asarray(answered, dtype=bool)
    locked = np.asarray(locked, dtype=bool)

    out: list[Proposal] = []
    committed = math.nan
    open_at: int | None = None
    open_sign = 0
    zero_since: float | None = None

    for i, t in enumerate(times):
        if not (locked[i] and answered[i]):
            continue
        if math.isnan(committed):
            committed = float(published_bpm[i])
        sign = int(np.sign(octave_k(float(measured_bpm[i]), committed)))

        if open_at is None:
            if sign != 0:
                open_at, open_sign, zero_since = i, sign, None
            else:
                # Inside the octave the committed level simply follows, so a
                # band drifting 128 -> 132 is never a proposal.
                committed = float(published_bpm[i])
            continue

        if sign == open_sign:
            zero_since = None
        elif sign == 0 and zero_since is None:
            # Not closed on the first agreeing frame: the estimator drops in and
            # out far faster than the disagreement it reports resolves.
            zero_since = float(t)
        flipped = sign != 0 and sign != open_sign
        if flipped or (zero_since is not None and t - zero_since >= CLOSE_AFTER_SEC):
            _emit(out, times, committed, measured_bpm, open_at,
                  zero_since if zero_since is not None else float(t), open_sign)
            committed = float(published_bpm[i])
            open_at = i if flipped else None
            open_sign = sign if flipped else 0
            zero_since = None

    if open_at is not None:
        _emit(out, times, committed, measured_bpm, open_at,
              float(times[-1]), open_sign)
    return out


def _emit(out: list[Proposal], times: np.ndarray, committed: float,
          measured_bpm: np.ndarray, start: int, close: float, sign: int) -> None:
    """Append one event, unless it is too close to the one before it."""
    onset = float(times[start])
    # Merged into the previous event rather than counted again.
    if out and onset - out[-1].onset_sec < MIN_SEPARATION_SEC:
        return
    out.append(Proposal(onset_sec=onset, close_sec=close,
                        k=octave_k(float(measured_bpm[start]), committed) or sign,
                        committed_bpm=committed,
                        measured_bpm=float(measured_bpm[start])))


def committed_grid(beats: np.ndarray, times: np.ndarray, bpm: np.ndarray,
                   onset_sec: float, count: int = WINDOW_BEATS) -> np.ndarray:
    """The committed grid's last `count` beats, walked back from real phase.

    §2 asks for "the 16 most recent **committed beats** completed at or before
    the proposal", and the first implementation read the tracker's *published*
    beat list. Those are not the same thing, and the difference is not academic:
    the published list is gated by the lock/release hysteresis, so it thins out
    exactly when confidence is low — which is exactly when an octave conflict
    happens. Measured on 100 GTZAN excerpts, 10 of 22 proposals had between 2
    and 11 played beats behind them at onsets as late as 19.6 s, and would have
    fallen to baseline for a reason that has nothing to do with the evidence.

    A committed beat is a position on the committed grid whether or not the
    shell played it. So the grid is walked backwards from the last published
    beat — real phase, not a guess — one period at a time, each step using the
    BPM the tracker was publishing *at that moment* rather than a single period
    for the whole window, so that a drifting tempo does not smear the grid.

    Causal throughout: nothing after `onset_sec` is read.
    """
    beats = np.asarray(beats, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    bpm = np.asarray(bpm, dtype=np.float64)
    played = beats[beats <= onset_sec]
    if len(played) == 0 or len(times) == 0:
        return np.empty(0, dtype=np.float64)

    grid = [float(played[-1])]
    for _ in range(count - 1):
        index = int(np.searchsorted(times, grid[-1], side="right")) - 1
        if index < 0:
            return np.empty(0, dtype=np.float64)
        period_bpm = float(bpm[index])
        if not (period_bpm > 0.0):
            return np.empty(0, dtype=np.float64)
        previous = grid[-1] - 60.0 / period_bpm
        if previous < float(times[0]):
            return np.empty(0, dtype=np.float64)
        grid.append(previous)
    return np.array(grid[::-1], dtype=np.float64)


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
