"""Offline beat tracking by dynamic programming (Ellis, 2007).

This is the *offline* back-end: it sees the whole ODF before deciding anything,
so it can pick the globally best beat sequence instead of committing frame by
frame. That is why the file path in the app analyses an imported track far more
accurately than the microphone path ever will, and why it is worth having two
trackers over one shared front-end.

The online tracker (particle filter) is a separate module and a later phase. It
solves a strictly harder problem — it must *predict* the next beat before it
arrives, because a click has to sound at the beat, not after it.

The objective, maximised over all beat sequences b_1..b_N:

    sum_i  odf(b_i)  +  tightness * sum_i  -( log( (b_i - b_{i-1}) / period ) )^2

The first term wants beats on onsets; the second wants the gaps between them to
stay near the estimated period. One pass forward with a backtrace solves it
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tempo import TempoConfig, estimate_tempo


@dataclass(frozen=True)
class TrackerConfig:
    # Weight of the tempo-consistency penalty. Higher keeps the beat grid rigid
    # through weak passages; lower lets it follow a rubato performer.
    tightness: float = 100.0
    # Trim beats at the very start and end that sit on no real onset — the DP
    # will happily extend its grid into silence to keep the sequence regular.
    trim: bool = True

    # How many of the estimator's leading tempo candidates to actually track
    # before choosing between them. The prior ranks candidates by plausibility;
    # it has no idea which one would *fit*. Measured against a Beat This!
    # reference on 106 recordings, the right tempo is nearly always somewhere in
    # this list and simply not first — an oracle picking the best of these
    # reaches CMLt 0.704 against 0.488 for the first choice, and 0.695 for
    # handing the tracker the reference tempo outright. The evidence to break
    # the tie is the objective the DP maximises anyway, so it costs one extra
    # pass per candidate and nothing else.
    #
    # Four rather than all eight: the tail of the candidate list is where the
    # implausible octaves live, and every extra hypothesis is another full pass.
    #
    # These two live on OfflineConfig in the core rather than on its
    # TrackerConfig, because there the search wraps the tracker instead of
    # living inside it. Here `track_beats` is the whole offline path — it is
    # what the parity gate runs against tt_offline — so they belong on this
    # config. See core/src/analysis/offline.hpp.
    tempo_hypotheses: int = 4
    # How much to trust the prior against the fit, as the exponent on candidate
    # strength in `strength**w * objective`.
    #
    # Zero would trust the fit alone, which is much worse rather than better:
    # the objective is a mean over beats and a half-speed grid raises it by
    # sitting only on the loudest onsets, so fit alone chases exactly the octave
    # error the prior exists to prevent.
    #
    # 1.5 was chosen by holding out each batch of the evaluation set and fitting
    # on the rest; both batches picked the same value and both gained on
    # material they had not been fitted to, 8 and 10 points of CMLt. The optimum
    # is broad — anything from 0.75 to 2.0 lands within 3 points — which is the
    # reason to believe it at all.
    tempo_fit_weight: float = 1.5

    def validate(self) -> None:
        if self.tightness <= 0.0:
            raise ValueError("tightness must be positive")
        if self.tempo_hypotheses < 1:
            raise ValueError("tempo_hypotheses must be at least 1")
        if self.tempo_fit_weight < 0.0:
            raise ValueError("tempo_fit_weight must be non-negative")


@dataclass
class BeatResult:
    beats: np.ndarray        # beat times, seconds
    frames: np.ndarray       # ODF frame index of each beat
    bpm: float               # period the tracker was run at
    tempo_confidence: float
    # The DP objective at the last surviving beat, divided by the number of
    # beats kept. The total is not comparable across tempi — a grid at half
    # speed collects roughly twice the score for doing nothing better — and per
    # beat is not neutral either, since a slow grid can sit on the loudest
    # onsets only. So this is evidence for weighing tempo hypotheses against a
    # prior, not a number to maximise on its own. Zero when no beats were found.
    objective_per_beat: float = 0.0

    def __len__(self) -> int:
        return len(self.beats)

    @property
    def intervals(self) -> np.ndarray:
        return np.diff(self.beats)


def _local_score(odf_values: np.ndarray, period: float) -> np.ndarray:
    """Smooth and normalise the ODF into a per-frame "beatiness" score.

    Smoothing over a fraction of the beat period stops the DP from latching onto
    single-frame spikes; normalising by the standard deviation makes `tightness`
    mean the same thing regardless of input level.
    """
    spread = odf_values.std(ddof=1) if len(odf_values) > 1 else 0.0
    if spread <= 0.0:
        return np.zeros_like(odf_values)

    half = max(1, int(round(period)))
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets * 32.0 / period) ** 2)

    return np.convolve(odf_values / spread, kernel, mode="same")


def _forward(local_score: np.ndarray, period: float, tightness: float):
    """Forward pass: best cumulative score ending on each frame, and backlinks."""
    n = len(local_score)
    cumulative = np.zeros(n, dtype=np.float64)
    backlink = np.full(n, -1, dtype=np.int64)

    # Candidate gaps to the previous beat: half the period to twice it. Outside
    # that range the transition penalty dominates anyway.
    lo = max(1, int(round(period / 2)))
    hi = max(lo + 1, int(round(2 * period)))
    gaps = np.arange(lo, hi + 1, dtype=np.int64)
    penalty = -tightness * np.log(gaps / period) ** 2

    for i in range(n):
        previous = i - gaps

        # Out-of-range predecessors keep the penalty alone, with no cumulative
        # score behind them. That is what lets a sequence *start*: early frames
        # can win without paying for a predecessor that does not exist.
        scores = penalty.copy()
        valid = previous >= 0
        scores[valid] += cumulative[previous[valid]]

        best = int(np.argmax(scores))
        cumulative[i] = local_score[i] + scores[best]
        backlink[i] = previous[best] if previous[best] >= 0 else -1

    return cumulative, backlink


def _last_beat(cumulative: np.ndarray) -> int:
    """Where to start the backtrace: the last convincing local maximum."""
    if len(cumulative) < 3:
        return len(cumulative) - 1

    peaks = np.zeros(len(cumulative), dtype=bool)
    peaks[1:-1] = (cumulative[1:-1] >= cumulative[:-2]) & (cumulative[1:-1] > cumulative[2:])
    if not peaks.any():
        return int(np.argmax(cumulative))

    # Root-mean-square of the peaks, halved: high enough to ignore the tail the
    # DP grows into trailing silence, low enough to keep a genuine final beat.
    threshold = 0.5 * np.sqrt(np.mean(cumulative[peaks] ** 2))
    strong = np.flatnonzero(peaks & (cumulative >= threshold))
    return int(strong[-1]) if len(strong) else int(np.argmax(cumulative))


def _trim(frames: np.ndarray, local_score: np.ndarray) -> np.ndarray:
    """Drop leading and trailing beats that sit on no actual onset."""
    if len(frames) == 0:
        return frames

    threshold = 0.5 * np.mean(local_score[frames] ** 2) ** 0.5
    keep = local_score[frames] >= threshold
    if not keep.any():
        return frames

    first, last = int(np.argmax(keep)), len(keep) - int(np.argmax(keep[::-1]))
    return frames[first:last]


def _track_at(
    odf_values: np.ndarray,
    times: np.ndarray,
    fps: float,
    bpm: float,
    config: TrackerConfig,
    confidence: float,
) -> BeatResult:
    """One dynamic-programming pass at a fixed tempo."""
    empty = BeatResult(
        beats=np.zeros(0),
        frames=np.zeros(0, dtype=np.int64),
        bpm=float(bpm),
        tempo_confidence=float(confidence),
    )
    if len(odf_values) < 3 or not np.any(odf_values > 0.0):
        return empty

    period = 60.0 * fps / bpm
    local_score = _local_score(odf_values, period)
    if not np.any(local_score > 0.0):
        return empty

    cumulative, backlink = _forward(local_score, period, config.tightness)

    frames: list[int] = []
    cursor = _last_beat(cumulative)
    while cursor >= 0:
        frames.append(cursor)
        cursor = int(backlink[cursor])
    frames.reverse()

    found = np.array(frames, dtype=np.int64)
    if config.trim:
        found = _trim(found, local_score)

    # Read the objective off the sequence that was kept, not off the whole
    # array: trim() may have dropped beats from either end, and the cumulative
    # score at the last surviving frame still includes everything the backtrace
    # passed through.
    objective = float(cumulative[found[-1]] / len(found)) if len(found) else 0.0

    return BeatResult(
        beats=times[found],
        frames=found,
        bpm=float(bpm),
        tempo_confidence=float(confidence),
        objective_per_beat=objective,
    )


def track_beats(
    odf_values: np.ndarray,
    times: np.ndarray,
    fps: float,
    bpm: float | None = None,
    config: TrackerConfig | None = None,
    tempo_config: TempoConfig | None = None,
) -> BeatResult:
    """Find the beat sequence in an ODF.

    `times` comes from the ODF rather than being derived from `fps`, because the
    ODF stamps each frame with its window centre. Recomputing the times here
    would reintroduce half a window of bias.

    If `bpm` is None the tempo is estimated first, and the leading candidates
    are each tracked and scored so that the grid which actually fits wins over
    the one the prior merely likes best. Pass a `bpm` to fix it: that is an
    instruction rather than a hypothesis, and searching around it would be
    overriding the caller — the app's manual mode depends on it.
    """
    config = config or TrackerConfig()
    config.validate()

    odf_values = np.asarray(odf_values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(odf_values) != len(times):
        raise ValueError("odf_values and times must be the same length")
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    if bpm is not None:
        if bpm <= 0.0:
            raise ValueError("bpm must be positive")
        return _track_at(odf_values, times, fps, bpm, config, 1.0)

    estimate = estimate_tempo(odf_values, fps, tempo_config)
    if estimate.bpm <= 0.0:
        raise ValueError("bpm must be positive")

    # Confidence stays the estimator's, not the winning hypothesis'. It answers
    # "how periodic is this signal", which the search does not change.
    confidence = estimate.confidence

    best: BeatResult | None = None
    best_score = 0.0
    for candidate_bpm, strength in estimate.top_candidates(config.tempo_hypotheses):
        if candidate_bpm <= 0.0 or strength <= 0.0:
            continue
        grid = _track_at(odf_values, times, fps, candidate_bpm, config, confidence)
        if len(grid.beats) == 0:
            continue

        score = strength**config.tempo_fit_weight * grid.objective_per_beat
        # Strictly greater, so a tie leaves the estimator's own ranking in
        # charge and the same audio always produces the same grid.
        if best is None or score > best_score:
            best, best_score = grid, score

    if best is not None:
        return best
    # No candidate produced a grid — fall back to the single estimate, which
    # keeps the failure identical to what it was before this existed.
    return _track_at(odf_values, times, fps, estimate.bpm, config, confidence)
