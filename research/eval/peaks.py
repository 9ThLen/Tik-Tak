"""Sparse time-frequency peaks over the network's own input features.

Adaptive whitening — dividing each band by its running peak — failed in a room
because that denominator follows the reverberant tail: the beat is divided by an
inflated number and compressed harder than the smear it was meant to remove
(`research/results/README.md`). A local maximum has no denominator, which is the
distinction this module exists to test.

Everything here is written to make a leak awkward rather than merely
discouraged:

* **the time window trails.** `future_frames` is the whole look-ahead and it is
  carried in the parameters so it can be priced in milliseconds rather than
  forgotten. `future_frames = 0` peeks at nothing. The band dimension is
  symmetric because frequency arrives all at once.
* **the density cap is a refractory period**, spent forward in time, band by
  band. Never a top-K over the file: that is a global statistic, and a global
  statistic is future context wearing a different hat.
* **nothing is normalised.** The contrast metric is a ratio of two numbers from
  the same signal, so it is already scale-free, and dividing by anything the
  whole track computed would remove the raised floor by definition — which is
  the quantity under measurement.
* **plateaus and ties resolve one way only** — earliest frame, then lowest
  band — so two runs cannot disagree about which cell of a flat region fired.

The two halves of the feature vector are *two channels, not one frequency
axis*: the filterbank is log10(1 + magnitude), the difference is its positive
rise. They are picked separately and merged by a named rule, because a peak
straddling the seam would correspond to nothing.

The `origin` arithmetic below was derived by testing scipy against a written-out
window rather than from its documentation: the sign is the opposite of the
obvious reading, and getting it backwards produces a *leading* window, which is
future context that would look exactly like success. `test_peaks.py` asserts the
window directly for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

MERGE_RULES = ("difference", "union", "sum")
COLLAPSE_RULES = ("count", "weighted", "novelty")
FPS = 50.0


@dataclass(frozen=True)
class PeakParams:
    """One point in the sweep. Hashable, so a fold can name its choice."""

    band_radius: int = 2
    past_frames: int = 6
    future_frames: int = 0
    refractory_frames: int = 3
    novelty_frames: int = 10
    merge: str = "difference"

    @property
    def lookahead_ms(self) -> float:
        """What this costs the live path, in the units the budget is kept in."""
        return self.future_frames * 1000.0 / FPS

    @property
    def causal(self) -> bool:
        """True when the map reads no frame later than the one it reports."""
        return self.future_frames == 0

    def label(self) -> str:
        return (f"b{self.band_radius}_p{self.past_frames}_f{self.future_frames}"
                f"_r{self.refractory_frames}_n{self.novelty_frames}_{self.merge}")

    def as_dict(self) -> dict:
        return {"band_radius": self.band_radius,
                "past_frames": self.past_frames,
                "future_frames": self.future_frames,
                "refractory_frames": self.refractory_frames,
                "novelty_frames": self.novelty_frames,
                "merge": self.merge,
                "lookahead_ms": self.lookahead_ms}


def windowed_max(values: np.ndarray, past_frames: int, future_frames: int,
                 band_radius: int) -> np.ndarray:
    """The maximum over [t - past, t + future] x [b - radius, b + radius].

    Truncated at the edges rather than padded: `cval=-inf` can never win, so a
    window that runs off the start reports only what is actually there. Padding
    with zeros would invent a rising edge at the head of every recording, and
    padding with the edge value would make the first frames trivially maximal.
    """
    length = past_frames + future_frames + 1
    out = maximum_filter1d(values, size=length, axis=0, mode="constant",
                           cval=-np.inf, origin=past_frames - length // 2)
    width = 2 * band_radius + 1
    return maximum_filter1d(out, size=width, axis=1, mode="constant",
                            cval=-np.inf, origin=0)


def local_maxima(values: np.ndarray, band_radius: int, past_frames: int,
                 future_frames: int) -> np.ndarray:
    """Cells that are the maximum of their window, one per plateau.

    A contiguous run of equal values is one plateau and yields one peak: the
    cell must be strictly greater than its immediate predecessor along each
    axis, which is "earliest frame, then lowest band" written so it vectorises.
    Without it a flat ridge across three bands counts three times and the
    density rules are measuring the ridge rather than the music.
    """
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool)

    peak = values >= windowed_max(values, past_frames, future_frames,
                                  band_radius)
    peak &= values > 0.0

    earlier = np.ones_like(peak)
    earlier[1:] = values[1:] > values[:-1]
    lower = np.ones_like(peak)
    lower[:, 1:] = values[:, 1:] > values[:, :-1]
    return peak & earlier & lower


def apply_refractory(mask: np.ndarray, refractory_frames: int) -> np.ndarray:
    """Spend a per-band budget forward in time.

    A band that just fired is closed for `refractory_frames`, which caps density
    using only what has already happened. The alternative — keeping the
    strongest K peaks in the file — reads the whole recording to decide about
    its first second.
    """
    if refractory_frames <= 0:
        return mask.copy()
    frames, bands = mask.shape
    out = np.zeros_like(mask)
    open_at = np.zeros(bands, dtype=np.int64)
    for t in range(frames):
        firing = np.flatnonzero(mask[t] & (open_at <= t))
        if len(firing):
            out[t, firing] = True
            open_at[firing] = t + refractory_frames + 1
    return out


def peak_map(filterbank: np.ndarray, difference: np.ndarray,
             params: PeakParams) -> tuple[np.ndarray, np.ndarray]:
    """Merged peak mask and the heights under it, for one recording."""
    if params.merge not in MERGE_RULES:
        raise ValueError(f"unknown merge rule {params.merge!r}")

    def picked(values: np.ndarray) -> np.ndarray:
        return apply_refractory(
            local_maxima(values, params.band_radius, params.past_frames,
                         params.future_frames),
            params.refractory_frames)

    if params.merge == "difference":
        return picked(difference), difference
    if params.merge == "union":
        return picked(difference) | picked(filterbank), difference
    combined = difference + filterbank
    return picked(combined), combined


def collapse(mask: np.ndarray, heights: np.ndarray, rule: str,
             params: PeakParams) -> np.ndarray:
    """Turn a peak map back into one value per frame.

    Shazam never needs this — it hashes peak pairs and looks for one consistent
    offset, and can afford almost every hash to die. A tracker needs a value for
    every frame, and which readout is right is not knowable in advance, so all
    three are measured. If all three fail that is a far stronger null than one
    failing, which is the point of carrying three.
    """
    if rule == "count":
        return mask.sum(axis=1).astype(np.float64)
    if rule == "weighted":
        return (mask * heights).sum(axis=1).astype(np.float64)
    if rule == "novelty":
        frames, bands = mask.shape
        out = np.zeros(frames, dtype=np.float64)
        last = np.full(bands, -(10 ** 9), dtype=np.int64)
        for t in range(frames):
            firing = np.flatnonzero(mask[t])
            if len(firing):
                out[t] = float(np.count_nonzero(
                    t - last[firing] > params.novelty_frames))
                last[firing] = t
        return out
    raise ValueError(f"unknown collapse rule {rule!r}")


def dense_signals(filterbank: np.ndarray,
                  difference: np.ndarray) -> dict[str, np.ndarray]:
    """The control, and it has to be a signal before it can be one.

    `count` over a dense map would be the band count in every frame — a
    constant — so the control is the per-frame mean of each half. The difference
    half is what the peak arms are compared against, because it is the half that
    carries onsets; the filterbank mean is reported beside it so a reader can
    see whether it was the better baseline all along.
    """
    return {"dense_difference": difference.mean(axis=1).astype(np.float64),
            "dense_filterbank": filterbank.mean(axis=1).astype(np.float64)}
