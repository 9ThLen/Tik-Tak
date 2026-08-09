"""Reader for the ``--dump-features`` file dump_analysis writes.

The dump is the network's own input: 136 log-filterbank bands at 24 per octave
from 30 Hz to 17 kHz, and the positive frame-to-frame rise of the same, at 50
frames a second.

**The two halves are two channels, not one frequency axis.** The filterbank is
log10(1 + magnitude); the difference is its positive rise. Reading the 272 as
contiguous frequencies would find peaks straddling the seam that correspond to
nothing, which is why they are returned separately and never as one array.

Layout, little-endian throughout::

    uint32   "TTFD"
    uint32   version
    uint32   0x01020304        byte order check
    uint32   bytes per value
    uint32   frames
    uint32   filterbank bands
    uint32   difference bands
    float64  frame rate
    float64  source sample rate
    float64  times[frames]
    float32  values[frames][filterbank + difference]
"""
import pathlib
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x44465454       # "TTFD" read as a little-endian uint32
BYTE_ORDER = 0x01020304
VERSION = 1
_HEADER = struct.Struct("<7I2d")


@dataclass(frozen=True)
class Features:
    """One recording's features, with the halves kept apart."""

    filterbank: np.ndarray   # (frames, bands), log10(1 + magnitude)
    difference: np.ndarray   # (frames, bands), positive rise
    times: np.ndarray        # (frames,), seconds, the frame's own reference time
    frame_rate: float
    sample_rate: float

    @property
    def frames(self) -> int:
        return len(self.times)


def read(path: pathlib.Path | str) -> Features:
    blob = pathlib.Path(path).read_bytes()
    if len(blob) < _HEADER.size:
        raise ValueError(f"{path}: shorter than a header")

    (magic, version, order, value_bytes, frames, filters, deltas,
     frame_rate, sample_rate) = _HEADER.unpack_from(blob)

    if magic != MAGIC:
        raise ValueError(f"{path}: not a feature dump")
    if order != BYTE_ORDER:
        raise ValueError(f"{path}: written by a machine of another byte order")
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, this reader knows {VERSION}")
    if value_bytes != 4:
        raise ValueError(f"{path}: {value_bytes}-byte values, expected float32")

    bands = filters + deltas
    want = _HEADER.size + frames * 8 + frames * bands * 4
    if len(blob) != want:
        raise ValueError(f"{path}: {len(blob)} bytes, header describes {want}")

    times = np.frombuffer(blob, dtype="<f8", count=frames,
                          offset=_HEADER.size)
    values = np.frombuffer(blob, dtype="<f4", count=frames * bands,
                           offset=_HEADER.size + frames * 8)
    values = values.reshape(frames, bands)

    return Features(filterbank=values[:, :filters],
                    difference=values[:, filters:],
                    times=times,
                    frame_rate=float(frame_rate),
                    sample_rate=float(sample_rate))
