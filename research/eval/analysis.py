"""Running the C++ analysis and reading back what it concluded.

Everything the evaluation scores comes through here, and it deliberately comes
from the shipping core rather than from a Python model of it. The Python
reference in ``research/tiktak/`` exists to design algorithms quickly and is
held to the C++ by ``tools/parity/check_parity.py``; it stops at beat tracking
and has no downbeat stage. Reimplementing one here to evaluate would produce
numbers about the reimplementation.

Build the tool first::

    cmake -S tools/eval -B tools/eval/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build tools/eval/build
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

__all__ = ["Estimate", "Analyser", "DEFAULT_BINARY"]

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _find_binary() -> pathlib.Path:
    """Where dump_analysis lands, across the layouts CMake produces.

    Windows adds .exe, and a multi-config generator (Visual Studio, Xcode) puts
    the binary in a per-configuration subdirectory rather than at the top of the
    build tree. Returning the first that exists — and the plain path when none
    do — keeps the error message pointing at the expected location.
    """
    build = ROOT / "tools" / "eval" / "build"
    names = ("dump_analysis", "dump_analysis.exe")
    folders = (build, *(build / c for c in
                        ("RelWithDebInfo", "Release", "Debug", "MinSizeRel")))
    for folder in folders:
        for name in names:
            if (candidate := folder / name).is_file():
                return candidate
    return build / names[0]


DEFAULT_BINARY = _find_binary()


@dataclass
class Estimate:
    """One analysis result. Mirrors the JSON dump_analysis prints.

    The two margins are separate because they answer separate questions and
    conflating them hid real errors — see eval/downbeat.py and the comment on
    DownbeatResult in the core.
    """

    beats: np.ndarray
    downbeats: np.ndarray
    beats_per_bar: int
    downbeat_strength: float
    downbeat_phase_margin: float
    downbeat_meter_margin: float
    # What the core itself concluded, under its own default thresholds. The
    # sweep applies its own instead — this is here so a run can report how the
    # shipped defaults would have behaved.
    downbeat_confident: bool = False
    bpm: float = 0.0
    confidence: float = 0.0
    sample_rate: float = 0.0
    duration_sec: float = 0.0
    # "cues" when the built-in scorer produced the salience, "file" when it was
    # injected — so a result can always say which backend it is a result *of*.
    salience_source: str = "cues"

    @classmethod
    def from_json(cls, payload: dict) -> "Estimate":
        return cls(
            beats=np.asarray(payload.get("beats", []), dtype=np.float64),
            downbeats=np.asarray(payload.get("downbeats", []), dtype=np.float64),
            beats_per_bar=int(payload.get("beats_per_bar", 0)),
            downbeat_strength=float(payload.get("downbeat_strength", 0.0)),
            downbeat_phase_margin=float(payload.get("downbeat_phase_margin", 0.0)),
            downbeat_meter_margin=float(payload.get("downbeat_meter_margin", 0.0)),
            downbeat_confident=bool(payload.get("downbeat_confident", False)),
            bpm=float(payload.get("bpm", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            sample_rate=float(payload.get("sample_rate", 0.0)),
            duration_sec=float(payload.get("duration_sec", 0.0)),
            salience_source=str(payload.get("salience_source", "cues")),
        )


class Analyser:
    """Calls ``dump_analysis`` and parses its JSON."""

    def __init__(self, binary: pathlib.Path | str = DEFAULT_BINARY):
        self.binary = pathlib.Path(binary)

    @property
    def available(self) -> bool:
        return self.binary.is_file()

    def _run(self, args: list[str]) -> Estimate:
        if not self.available:
            raise FileNotFoundError(
                f"{self.binary} not found — build it with:\n"
                f"  cmake -S tools/eval -B tools/eval/build "
                f"-DCMAKE_BUILD_TYPE=RelWithDebInfo\n"
                f"  cmake --build tools/eval/build"
            )
        completed = subprocess.run(
            [str(self.binary), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"dump_analysis failed on {args[0]}: {completed.stderr.strip()}"
            )
        return Estimate.from_json(json.loads(completed.stdout))

    def analyse_file(self, path: pathlib.Path | str,
                     salience: "np.ndarray | None" = None,
                     salience_min_range: "float | None" = None) -> Estimate:
        """Analyses an encoded audio file (WAV, FLAC or MP3).

        ``salience`` swaps the built-in per-beat scorer for the values given —
        one per beat of *this file's* grid, so the caller has usually analysed
        once already to learn the beat times. This is how a model is scored
        through the shipping resolver before it is ported: sample its activation
        at the beat times, pass the array here. The count must match the beat
        count; the tool refuses a mismatch rather than aligning by guesswork.
        ``salience_min_range`` is the backend's calibrated evidence gate in its
        own units; the built-in cue default is used when it is omitted.
        """
        return self._with_salience([str(path)], salience, salience_min_range)

    def analyse_audio(self, audio: np.ndarray, sample_rate: float,
                      salience: "np.ndarray | None" = None,
                      salience_min_range: "float | None" = None) -> Estimate:
        """Analyses samples already in memory, via a raw float32 temporary.

        Raw rather than a WAV: this path exists for the synthetic clips, and
        writing a container would put a second encoder between the material and
        the thing being measured for no gain.
        """
        # delete=False, then unlink: on Windows a NamedTemporaryFile still open
        # here cannot be opened by the child process at all.
        handle = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
        try:
            handle.write(np.asarray(audio, dtype=np.float32).tobytes())
            handle.close()
            return self._with_salience([handle.name, repr(float(sample_rate))],
                                       salience, salience_min_range)
        finally:
            pathlib.Path(handle.name).unlink(missing_ok=True)

    def _with_salience(self, args: list[str],
                       salience: "np.ndarray | None",
                       salience_min_range: "float | None") -> Estimate:
        if salience is None:
            if salience_min_range is not None:
                raise ValueError("salience_min_range requires salience")
            return self._run(args)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            handle.write("\n".join(
                repr(float(v)) for v in np.asarray(salience, dtype=np.float64)))
            handle.close()
            command = [*args, "--salience", handle.name]
            if salience_min_range is not None:
                command.extend(
                    ["--salience-min-range", repr(float(salience_min_range))]
                )
            return self._run(command)
        finally:
            pathlib.Path(handle.name).unlink(missing_ok=True)
