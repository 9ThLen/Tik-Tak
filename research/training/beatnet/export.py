"""Use the existing TTBN v1 exporter for frozen and trained checkpoints."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import torch

from .model import BeatNetTrainable


def save_state_dict(path: pathlib.Path, model: BeatNetTrainable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().cpu()
                for name, value in model.state_dict().items()}, path)


def export_ttbn(checkpoint: pathlib.Path, output: pathlib.Path,
                *, repository: pathlib.Path) -> None:
    exporter = repository / "models" / "export_beatnet.py"
    completed = subprocess.run(
        [sys.executable, str(exporter), str(checkpoint), str(output)],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"TTBN exporter failed: {completed.stderr.strip()}")
    if not output.is_file():
        raise RuntimeError("TTBN exporter reported success without output")
