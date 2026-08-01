#!/usr/bin/env python3
"""What a number needs beside it to still mean something in six months.

A rate quoted without the commit, the binary and the corpus it was measured on
is neither reproducible nor falsifiable: a later disagreement cannot be settled,
because nobody can tell whether the code moved or the corpus did. Two of the
scripts here shipped results without any of it, which is how
``oracle_activation.json`` came to be cited in a production header while
carrying nothing that identifies what produced it.

`tree_clean` is the field that matters most and is easiest to skim past. A run
from a dirty tree records a commit that does **not** describe the binary, so the
artifact is a record of a measurement rather than a reproducible one. Committing
afterwards does not repair it — the run has to be repeated from a clean tree.
"""

from __future__ import annotations

import hashlib
import pathlib
import platform
import subprocess
import time
from typing import Any, Mapping

__all__ = ["digest", "provenance"]


def digest(path: pathlib.Path | None) -> dict | None:
    if path is None:
        return None
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError:
        return None
    return {"name": pathlib.Path(path).name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def provenance(repository: pathlib.Path,
               files: Mapping[str, pathlib.Path | None] | None = None,
               **extra: Any) -> dict:
    def git(*command: str) -> str | None:
        try:
            done = subprocess.run(("git", "-C", str(repository)) + command,
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git("rev-parse", "HEAD"),
        # False means the commit above does not identify the binary that ran.
        "tree_clean": git("status", "--porcelain") == "",
        "python": platform.python_version(),
        "platform": platform.system(),
        **{name: digest(path) for name, path in (files or {}).items()},
        **extra,
    }
