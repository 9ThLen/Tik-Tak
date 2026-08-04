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

It has three values, not two. `true` and `false` are answers; `null` means git
could not be consulted and nothing is known either way. An earlier version
collapsed the third case into `false`, which reads as "the tree was dirty" and
is a different and much less alarming statement than "this artifact has no
provenance at all". That is not hypothetical: on 2026-08-04 a run lost its git
answers to a timeout while six workers saturated the machine, and the artifact
recorded `tree_clean: false` beside a `commit` of `null` — the same run had also
silently dropped a third of its corpus. `provenance_error` now says which call
failed and why.
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
    failures: list[str] = []

    def git(*command: str) -> str | None:
        # Two minutes rather than thirty seconds. `git status` walks the working
        # tree, and a benchmark run has every core busy and the disk saturated —
        # the call that failed in practice was not slow git, it was git waiting
        # behind eight decoders. A timeout here costs an artifact's provenance,
        # which is worth far more than the two minutes.
        try:
            done = subprocess.run(("git", "-C", str(repository)) + command,
                                  capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as error:
            failures.append(f"git {' '.join(command)}: {type(error).__name__}")
            return None
        if done.returncode != 0:
            failures.append(
                f"git {' '.join(command)}: exit {done.returncode} "
                f"{done.stderr.strip()[:200]}")
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        # True and False are answers. None means git could not be consulted, so
        # nothing is known — which is worse than a dirty tree, not better, and
        # must not be spelled the same way.
        "tree_clean": None if status is None else status == "",
        "python": platform.python_version(),
        "platform": platform.system(),
        **{name: digest(path) for name, path in (files or {}).items()},
        **extra,
    }
    if failures:
        record["provenance_error"] = failures
    return record
