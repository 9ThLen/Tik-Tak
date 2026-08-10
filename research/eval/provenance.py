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
import re
import subprocess
import time
from typing import Any, Mapping

__all__ = [
    "PROVENANCE_SCHEMA", "digest", "experiment_provenance", "provenance",
]


PROVENANCE_SCHEMA = "tiktak.provenance/v1"

# Captured on Windows when Git can traverse the repository but cannot read the
# user's global excludes file. This warning does not make `git status` partial.
# Keep this list deliberately narrow: an unknown warning is evidence we do not
# understand, so provenance must fail closed rather than guess that it is safe.
_BENIGN_GIT_STDERR = (
    re.compile(
        r"^warning: unable to access '.*[\\/]\.config[\\/]git[\\/]ignore': "
        r"Permission denied$"
    ),
)


def _unexpected_git_stderr(stderr: str) -> list[str]:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return [
        line for line in lines
        if not any(pattern.fullmatch(line) for pattern in _BENIGN_GIT_STDERR)
    ]


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
        unexpected = _unexpected_git_stderr(done.stderr)
        if unexpected:
            failures.append(
                f"git {' '.join(command)}: unexpected stderr "
                f"{' | '.join(unexpected)[:200]}")
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    record = {
        "schema": PROVENANCE_SCHEMA,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        # True and False are answers. None means git could not be consulted, so
        # nothing is known — which is worse than a dirty tree, not better, and
        # must not be spelled the same way.
        "tree_clean": (
            None if commit is None or status is None else status == ""
        ),
        "python": platform.python_version(),
        "platform": platform.system(),
        **{name: digest(path) for name, path in (files or {}).items()},
        **extra,
    }
    if failures:
        record["provenance_error"] = failures
    return record


def experiment_provenance(
        repository: pathlib.Path,
        files: Mapping[str, pathlib.Path | None] | None = None,
        **extra: Any) -> dict:
    """Return provenance only when the checked-out commit describes the run.

    Experimental artifacts are evidence, not diagnostics. A dirty tree and an
    unknown tree are therefore both hard failures. Call ``provenance`` directly
    only for an explicitly labelled diagnostic whose schema prevents it from
    being accepted as an experiment result.
    """
    record = provenance(repository, files, **extra)
    unreadable = [name for name, path in (files or {}).items()
                  if path is not None and record.get(name) is None]
    if unreadable:
        raise RuntimeError(
            "experimental run has unreadable provenance inputs: "
            + ", ".join(sorted(unreadable)))
    if record["tree_clean"] is not True:
        detail = record.get("provenance_error", [])
        suffix = f": {'; '.join(detail)}" if detail else ""
        raise RuntimeError(
            f"experimental run requires tree_clean is True; got "
            f"{record['tree_clean']!r}{suffix}")
    return record
