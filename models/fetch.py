#!/usr/bin/env python3
"""Pin and verify model artifacts, so a weight file is a known quantity.

    python models/fetch.py list
    python models/fetch.py pin beatnet_model_1 /path/to/model_1_weights.pt
    python models/fetch.py pin beat_this_small https://…   # downloads first
    python models/fetch.py verify

Stdlib only, deliberately: this script runs on whatever machine happens to have
network access, before the artifact ever reaches the development environment,
and a dependency it had to install first would defeat that.

**Why trust-on-first-use.** The development environment has no network, so the
checksums cannot be pinned from here, and a checksum typed in from memory or
from a web page is a guess wearing hexadecimal. Instead the *first* successful
fetch computes the hash and writes it into the manifest; committing that change
is the act of vouching for the artifact, and every later fetch or verify is
held to it. The threat this defends against is not a targeted attack — it is
the ordinary one: a re-download that silently got a different file, a proxy
that returned an error page with status 200, a share link whose content moved.

**What refuses loudly.** Pinning content that looks like HTML (the error-page
trap), a size wildly off the documented one, replacing an existing pin with
different bytes without --force, and any verify where an artifact exists
unpinned, is missing despite a pin, or hashes differently. Supplying the same
trusted bytes restores a missing artifact in a fresh checkout without changing
the manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import errno
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import urllib.request

MANIFEST = pathlib.Path(__file__).resolve().parent / "manifest.json"

# A pinned artifact may be this many times smaller or larger than the manifest
# says before the pin is refused. Generous on purpose: expected_mb documents an
# order of magnitude, not a byte count — the trap being caught is a 4 KB error
# page standing in for an 8 MB model, not a checkpoint that grew ten percent.
SIZE_TOLERANCE = 3.0


@contextlib.contextmanager
def artifact_lock(manifest_path: pathlib.Path):
    """Serialize every manifest/artifact transaction across processes."""
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    with open(lock_path, "a+b") as handle:
        # msvcrt locks a byte range. Initialising one byte before contending is
        # harmless even if two first-time callers race: every caller still
        # locks byte zero.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)

        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {
                            errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load(manifest_path: pathlib.Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save(manifest_path: pathlib.Path, manifest: dict) -> None:
    # A killed process must leave either the old manifest or the complete new
    # one, never a truncated JSON file.
    staged: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=manifest_path.parent, prefix=f".{manifest_path.name}.",
                suffix=".tmp", delete=False) as handle:
            staged = pathlib.Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        staged.replace(manifest_path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def looks_like_html(path: pathlib.Path) -> bool:
    """The exact trap a proxy sets: an error page saved under a model's name."""
    with open(path, "rb") as handle:
        head = handle.read(512).lstrip().lower()
    return head.startswith((b"<!doctype", b"<html", b"<head", b"<?xml"))


def provenance_from_manifest(entry: dict) -> dict:
    """Snapshot the durable source description, never the caller's local path."""
    source = entry.get("source") or {}
    if not isinstance(source, dict) or not source:
        raise ValueError("the artifact has no canonical source in the manifest")
    provenance = {"source": copy.deepcopy(source)}
    conversion = entry.get("conversion")
    if conversion is not None:
        if not isinstance(conversion, dict):
            raise ValueError("artifact conversion metadata must be an object")
        provenance["conversion"] = copy.deepcopy(conversion)
    return provenance


def stage_origin(origin: str, target: pathlib.Path) -> pathlib.Path | None:
    """Copy or download origin beside target, without touching target."""
    if not origin.startswith(("http://", "https://")):
        source = pathlib.Path(origin)
        if not source.is_file():
            print(f"{source} is not a file")
            return None

    with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False) as handle:
        staged = pathlib.Path(handle.name)
    try:
        if origin.startswith(("http://", "https://")):
            with urllib.request.urlopen(origin) as response, \
                    open(staged, "wb") as out:
                shutil.copyfileobj(response, out)
        else:
            shutil.copyfile(pathlib.Path(origin), staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def install_and_save(target: pathlib.Path, staged: pathlib.Path,
                     manifest_path: pathlib.Path, manifest: dict) -> None:
    """Install validated bytes and roll them back if saving the pin fails."""
    backup: pathlib.Path | None = None
    keep_backup = False
    try:
        if target.is_file():
            with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=f".{target.name}.",
                    suffix=".backup", delete=False) as handle:
                backup = pathlib.Path(handle.name)
            shutil.copyfile(target, backup)

        staged.replace(target)
        try:
            save(manifest_path, manifest)
        except BaseException:
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                try:
                    backup.replace(target)
                except BaseException as rollback_error:
                    keep_backup = True
                    raise RuntimeError(
                        f"could not restore {target}; the previous artifact "
                        f"is preserved at {backup}") from rollback_error
            raise
    finally:
        staged.unlink(missing_ok=True)
        if backup is not None and not keep_backup:
            backup.unlink(missing_ok=True)


def _pin_locked(manifest_path: pathlib.Path, name: str, origin: str,
                force: bool = False) -> int:
    manifest = load(manifest_path)
    entry = manifest["artifacts"].get(name)
    if entry is None:
        known = ", ".join(sorted(manifest["artifacts"]))
        print(f"unknown artifact {name!r} — the manifest knows: {known}")
        return 2

    target = manifest_path.parent / entry["file"]

    staged = stage_origin(origin, target)
    if staged is None:
        return 2

    try:
        if looks_like_html(staged):
            print(f"refusing to pin {name}: the content is an HTML page, not a "
                  f"model — most likely an error page delivered with status 200")
            return 1

        size = staged.stat().st_size
        expected = entry.get("expected_mb")
        if expected:
            ratio = size / (expected * 1024 * 1024)
            if ratio > SIZE_TOLERANCE or ratio < 1.0 / SIZE_TOLERANCE:
                print(f"refusing to pin {name}: {size} bytes against an expected "
                      f"~{expected} MB — off by more than {SIZE_TOLERANCE}x, which "
                      f"is a wrong file, not a new version")
                return 1

        digest = sha256_of(staged)
        trusted = entry.get("pinned")
        if trusted is not None and not force:
            if digest != trusted["sha256"]:
                print(
                    f"refusing to replace {name}: supplied sha256 "
                    f"{digest[:12]}… does not match the trusted "
                    f"{trusted['sha256'][:12]}…; use --force only when "
                    f"intentionally changing the trust root")
                return 2
            staged.replace(target)
            print(
                f"restored {name}: supplied bytes match trusted sha256 "
                f"{digest[:12]}…; manifest unchanged")
            return 0

        entry["pinned"] = {
            "sha256": digest,
            "bytes": size,
            "provenance": provenance_from_manifest(entry),
            "date": datetime.date.today().isoformat(),
        }
        install_and_save(target, staged, manifest_path, manifest)
    finally:
        staged.unlink(missing_ok=True)
    print(f"pinned {name}: sha256 {entry['pinned']['sha256'][:12]}…, "
          f"{size} bytes. Commit the manifest — that commit is the provenance.")
    return 0


def pin(manifest_path: pathlib.Path, name: str, origin: str,
        force: bool = False) -> int:
    manifest_path = manifest_path.resolve()
    with artifact_lock(manifest_path):
        return _pin_locked(manifest_path, name, origin, force)


def _verify_locked(manifest_path: pathlib.Path,
                   names: list[str] | None = None) -> int:
    manifest = load(manifest_path)
    artifacts = manifest["artifacts"]
    selected = names or sorted(artifacts)
    failures = 0
    for name in selected:
        entry = artifacts.get(name)
        if entry is None:
            print(f"  ? {name}: not in the manifest")
            failures += 1
            continue
        pinned = entry.get("pinned")
        target = manifest_path.parent / entry["file"]
        if pinned is None:
            if target.exists():
                print(f"  ! {name}: {entry['file']} exists but is not pinned; "
                      f"no manifest hash vouches for these bytes")
                failures += 1
                continue
            # An absent, unpinned artifact is future work, not a failure.
            print(f"  - {name}: not pinned yet")
            continue
        if not target.is_file():
            print(f"  ! {name}: pinned but {entry['file']} is missing")
            failures += 1
            continue
        actual = sha256_of(target)
        if actual != pinned["sha256"]:
            print(f"  ! {name}: sha256 mismatch — the file is not the one "
                  f"that was pinned\n"
                  f"      pinned  {pinned['sha256']}\n"
                  f"      actual  {actual}")
            failures += 1
            continue
        print(f"  ok {name}: {pinned['bytes']} bytes, "
              f"sha256 {pinned['sha256'][:12]}…")
    return 1 if failures else 0


def verify(manifest_path: pathlib.Path, names: list[str] | None = None) -> int:
    manifest_path = manifest_path.resolve()
    with artifact_lock(manifest_path):
        return _verify_locked(manifest_path, names)


def _list_artifacts_locked(manifest_path: pathlib.Path) -> int:
    manifest = load(manifest_path)
    for name, entry in sorted(manifest["artifacts"].items()):
        pinned = entry.get("pinned")
        target = manifest_path.parent / entry["file"]
        if pinned is None:
            state = ("UNTRUSTED file present" if target.exists()
                     else "not pinned")
        elif not target.is_file():
            state = "pinned, file missing"
        else:
            state = "pinned, present"
        print(f"  {name:<22} {state:<22} {entry['purpose']}")
    return 0


def list_artifacts(manifest_path: pathlib.Path) -> int:
    manifest_path = manifest_path.resolve()
    with artifact_lock(manifest_path):
        return _list_artifacts_locked(manifest_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list")

    pin_cmd = commands.add_parser("pin")
    pin_cmd.add_argument("name")
    pin_cmd.add_argument("origin", help="a local file or an http(s) URL")
    pin_cmd.add_argument("--force", action="store_true")

    verify_cmd = commands.add_parser("verify")
    verify_cmd.add_argument("names", nargs="*")

    args = parser.parse_args(argv)
    if args.command == "list":
        return list_artifacts(args.manifest)
    if args.command == "pin":
        return pin(args.manifest, args.name, args.origin, args.force)
    return verify(args.manifest, args.names or None)


if __name__ == "__main__":
    sys.exit(main())
