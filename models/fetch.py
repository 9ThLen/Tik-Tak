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
trap), a size wildly off the documented one, re-pinning over an existing pin
without --force, and any verify where a pinned file is missing or hashes
differently.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import urllib.request

MANIFEST = pathlib.Path(__file__).resolve().parent / "manifest.json"

# A pinned artifact may be this many times smaller or larger than the manifest
# says before the pin is refused. Generous on purpose: expected_mb documents an
# order of magnitude, not a byte count — the trap being caught is a 4 KB error
# page standing in for an 8 MB model, not a checkpoint that grew ten percent.
SIZE_TOLERANCE = 3.0


def load(manifest_path: pathlib.Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save(manifest_path: pathlib.Path, manifest: dict) -> None:
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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


def pin(manifest_path: pathlib.Path, name: str, origin: str,
        force: bool = False) -> int:
    manifest = load(manifest_path)
    entry = manifest["artifacts"].get(name)
    if entry is None:
        known = ", ".join(sorted(manifest["artifacts"]))
        print(f"unknown artifact {name!r} — the manifest knows: {known}")
        return 2
    if entry.get("pinned") and not force:
        print(f"{name} is already pinned to {entry['pinned']['sha256'][:12]}…; "
              f"a different file needs --force, and the diff of the manifest "
              f"is the record of why")
        return 2

    target = manifest_path.parent / entry["file"]

    if origin.startswith(("http://", "https://")):
        # Download to a temporary name first: a failed download must never
        # leave a half-written file where the artifact belongs.
        with tempfile.NamedTemporaryFile(dir=manifest_path.parent,
                                         delete=False) as handle:
            partial = pathlib.Path(handle.name)
        try:
            with urllib.request.urlopen(origin) as response, \
                    open(partial, "wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    else:
        source = pathlib.Path(origin)
        if not source.is_file():
            print(f"{source} is not a file")
            return 2
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)

    if looks_like_html(target):
        target.unlink()
        print(f"refusing to pin {name}: the content is an HTML page, not a "
              f"model — most likely an error page delivered with status 200")
        return 1

    size = target.stat().st_size
    expected = entry.get("expected_mb")
    if expected:
        ratio = size / (expected * 1024 * 1024)
        if ratio > SIZE_TOLERANCE or ratio < 1.0 / SIZE_TOLERANCE:
            target.unlink()
            print(f"refusing to pin {name}: {size} bytes against an expected "
                  f"~{expected} MB — off by more than {SIZE_TOLERANCE}x, which "
                  f"is a wrong file, not a new version")
            return 1

    entry["pinned"] = {
        "sha256": sha256_of(target),
        "bytes": size,
        "origin": origin,
        "date": datetime.date.today().isoformat(),
    }
    save(manifest_path, manifest)
    print(f"pinned {name}: sha256 {entry['pinned']['sha256'][:12]}…, "
          f"{size} bytes. Commit the manifest — that commit is the provenance.")
    return 0


def verify(manifest_path: pathlib.Path, names: list[str] | None = None) -> int:
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
            # Not a failure: an unpinned artifact is future work, and failing
            # on it would teach people to pin placeholder hashes to get green.
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


def list_artifacts(manifest_path: pathlib.Path) -> int:
    manifest = load(manifest_path)
    for name, entry in sorted(manifest["artifacts"].items()):
        pinned = entry.get("pinned")
        target = manifest_path.parent / entry["file"]
        if pinned is None:
            state = "not pinned"
        elif not target.is_file():
            state = "pinned, file missing"
        else:
            state = "pinned, present"
        print(f"  {name:<22} {state:<22} {entry['purpose']}")
    return 0


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
