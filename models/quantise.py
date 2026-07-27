"""Dynamic int8 quantisation for an ONNX model, with the same pinning rules.

Measured on the full Beat This! export, quantisation is close to free: the file
drops from 83.1 MB to 22.9 MB, single-thread inference from RTF 0.123 to 0.078,
and the bar lines do not move — F 0.946 against the float32 model's own
downbeats, where the float32 model scores 0.947 against itself, with 99.5% of
its beats recovered. The reference click track still comes back at exactly
120.00 BPM and exactly 4.00 beats per bar through both.

That result is about the full model, which docs/ml-models.md designates as the
pseudo-label source rather than a product artifact. Whether it carries over to
small0 is untested — small0's checkpoint host is not reachable from the
research container, and inferring one model's quantisation behaviour from
another's is exactly the kind of arithmetic-instead-of-measurement this
project keeps having to retract.

Weights only, no calibration set: dynamic quantisation picks activation scales
at run time, which costs a little speed against static quantisation and needs
no representative audio to be chosen, argued about, or licensed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantise(source: pathlib.Path, target: pathlib.Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "onnxruntime is required to quantise; pip install onnxruntime"
        ) from error

    if not source.is_file():
        raise SystemExit(f"{source} is not present — see models/README.md")

    # Stage beside the target and rename, so an interrupted run cannot leave a
    # half-written model somewhere a later run would trust. Same rule as fetch.
    staged = target.with_name(f".{target.name}.tmp")
    try:
        quantize_dynamic(str(source), str(staged), weight_type=QuantType.QInt8)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


def record(target: pathlib.Path, source: pathlib.Path) -> dict:
    """The provenance a quantised artifact needs to be reproducible.

    The source *digest*, not its path: what makes this file reproducible is
    which bytes went in, and a local path says nothing to anyone else.
    """
    return {
        "file": target.name,
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
        "derived_from_sha256": sha256(source),
        "method": "onnxruntime.quantization.quantize_dynamic, QuantType.QInt8",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("target", type=pathlib.Path, nargs="?")
    parser.add_argument("--print-only", action="store_true",
                        help="report what would be recorded, write no manifest")
    args = parser.parse_args(argv)

    source = args.source if args.source.is_absolute() else HERE / args.source
    target = args.target or source.with_name(source.stem + "_int8.onnx")
    target = target if target.is_absolute() else HERE / target

    quantise(source, target)
    entry = record(target, source)
    ratio = source.stat().st_size / entry["bytes"]
    print(f"{source.name} -> {target.name}   "
          f"{source.stat().st_size/1e6:.1f} MB -> {entry['bytes']/1e6:.1f} MB "
          f"({ratio:.1f}x smaller)")
    print(json.dumps(entry, indent=2))
    if args.print_only:
        return 0

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    manifest.setdefault("artifacts", {})[target.stem] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"pinned in {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
