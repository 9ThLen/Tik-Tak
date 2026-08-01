#!/usr/bin/env python3
"""Convert BeatNet's published checkpoint into a flat weight file the core reads.

    research/.venv/bin/python models/export_beatnet.py \
        models/beatnet_model_1_weights.pt models/beatnet_model_1.ttw

BeatNet is Mojtaba Heydari, Frank Cwitkowitz and Zhiyao Duan's, CC BY 4.0, and
this is a modification in the sense the licence means: the weights are the
published ones, unretrained and unquantised, rewritten from a pickled torch
state dict into little-endian float32 in a fixed order. See NOTICE.md.

**Why not ONNX**, when Beat This! is exported to ONNX two files over. Because
the two models are asked to do different jobs under different constraints.
Beat This! is a transformer of 11 MB that runs once over a whole file off the
audio thread, so a runtime that brings its own graph executor, operator set and
threading is a fair trade. BeatNet is 402,325 parameters — one 1-D convolution,
two linear layers and a two-layer LSTM — that has to run every 20 ms with its
recurrent state carried between calls, forever, on the live path. Against that:

* ONNX Runtime Mobile is a multi-megabyte static library, several times the
  size of the model it would be hosting, and `tiktak_core` has no third-party
  dependencies at all — a property ADR 0001 treats as load-bearing, because it
  is what lets the analysis cross-compile to iOS and Android unchanged.
* Streaming a recurrent net through ONNX means lifting the LSTM state out as
  graph inputs and outputs and threading it back in by hand each frame, which
  is more code at the call site than the LSTM forward pass is in total.
* 20 MMAC/s does not need a graph executor.

So the core carries the forward pass itself, and this script's only job is to
get the numbers out of Python without reordering any of them. ONNX Runtime
remains the right answer for Beat This!, and nothing here argues otherwise.

The format is deliberately dull: a magic, a version, the seven shape numbers
the core checks its expectations against, then the tensors end to end. It is
not a container format and should not grow into one — a second model gets a
second exporter, not a schema.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

MAGIC = b"TTBN"
VERSION = 1

FEATURES = 272
CONV_CHANNELS = 2
KERNEL = 10
HIDDEN = 150
LAYERS = 2
CLASSES = 3

# Written in this order and read back in this order. The core hard-codes the
# same sequence: a weight file that disagrees is a wrong file, not a variant to
# negotiate with, so there are no names or offsets to look tensors up by.
TENSORS = [
    ("conv1.weight", (CONV_CHANNELS, 1, KERNEL)),
    ("conv1.bias", (CONV_CHANNELS,)),
    ("linear0.weight", (HIDDEN, CONV_CHANNELS * ((FEATURES - KERNEL + 1) // 2))),
    ("linear0.bias", (HIDDEN,)),
    ("lstm.weight_ih_l0", (4 * HIDDEN, HIDDEN)),
    ("lstm.weight_hh_l0", (4 * HIDDEN, HIDDEN)),
    ("lstm.bias_ih_l0", (4 * HIDDEN,)),
    ("lstm.bias_hh_l0", (4 * HIDDEN,)),
    ("lstm.weight_ih_l1", (4 * HIDDEN, HIDDEN)),
    ("lstm.weight_hh_l1", (4 * HIDDEN, HIDDEN)),
    ("lstm.bias_ih_l1", (4 * HIDDEN,)),
    ("lstm.bias_hh_l1", (4 * HIDDEN,)),
    ("linear.weight", (CLASSES, HIDDEN)),
    ("linear.bias", (CLASSES,)),
]

HEADER = struct.Struct("<4s7I")


def header() -> bytes:
    return HEADER.pack(MAGIC, VERSION, FEATURES, CONV_CHANNELS, KERNEL,
                       HIDDEN, LAYERS, CLASSES)


def convert(state: dict) -> bytes:
    """Header plus every tensor, shapes checked against what the core expects.

    Checking the shapes here rather than trusting them is the point of the
    exercise: the first version of the Python front end fed this network 84
    filters where it wanted 136, and the only thing that would have caught it
    is a shape that refused to fit.
    """
    import numpy as np

    missing = [name for name, _ in TENSORS if name not in state]
    if missing:
        raise SystemExit(f"checkpoint is missing {', '.join(missing)}")

    out = [header()]
    for name, shape in TENSORS:
        tensor = state[name]
        actual = tuple(tensor.shape)
        if actual != shape:
            raise SystemExit(f"{name}: checkpoint has {actual}, core expects {shape}")
        out.append(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype="<f4").tobytes())

    extra = sorted(set(state) - {name for name, _ in TENSORS})
    if extra:
        print(f"note: ignoring {len(extra)} tensor(s) the core does not use: "
              f"{', '.join(extra)}", file=sys.stderr)
    return b"".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args(argv)

    import torch

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    payload = convert(state)
    args.output.write_bytes(payload)

    parameters = (len(payload) - HEADER.size) // 4
    print(f"wrote {args.output} — {parameters} parameters, {len(payload)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
