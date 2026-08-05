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
import collections
import io
import pathlib
import pickle
import struct
import sys
import zipfile

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


# ---------------------------------------------------------------- reading --
#
# The checkpoint is read without torch. That is not thrift for its own sake:
# this script's entire job is to get fourteen float32 tensors out of a file,
# and requiring a multi-gigabyte deep-learning framework to do it means the
# export cannot be reproduced anywhere the framework is not already installed —
# including, as it turned out, this project's own environment after a rebuild.
#
# The format is a zip: `archive/data.pkl` is a pickled OrderedDict whose values
# are calls to torch._utils._rebuild_tensor_v2, and `archive/data/<n>` are the
# raw little-endian storages those calls refer to. Nothing here executes code
# from the pickle — the unpickler below refuses every global except the two it
# needs, which is also why it is safe to point at a downloaded file.
#
# Correctness is not argued, it is checked: the bytes this produces hash to the
# same sha256 the torch-based reader produced, and models/manifest.json already
# vouches for that hash.

_ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"),
    ("collections", "OrderedDict"),
}

_STORAGE_DTYPES = {
    "FloatStorage": ("<f4", 4),
    "DoubleStorage": ("<f8", 8),
    "LongStorage": ("<i8", 8),
    "IntStorage": ("<i4", 4),
    "HalfStorage": ("<f2", 2),
}


class _Tensor:
    """What _rebuild_tensor_v2 is replaced by: a description, not an object."""

    def __init__(self, storage, offset, shape, stride, *ignored):
        # _rebuild_tensor_v2 also carries requires_grad, backward hooks and,
        # in newer torch, a metadata dict. None of it describes the numbers.
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape)
        self.stride = tuple(stride)


class _Storage:
    def __init__(self, key, dtype, itemsize):
        self.key = key
        self.dtype = dtype
        self.itemsize = itemsize


def _read_state_dict(path: pathlib.Path) -> dict:
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickled = next(n for n in names if n.endswith("data.pkl"))
        prefix = pickled[: -len("data.pkl")]

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) in _ALLOWED:
                    return _Tensor if name == "_rebuild_tensor_v2" else collections.OrderedDict
                if module == "torch" and name in _STORAGE_DTYPES:
                    return name
                raise pickle.UnpicklingError(
                    f"{path.name} wants {module}.{name}, which this reader does not provide")

            def persistent_load(self, saved_id):
                kind, storage_type, key, _location, _numel = saved_id
                if kind != "storage":
                    raise pickle.UnpicklingError(f"unexpected persistent id {kind!r}")
                dtype, itemsize = _STORAGE_DTYPES[storage_type]
                return _Storage(key, dtype, itemsize)

        described = Unpickler(io.BytesIO(archive.read(pickled))).load()

        out = {}
        for name, tensor in described.items():
            raw = archive.read(f"{prefix}data/{tensor.storage.key}")
            flat = np.frombuffer(raw, dtype=tensor.storage.dtype)
            count = 1
            for dim in tensor.shape:
                count *= dim
            # Only contiguous tensors, which is what a saved state dict holds.
            # A non-contiguous one would need the strides honoured, and silently
            # reading it as contiguous is the kind of wrong that still loads.
            expected = []
            running = 1
            for dim in reversed(tensor.shape):
                expected.append(running)
                running *= dim
            if tensor.stride != tuple(reversed(expected)):
                raise SystemExit(f"{name}: not contiguous, which this reader does not handle")
            out[name] = flat[tensor.offset:tensor.offset + count].reshape(tensor.shape)
        return out


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
        actual = tuple(int(d) for d in tensor.shape)
        if actual != shape:
            raise SystemExit(f"{name}: checkpoint has {actual}, core expects {shape}")
        values = tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()
        out.append(np.ascontiguousarray(values, dtype="<f4").tobytes())

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

    payload = convert(_read_state_dict(args.checkpoint))
    args.output.write_bytes(payload)

    parameters = (len(payload) - HEADER.size) // 4
    print(f"wrote {args.output} — {parameters} parameters, {len(payload)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
