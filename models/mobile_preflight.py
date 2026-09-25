#!/usr/bin/env python3
"""P1-D: what can be known about an ONNX model on a phone without a phone.

    python models/mobile_preflight.py models/small0_int8.onnx \
        --shape 1,1500,128 --output <outside-repository>/preflight.json

plan.md:306 asks for the static half of the mobile budget before any
architecture is chosen: operator coverage under ONNX Runtime Mobile,
parameters, MACs and model size. The dynamic half -- RTF, RAM, cold start,
energy, thermal -- needs a device and stays `estimated` until P5.

Operator coverage is not reimplemented here. ONNX Runtime ships its own
answer, `python -m onnxruntime.tools.check_onnx_model_mobile_usability`, which
partitions the graph against the op lists NNAPI and both CoreML formats
support; this calls the same functions and keeps the numbers instead of a log.
Its op lists are version-specific, so the version travels with the result.

The input shape is fixed before anything is counted, because both questions
depend on it: NNAPI refuses dynamic shapes outright, CoreML pays for them, and
a MAC count over a symbolic time axis is not a number. The shape given should
be the one the shell will actually feed -- for Beat This! the 1500-frame chunk
`export_beat_this.py` exports and the harness runs.

**MACs count MatMul, Gemm, Conv and LSTM, and MatMulInteger, and nothing
else.** Any other op that takes a weight matrix -- an initializer of rank two
or more, such as ConvInteger or DynamicQuantizeLSTM in a quantised graph -- is
named in `macs_uncounted_ops`, so a partial number cannot pass for a complete
one. Counted on the graph as exported, before ORT fuses anything.

`parameters` is every initializer element, so a quantised graph's scales and
zero points are included in it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
import tempfile

import onnx
import onnxruntime
from onnx import numpy_helper
from onnxruntime.tools.mobile_helpers import usability_checker
from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from research.eval.provenance import provenance  # noqa: E402

SCHEMA = "tiktak.mobile_preflight/v1"

COUNTED = {"MatMul", "MatMulInteger", "Gemm", "Conv", "LSTM"}


def _dims(value: onnx.ValueInfoProto) -> list[int] | None:
    shape = value.type.tensor_type.shape
    dims = [d.dim_value if d.HasField("dim_value") else None for d in shape.dim]
    return None if None in dims else dims


def _macs(model: onnx.ModelProto) -> tuple[int, dict[str, int]]:
    graph = model.graph
    shapes = {v.name: _dims(v) for v in (*graph.input, *graph.value_info,
                                         *graph.output)}
    shapes.update({t.name: list(t.dims) for t in graph.initializer})
    by_op: collections.Counter[str] = collections.Counter()
    for node in graph.node:
        op = node.op_type
        if op in ("MatMul", "MatMulInteger"):
            a, out = shapes.get(node.input[0]), shapes.get(node.output[0])
            if a and out:
                by_op[op] += math.prod(out) * a[-1]
        elif op == "Gemm":
            a, out = shapes.get(node.input[0]), shapes.get(node.output[0])
            trans_a = any(x.name == "transA" and x.i for x in node.attribute)
            if a and out:
                by_op[op] += math.prod(out) * (a[0] if trans_a else a[1])
        elif op == "Conv":
            w, out = shapes.get(node.input[1]), shapes.get(node.output[0])
            if w and out:
                # w is (out_ch, in_ch / group, *kernel): each output element
                # reads exactly w[1:] inputs, grouping included.
                by_op[op] += math.prod(out) * math.prod(w[1:])
        elif op == "LSTM":
            x, w, r = (shapes.get(n) for n in node.input[:3])
            if x and w and r:
                # x (seq, batch, input); w (dirs, 4H, input); r (dirs, 4H, H)
                by_op[op] += x[0] * x[1] * w[0] * w[1] * (w[2] + r[2])
    return sum(by_op.values()), dict(by_op)


def _uncounted(graph: onnx.GraphProto) -> list[str]:
    weights = {t.name for t in graph.initializer if len(t.dims) >= 2}
    return sorted({node.op_type for node in graph.node
                   if node.op_type not in COUNTED
                   and weights.intersection(node.input)})


def _partitions(info) -> dict:
    return {
        "nodes": info.num_nodes,
        "supported_nodes": info.num_supported_nodes,
        "partitions": info.num_partitions,
        "unsupported_ops": sorted(info.unsupported_ops),
        "suitability": info.suitability().name,
    }


def preflight(model_path: pathlib.Path, shape: list[int]) -> dict:
    model = onnx.load(str(model_path))
    inputs = [i for i in model.graph.input
              if i.name not in {t.name for t in model.graph.initializer}]
    if len(inputs) != 1:
        raise ValueError(f"expected one graph input, found {len(inputs)}")
    make_input_shape_fixed(model.graph, inputs[0].name, shape)

    parameters = 0
    parameter_bytes: collections.Counter[str] = collections.Counter()
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        parameters += array.size
        parameter_bytes[str(array.dtype)] += array.nbytes
    ops = collections.Counter(node.op_type for node in model.graph.node)

    macs, macs_by_op = _macs(onnx.shape_inference.infer_shapes(model))

    # The same path ORT's own checker takes: basic optimisation, then shape
    # inference, then partitioning against each EP's op list.
    with tempfile.TemporaryDirectory() as tmp:
        fixed = pathlib.Path(tmp) / "fixed.onnx"
        optimised = pathlib.Path(tmp) / "optimised.onnx"
        onnx.save_model(model, str(fixed))
        usability_checker.optimize_model(fixed, optimised,
                                         use_external_initializers=True)
        checked = usability_checker.ModelProtoWithShapeInfo(
            optimised).model_with_shape_info
        _, dynamic_values = usability_checker.check_shapes(checked.graph)
        eps = {
            "nnapi": usability_checker.check_nnapi_partitions(checked, True),
            "coreml_neuralnetwork": usability_checker.check_coreml_partitions(
                checked, True, "coreml_supported_neuralnetwork_ops.md"),
            "coreml_mlprogram": usability_checker.check_coreml_partitions(
                checked, True, "coreml_supported_mlprogram_ops.md"),
        }

    return {
        "schema": SCHEMA,
        "provenance": provenance(REPOSITORY, {"model": model_path},
                                 onnx=onnx.__version__,
                                 onnxruntime=onnxruntime.__version__),
        "input": {"name": inputs[0].name, "shape": shape},
        # Values whose shape is still symbolic after the input was fixed. ORT's
        # per-EP count of nodes "unsupported due to dynamic input" is not
        # reported: it treats an omitted optional input ('') as dynamic, so it
        # flags every LSTM however fixed the graph is.
        "dynamic_values": dynamic_values,
        "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "parameters": parameters,
        "parameter_bytes": dict(parameter_bytes),
        "ops": dict(sorted(ops.items())),
        "macs": macs,
        "macs_by_op": macs_by_op,
        "macs_uncounted_ops": _uncounted(model.graph),
        "execution_providers": {name: _partitions(info)
                                for name, info in eps.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--shape", required=True,
                        type=lambda s: [int(d) for d in s.split(",")],
                        help="fixed input shape, e.g. 1,1500,128")
    parser.add_argument("--output", type=pathlib.Path,
                        help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    text = json.dumps(preflight(args.model, args.shape), indent=1) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
