"""The weight file's format, pinned from the writing side.

The core reads these bytes with no names and no offsets — a fixed header
followed by a fixed sequence of tensors — which is fast, simple, and completely
unforgiving. If the two sides ever disagree about the order, nothing crashes:
the network loads a convolution kernel into its output layer and produces
plausible-looking numbers that mean nothing. So the layout is asserted here as
well as in core/tests/test_beatnet.cpp, and the two sets of constants are
written out independently rather than derived from one another. A single shared
definition would agree with itself while both sides were wrong.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import struct

import numpy as np

import pytest

EXPORT_PY = pathlib.Path(__file__).resolve().parents[2] / "models" / "export_beatnet.py"

spec = importlib.util.spec_from_file_location("export_beatnet", EXPORT_PY)
export_beatnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_beatnet)


def test_the_header_says_what_the_core_checks():
    header = export_beatnet.header()
    assert len(header) == 32, "core/src/ml/beatnet.hpp sizes kHeaderBytes at 32"

    magic, version, features, channels, kernel, hidden, layers, classes = \
        struct.unpack("<4s7I", header)
    assert magic == b"TTBN"
    assert version == 1
    assert (features, channels, kernel, hidden, layers, classes) == (272, 2, 10, 150, 2, 3)


def test_the_file_is_the_size_the_core_expects():
    parameters = sum(
        int.__mul__(1, 1) * _product(shape) for _, shape in export_beatnet.TENSORS)
    assert parameters == 402325, "the published BeatNet has 402325 parameters"
    assert 32 + parameters * 4 == 1609332


def _product(shape):
    total = 1
    for value in shape:
        total *= value
    return total


def test_the_tensors_run_in_the_order_the_forward_pass_reads_them():
    # Written out rather than generated. The order *is* the format.
    assert [name for name, _ in export_beatnet.TENSORS] == [
        "conv1.weight",
        "conv1.bias",
        "linear0.weight",
        "linear0.bias",
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "lstm.bias_ih_l0",
        "lstm.bias_hh_l0",
        "lstm.weight_ih_l1",
        "lstm.weight_hh_l1",
        "lstm.bias_ih_l1",
        "lstm.bias_hh_l1",
        "linear.weight",
        "linear.bias",
    ]


def test_the_flattened_convolution_width_follows_from_the_kernel():
    # 272 features, a kernel of 10 and no padding leave 263 positions; pooling
    # in pairs drops the ragged last one and leaves 131 per channel. The linear
    # layer's 262 columns are that number twice, and if this arithmetic is ever
    # wrong the checkpoint simply will not fit — which is the point of asserting
    # it against the published shape rather than computing it in both places.
    width = next(shape for name, shape in export_beatnet.TENSORS
                 if name == "linear0.weight")[1]
    assert width == 262


def zeros():
    return {name: np.zeros(shape, dtype="float32")
            for name, shape in export_beatnet.TENSORS}


def test_a_checkpoint_with_the_wrong_shape_is_refused():
    state = zeros()
    assert len(export_beatnet.convert(state)) == 1609332

    state["linear0.weight"] = np.zeros((150, 84), dtype="float32")
    with pytest.raises(SystemExit, match="linear0.weight"):
        export_beatnet.convert(state)


def test_a_checkpoint_missing_a_tensor_is_refused():
    state = zeros()
    del state["lstm.bias_hh_l1"]
    with pytest.raises(SystemExit, match="lstm.bias_hh_l1"):
        export_beatnet.convert(state)


def test_the_reader_needs_no_torch_and_gets_the_same_bytes():
    """The published checkpoint, read without the framework that wrote it.

    Correctness is not argued here, it is checked against a hash the manifest
    already vouches for — produced, when it was pinned, by the torch-based
    reader this one replaces. Byte-identical or it is wrong.
    """
    checkpoint = EXPORT_PY.parent / "beatnet_model_1_weights.pt"
    if not checkpoint.is_file():
        pytest.skip("the published checkpoint is not in this checkout")

    payload = export_beatnet.convert(export_beatnet._read_state_dict(checkpoint))
    assert len(payload) == 1609332
    assert payload[:4] == b"TTBN"
    assert (hashlib.sha256(payload).hexdigest()
            == "812ed11af745885127cfb967e7db847c9bdef44b8e2c80c79cf875f790b978f1")


def test_the_reader_refuses_a_pickle_that_asks_for_anything_else(tmp_path):
    """A checkpoint is data, and this reader will not let it become code.

    The file is downloaded from the internet, so an unpickler that imports
    whatever it is told to is a remote-execution hole wearing a file format.
    """
    import pickle
    import zipfile

    path = tmp_path / "hostile.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(__import__("os").system))

    with pytest.raises(pickle.UnpicklingError, match="does not provide"):
        export_beatnet._read_state_dict(path)
