"""The mobile preflight's counts, held to a network whose answer is known.

BeatNet's arithmetic is written out in core/src/ml/beatnet.hpp and pinned by
test_export_beatnet.py: 402,325 parameters and, per frame, a 10-tap
convolution into 2 x 263, a 262 -> 150 linear, two LSTM(150) layers and a
150 -> 3 output -- 405,010 multiply-accumulates. A module of the same shapes,
exported to ONNX, must come back with exactly those numbers, or the counter is
wrong in a way no real model would reveal.

Skipped without torch, onnx and onnxruntime, none of which the core or the
research harness needs.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

PREFLIGHT_PY = pathlib.Path(__file__).resolve().parents[2] / "models" / "mobile_preflight.py"

spec = importlib.util.spec_from_file_location("mobile_preflight", PREFLIGHT_PY)
mobile_preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mobile_preflight)

FRAMES = 50


class BeatNetShaped(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 2, 10)
        self.pool = torch.nn.MaxPool1d(2)
        self.linear0 = torch.nn.Linear(262, 150)
        self.lstm = torch.nn.LSTM(150, 150, num_layers=2)
        self.out = torch.nn.Linear(150, 3)

    def forward(self, x):  # (frames, 272)
        h = self.pool(self.conv(x.unsqueeze(1))).flatten(1)
        h, _ = self.lstm(self.linear0(h).unsqueeze(1))
        return torch.softmax(self.out(h.squeeze(1)), dim=-1)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    path = tmp_path_factory.mktemp("preflight") / "beatnet_shaped.onnx"
    torch.onnx.export(BeatNetShaped().eval(), torch.zeros(FRAMES, 272),
                      str(path), input_names=["features"],
                      dynamic_axes={"features": {0: "frames"}},
                      opset_version=17, dynamo=False)
    return mobile_preflight.preflight(path, [FRAMES, 272])


def test_parameters_match_the_published_beatnet(report):
    assert report["parameters"] == 402325


def test_macs_per_frame_match_the_core_arithmetic(report):
    assert report["macs"] == 405010 * FRAMES
    assert report["macs_uncounted_ops"] == []


def test_every_execution_provider_is_reported(report):
    providers = report["execution_providers"]
    assert set(providers) == {"nnapi", "coreml_neuralnetwork",
                              "coreml_mlprogram"}
    for info in providers.values():
        assert 0 <= info["supported_nodes"] <= info["nodes"]
        assert info["suitability"] in {"YES", "MAYBE", "NO"}


def test_the_input_shape_is_fixed_before_counting(report):
    assert report["input"] == {"name": "features", "shape": [FRAMES, 272]}
    assert report["dynamic_values"] == 0
