"""Exact trainable form of the 0.40 M-parameter BeatNet CRNN."""

from __future__ import annotations

import hashlib
import pathlib

import torch
from torch import nn


FEATURES = 272
CONV_CHANNELS = 2
KERNEL = 10
HIDDEN = 150
LAYERS = 2
CLASSES = 3
SOURCE_SHA256 = (
    "619091bc317ca3e83b45591d46f6de3d5a41588bcb39fe9fe7be30cffa6aca84")


def file_sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class BeatNetTrainable(nn.Module):
    """Published BeatNet topology with explicit recurrent state input/output."""

    def __init__(self) -> None:
        super().__init__()
        pooled = (FEATURES - KERNEL + 1) // 2
        self.conv1 = nn.Conv1d(1, CONV_CHANNELS, KERNEL)
        self.linear0 = nn.Linear(CONV_CHANNELS * pooled, HIDDEN)
        self.lstm = nn.LSTM(
            HIDDEN, HIDDEN, num_layers=LAYERS, batch_first=True,
            bidirectional=False)
        self.linear = nn.Linear(HIDDEN, CLASSES)

    @classmethod
    def from_checkpoint(cls, path: pathlib.Path,
                        *, verify_source: bool = True) -> "BeatNetTrainable":
        if verify_source and file_sha256(path) != SOURCE_SHA256:
            raise ValueError("BeatNet source checkpoint digest changed")
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    def zero_state(self, batch: int, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        shape = (LAYERS, batch, HIDDEN)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor, state=None):
        if features.ndim != 3 or features.shape[-1] != FEATURES:
            raise ValueError(f"expected (batch, frames, {FEATURES}) features")
        batch, frames, _ = features.shape
        x = features.reshape(batch * frames, 1, FEATURES)
        x = torch.nn.functional.max_pool1d(torch.relu(self.conv1(x)), 2)
        x = self.linear0(x.reshape(batch * frames, -1))
        x = x.reshape(batch, frames, HIDDEN)
        output, state = self.lstm(x, state)
        return self.linear(output), state

    def probabilities(self, features: torch.Tensor, state=None):
        logits, state = self(features, state)
        return torch.softmax(logits, dim=-1), state


def configure_a3(model: BeatNetTrainable) -> list[str]:
    """Freeze everything except LSTM layer 1 and the existing output head."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith("linear.") or (
                name.startswith("lstm.") and name.endswith("_l1")):
            parameter.requires_grad_(True)
            trainable.append(name)
    expected = {
        "lstm.weight_ih_l1", "lstm.weight_hh_l1",
        "lstm.bias_ih_l1", "lstm.bias_hh_l1",
        "linear.weight", "linear.bias",
    }
    if set(trainable) != expected:
        raise RuntimeError(f"A3 trainable parameter contract changed: {trainable}")
    return trainable
