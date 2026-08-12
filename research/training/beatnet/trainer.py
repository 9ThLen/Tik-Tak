"""Deterministic A3 reset/stateful TBPTT primitives and epoch checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .data import Recording, contiguous_batches
from .model import BeatNetTrainable


ARMS = ("A3_reset", "A3_stateful")
CLASS_WEIGHTS = (50.0, 400.0, 5.0)


def set_deterministic(seed: int) -> None:
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "S1 requires a deterministic CUBLAS_WORKSPACE_CONFIG")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def detach_state(state):
    return tuple(value.detach() for value in state)


def _stack_states(model: BeatNetTrainable, batch, states, arm, device):
    columns = []
    for item in batch:
        if arm == "A3_reset" or item.reset or item.slot not in states:
            columns.append(model.zero_state(1, device=device))
        else:
            columns.append(states[item.slot])
    return tuple(torch.cat([column[index] for column in columns], dim=1)
                 for index in range(2))


@dataclass(frozen=True)
class EpochResult:
    loss: float
    supervised_frames: int
    blocks: int
    frames: int


def train_epoch(model: BeatNetTrainable, optimizer: torch.optim.Optimizer,
                recordings: list, *, arm: str, seed: int,
                batch_size: int, device: torch.device,
                gradient_clip: float = 5.0,
                loader: Callable[[object], Recording] | None = None) -> EpochResult:
    if arm not in ARMS:
        raise ValueError(f"unknown S1 arm: {arm}")
    model.train()
    weights = torch.tensor(CLASS_WEIGHTS, device=device)
    states = {}
    loss_sum = 0.0
    supervised = blocks = frames = 0
    for batch in contiguous_batches(
            recordings, batch_size=batch_size, seed=seed, loader=loader):
        features = torch.from_numpy(np.stack(
            [item.features for item in batch])).to(device)
        labels = torch.from_numpy(np.stack(
            [item.labels for item in batch])).to(device)
        mask = torch.from_numpy(np.stack(
            [item.mask for item in batch])).to(device)
        incoming = _stack_states(model, batch, states, arm, device)
        optimizer.zero_grad(set_to_none=True)
        logits, outgoing = model(features, incoming)
        selected = mask.reshape(-1)
        count = int(selected.sum().item())
        if count:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3)[selected], labels.reshape(-1)[selected],
                weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite S1 loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.all(
                        torch.isfinite(parameter.grad)):
                    raise RuntimeError("non-finite S1 gradient")
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                gradient_clip)
            optimizer.step()
        outgoing = detach_state(outgoing)
        for column, item in enumerate(batch):
            if item.end:
                states.pop(item.slot, None)
            else:
                states[item.slot] = tuple(
                    value[:, column:column + 1].contiguous()
                    for value in outgoing)
        if count:
            loss_sum += float(loss.detach()) * count
        supervised += count
        blocks += len(batch)
        frames += sum(len(item.features) for item in batch)
    if states:
        raise RuntimeError("state survived the end of an S1 epoch")
    if not supervised:
        raise RuntimeError("S1 epoch has no supervised frames")
    return EpochResult(loss_sum / supervised, supervised, blocks, frames)


@torch.no_grad()
def validation_loss(model: BeatNetTrainable, recordings: list, *, arm: str,
                    seed: int, batch_size: int, device: torch.device,
                    loader: Callable[[object], Recording] | None = None) -> EpochResult:
    """Masked neural loss with the same scheduling/state contract as training."""
    if arm not in ARMS:
        raise ValueError(f"unknown S1 arm: {arm}")
    model.eval()
    weights = torch.tensor(CLASS_WEIGHTS, device=device)
    states = {}
    loss_sum = 0.0
    supervised = blocks = frames = 0
    for batch in contiguous_batches(
            recordings, batch_size=batch_size, seed=seed, loader=loader):
        features = torch.from_numpy(np.stack(
            [item.features for item in batch])).to(device)
        labels = torch.from_numpy(np.stack(
            [item.labels for item in batch])).to(device)
        mask = torch.from_numpy(np.stack(
            [item.mask for item in batch])).to(device)
        incoming = _stack_states(model, batch, states, arm, device)
        logits, outgoing = model(features, incoming)
        selected = mask.reshape(-1)
        count = int(selected.sum().item())
        if count:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3)[selected], labels.reshape(-1)[selected],
                weight=weights)
        outgoing = detach_state(outgoing)
        for column, item in enumerate(batch):
            if item.end:
                states.pop(item.slot, None)
            else:
                states[item.slot] = tuple(
                    value[:, column:column + 1].contiguous()
                    for value in outgoing)
        if count:
            loss_sum += float(loss) * count
        supervised += count
        blocks += len(batch)
        frames += sum(len(item.features) for item in batch)
    if states:
        raise RuntimeError("state survived the end of S1 validation")
    if not supervised:
        raise RuntimeError("S1 validation has no supervised frames")
    return EpochResult(loss_sum / supervised, supervised, blocks, frames)


def checkpoint_identity(config: dict, *, source_sha256: str,
                        split_sha256: str, cache_sha256: str,
                        commit: str) -> dict:
    return {
        "schema": "tiktak.s1_checkpoint/v1", "commit": commit,
        "source_sha256": source_sha256, "split_sha256": split_sha256,
        "cache_sha256": cache_sha256, "config": config,
    }


def identity_sha256(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(path: pathlib.Path, model: BeatNetTrainable,
                    optimizer: torch.optim.Optimizer, *, epoch: int,
                    identity: dict, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tiktak.s1_checkpoint/v1", "epoch": epoch,
        "identity": identity, "identity_sha256": identity_sha256(identity),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "metadata": metadata or {},
    }
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary_path = pathlib.Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint_payload(path: pathlib.Path, model: BeatNetTrainable,
                            optimizer: torch.optim.Optimizer,
                            *, identity: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != "tiktak.s1_checkpoint/v1"
            or payload.get("identity") != identity
            or payload.get("identity_sha256") != identity_sha256(identity)):
        raise ValueError("S1 checkpoint identity mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return payload
