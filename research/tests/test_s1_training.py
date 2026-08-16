import json
import pathlib
import subprocess
import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.beatnet_onnx import BeatNet
from eval.beatnet_onnx import log_filtered_spectrogram
from eval.octave_veto_replay import run
from training.beatnet.data import (
    FEATURES, Recording, contiguous_batches, frame_labels)
from training.beatnet.export import export_ttbn, save_state_dict
from training.beatnet.model import BeatNetTrainable, configure_a3
from training.beatnet.run import _eligible_key, _validate_config
from training.beatnet.evaluate import validate_product_binary
from training.beatnet.summarise import summarise
from training.beatnet.cache import _atomic_json
from training.beatnet.data import file_sha256
from training.beatnet.trainer import (
    checkpoint_identity, load_checkpoint_payload, save_checkpoint,
    set_deterministic, train_epoch)


def _main_models() -> pathlib.Path:
    repository = pathlib.Path(__file__).resolve().parents[2]
    common = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--path-format=absolute",
         "--git-common-dir"], capture_output=True, text=True, check=True)
    return pathlib.Path(common.stdout.strip()).parent / "models"


def _record(identity: str, frames: int = 800, *, value: float = 0.0,
            target: int = 1) -> Recording:
    features = np.full((frames, FEATURES), value, dtype=np.float32)
    labels = np.full(frames, target, dtype=np.int64)
    return Recording(identity, identity, features, labels,
                     np.ones(frames, dtype=bool))


def test_source_forward_and_export_parity(tmp_path):
    models = _main_models()
    checkpoint = models / "beatnet_model_1_weights.pt"
    frozen = models / "beatnet_model_1.ttw"
    if not checkpoint.is_file() or not frozen.is_file():
        pytest.skip("pinned BeatNet model cache is unavailable")
    rng = np.random.default_rng(20260812)
    features = rng.normal(size=(2, 811, FEATURES))
    candidate = BeatNetTrainable.from_checkpoint(checkpoint).double().eval()
    reference = BeatNet(checkpoint)
    expected = []
    with torch.no_grad():
        actual = candidate.probabilities(torch.from_numpy(features))[0].numpy()
        for sequence in features:
            x = torch.from_numpy(sequence).unsqueeze(1)
            x = torch.nn.functional.conv1d(
                x, reference.state["conv1.weight"],
                reference.state["conv1.bias"])
            x = torch.nn.functional.max_pool1d(torch.relu(x), 2)
            x = torch.nn.functional.linear(
                x.reshape(len(sequence), -1),
                reference.state["linear0.weight"],
                reference.state["linear0.bias"])
            output, _ = reference._lstm(x.unsqueeze(0))
            logits = torch.nn.functional.linear(
                output.squeeze(0), reference.state["linear.weight"],
                reference.state["linear.bias"])
            expected.append(torch.softmax(logits, dim=1).numpy())
    assert np.max(np.abs(actual - np.stack(expected))) <= 2e-6

    saved = tmp_path / "source.pt"
    exported = tmp_path / "source.ttw"
    save_state_dict(saved, candidate.float())
    export_ttbn(saved, exported,
                repository=pathlib.Path(__file__).resolve().parents[2])
    assert exported.read_bytes() == frozen.read_bytes()


def test_a3_trainable_set_is_exact():
    model = BeatNetTrainable()
    names = configure_a3(model)
    assert set(names) == {name for name, parameter in model.named_parameters()
                         if parameter.requires_grad}
    assert all(name.startswith(("lstm.", "linear.")) for name in names)
    assert not any(name.endswith("_l0") for name in names)


def test_frame_labels_use_earlier_tie_and_exclude_unsupported_segments():
    annotation = {
        "times": np.asarray([0.01, 0.03, 0.05, 1.0, 1.5]),
        "positions": np.asarray([1, 2, 3, 1, 2]),
        "segments": np.asarray([0, 0, 0, 1, 1]),
        "supported": np.asarray([True, True, True, False, False]),
    }
    labels, mask = frame_labels(100, annotation)
    # 0.01 s is exactly between frames 0 and 1 and resolves earlier.
    assert labels[0] == 1
    assert labels[1] == 0
    assert labels[2] == 0
    assert not np.any(mask[50:76])


def test_scheduler_pairs_order_masks_and_never_leaks_slots():
    recordings = [_record("a", 850), _record("b", 410), _record("c", 900)]
    left = list(contiguous_batches(recordings, batch_size=2, seed=17))
    right = list(contiguous_batches(recordings, batch_size=2, seed=17))
    signature = lambda batches: [[
        (item.slot, item.recording, item.index, item.reset, item.end,
         item.mask.tobytes()) for item in batch] for batch in batches]
    assert signature(left) == signature(right)
    active = {}
    for batch in left:
        for item in batch:
            if item.reset:
                assert item.slot not in active
                active[item.slot] = item.recording
            assert active[item.slot] == item.recording
            assert not np.any(item.mask[:100])
            if item.end:
                del active[item.slot]
    assert not active


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_checkpoint_resume_is_tensor_identical(tmp_path, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    config = {"test": "resume"}
    identity = checkpoint_identity(
        config, source_sha256="a", split_sha256="b", cache_sha256="c",
        commit="d")
    recordings = [_record("one", value=0.1), _record("two", value=-0.1)]

    def build():
        set_deterministic(29)
        model = BeatNetTrainable()
        configure_a3(model)
        model.to(device)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=5e-4)
        return model, optimizer

    uninterrupted, optimizer_a = build()
    train_epoch(uninterrupted, optimizer_a, recordings, arm="A3_stateful",
                seed=29, batch_size=2, device=device)
    train_epoch(uninterrupted, optimizer_a, recordings, arm="A3_stateful",
                seed=30, batch_size=2, device=device)

    resumed, optimizer_b = build()
    train_epoch(resumed, optimizer_b, recordings, arm="A3_stateful",
                seed=29, batch_size=2, device=device)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, resumed, optimizer_b, epoch=0,
                    identity=identity, metadata={"marker": 1})
    loaded, optimizer_c = build()
    payload = load_checkpoint_payload(
        checkpoint, loaded, optimizer_c, identity=identity)
    assert payload["metadata"] == {"marker": 1}
    train_epoch(loaded, optimizer_c, recordings, arm="A3_stateful",
                seed=30, batch_size=2, device=device)
    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, loaded.state_dict()[name]), name


def test_product_binary_is_fixed_before_corpus_work(tmp_path):
    wrong = tmp_path / "dump_analysis.exe"
    wrong.write_bytes(b"not the registered binary")
    with pytest.raises(ValueError, match="M0e product binary"):
        validate_product_binary(wrong)


def test_both_arms_overfit_tiny_set_and_state_changes_second_block():
    models = _main_models()
    checkpoint = models / "beatnet_model_1_weights.pt"
    if not checkpoint.is_file():
        pytest.skip("pinned BeatNet model cache is unavailable")
    rng = np.random.default_rng(7)
    recording = Recording(
        "tiny", "tiny", rng.normal(size=(800, FEATURES)).astype(np.float32),
        np.ones(800, dtype=np.int64), np.ones(800, dtype=bool))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arm in ("A3_reset", "A3_stateful"):
        set_deterministic(17)
        model = BeatNetTrainable.from_checkpoint(checkpoint)
        configure_a3(model)
        model.to(device)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=5e-4)
        first = train_epoch(
            model, optimizer, [recording], arm=arm, seed=17,
            batch_size=1, device=device).loss
        second = train_epoch(
            model, optimizer, [recording], arm=arm, seed=18,
            batch_size=1, device=device).loss
        assert second <= 0.5 * first

    model = BeatNetTrainable.from_checkpoint(checkpoint).to(device).eval()
    frames = torch.from_numpy(recording.features).unsqueeze(0).to(device)
    with torch.no_grad():
        _, state = model(frames[:, :400])
        carried, _ = model(frames[:, 400:], state)
        reset, _ = model(frames[:, 400:], model.zero_state(1, device=device))
    assert not torch.equal(carried, reset)
    assert all(not value.requires_grad for value in state)


def test_trained_ttbn_cpp_probability_parity(tmp_path):
    models = _main_models()
    checkpoint = models / "beatnet_model_1_weights.pt"
    binary = (_main_models().parent / "tools" / "eval" / "build"
              / "RelWithDebInfo" / "dump_analysis.exe")
    if not checkpoint.is_file() or not binary.is_file():
        pytest.skip("pinned model or C++ evaluation binary is unavailable")
    set_deterministic(43)
    model = BeatNetTrainable.from_checkpoint(checkpoint)
    configure_a3(model)
    # A deterministic parameter change makes this a trained/checkpoint path,
    # rather than merely repeating frozen-source parity.
    with torch.no_grad():
        model.linear.bias.add_(torch.tensor([0.01, -0.02, 0.03]))
    state = tmp_path / "trained.pt"
    ttbn = tmp_path / "trained.ttw"
    save_state_dict(state, model)
    export_ttbn(state, ttbn,
                repository=pathlib.Path(__file__).resolve().parents[2])

    rate = 22050
    seconds = 4
    sample = np.arange(rate * seconds, dtype=np.float64) / rate
    audio = (0.2 * np.sin(2 * np.pi * 220 * sample)
             + 0.05 * np.sin(2 * np.pi * 733 * sample))
    wav = tmp_path / "fixture.wav"
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    decoded = pcm.astype(np.float64) / 32768.0
    features = log_filtered_spectrogram(decoded).astype(np.float32)
    with torch.no_grad():
        probabilities = model.probabilities(
            torch.from_numpy(features).unsqueeze(0))[0][0].numpy()
    payload = run(binary, wav, ttbn, extra=["--live-bars"])
    beat = np.asarray(payload["activation_beat"])
    downbeat = np.asarray(payload["activation_downbeat"])
    count = min(len(probabilities), len(beat))
    assert count > 100
    assert np.max(np.abs(
        probabilities[:count, 0] + probabilities[:count, 1] - beat[:count])) <= 2e-5
    assert np.max(np.abs(probabilities[:count, 1] - downbeat[:count])) <= 2e-5


def test_registered_config_and_checkpoint_selection():
    config_path = (pathlib.Path(__file__).resolve().parents[1] / "training"
                   / "beatnet" / "s1_a3.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    baseline = {"means": {"beat_f1": 0.80}}
    assert _eligible_key({"means": {
        "beat_f1": 0.79, "phase_f1": 0.5, "downbeat_f1": 0.4}},
        baseline) == (0.5, 0.4, 0.79)
    assert _eligible_key({"means": {
        "beat_f1": 0.789, "phase_f1": 0.9, "downbeat_f1": 0.9}},
        baseline) is None


def test_summary_recomputes_registered_work_pairing(tmp_path):
    paths = []
    metrics = (
        "phase_f1", "beat_f1", "downbeat_f1", "stable_exact_position",
        "false_switches_per_5min", "long_wrong_episodes_per_5min",
        "beat_precision", "beat_recall", "downbeat_precision",
        "downbeat_recall", "usable_strict", "position_accuracy",
        "grouping_balanced_accuracy", "coverage", "false_confident_share",
        "unnecessary_unknown_share", "wrong_episodes_per_5min",
        "resolver_state_changes_per_5min", "held_state_changes_per_5min",
        "acquisition_latency_sec")
    work_corpora = {f"work-{index}": "rwc2" for index in range(84)}
    baseline_path = tmp_path / "baseline.json"
    _atomic_json(baseline_path, {
        "schema": "tiktak.s1_evaluation/v1", "dev_works": 84,
        "provenance": {"tree_clean": True},
        "work_metrics": {
            f"work-{index}": {metric: 0.5 for metric in metrics}
            for index in range(84)
        },
        "work_corpora": work_corpora,
    })
    baseline_sha256 = file_sha256(baseline_path)
    for arm in ("A3_reset", "A3_stateful"):
        for seed in (17, 29, 43):
            root = tmp_path / f"{arm}-{seed}"
            epochs = (5, 10) if arm == "A3_stateful" else (5,)
            history = []
            for epoch in epochs:
                evaluation_path = (root / "candidates"
                                   / f"epoch-{epoch:03d}" / "evaluation.json")
                work_metrics = {}
                for index in range(84):
                    row = {metric: 0.5 for metric in metrics}
                    if arm == "A3_stateful":
                        row["phase_f1"] += -0.01 if epoch == 5 else 0.04
                    work_metrics[f"work-{index}"] = row
                evaluation = {
                    "schema": "tiktak.s1_evaluation/v1", "arm": arm,
                    "seed": seed, "dev_works": 84,
                    "work_metrics": work_metrics,
                    "work_corpora": work_corpora,
                }
                _atomic_json(evaluation_path, evaluation)
                history.append({
                    "epoch": epoch, "eligible": True,
                    "evaluation_sha256": file_sha256(evaluation_path),
                })
            selected_epoch = epochs[-1]
            result = {
                "schema": "tiktak.s1_training/v1", "complete": True,
                "provenance": {"tree_clean": True}, "arm": arm, "seed": seed,
                "identity": {"same": True, "arm": arm, "seed": seed,
                             "baseline_sha256": baseline_sha256},
                "best": {
                    "epoch": selected_epoch,
                    "evaluation": (
                        f"candidates/epoch-{selected_epoch:03d}/evaluation.json"),
                },
                "history": history,
            }
            result_path = root / "result.json"
            _atomic_json(result_path, result)
            paths.append(result_path)
    summary = summarise(paths, baseline_path)
    assert summary["paired_effects"]["phase_f1"]["mean"] == pytest.approx(0.04)
    assert summary["selection_diagnostics"]["A3_reset"]["17"] == {
        "validation_points": 1, "eligible_points": 1, "selected_epoch": 5,
    }
    assert summary["selection_diagnostics"]["A3_stateful"]["17"] == {
        "validation_points": 2, "eligible_points": 2, "selected_epoch": 10,
    }
    assert summary["common_epoch_endpoint"]["paired_effects"][
        "phase_f1"]["mean"] == pytest.approx(-0.01)
    assert summary["common_epoch_endpoint"]["non_gating"] is True
    assert summary["a0_diagnostics"]["paired_effects"]["A3_reset"][
        "phase_f1"]["mean"] == pytest.approx(0.0)
    assert summary["a0_diagnostics"]["paired_effects"]["A3_stateful"][
        "phase_f1"]["mean"] == pytest.approx(0.04)
    assert summary["interpretation"] == "stateful_training_positive"


def test_summary_reports_inconclusive_when_an_arm_has_no_eligible_checkpoint(
        tmp_path):
    paths = []
    metrics = (
        "phase_f1", "beat_f1", "downbeat_f1", "stable_exact_position",
        "false_switches_per_5min", "long_wrong_episodes_per_5min",
        "beat_precision", "beat_recall", "downbeat_precision",
        "downbeat_recall", "usable_strict", "position_accuracy",
        "grouping_balanced_accuracy", "coverage", "false_confident_share",
        "unnecessary_unknown_share", "wrong_episodes_per_5min",
        "resolver_state_changes_per_5min", "held_state_changes_per_5min")
    work_corpora = {f"work-{index}": "rwc2" for index in range(84)}
    work_metrics = {
        f"work-{index}": {metric: 0.5 for metric in metrics}
        for index in range(84)
    }
    baseline_path = tmp_path / "baseline.json"
    _atomic_json(baseline_path, {
        "schema": "tiktak.s1_evaluation/v1", "dev_works": 84,
        "provenance": {"tree_clean": True},
        "work_metrics": work_metrics, "work_corpora": work_corpora,
    })
    baseline_sha256 = file_sha256(baseline_path)
    for arm in ("A3_reset", "A3_stateful"):
        for seed in (17, 29, 43):
            root = tmp_path / f"missing-{arm}-{seed}"
            evaluation_path = (
                root / "candidates" / "epoch-005" / "evaluation.json")
            _atomic_json(evaluation_path, {
                "schema": "tiktak.s1_evaluation/v1", "arm": arm,
                "seed": seed, "dev_works": 84,
                "work_metrics": work_metrics, "work_corpora": work_corpora,
            })
            result = {
                "schema": "tiktak.s1_training/v1", "complete": True,
                "provenance": {"tree_clean": True}, "arm": arm, "seed": seed,
                "identity": {"same": True, "arm": arm, "seed": seed,
                             "baseline_sha256": baseline_sha256},
                "best": None, "history": [{
                    "epoch": 5, "eligible": False,
                    "evaluation_sha256": file_sha256(evaluation_path),
                }],
            }
            path = root / "result.json"
            _atomic_json(path, result)
            paths.append(path)
    summary = summarise(paths, baseline_path)
    assert summary["complete"] is False
    assert summary["interpretation"] == "inconclusive"
    assert summary["reason"] == (
        "one or more arms had no beat-noninferior checkpoint")
    assert len(summary["ineligible_runs"]) == 6
