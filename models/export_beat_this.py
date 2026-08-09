"""Export an official Beat This! checkpoint to a single self-contained ONNX file.

The full-model ONNX this project has been measuring came pre-converted from a
third party, which was fine for answering "does this export at all" and not
fine for anything that has to be trusted. This builds the graph from the
official checkpoint and the official model code, and then checks the result
against PyTorch rather than assuming the exporter did what it said.

    python models/export_beat_this.py small0.ckpt models/small0.onnx

Needs torch, einops, rotary-embedding-torch, onnx, onnxscript and the
beat_this package (pip install -e from github.com/CPJKU/beat_this --no-deps).
None of that is a runtime dependency: the product ships the ONNX, and the
research harness reads it with onnxruntime alone.

Three details that are easy to get wrong and silent when wrong:

* Lightning stores every parameter under a "model." prefix, so loading with
  strict=True against the bare module fails, and loading carelessly with
  strict=False succeeds while initialising nothing.
* torch writes weights to a sidecar .onnx.data file by default, which makes
  the export look five times smaller than it is and breaks the moment the file
  is moved on its own.
* The two export paths do not agree. The legacy tracer (dynamo=False) produces
  a graph that differs from the module by about 6.0 on the raw logits, which
  is not round-off — it is a different function. Only the dynamo path matches.
  That is the whole reason verify() exists rather than a print statement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from beat_this.model.beat_tracker import BeatThis

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from research.eval.beat_this_onnx import (  # noqa: E402
    SAMPLE_RATE,
    log_mel_spectrogram,
)
from research.eval.provenance import digest, provenance  # noqa: E402

INPUT_NAME = "input_spectrogram"   # what eval/beat_this_onnx.py feeds
CHUNK_FRAMES = 1500

# Round-off through a different kernel set is order 1e-6. Anything approaching
# this is a divergence rather than arithmetic.
MAX_ACCEPTABLE_DIFF = 1e-4
PARITY_SCHEMA = "tiktak.beat_this_parity/v1"
PARITY_AUDIO_SECONDS = 8
PARITY_AUDIO_SEED = 20260809


class TwoHeads(torch.nn.Module):
    """The model's dict output as a tuple, which is what ONNX can express.

    Defined at module scope on purpose: the tracer follows a locally defined
    nn.Module poorly and fails with an unhelpful assertion about compiled
    functions.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, spectrogram):
        out = self.inner(spectrogram)
        return out["beat"], out["downbeat"]


def build(checkpoint: pathlib.Path):
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyper = loaded["hyper_parameters"]
    model = BeatThis(
        spect_dim=hyper["spect_dim"], transformer_dim=hyper["transformer_dim"],
        ff_mult=hyper["ff_mult"], n_layers=hyper["n_layers"],
        head_dim=hyper["head_dim"], stem_dim=hyper["stem_dim"],
        dropout=hyper["dropout"],
        # Published final checkpoints predate these keys in Lightning's saved
        # hyperparameters. Their pinned v1.1.0 constructor defaults are the
        # architecture used by those checkpoints; strict state loading below
        # still rejects any wrong interpretation.
        sum_head=hyper.get("sum_head", True),
        partial_transformers=hyper.get("partial_transformers", True),
    )
    prefix = "model."
    state = {k[len(prefix):]: v for k, v in loaded["state_dict"].items()
             if k.startswith(prefix)}
    # Strict, so a checkpoint whose shape does not match the hyperparameters it
    # carries fails here rather than exporting a partly random model.
    model.load_state_dict(state)
    model.eval()
    return model, hyper


def export(checkpoint: pathlib.Path, target: pathlib.Path) -> None:
    model, hyper = build(checkpoint)
    dummy = torch.zeros(1, CHUNK_FRAMES, hyper["spect_dim"])

    # Run it once before exporting. The rotary embedding fills a cache on its
    # first forward pass, and tracing the uncached path fails inside dynamo
    # with "expected compiled_fn to be GraphModule, got function" — an error
    # that names nothing to do with rotary embeddings or caches. Without this
    # line the export simply does not work, and the reason is not guessable
    # from the message.
    with torch.no_grad():
        model(dummy)

    staged = target.with_name(f".{target.name}.tmp")
    try:
        torch.onnx.export(
            TwoHeads(model), dummy, str(staged),
            input_names=[INPUT_NAME], output_names=["beat", "downbeat"],
            dynamic_axes={INPUT_NAME: {0: "batch", 1: "frames"},
                          "beat": {0: "batch", 1: "frames"},
                          "downbeat": {0: "batch", 1: "frames"}},
            opset_version=17,
        )
        # Re-saving with the weights inlined collapses whatever sidecar the
        # exporter wrote back into one file; loading first pulls it in.
        onnx.save_model(onnx.load(str(staged)), str(target),
                        save_as_external_data=False)
    finally:
        for leftover in staged.parent.glob(f"{staged.name}*"):
            leftover.unlink()

    verify(model, target, hyper)


def verify(model: torch.nn.Module, target: pathlib.Path, hyper) -> float:
    """Check the exported graph against the module it came from.

    An export that quietly differs would invalidate every measurement taken
    downstream of it, and nothing else in the pipeline would notice — the
    activations would simply be a little worse, and the model would look a
    little less good than it is.
    """
    sample = np.random.default_rng(0).normal(
        size=(1, CHUNK_FRAMES, hyper["spect_dim"])).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(sample))

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    beat, downbeat = session.run(["beat", "downbeat"], {INPUT_NAME: sample})

    worst = float(max(np.abs(expected["beat"].numpy() - beat).max(),
                      np.abs(expected["downbeat"].numpy() - downbeat).max()))
    if worst > MAX_ACCEPTABLE_DIFF:
        raise SystemExit(f"export disagrees with PyTorch by {worst:.2e} — "
                         f"the graph is not the model, do not measure with it")
    print(f"verified against PyTorch: max abs diff {worst:.2e}")
    return worst


def _fixed_parity_audio() -> np.ndarray:
    """A deterministic, non-musical signal with tonal and transient content."""
    frames = SAMPLE_RATE * PARITY_AUDIO_SECONDS
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(PARITY_AUDIO_SEED)
    audio = (
        0.20 * np.sin(2 * np.pi * 110.0 * time)
        + 0.12 * np.sin(2 * np.pi * 440.0 * time + 0.3)
        + 0.01 * rng.standard_normal(frames)
    )
    # Deterministic impulses exercise the broadband path without depending on
    # a copyrighted or locally mounted corpus file.
    audio[np.arange(SAMPLE_RATE // 2, frames, SAMPLE_RATE // 2)] += 0.6
    return audio.astype(np.float32)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _head_differences(expected: dict, actual: tuple[np.ndarray, np.ndarray]) -> dict:
    details = {}
    for name, observed in zip(("beat", "downbeat"), actual):
        reference = expected[name].detach().cpu().numpy()
        observed = np.asarray(observed)
        if reference.shape != observed.shape:
            raise SystemExit(
                f"{name} shape differs: source {reference.shape}, ONNX {observed.shape}")
        delta = np.abs(reference - observed)
        details[name] = {
            "frames": int(delta.size),
            "max_abs_diff": float(delta.max(initial=0.0)),
            "mean_abs_diff": float(delta.mean()) if delta.size else 0.0,
        }
    return details


def _manifest_contract(checkpoint: pathlib.Path, runtime: pathlib.Path) -> dict:
    manifest_path = REPOSITORY / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]
    source = manifest["beat_this_final0"]
    target = manifest["beat_this_cpp_onnx"]
    source_digest = digest(checkpoint)
    target_digest = digest(runtime)
    expected_source = source["pinned"]["sha256"]
    expected_target = target["pinned"]["sha256"]
    conversion_source = target["conversion"]["source_weights_sha256"]
    if source_digest is None or source_digest["sha256"] != expected_source:
        raise SystemExit("source checkpoint does not match pinned beat_this_final0")
    if target_digest is None or target_digest["sha256"] != expected_target:
        raise SystemExit("runtime ONNX does not match pinned beat_this_cpp_onnx")
    if conversion_source != expected_source:
        raise SystemExit("manifest conversion does not point to pinned beat_this_final0")
    return {
        "manifest": digest(manifest_path),
        "source_artifact": "beat_this_final0",
        "runtime_artifact": "beat_this_cpp_onnx",
        "conversion": target["conversion"],
    }


def verify_runtime(checkpoint: pathlib.Path, runtime: pathlib.Path,
                   report_path: pathlib.Path, allow_dirty_diagnostic: bool) -> dict:
    """Verify the pinned runtime ONNX against the pinned official checkpoint."""
    contract = _manifest_contract(checkpoint, runtime)
    run = provenance(
        REPOSITORY,
        files={"source_checkpoint": checkpoint, "runtime_onnx": runtime},
    )
    if run["tree_clean"] is not True and not allow_dirty_diagnostic:
        raise SystemExit(
            "parity refused: repository provenance is not clean; commit first or "
            "use --allow-dirty-diagnostic for a non-acceptance diagnostic")

    try:
        report_path.resolve().relative_to(REPOSITORY.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("parity report must be written outside the repository")

    model, hyper = build(checkpoint)
    audio = _fixed_parity_audio()
    sample = log_mel_spectrogram(audio)[None, ...]
    with torch.no_grad():
        expected = model(torch.from_numpy(sample))
    session = ort.InferenceSession(str(runtime), providers=["CPUExecutionProvider"])
    actual = session.run(["beat", "downbeat"], {INPUT_NAME: sample})
    heads = _head_differences(expected, (actual[0], actual[1]))
    worst = max(head["max_abs_diff"] for head in heads.values())
    passed = worst <= MAX_ACCEPTABLE_DIFF

    report = {
        "schema": PARITY_SCHEMA,
        "accepted": bool(passed and run["tree_clean"] is True),
        "diagnostic_only": run["tree_clean"] is not True,
        "provenance": run,
        "contract": contract,
        "input": {
            "kind": "deterministic synthetic audio through project log-mel frontend",
            "sample_rate": SAMPLE_RATE,
            "seconds": PARITY_AUDIO_SECONDS,
            "seed": PARITY_AUDIO_SEED,
            "audio_sha256": _sha256_bytes(audio.tobytes()),
            "spectrogram_sha256": _sha256_bytes(sample.tobytes()),
            "spectrogram_shape": list(sample.shape),
        },
        "tolerance": {"metric": "max_abs_diff", "maximum": MAX_ACCEPTABLE_DIFF},
        "heads": heads,
        "max_abs_diff": worst,
        "passed": passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staged = report_path.with_name(f".{report_path.name}.tmp")
    staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    staged.replace(report_path)
    if not passed:
        raise SystemExit(
            f"runtime ONNX disagrees with the official checkpoint by {worst:.2e}; "
            f"tolerance is {MAX_ACCEPTABLE_DIFF:.1e}")
    print(f"runtime parity passed: max abs diff {worst:.2e}")
    print(f"wrote {report_path} ({'accepted' if report['accepted'] else 'diagnostic only'})")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("target", type=pathlib.Path)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="compare pinned final0 to pinned runtime ONNX instead of exporting",
    )
    parser.add_argument(
        "--report", type=pathlib.Path,
        help="outside-repository JSON report required by --verify-only",
    )
    parser.add_argument(
        "--allow-dirty-diagnostic", action="store_true",
        help="run on a dirty tree but mark the report diagnostic-only",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"{args.checkpoint} is not present")
    if args.verify_only:
        if not args.target.is_file():
            raise SystemExit(f"{args.target} is not present")
        if args.report is None:
            raise SystemExit("--verify-only requires --report outside the repository")
        verify_runtime(
            args.checkpoint, args.target, args.report,
            allow_dirty_diagnostic=args.allow_dirty_diagnostic,
        )
        return 0
    if args.report is not None or args.allow_dirty_diagnostic:
        raise SystemExit("--report and --allow-dirty-diagnostic require --verify-only")
    export(args.checkpoint, args.target)
    print(f"wrote {args.target} ({args.target.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
