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
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from beat_this.model.beat_tracker import BeatThis

INPUT_NAME = "input_spectrogram"   # what eval/beat_this_onnx.py feeds
CHUNK_FRAMES = 1500

# Round-off through a different kernel set is order 1e-6. Anything approaching
# this is a divergence rather than arithmetic.
MAX_ACCEPTABLE_DIFF = 1e-4


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
        dropout=hyper["dropout"], sum_head=hyper["sum_head"],
        partial_transformers=hyper["partial_transformers"],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("target", type=pathlib.Path)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"{args.checkpoint} is not present")
    export(args.checkpoint, args.target)
    print(f"wrote {args.target} ({args.target.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
