"""BeatNet's causal beat/downbeat activations, without madmom.

BeatNet is the online counterpart to Beat This!: a two-layer unidirectional
LSTM over a log-filtered spectrogram, 0.40 M parameters in 1.6 MB, published by
Mojtaba Heydari, Frank Cwitkowitz and Zhiyao Duan under CC BY 4.0. See NOTICE.md
— attribution is a condition of the licence, and so is saying what was changed:
nothing here is retrained, this is a reimplementation of the published feature
pipeline plus the published weights.

Why it exists. The live tracker's confidence was taken apart layer by layer and
what remains is the observation model: spectral flux does not concentrate on
the beat in a produced mix, so a causal tracker fed by it stays honestly
silent. Everything measured points at a learned causal activation feeding the
particle filter that already exists. This is that activation, in a form the
research harness can score before anyone commits to porting it.

The preprocessing is madmom's, transcribed rather than imported: madmom does
not build against current numpy here, and its models carry a commercial licence
this project has already decided to stay clear of. Transcribing it means the
constants below are load-bearing, which is what test_beatnet_onnx.py is for.

  22050 Hz mono, 64 ms frames every 20 ms — 1411 samples every 441, which is
  exactly 50 frames a second — with a Hann window; a logarithmic filterbank of
  24 bands per octave from 30 Hz to 17 kHz, triangular, each normalised to unit
  area, on the FFT bins of the frame; log10(1 + magnitude); and the positive
  difference to the previous frame stacked alongside, giving 272 features.

  Those numbers come from BeatNet.py rather than from the defaults on the
  feature class it calls, which are different in every one of frame size, hop
  and bands. Taking the defaults gives 84 filters where the network's first
  layer expects 136, so the mistake is caught by the weights refusing to fit —
  but only if the shape is checked, which is why the test does.

The network reads one frame at a time and carries its LSTM state forward, so
its output at frame k depends on nothing after k. That is the property the
microphone path needs and the reason to prefer it over Beat This! online,
whatever the offline numbers say.
"""

from __future__ import annotations

import pathlib

import numpy as np

__all__ = [
    "SAMPLE_RATE", "FRAME_SIZE", "HOP", "FPS", "FEATURES",
    "WEIGHTS_PATH", "log_filtered_spectrogram", "filterbank", "BeatNet",
]

SAMPLE_RATE = 22050
FRAME_SIZE = 1411                # 64 ms at 22050 Hz
HOP = 441                        # 20 ms, so exactly 50 frames a second
FPS = SAMPLE_RATE / HOP
BANDS_PER_OCTAVE = 24
F_MIN = 30.0
F_MAX = 17000.0
F_REF = 440.0                    # madmom's A4; at 12 bands an octave this would be MIDI spacing
FEATURES = 272                   # 136 filters, then the same again as differences

WEIGHTS_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / "models" / "beatnet_model_1_weights.pt")


def _log_frequencies() -> np.ndarray:
    """Filter centres on a logarithmic scale, as madmom places them."""
    left = np.floor(np.log2(F_MIN / F_REF) * BANDS_PER_OCTAVE)
    right = np.ceil(np.log2(F_MAX / F_REF) * BANDS_PER_OCTAVE)
    frequencies = F_REF * 2.0 ** (np.arange(left, right) / BANDS_PER_OCTAVE)
    # floor and ceil overshoot on both sides, so trim back to the range asked for.
    frequencies = frequencies[np.searchsorted(frequencies, F_MIN):]
    return frequencies[:np.searchsorted(frequencies, F_MAX, "right")]


def _frequencies_to_bins(frequencies: np.ndarray, bin_frequencies: np.ndarray) -> np.ndarray:
    """Nearest FFT bin per frequency, duplicates removed.

    Removing duplicates is not tidiness: at the bottom of the range several
    filter centres fall in one bin, and keeping them would stack several
    filters on the same handful of bins and give the low end many times the
    weight the design intends.
    """
    indices = np.searchsorted(bin_frequencies, frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices = indices - (frequencies - left < right - frequencies)
    return np.unique(indices)


def filterbank(bins: int = FRAME_SIZE // 2 + 1,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """(bins, filters) triangles, each normalised to unit area.

    Unit *area*, not unit peak — this is the one place BeatNet's front end
    differs in spirit from Beat This!'s, and getting it wrong scales every
    band by its own width.
    """
    bin_frequencies = np.fft.rfftfreq(FRAME_SIZE, 1.0 / sample_rate)
    centres = _frequencies_to_bins(_log_frequencies(), bin_frequencies)

    bank = np.zeros((bins, max(0, len(centres) - 2)))
    for i in range(len(centres) - 2):
        start, centre, stop = (int(centres[i]), int(centres[i + 1]), int(centres[i + 2]))
        if stop - start < 2:
            centre, stop = start, start + 1
        triangle = np.zeros(stop - start)
        rise = centre - start
        triangle[:rise] = np.linspace(0, 1, rise, endpoint=False)
        triangle[rise:] = np.linspace(1, 0, stop - centre, endpoint=False)
        total = triangle.sum()
        if total > 0:
            triangle = triangle / total
        bank[start:stop, i] = triangle
    return bank


def _frames(audio: np.ndarray) -> np.ndarray:
    """Frames centred on their reference sample, zero-padded at the edges.

    madmom's reference sample for frame 0 is sample 0 and the window is centred
    on it, so the first frame is half padding. Aligning to the frame's start
    instead would put every activation half a window — 32 ms — early, which at
    120 BPM is a sixteenth of a beat and would look like a tracker that rushes.
    """
    half = FRAME_SIZE // 2
    padded = np.concatenate([np.zeros(half), audio, np.zeros(FRAME_SIZE)])
    count = int(np.ceil(len(audio) / HOP))
    out = np.empty((count, FRAME_SIZE))
    for i in range(count):
        out[i] = padded[i * HOP:i * HOP + FRAME_SIZE]
    return out


def log_filtered_spectrogram(audio: np.ndarray) -> np.ndarray:
    """(frames, 272): the log filtered spectrogram and its positive difference."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < HOP:
        return np.zeros((0, FEATURES))

    window = np.hanning(FRAME_SIZE)
    spectrum = np.abs(np.fft.rfft(_frames(audio) * window, axis=1))
    spec = np.log10(spectrum @ filterbank() + 1.0)

    # One frame back. madmom derives this from the window rather than fixing
    # it: the difference is taken to the frame half a window's half-power width
    # away, which for a Hann window of 1411 with a 441 hop rounds to exactly
    # one frame. A different frame or hop would not, so this is transcribed as
    # a consequence and pinned by a test rather than left as a magic 1.
    diff = np.zeros_like(spec)
    diff[1:] = np.maximum(spec[1:] - spec[:-1], 0.0)
    return np.hstack([spec, diff])


def resample_to_model_rate(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    if int(sample_rate) == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float64)
    duration = len(audio) / float(sample_rate)
    target = int(round(duration * SAMPLE_RATE))
    source_times = np.arange(len(audio)) / float(sample_rate)
    return np.interp(np.arange(target) / SAMPLE_RATE, source_times,
                     np.asarray(audio, dtype=np.float64))


class BeatNet:
    """The published network, run frame by frame with its state carried forward.

    Torch is a research-time dependency only; the product would carry the ONNX
    export, as Beat This! does. This class exists to answer whether the model
    is worth exporting at all.
    """

    def __init__(self, weights: pathlib.Path | None = None) -> None:
        import torch

        path = weights or WEIGHTS_PATH
        if not path.is_file():
            raise SystemExit(f"{path} is not present — run models/fetch.py pin")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._torch = torch
        self.state = {k: v.double() for k, v in state.items()}

    def activations(self, audio: np.ndarray, sample_rate: float) -> np.ndarray:
        """(frames, 3): probabilities of beat, downbeat and neither, in that order.

        The order is read off BeatNet.py, which keeps `preds[:2]` and discards
        the third — so the two it keeps are the beat and the downbeat, and the
        one it drops is the null class. Assuming the intuitive order instead
        puts the null class where the downbeat should be, and since the null
        class is high almost all the time the mistake looks like a model that
        hears a downbeat everywhere rather than like an index error.

        The published model applies a softmax across the three, so they sum to
        one per frame: a beat is a beat *instead of* nothing happening, not in
        addition to it.
        """
        torch = self._torch
        features = log_filtered_spectrogram(
            resample_to_model_rate(audio, sample_rate))
        if len(features) == 0:
            return np.zeros((0, 3))

        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(1)           # (frames, 1, 272)
            x = torch.nn.functional.conv1d(x, self.state["conv1.weight"],
                                           self.state["conv1.bias"])
            x = torch.nn.functional.max_pool1d(torch.relu(x), 2)
            x = x.reshape(len(features), -1)
            x = torch.nn.functional.linear(x, self.state["linear0.weight"],
                                           self.state["linear0.bias"])
            # Built as a module rather than called through the functional
            # LSTM: the latter's signature is a private detail that has changed
            # between torch releases, and this has to keep working.
            lstm = torch.nn.LSTM(input_size=150, hidden_size=150, num_layers=2,
                                 batch_first=True, bidirectional=False).double()
            lstm.load_state_dict({k[len("lstm."):]: v
                                  for k, v in self.state.items()
                                  if k.startswith("lstm.")})
            lstm.eval()
            out, _ = lstm(x.unsqueeze(0))
            logits = torch.nn.functional.linear(out.squeeze(0),
                                                self.state["linear.weight"],
                                                self.state["linear.bias"])
            return torch.softmax(logits, dim=1).numpy()

    def beat_activation(self, audio: np.ndarray, sample_rate: float) -> np.ndarray:
        """Beat probability per frame, downbeats included — a downbeat is a beat."""
        probabilities = self.activations(audio, sample_rate)
        if len(probabilities) == 0:
            return probabilities
        return probabilities[:, 0] + probabilities[:, 1]

    def downbeat_activation(self, audio: np.ndarray, sample_rate: float) -> np.ndarray:
        probabilities = self.activations(audio, sample_rate)
        return probabilities[:, 1] if len(probabilities) else probabilities
