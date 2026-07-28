#pragma once

#include <cstddef>
#include <vector>

#include "dsp/fft.hpp"
#include "dsp/mel.hpp"

namespace tiktak::ml {

// Beat This!'s input features, computed in the core.
//
// The work is Francesco Foscarin, Jan Schlüter and Gerhard Widmer's, MIT
// licensed; see NOTICE.md. Nothing is retrained here — this is the published
// preprocessing, and the network itself runs from the exported ONNX.
//
// **Why the spectrogram is ours and the network is not.** The exported graph
// begins at `input_spectrogram`: the mel front end was never part of the model,
// so somebody has to compute it, and it may as well be the code that already
// owns an FFT and a filterbank. That also means this file is the one place a
// transcription error can hide, which is why every constant below is checked
// against the reference in tools/parity rather than trusted.
//
// The awkward details, each of which is silently wrong-looking-right if missed:
//
//   * The Hann window is **periodic** — divide by N, not N-1.
//   * The mel scale is **Slaney's**, not the 1127*ln form the ODF uses.
//   * Triangles are plain, peaking at one; there is no area normalisation.
//   * The spectrum is an **amplitude**, not a power, divided by sqrt(1024).
//   * Compression is log1p(1000 * energy), not decibels.
//   * The signal is **reflect-padded** by half a window at both ends, so frame
//     k is centred on sample 441k — the same centring BeatNet uses, for the
//     same reason: aligning to the frame's start would put every activation
//     half a window early.
class BeatThisFeatures {
public:
    static constexpr double kModelRate = 22050.0;
    static constexpr std::size_t kFftSize = 1024;
    static constexpr std::size_t kHopSize = 441;   // exactly 50 frames a second
    static constexpr std::size_t kMels = 128;
    static constexpr double kFrameRate = kModelRate / static_cast<double>(kHopSize);
    static constexpr double kMinHz = 30.0;
    static constexpr double kMaxHz = 11000.0;
    static constexpr double kLogMultiplier = 1000.0;
    static constexpr double kFloor = 1e-10;

    BeatThisFeatures();

    // Frames this many samples of model-rate audio would produce.
    static std::size_t frameCount(std::size_t samples);

    // (frames, 128) row-major log-mel, for `samples` of mono audio already at
    // kModelRate. Whole-file rather than streaming on purpose: Beat This! is
    // not causal and there is nothing to stream it into.
    std::vector<float> compute(const float* samples, std::size_t count);

private:
    dsp::Fft fft_;
    dsp::MelFilterbank bank_;
    std::vector<float> window_;
    std::vector<float> padded_;
    std::vector<float> block_;
    std::vector<float> spectrum_;
};

}  // namespace tiktak::ml
