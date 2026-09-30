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
// transcription error can hide, which is why every constant below is pinned
// by reference tests rather than trusted.
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

// Beat and downbeat times, in seconds, from the model's per-frame logits.
//
// Transcribed from the reference port rather than invented, because a different
// peak picker changes every number downstream and makes any comparison with
// published results meaningless. Three steps, each of which matters:
//
//   * A frame is a peak if it equals the maximum over a seven-frame window
//     centred on it — 140 ms, comfortably under any beat period.
//   * And if its logit is above zero, which is to say a probability above a
//     half. This is the only threshold, and it is the model's own.
//   * Peaks one frame apart are collapsed to the first. Two beats 20 ms apart
//     is not a tempo, it is a plateau the pooling could not separate.
//
// Downbeats are snapped onto the beat nearest each downbeat peak. The two heads
// are independent, so a downbeat can land a frame off its own beat; leaving it
// there would put a bar line between two beats, where no bar line can be.
struct BeatGrid {
    std::vector<double> beats;
    std::vector<double> downbeats;
};

BeatGrid pickBeats(const float* beat_logits, const float* downbeat_logits,
                   std::size_t frames, double frameRate = BeatThisFeatures::kFrameRate);

// --------------------------------------------------------- the inference seam

// The network, one chunk at a time: `spectrogram` holds `frames` rows of
// `mels` log-mel values, zero past the end of the song, and the runner writes
// `frames` beat logits and `frames` downbeat logits. False on any failure.
//
// This is all of Beat This! a platform has to supply. Features, chunking,
// stitching and peak picking are the core's, so ONNX Runtime on a desktop and
// Core ML on a phone differ in this one function and cannot drift anywhere
// else. A plain function pointer and a context rather than a std::function,
// because the same thing has to cross the C API.
using ChunkRunner = bool (*)(void* context, const float* spectrogram, std::size_t frames,
                             std::size_t mels, float* beat_logits, float* downbeat_logits);

// Frames per inference chunk, and the border discarded at each edge.
// Transcribed from the reference port rather than chosen: the model's answer
// near a chunk boundary is worse than in the middle, and these two numbers are
// how the reference arranges for no frame to be read from an edge if any chunk
// covers it away from one.
constexpr std::size_t kChunkFrames = 1500;
constexpr std::size_t kBorderFrames = 6;

struct Activations {
    // Raw logits, one per frame at 50 frames a second. Logits rather than
    // probabilities because the peak picker's threshold is "above zero", and
    // passing these through a sigmoid first would only move where that
    // threshold has to be written.
    std::vector<float> beat;
    std::vector<float> downbeat;
};

// The whole spectrogram through `runner`, arranged and stitched exactly as the
// reference port does it: chunks of kChunkFrames starting from -kBorderFrames,
// the last one pulled back to end on the tail, and overlaps resolved by keeping
// the earlier chunk, whose answer was computed further from its edge. False,
// with `out` left empty, when any chunk fails — half a song's activations are
// not an answer.
bool runChunked(const float* spectrogram, std::size_t frames, std::size_t mels,
                ChunkRunner runner, void* context, Activations& out);

}  // namespace tiktak::ml
