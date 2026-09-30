#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ml/beat_this.hpp"

namespace tiktak::ml {

// Beat This! inference, through ONNX Runtime.
//
// **This is not part of tiktak_core, and the separation is the point.** The
// core has no third-party dependencies, which is what lets the analysis
// cross-compile to iOS and Android without a vendoring story; an inference
// runtime cannot have that property. So it lives beside the core exactly as
// tiktak_decode does, and a platform that would rather bring its own runtime
// links the core alone.
//
// Contrast with ml/beatnet, which the core *does* carry. That model is 0.40 M
// parameters of convolution and LSTM at 20 MMAC/s, smaller than the runtime
// that would host it. This one is a transformer of tens of millions of
// parameters run once over a whole file off the audio thread. Same project,
// opposite answers, and the size of the model against the size of the runtime
// is what decides it.
//
// The work is Francesco Foscarin, Jan Schlüter and Gerhard Widmer's, MIT
// licensed; see NOTICE.md. Nothing here is retrained.
class BeatThisSession {
public:
    // The core's, where the chunking now lives (ml::runChunked); kept here so
    // callers that named them through this class still compile.
    static constexpr std::size_t kChunkFrames = ml::kChunkFrames;
    static constexpr std::size_t kBorderFrames = ml::kBorderFrames;

    BeatThisSession();
    ~BeatThisSession();

    BeatThisSession(const BeatThisSession&) = delete;
    BeatThisSession& operator=(const BeatThisSession&) = delete;

    // Loads the graph. False on any failure, with reason() set — a missing or
    // unreadable model is an ordinary condition here, not an exception: the
    // artifact is fetched separately and deliberately not in git.
    bool open(const std::string& model_path);
    bool isOpen() const;
    const std::string& reason() const { return reason_; }

    using Activations = ml::Activations;

    // `spectrogram` is (frames, mels) row-major, as BeatThisFeatures produces.
    // Empty when the session is closed or any chunk fails.
    Activations run(const float* spectrogram, std::size_t frames, std::size_t mels);

    // One chunk through the network, in the shape ml::ChunkRunner asks for,
    // with `context` the session. This is what an OfflineAnalyzer is handed to
    // run the learned front end on the desktop; a phone hands it Core ML
    // instead, through the same signature.
    static bool runChunk(void* context, const float* spectrogram, std::size_t frames,
                         std::size_t mels, float* beat_logits, float* downbeat_logits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string reason_;
};

}  // namespace tiktak::ml
