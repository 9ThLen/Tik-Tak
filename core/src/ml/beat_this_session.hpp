#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

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
    // Frames per inference chunk, and the border discarded at each edge.
    // Transcribed from the reference port rather than chosen: the model's
    // answer near a chunk boundary is worse than in the middle, and these two
    // numbers are how the reference arranges for no frame to be read from an
    // edge if any chunk covers it away from one.
    static constexpr std::size_t kChunkFrames = 1500;
    static constexpr std::size_t kBorderFrames = 6;

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

    struct Activations {
        // Raw logits, one per frame at 50 frames a second. Logits rather than
        // probabilities because the peak picker's threshold is "above zero",
        // and passing these through a sigmoid first would only move where that
        // threshold has to be written.
        std::vector<float> beat;
        std::vector<float> downbeat;
    };

    // `spectrogram` is (frames, mels) row-major, as BeatThisFeatures produces.
    Activations run(const float* spectrogram, std::size_t frames, std::size_t mels);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string reason_;
};

}  // namespace tiktak::ml
