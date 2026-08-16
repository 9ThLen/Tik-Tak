#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "dsp/dft.hpp"
#include "dsp/logfilt.hpp"

namespace tiktak::ml {

// BeatNet — a causal beat/downbeat network, running in the core.
//
// The work is Mojtaba Heydari, Frank Cwitkowitz and Zhiyao Duan's, published
// under CC BY 4.0; see NOTICE.md. The weights are unchanged: nothing is
// retrained or quantised, only rewritten into a flat float32 file.
//
// **Why it is here.** The live tracker's silence on real music was taken apart
// layer by layer and what remained was the evidence itself. Spectral flux does
// not concentrate on the beat in a produced mix: the filter's coincidence term
// sat at 0.226 where perfect tracking of that same evidence could only have
// reached 0.39, so the publishing gate stayed shut on exactly the material the
// product exists for — a singer practising against a backing track playing from
// another device, which cannot be analysed in advance and must be followed as
// it arrives. Swapping in this network's activation, with the filter, the
// gating and thresholds untouched, materially improves the causal tracker's
// accuracy and coverage. This is the implementation that was evaluated.
//
// **Why the forward pass is written out rather than run through ONNX.** The
// network is 402,325 parameters: one 1-D convolution, two linear layers and a
// two-layer LSTM, about 405k multiply-accumulates per frame at 50 frames a
// second. A mobile inference runtime is several megabytes of library — larger
// than the model — and tiktak_core has no third-party dependencies at all,
// which is the property that lets it cross-compile to iOS and Android
// unchanged (ADR 0001). Streaming a recurrent network through a graph runtime
// also means lifting the LSTM state out as graph inputs and threading it back
// each frame, which is more code at the call site than the LSTM is here.
// Beat This! keeps its ONNX export: 11 MB of transformer run once over a whole
// file off the audio thread is a different job with different arithmetic.

// The published weights, checked against the shapes the forward pass assumes.
//
// Loading is from bytes rather than from a path on purpose: the core does no
// I/O, on the audio thread or off it, and where the weights come from —
// an app bundle, an asset manager, a test fixture — is the shell's business.
class BeatNetWeights {
public:
    static constexpr std::size_t kFeatures = 272;
    static constexpr std::size_t kConvChannels = 2;
    static constexpr std::size_t kKernel = 10;
    static constexpr std::size_t kConvOut = kFeatures - kKernel + 1;   // 263
    static constexpr std::size_t kPooled = kConvOut / 2;               // 131
    static constexpr std::size_t kFlat = kConvChannels * kPooled;      // 262
    static constexpr std::size_t kHidden = 150;
    static constexpr std::size_t kLayers = 2;
    static constexpr std::size_t kClasses = 3;
    static constexpr std::size_t kParameters = 402325;

    // Header: "TTBN", version, then features, conv channels, kernel, hidden,
    // layers, classes — the numbers this class would otherwise have to assume.
    static constexpr std::size_t kHeaderBytes = 4 + 7 * sizeof(std::uint32_t);
    static constexpr std::size_t kFileBytes = kHeaderBytes + kParameters * sizeof(float);

    // True when `bytes` held a weight file this build can run. A file that
    // disagrees about any shape is refused rather than reinterpreted: the whole
    // reason for writing the shapes down is that the first version of the
    // feature front end produced 84 filters where the network wanted 136, and
    // silent reinterpretation is how that mistake survives.
    bool load(const void* data, std::size_t bytes);

    bool valid() const { return !storage_.empty(); }

    const float* conv_weight = nullptr;    // (2, 10)
    const float* conv_bias = nullptr;      // (2)
    const float* linear0_weight = nullptr; // (150, 262)
    const float* linear0_bias = nullptr;   // (150)
    const float* lstm_weight_ih[kLayers] = {nullptr, nullptr};  // (600, 150)
    const float* lstm_weight_hh[kLayers] = {nullptr, nullptr};  // (600, 150)
    const float* lstm_bias_ih[kLayers] = {nullptr, nullptr};    // (600)
    const float* lstm_bias_hh[kLayers] = {nullptr, nullptr};    // (600)
    const float* out_weight = nullptr;     // (3, 150)
    const float* out_bias = nullptr;       // (3)

private:
    std::vector<float> storage_;
};

// One frame in, three class probabilities out, recurrent state carried.
//
// Real-time safe: every buffer is sized in the constructor and forward()
// allocates nothing.
class BeatNetModel {
public:
    // The weights must outlive the model and must be valid().
    explicit BeatNetModel(const BeatNetWeights& weights);

    // `features` holds kFeatures values; `probabilities` receives three, in the
    // order **beat, downbeat, null**.
    //
    // That order is read off the published BeatNet.py, which keeps preds[:2]
    // and discards the third — so the two it keeps are the beat and the
    // downbeat, and the one it drops is the null class. Assuming the intuitive
    // order instead puts the null class where the downbeat should be, and since
    // the null class is high almost all the time the mistake reads as a model
    // that hears a downbeat everywhere rather than as an index error. It cost
    // an afternoon once already.
    void forward(const float* features, float* probabilities);

    // Forgets the recurrent state. The tracker calls this when the stream
    // restarts; carrying an LSTM across a gap in the audio is carrying a
    // memory of music that is no longer playing.
    void reset();

private:
    const BeatNetWeights& weights_;

    std::vector<float> pooled_;   // 262
    std::vector<float> layer_in_; // 150, the input to the current LSTM layer
    std::vector<float> gates_;    // 600
    std::vector<float> hidden_;   // layers * 150
    std::vector<float> cell_;     // layers * 150
};

// The network's input features, computed from audio as it arrives.
//
// 22050 Hz mono, 64 ms frames every 20 ms — 1411 samples every 441, which is
// exactly 50 frames a second — Hann windowed; a logarithmic filterbank of 24
// bands per octave from 30 Hz to 17 kHz, area-normalised, giving 136 values;
// log10(1 + magnitude); and the positive difference to the previous frame
// stacked alongside, for 272 features in all.
//
// Frames are centred on their reference sample, as madmom centres them, so
// frame k is the 1411 samples around sample 441k and the first frame is half
// padding. Aligning to the frame's start instead would put every activation
// half a window — 32 ms — early, which at 120 BPM is a sixteenth of a beat and
// would look like a tracker that rushes.
//
// The flip side of centring is that frame k cannot be computed until 705
// samples past its own timestamp have arrived, so the activation reaches the
// filter 32 ms after the instant it describes. The timestamp handed to the
// caller is the frame's own, not the moment it was computed, so the filter
// still places the evidence correctly — it simply places it slightly in the
// past, which a particle filter over continuous time handles and a
// single-hypothesis tracker would not. The upstream project notes the same
// delay in its own streaming mode.
//
// Real-time safe: process() allocates nothing.
class BeatNetFeatures {
public:
    static constexpr std::size_t kFeatures = BeatNetWeights::kFeatures;
    static constexpr std::size_t kFilters = 136;
    static constexpr double kModelRate = 22050.0;
    static constexpr std::size_t kFrameSize = 1411;   // 64 ms
    static constexpr std::size_t kHopSize = 441;      // 20 ms, so exactly 50 fps
    static constexpr double kFrameRate = kModelRate / static_cast<double>(kHopSize);

    // Upsampling by more than this is a capture rate the model has no business
    // being fed; 8 covers everything down to 3 kHz.
    static constexpr std::size_t kMaxPerSample = 8;

    explicit BeatNetFeatures(double sampleRate);

    void reset();

    // Feeds captured mono audio at the construction sample rate, invoking
    //     onFrame(const float* features, std::size_t count, double time_sec)
    // once per completed frame. `time_sec` is the frame's reference time
    // measured from the last reset().
    template <typename Fn>
    void process(const float* samples, std::size_t n, Fn&& onFrame) {
        float staged[kMaxPerSample];
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t produced = resample(samples[i], staged);
            for (std::size_t j = 0; j < produced; ++j) {
                if (accept(staged[j])) {
                    onFrame(features_.data(), features_.size(), frameTimeSec());
                    advance();
                }
            }
        }
    }

private:
    // One captured sample in, up to kMaxPerSample model-rate samples out.
    //
    // Linear interpolation, matching the Python reference exactly, which is
    // what every measurement so far was made through. It is worth being plain
    // about what that means: decimating 48 kHz this way has no anti-alias
    // filter, so content above 11 kHz folds back into the band the network
    // reads. Whether a proper polyphase decimator moves the numbers is an open
    // question and a measurable one; it is not something to change quietly
    // underneath results that were obtained without it.
    std::size_t resample(float sample, float* out);

    // One model-rate sample in; true when a frame is complete in features_.
    bool accept(float sample);
    void advance();
    double frameTimeSec() const {
        return static_cast<double>(frame_index_) * static_cast<double>(kHopSize) / kModelRate;
    }

    double ratio_;               // captured samples per model sample
    dsp::RealDft dft_;
    dsp::LogFilterbank bank_;
    std::vector<float> window_;

    std::vector<float> buffer_;    // one analysis frame of model-rate audio
    std::vector<float> windowed_;
    std::vector<float> spectrum_;
    std::vector<float> previous_;  // last frame's log-filtered band values
    std::vector<float> features_;

    std::size_t fill_ = 0;
    std::size_t frame_index_ = 0;
    bool seen_frame_ = false;

    // Resampler position, in captured samples since reset.
    std::size_t input_index_ = 0;
    std::size_t output_index_ = 0;
    float previous_sample_ = 0.0f;
};

// Audio in, beat activation out: the features and the network wired together.
//
// This is what the live tracker holds. Real-time safe throughout — no
// allocation, no locks, no I/O — though a frame's arithmetic lands in one
// audio callback every 20 ms rather than being spread evenly, so a shell with
// a very small buffer should measure before assuming.
// **More than one set of weights averages them.** BeatNet publishes three
// checkpoints, each withholding a different one of its five training corpora,
// and the mean of their activations is a materially better observation than any
// one of them: measured through the research seam on 581 full-length recordings
// it lifts the share with no wrong-level episode from 40.8% to 50.6% and beat F
// from 0.803 to 0.845, every comparison significant after Holm correction. See
// research/results/README.md, and eval/PREREGISTERED_ensemble_in_core.md for
// what this class is required to reproduce.
//
// The averaging happens here, over one front end, rather than in the tracker
// over several activations. The features are identical for every checkpoint —
// same rate, same filterbank, same frame — so computing them once is not an
// optimisation to be verified, it is the only arrangement that is correct;
// three front ends would be three copies of one answer. It also means the cost
// of N networks is N times the *network* and once the spectrum, which is why
// the real-time factor has to be measured rather than assumed to triple.
//
// A mean and not a max. They fail in opposite directions — a mean is dragged
// down by a checkpoint that is unsure, a max by one that is wrong and confident
// — and on RWC and Harmonix alike the max arm scores below every single fold on
// the usable rate. What the mean suppresses is the failure the corpora contain.
class BeatNetActivation {
public:
    BeatNetActivation(double sampleRate, const BeatNetWeights& weights);

    // `weights` is `count` pointers, each to a valid() set that must outlive
    // this object; the core does no I/O and does not own them. A count of zero
    // or a null entry is a programming error and is asserted, not tolerated:
    // quietly falling back to fewer networks than the caller asked for would
    // produce a working tracker that is not the one being measured, and the
    // difference between those two is several points of the headline.
    BeatNetActivation(double sampleRate,
                      const BeatNetWeights* const* weights, std::size_t count);

    // How many networks are being averaged. One, for the single-weight form.
    std::size_t networks() const { return models_.size(); }

    void reset();

    // Forget only the LSTM hidden/cell state. Feature framing, resampler
    // history, previous spectral frame and frame clock continue untouched.
    // This is narrower than reset() on purpose: the S0 experiment asks whether
    // recurrent memory helps, and resetting the feature stream would measure an
    // artificial audio discontinuity instead.
    void resetModelState();

    // Feeds captured mono audio, invoking
    //     onActivation(double time_sec, double beat, double downbeat)
    // once per frame, 50 times a second of audio. `beat` includes downbeats,
    // because a downbeat is a beat; `downbeat` is the downbeat alone.
    //
    // `time_sec` is measured from the last reset(), as dsp::Odf measures its
    // frames, and for the same reason: the caller owns the stream clock. A
    // device that drops a buffer moves the audio's timestamps without moving
    // its sample count, and only the caller can see that happen.
    template <typename Fn>
    void process(const float* samples, std::size_t n, Fn&& onActivation) {
        process(samples, n, [](double) { return false; },
                std::forward<Fn>(onActivation));
    }

    // The predicate runs immediately before the network consumes a completed
    // feature frame. Returning true resets only recurrent model state, so a
    // caller can place a reset at "the first frame whose timestamp reaches H"
    // without approximating it at an audio-buffer boundary.
    template <typename ResetFn, typename Fn>
    void process(const float* samples, std::size_t n, ResetFn&& shouldReset,
                 Fn&& onActivation) {
        features_.process(samples, n, [&](const float* f, std::size_t count, double t) {
            (void)count;
            if (shouldReset(t)) resetModelState();
            // Averaged in probability space, which is where the research seam
            // averaged them and therefore what the measured numbers describe.
            // Averaging the logits instead is a geometric mean of the odds and
            // is a different estimator with different numbers; it has not been
            // measured and must not be substituted here on the grounds that it
            // is more principled.
            double beat = 0.0;
            double downbeat = 0.0;
            for (BeatNetModel& model : models_) {
                model.forward(f, probabilities_);
                beat += static_cast<double>(probabilities_[0]) +
                        static_cast<double>(probabilities_[1]);
                downbeat += static_cast<double>(probabilities_[1]);
            }
            const double scale = 1.0 / static_cast<double>(models_.size());
            onActivation(t, beat * scale, downbeat * scale);
        });
    }

private:
    BeatNetFeatures features_;
    // Sized once, in the constructor. `process` walks it and allocates nothing,
    // which is what keeps this usable from an audio callback.
    std::vector<BeatNetModel> models_;
    float probabilities_[BeatNetWeights::kClasses] = {0.0f, 0.0f, 0.0f};
};

}  // namespace tiktak::ml
