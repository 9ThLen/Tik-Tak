#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "dsp/chroma.hpp"
#include "dsp/mel.hpp"
#include "dsp/stft.hpp"

namespace tiktak::dsp {

struct OdfConfig {
    double sampleRate = 48000.0;
    std::size_t frameSize = 2048;
    std::size_t hopSize = 512;
    std::size_t melBands = 81;
    double melMinHz = 27.5;      // A0
    double melMaxHz = 16000.0;   // clamped to Nyquist
    double lowBandHz = 200.0;    // kick and bass live below here
    double highBandHz = 4000.0;  // hi-hat and cymbals live above here
    bool whitening = true;
    double whiteningTau = 1.0;   // seconds

    // Whitening exponent in [0, 1]: each band is divided by peak^strength.
    //
    // 1.0 is textbook adaptive whitening — every band is normalised to its own
    // running peak, which equalises loud and quiet instruments but also forces
    // any band on a rising edge to exactly full scale. That erases the relative
    // balance between bands, and the low/high outputs exist precisely to carry
    // it. 0.0 disables normalisation and lets absolute level back in.
    //
    // The default splits the difference. Phase 2 tunes it against mir_eval
    // rather than intuition; see docs/PLAN.md.
    double whiteningStrength = 0.5;

    // Whitening floor, relative to the loudest mel band seen recently. Bands
    // quieter than this fraction of the current peak are treated as noise
    // rather than being normalised up to full scale. Must be in (0, 1).
    double whiteningFloorRel = 1e-3;  // -60 dB

    // Also produce a pitch class profile per frame, for the downbeat analysis.
    //
    // Off by default because it is dead weight on the microphone path: the live
    // tracker only wants the beat, and harmony is a bar-level cue that costs
    // one extra pass over the spectrum every hop. The offline analyser turns it
    // on, which is where bar lines are decided anyway.
    bool chroma = false;
    double chromaMinHz = 55.0;    // A1 — below this, pitch is mostly inharmonic
    double chromaMaxHz = 2093.0;  // C7 — above this it is overtones of the rest

    bool valid() const;
};

struct OdfFrame {
    double timeSec = 0.0;
    float full = 0.0f;
    float low = 0.0f;
    float high = 0.0f;
};

// Onset detection function: half-wave rectified spectral flux over a log mel
// spectrogram, with optional adaptive whitening.
//
// Three bands are produced in parallel from the same spectrum because they
// answer different questions downstream: `full` drives tempo estimation and beat
// phase, `low` is the strongest downbeat cue, `high` the strongest subdivision
// cue. Computing them together costs one extra pass over the mel bands.
//
// Each band reports the *mean* rise across its mel bands, not the sum. The
// high band spans several times more mel bands than the low one, so summing
// would make `high` structurally larger than `low` regardless of the audio and
// leave the two incomparable.
//
// Every buffer is allocated in the constructor; process() allocates nothing.
class Odf {
public:
    explicit Odf(const OdfConfig& config);

    const OdfConfig& config() const { return config_; }
    std::size_t melBands() const { return mel_.bands(); }

    // The current frame's pitch class profile: ChromaFilterbank::kBins values,
    // or nullptr when OdfConfig::chroma is off.
    //
    // Read it from inside the process() callback. It points at storage owned by
    // this object and is overwritten by the next frame — returning it by value
    // would put twelve floats on the hot path for the benefit of the one caller
    // that wants them.
    const float* chroma() const { return chroma_.empty() ? nullptr : chroma_.data(); }

    std::size_t framesAvailable(std::size_t n) const { return stft_.framesAvailable(n); }

    // How far behind the newest sample an emitted frame's timestamp sits.
    double latencySec() const {
        return static_cast<double>(config_.frameSize) * 0.5 / config_.sampleRate;
    }

    // Clears buffered audio, the previous spectrum and the whitening state, and
    // restarts the stream clock. Use on a seek, a new file, or a mic restart.
    void reset();

    // Feeds `n` mono samples, invoking onFrame(const OdfFrame&) per hop.
    template <typename Fn>
    void process(const float* samples, std::size_t n, Fn&& onFrame) {
        stft_.process(samples, n,
                      [&](const float* magnitude, std::size_t bins, std::size_t frameStart) {
                          (void)bins;
                          const OdfFrame frame = computeFrame(magnitude, frameStart);
                          onFrame(frame);
                      });
    }

private:
    OdfFrame computeFrame(const float* magnitude, std::size_t frameStartSample);

    OdfConfig config_;
    Stft stft_;
    MelFilterbank mel_;
    std::unique_ptr<ChromaFilterbank> chroma_bank_;
    std::vector<float> chroma_;

    std::size_t lowSplit_ = 0;   // bands [0, lowSplit_) form the low band
    std::size_t highSplit_ = 0;  // bands [highSplit_, melBands) form the high band

    float whiteningDecay_ = 0.0f;
    float globalDecay_ = 0.0f;
    float globalPeak_ = 0.0f;
    std::vector<float> whitenPeak_;
    std::vector<float> melBuf_;
    std::vector<float> logMel_;
    std::vector<float> prevLogMel_;
    bool hasPrev_ = false;
};

}  // namespace tiktak::dsp
