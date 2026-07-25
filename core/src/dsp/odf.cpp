#include "dsp/odf.hpp"

#include <algorithm>

namespace tiktak::dsp {
namespace {

// Guards the whitening division in true digital silence, where even the
// relative floor collapses to zero. Small enough never to bite on real audio.
constexpr float kEpsilon = 1e-12f;

// The global peak that sets the relative floor is deliberately slower to decay
// than the per-band peaks: it represents "how loud is this material", which
// should survive a rest, while per-band peaks track individual instruments.
constexpr double kGlobalTauMultiplier = 4.0;

}  // namespace

bool OdfConfig::valid() const {
    if (!(sampleRate > 0.0)) return false;
    if (!Fft::isPowerOfTwo(frameSize)) return false;
    if (hopSize == 0 || hopSize > frameSize) return false;
    if (melBands == 0) return false;
    if (!(melMinHz >= 0.0)) return false;
    if (!(melMaxHz > melMinHz)) return false;
    if (melMinHz >= sampleRate * 0.5) return false;
    if (!(lowBandHz > 0.0) || !(highBandHz > lowBandHz)) return false;
    if (whitening && !(whiteningTau > 0.0)) return false;
    if (whitening && !(whiteningFloorRel > 0.0 && whiteningFloorRel < 1.0)) return false;
    if (whitening && !(whiteningStrength >= 0.0 && whiteningStrength <= 1.0)) return false;
    return true;
}

Odf::Odf(const OdfConfig& config)
    : config_(config),
      stft_(config.frameSize, config.hopSize),
      mel_(config.frameSize, config.sampleRate, config.melBands, config.melMinHz,
           config.melMaxHz) {
    const std::size_t bands = mel_.bands();

    lowSplit_ = mel_.bandAtOrAbove(config_.lowBandHz);
    highSplit_ = mel_.bandAtOrAbove(config_.highBandHz);

    // Guarantee both sub-bands are non-empty even on odd configurations, so that
    // downstream consumers never see a permanently silent band.
    lowSplit_ = std::clamp<std::size_t>(lowSplit_, 1, bands);
    highSplit_ = std::clamp<std::size_t>(highSplit_, 0, bands - 1);

    const double framesPerSecond = config_.sampleRate / static_cast<double>(config_.hopSize);
    whiteningDecay_ = static_cast<float>(
        std::exp(-1.0 / (config_.whiteningTau * framesPerSecond)));
    globalDecay_ = static_cast<float>(
        std::exp(-1.0 / (config_.whiteningTau * kGlobalTauMultiplier * framesPerSecond)));

    whitenPeak_.assign(bands, 0.0f);
    melBuf_.assign(bands, 0.0f);
    logMel_.assign(bands, 0.0f);
    prevLogMel_.assign(bands, 0.0f);
}

void Odf::reset() {
    stft_.reset();
    globalPeak_ = 0.0f;
    std::fill(whitenPeak_.begin(), whitenPeak_.end(), 0.0f);
    std::fill(melBuf_.begin(), melBuf_.end(), 0.0f);
    std::fill(logMel_.begin(), logMel_.end(), 0.0f);
    std::fill(prevLogMel_.begin(), prevLogMel_.end(), 0.0f);
    hasPrev_ = false;
}

OdfFrame Odf::computeFrame(const float* magnitude, std::size_t frameStartSample) {
    const std::size_t bands = mel_.bands();

    mel_.apply(magnitude, melBuf_.data());

    if (config_.whitening) {
        // Per-band running peak with exponential decay (Stowell & Plumbley 2007).
        // This is what lets a quiet guitar and a loud drum contribute comparably:
        // each band is measured against its own recent maximum, not an absolute
        // level, so the flux stops tracking overall loudness.
        //
        // The floor is relative to the loudest band rather than an absolute
        // constant. With an absolute floor, a band that is merely picking up
        // spectral leakage gets divided by its own tiny peak and normalises
        // straight to full scale — so after a silent passage every empty band
        // reports a full-strength onset. Tying the floor to the current overall
        // level keeps quiet bands quiet at any input gain.
        float frameMax = 0.0f;
        for (std::size_t b = 0; b < bands; ++b) frameMax = std::max(frameMax, melBuf_[b]);
        globalPeak_ = std::max(frameMax, globalPeak_ * globalDecay_);

        const float floor = globalPeak_ * static_cast<float>(config_.whiteningFloorRel);

        const float strength = static_cast<float>(config_.whiteningStrength);

        for (std::size_t b = 0; b < bands; ++b) {
            const float decayed = whitenPeak_[b] * whiteningDecay_;
            const float peak = std::max({melBuf_[b], decayed, floor, kEpsilon});
            whitenPeak_[b] = std::max(melBuf_[b], decayed);
            melBuf_[b] /= std::pow(peak, strength);
        }
    }

    for (std::size_t b = 0; b < bands; ++b) {
        logMel_[b] = std::log1p(melBuf_[b]);
    }

    OdfFrame frame;
    // The window's centre is what the frame describes, so that is what it is
    // timestamped with — a beat tracker downstream compares these against
    // predicted beat times and half a window of systematic bias would matter.
    frame.timeSec = (static_cast<double>(frameStartSample) +
                     static_cast<double>(config_.frameSize) * 0.5) /
                    config_.sampleRate;

    if (hasPrev_) {
        // Half-wave rectified flux: only rising energy marks an onset. Falling
        // energy is a note ending, which is not a rhythmic event we want.
        float full = 0.0f;
        float low = 0.0f;
        float high = 0.0f;

        for (std::size_t b = 0; b < bands; ++b) {
            const float rise = logMel_[b] - prevLogMel_[b];
            if (rise <= 0.0f) continue;
            full += rise;
            if (b < lowSplit_) low += rise;
            if (b >= highSplit_) high += rise;
        }

        // Mean, not sum — see the class comment. Band counts differ by an order
        // of magnitude, and the three outputs are meant to be compared.
        frame.full = full / static_cast<float>(bands);
        frame.low = low / static_cast<float>(lowSplit_);
        frame.high = high / static_cast<float>(bands - highSplit_);
    }

    std::swap(prevLogMel_, logMel_);
    hasPrev_ = true;
    return frame;
}

}  // namespace tiktak::dsp
