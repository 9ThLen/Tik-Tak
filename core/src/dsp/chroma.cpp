#include "dsp/chroma.hpp"

#include <algorithm>
#include <cmath>

namespace tiktak::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Width of the band within half a semitone of a pitch, as a fraction of that
// pitch: 2^(1/24) - 2^(-1/24).
const double kSemitoneWidth = std::pow(2.0, 1.0 / 24.0) - std::pow(2.0, -1.0 / 24.0);

}  // namespace

double ChromaFilterbank::resolvedMinHz(std::size_t fftSize, double sampleRate,
                                       double binsPerSemitone) {
    if (fftSize == 0 || !(sampleRate > 0.0) || !(binsPerSemitone > 0.0)) return 0.0;
    const double binHz = sampleRate / static_cast<double>(fftSize);
    return binsPerSemitone * binHz / kSemitoneWidth;
}

ChromaFilterbank::ChromaFilterbank(std::size_t fftSize, double sampleRate, double minHz,
                                   double maxHz, double floorRel)
    : fftSize_(fftSize), floorRel_(std::max(floorRel, 0.0)) {
    const double nyquist = sampleRate * 0.5;
    const double top = std::min(maxHz, nyquist);
    const double bottom = std::max({minHz, 1.0, resolvedMinHz(fftSize, sampleRate)});
    maxHz_ = top;
    minHz_ = bottom;
    if (fftSize_ == 0 || bottom >= top) return;

    const std::size_t bins = spectrumSize();
    for (std::size_t k = 1; k < bins; ++k) {
        const double hz = static_cast<double>(k) * sampleRate / static_cast<double>(fftSize_);
        if (hz < bottom || hz > top) continue;

        // MIDI number, fractional. 69 is A4 = 440 Hz.
        const double midi = 69.0 + 12.0 * std::log2(hz / 440.0);
        const double nearest = std::round(midi);
        const double deviation = midi - nearest;  // in [-0.5, 0.5] semitones

        // 1 dead on a semitone, 0 exactly between two. Squared so the falloff
        // is smooth at both ends rather than kinked at the centre.
        const double w = std::cos(kPi * deviation);
        if (w <= 0.0) continue;

        const auto semitone = static_cast<long long>(nearest);
        const auto pc = static_cast<std::size_t>(((semitone % 12) + 12) % 12);

        bin_.push_back(k);
        pitch_.push_back(pc);
        weight_.push_back(static_cast<float>(w * w));
    }
}

void ChromaFilterbank::apply(const float* magnitude, float* out) const {
    for (std::size_t i = 0; i < kBins; ++i) out[i] = 0.0f;
    if (magnitude == nullptr) return;

    for (std::size_t i = 0; i < bin_.size(); ++i) {
        out[pitch_[i]] += weight_[i] * magnitude[bin_[i]];
    }

    double energy = 0.0;
    for (std::size_t i = 0; i < kBins; ++i) energy += static_cast<double>(out[i]) * out[i];

    // Normalising is what makes the profile a direction rather than a level,
    // and it is also a trap: divide by a vanishing length and whatever noise
    // was there arrives at full scale, so a bass note an octave below the
    // resolvable range comes back as a confident chord assembled from its own
    // spectral leakage. The guard is relative to everything in the spectrum,
    // not an absolute level, so it holds at any input gain — the same reason
    // the onset detector floors its whitening against the loudest band rather
    // than against a constant.
    double total = 0.0;
    const std::size_t bins = spectrumSize();
    for (std::size_t k = 0; k < bins; ++k) {
        total += static_cast<double>(magnitude[k]) * magnitude[k];
    }
    if (energy <= 0.0 || energy < floorRel_ * total) {
        for (std::size_t i = 0; i < kBins; ++i) out[i] = 0.0f;
        return;
    }

    const auto scale = static_cast<float>(1.0 / std::sqrt(energy));
    for (std::size_t i = 0; i < kBins; ++i) out[i] *= scale;
}

double chromaDistance(const float* a, const float* b) {
    if (a == nullptr || b == nullptr) return 0.0;

    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < ChromaFilterbank::kBins; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;

    const double cosine = dot / std::sqrt(na * nb);
    return std::clamp(1.0 - cosine, 0.0, 1.0);
}

}  // namespace tiktak::dsp
