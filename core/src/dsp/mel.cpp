#include "dsp/mel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace tiktak::dsp {

namespace {

// Slaney's constants, as they appear in the Auditory Toolbox and in librosa:
// 3 mel per 20 Hz below 1 kHz, then a logarithm chosen to meet it smoothly at
// 1 kHz and cover 6.4 octaves in 27 further steps.
constexpr double kSlaneyFSp = 200.0 / 3.0;
constexpr double kSlaneyBreakHz = 1000.0;
const double kSlaneyBreakMel = kSlaneyBreakHz / kSlaneyFSp;
const double kSlaneyLogStep = std::log(6.4) / 27.0;

}  // namespace

// Natural-log form of the usual 2595*log10(1 + f/700) mel scale.
double hzToMel(double hz, MelScale scale) {
    if (scale == MelScale::Slaney) {
        if (hz < kSlaneyBreakHz) return hz / kSlaneyFSp;
        return kSlaneyBreakMel + std::log(hz / kSlaneyBreakHz) / kSlaneyLogStep;
    }
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

double melToHz(double mel, MelScale scale) {
    if (scale == MelScale::Slaney) {
        if (mel < kSlaneyBreakMel) return kSlaneyFSp * mel;
        return kSlaneyBreakHz * std::exp(kSlaneyLogStep * (mel - kSlaneyBreakMel));
    }
    return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}

MelFilterbank::MelFilterbank(std::size_t fftSize, double sampleRate, std::size_t bands,
                             double minHz, double maxHz, MelScale scale)
    : fftSize_(fftSize), bands_(bands) {
    assert(fftSize >= 2 && bands >= 1 && sampleRate > 0.0);

    const double nyquist = sampleRate * 0.5;
    maxHz = std::min(maxHz, nyquist);
    minHz = std::max(minHz, 0.0);
    assert(minHz < maxHz);

    // bands + 2 edges: each filter spans [edge[i], edge[i+2]] with its peak at
    // edge[i+1], so neighbouring triangles overlap by half.
    const double melMin = hzToMel(minHz, scale);
    const double melMax = hzToMel(maxHz, scale);
    std::vector<double> edges(bands_ + 2);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(bands_ + 1);
        edges[i] = melToHz(melMin + t * (melMax - melMin), scale);
    }

    centres_.resize(bands_);
    for (std::size_t b = 0; b < bands_; ++b) centres_[b] = edges[b + 1];

    const std::size_t bins = spectrumSize();
    const double binToHz = sampleRate / static_cast<double>(fftSize_);

    start_.resize(bands_);
    length_.resize(bands_);
    offset_.resize(bands_);
    weights_.clear();
    weights_.reserve(bands_ * 8);

    for (std::size_t b = 0; b < bands_; ++b) {
        const double left = edges[b];
        const double centre = edges[b + 1];
        const double right = edges[b + 2];

        // Bins strictly inside (left, right) get non-zero weight. Guard the
        // denominators: at low frequencies with a short FFT, adjacent mel edges
        // can collapse onto the same bin.
        std::size_t first = bins;
        std::size_t last = 0;
        std::vector<float> triangle;
        triangle.reserve(16);

        for (std::size_t k = 0; k < bins; ++k) {
            const double hz = static_cast<double>(k) * binToHz;
            float w = 0.0f;
            if (hz > left && hz < centre && centre > left) {
                w = static_cast<float>((hz - left) / (centre - left));
            } else if (hz >= centre && hz < right && right > centre) {
                w = static_cast<float>((right - hz) / (right - centre));
            }
            if (w > 0.0f) {
                if (first == bins) first = k;
                last = k;
                triangle.push_back(w);
            } else if (first != bins) {
                // Triangles are contiguous, so the first zero after the run ends it.
                break;
            }
        }

        offset_[b] = weights_.size();
        if (first == bins) {
            // Degenerate filter: the triangle fell between two FFT bins. Snap it
            // to the nearest bin so no band is silent.
            const std::size_t k = std::min(
                bins - 1, static_cast<std::size_t>(std::lround(centre / binToHz)));
            start_[b] = k;
            length_[b] = 1;
            weights_.push_back(1.0f);
        } else {
            start_[b] = first;
            length_[b] = last - first + 1;
            weights_.insert(weights_.end(), triangle.begin(), triangle.end());
        }
    }
}

std::size_t MelFilterbank::bandAtOrAbove(double hz) const {
    for (std::size_t b = 0; b < bands_; ++b) {
        if (centres_[b] >= hz) return b;
    }
    return bands_;
}

void MelFilterbank::apply(const float* magnitude, float* out) const {
    for (std::size_t b = 0; b < bands_; ++b) {
        const float* w = weights_.data() + offset_[b];
        const float* m = magnitude + start_[b];
        const std::size_t n = length_[b];

        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) sum += w[i] * m[i];
        out[b] = sum;
    }
}

}  // namespace tiktak::dsp
