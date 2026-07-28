#include "dsp/logfilt.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace tiktak::dsp {
namespace {

// Filter centre frequencies: `bandsPerOctave` per octave off `refHz`, trimmed
// to [minHz, maxHz].
//
// The floor and ceil deliberately overshoot the requested range on both sides
// and the trim brings it back. Rounding inwards instead would drop the band
// containing minHz whenever minHz falls between two centres, which is a filter
// the network expects to receive.
std::vector<double> centreFrequencies(std::size_t bandsPerOctave, double minHz,
                                      double maxHz, double refHz) {
    const double bands = static_cast<double>(bandsPerOctave);
    const double left = std::floor(std::log2(minHz / refHz) * bands);
    const double right = std::ceil(std::log2(maxHz / refHz) * bands);

    std::vector<double> out;
    for (double k = left; k < right; k += 1.0) {
        const double hz = refHz * std::pow(2.0, k / bands);
        if (hz >= minHz && hz <= maxHz) out.push_back(hz);
    }
    return out;
}

// Nearest FFT bin per centre frequency, duplicates removed.
//
// Removing duplicates is not tidiness. At the bottom of the range several
// centres round to the same bin, and keeping them would stack several triangles
// on the same handful of bins — giving the low end many times the weight the
// design intends, and, worse, changing how many filters come out.
std::vector<std::size_t> nearestBins(const std::vector<double>& centres,
                                     double binWidth, std::size_t bins) {
    std::vector<std::size_t> out;
    for (double hz : centres) {
        const double exact = hz / binWidth;
        std::size_t bin = static_cast<std::size_t>(std::lround(exact));
        if (bin >= bins) bin = bins - 1;
        if (out.empty() || out.back() != bin) out.push_back(bin);
    }
    return out;
}

}  // namespace

LogFilterbank::LogFilterbank(std::size_t dftSize, double sampleRate,
                             std::size_t bandsPerOctave, double minHz, double maxHz,
                             double refHz)
    : dftSize_(dftSize) {
    assert(dftSize >= 2 && sampleRate > 0.0 && bandsPerOctave > 0);
    assert(minHz > 0.0 && maxHz > minHz && refHz > 0.0);

    const std::size_t bins = spectrumSize();
    const double binWidth = sampleRate / static_cast<double>(dftSize);
    const std::vector<std::size_t> centres =
        nearestBins(centreFrequencies(bandsPerOctave, minHz, maxHz, refHz), binWidth, bins);
    if (centres.size() < 3) return;

    // Each triangle spans three consecutive centres: it rises from the first to
    // the second and falls back to the third, and its neighbours overlap it by
    // half. Both slopes are open at their far end, so no bin is counted twice
    // where two triangles meet.
    bands_ = centres.size() - 2;
    start_.reserve(bands_);
    length_.reserve(bands_);
    offset_.reserve(bands_);

    std::vector<float> triangle;
    for (std::size_t i = 0; i < bands_; ++i) {
        std::size_t begin = centres[i];
        std::size_t peak = centres[i + 1];
        std::size_t end = centres[i + 2];
        // Three centres inside two bins leave no room for a triangle; the band
        // collapses to the single bin it started on rather than vanishing.
        if (end - begin < 2) {
            peak = begin;
            end = begin + 1;
        }

        triangle.assign(end - begin, 0.0f);
        const std::size_t rise = peak - begin;
        for (std::size_t n = 0; n < rise; ++n) {
            triangle[n] = static_cast<float>(static_cast<double>(n) /
                                             static_cast<double>(rise));
        }
        const std::size_t fall = end - peak;
        for (std::size_t n = 0; n < fall; ++n) {
            triangle[rise + n] = static_cast<float>(1.0 - static_cast<double>(n) /
                                                              static_cast<double>(fall));
        }

        double total = 0.0;
        for (float w : triangle) total += w;
        if (total > 0.0) {
            for (float& w : triangle) w = static_cast<float>(w / total);
        }

        start_.push_back(begin);
        length_.push_back(triangle.size());
        offset_.push_back(weights_.size());
        weights_.insert(weights_.end(), triangle.begin(), triangle.end());
    }
}

void LogFilterbank::apply(const float* magnitude, float* out) const {
    assert(magnitude != nullptr && out != nullptr);
    for (std::size_t band = 0; band < bands_; ++band) {
        const float* weight = weights_.data() + offset_[band];
        const float* bin = magnitude + start_[band];
        float sum = 0.0f;
        for (std::size_t n = 0; n < length_[band]; ++n) sum += weight[n] * bin[n];
        out[band] = sum;
    }
}

}  // namespace tiktak::dsp
