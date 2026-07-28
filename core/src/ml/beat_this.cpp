#include "ml/beat_this.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace tiktak::ml {
namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

BeatThisFeatures::BeatThisFeatures()
    : fft_(kFftSize),
      bank_(kFftSize, kModelRate, kMels, kMinHz, kMaxHz, dsp::MelScale::Slaney),
      window_(kFftSize),
      block_(kFftSize, 0.0f),
      spectrum_(kFftSize / 2 + 1, 0.0f) {
    for (std::size_t n = 0; n < kFftSize; ++n) {
        window_[n] = static_cast<float>(
            0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(n) /
                                  static_cast<double>(kFftSize))));
    }
}

std::size_t BeatThisFeatures::frameCount(std::size_t samples) {
    const std::size_t pad = kFftSize / 2;
    if (samples <= pad) return 0;
    // Reflect-padding adds half a window at each end, so the padded length is
    // samples + 2*pad and the usual (length - window)/hop + 1 applies.
    const std::size_t padded = samples + 2 * pad;
    if (padded < kFftSize) return 0;
    return (padded - kFftSize) / kHopSize + 1;
}

std::vector<float> BeatThisFeatures::compute(const float* samples, std::size_t count) {
    assert(samples != nullptr || count == 0);

    const std::size_t frames = frameCount(count);
    if (frames == 0) return {};

    // Reflection that does not repeat the edge sample: padded[pad-1] is
    // samples[1], not samples[0]. numpy calls this "reflect"; the alternative
    // ("symmetric") doubles the first sample and shifts every frame by a
    // fraction of a hop, which survives all the way to the activation.
    const std::size_t pad = kFftSize / 2;
    padded_.assign(count + 2 * pad, 0.0f);
    for (std::size_t i = 0; i < pad; ++i) {
        padded_[i] = samples[std::min(count - 1, pad - i)];
        padded_[padded_.size() - 1 - i] = samples[count - 1 - std::min(count - 1, pad - i)];
    }
    std::copy(samples, samples + count, padded_.begin() + static_cast<std::ptrdiff_t>(pad));

    const float scale = 1.0f / std::sqrt(static_cast<float>(kFftSize));
    std::vector<float> out(frames * kMels);

    for (std::size_t f = 0; f < frames; ++f) {
        const float* begin = padded_.data() + f * kHopSize;
        for (std::size_t n = 0; n < kFftSize; ++n) block_[n] = begin[n] * window_[n];

        fft_.magnitudeReal(block_.data(), spectrum_.data());
        for (float& v : spectrum_) v *= scale;

        float* row = out.data() + f * kMels;
        bank_.apply(spectrum_.data(), row);
        for (std::size_t m = 0; m < kMels; ++m) {
            const double energy = std::max(static_cast<double>(row[m]), kFloor);
            row[m] = static_cast<float>(std::log1p(kLogMultiplier * energy));
        }
    }
    return out;
}

}  // namespace tiktak::ml
