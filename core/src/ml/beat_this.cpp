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

namespace {

// Frame indices where the logit is the maximum of the seven-frame window
// centred on it and is positive, with runs of adjacent peaks collapsed.
std::vector<std::size_t> peakFrames(const float* logits, std::size_t frames) {
    std::vector<std::size_t> out;
    if (logits == nullptr || frames == 0) return out;

    constexpr std::size_t kHalf = 3;  // a seven-frame window
    for (std::size_t f = 0; f < frames; ++f) {
        if (!(logits[f] > 0.0f)) continue;
        const std::size_t begin = f > kHalf ? f - kHalf : 0;
        const std::size_t end = std::min(frames, f + kHalf + 1);
        bool highest = true;
        for (std::size_t k = begin; k < end && highest; ++k) {
            if (logits[k] > logits[f]) highest = false;
        }
        if (!highest) continue;
        // Adjacent survivors are one plateau, not two beats.
        if (!out.empty() && f - out.back() <= 1) continue;
        out.push_back(f);
    }
    return out;
}

}  // namespace

BeatGrid pickBeats(const float* beat_logits, const float* downbeat_logits,
                   std::size_t frames, double frameRate) {
    BeatGrid grid;
    if (frameRate <= 0.0) return grid;

    const std::vector<std::size_t> beats = peakFrames(beat_logits, frames);
    grid.beats.reserve(beats.size());
    for (std::size_t f : beats) grid.beats.push_back(static_cast<double>(f) / frameRate);

    const std::vector<std::size_t> downbeats = peakFrames(downbeat_logits, frames);
    if (beats.empty() || downbeats.empty()) return grid;

    // Snap each downbeat onto its nearest beat, then drop duplicates: two
    // downbeat peaks either side of one beat are one bar line.
    // The search is in seconds rather than in frames, which looks like the
    // clumsier choice and is the right one. A downbeat peak can land exactly
    // halfway between two beats, and in frames that is an exact tie broken by
    // whichever comparison happens to be written; in seconds the same division
    // the reference performs resolves it the same way. Ties here are arbitrary
    // either way — what matters is that the two implementations are not
    // arbitrary in different directions, because then every comparison between
    // them carries a difference that means nothing.
    grid.downbeats.reserve(downbeats.size());
    for (std::size_t d : downbeats) {
        const double when = static_cast<double>(d) / frameRate;
        std::size_t nearest = 0;
        double best = -1.0;
        for (std::size_t i = 0; i < grid.beats.size(); ++i) {
            const double distance = std::fabs(grid.beats[i] - when);
            if (best < 0.0 || distance < best) {
                best = distance;
                nearest = i;
            }
        }
        grid.downbeats.push_back(grid.beats[nearest]);
    }

    // Sorted and deduplicated as a set, not as a running comparison against the
    // previous entry. The snap is not guaranteed monotonic: two downbeat peaks
    // close together can pick beats in the opposite order, and dropping the one
    // that arrives out of order would lose a real bar line rather than a
    // duplicate. Sorting first makes the two cases distinguishable.
    std::sort(grid.downbeats.begin(), grid.downbeats.end());
    grid.downbeats.erase(std::unique(grid.downbeats.begin(), grid.downbeats.end()),
                         grid.downbeats.end());
    return grid;
}

}  // namespace tiktak::ml
