#include "ml/beat_this.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

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

bool runChunked(const float* spectrogram, std::size_t frames, std::size_t mels,
                ChunkRunner runner, void* context, Activations& out) {
    out.beat.clear();
    out.downbeat.clear();
    if (spectrogram == nullptr || runner == nullptr || frames == 0 || mels == 0) return false;

    // -1000 is the reference's "no chunk has spoken for this frame yet". It
    // survives into the result only if the aggregation below has a hole, which
    // the choice of starts is arranged to prevent — so a -1000 in the output is
    // a bug rather than a value, and is worth being able to see.
    std::vector<float> beat(frames, -1000.0f);
    std::vector<float> downbeat(frames, -1000.0f);

    // Starts run from -border, so the first real frame is never at a chunk
    // edge, and the last start is pulled back to cover the tail exactly once.
    // That is what makes keep-first aggregation total rather than merely
    // usually total.
    const std::size_t step = kChunkFrames - 2 * kBorderFrames;
    std::vector<long long> starts;
    for (long long s = -static_cast<long long>(kBorderFrames);
         s < static_cast<long long>(frames) - static_cast<long long>(kBorderFrames);
         s += static_cast<long long>(step)) {
        starts.push_back(s);
    }
    if (starts.empty()) starts.push_back(-static_cast<long long>(kBorderFrames));
    if (frames > step) {
        starts.back() = static_cast<long long>(frames) -
                        static_cast<long long>(kChunkFrames - kBorderFrames);
    }

    std::vector<float> chunk(kChunkFrames * mels);
    std::vector<std::vector<float>> beat_chunks(starts.size());
    std::vector<std::vector<float>> downbeat_chunks(starts.size());

    for (std::size_t c = 0; c < starts.size(); ++c) {
        std::fill(chunk.begin(), chunk.end(), 0.0f);
        for (std::size_t j = 0; j < kChunkFrames; ++j) {
            const long long source = starts[c] + static_cast<long long>(j);
            if (source < 0 || source >= static_cast<long long>(frames)) continue;
            std::copy(spectrogram + static_cast<std::size_t>(source) * mels,
                      spectrogram + static_cast<std::size_t>(source + 1) * mels,
                      chunk.begin() + static_cast<std::ptrdiff_t>(j * mels));
        }
        beat_chunks[c].assign(kChunkFrames, 0.0f);
        downbeat_chunks[c].assign(kChunkFrames, 0.0f);
        if (!runner(context, chunk.data(), kChunkFrames, mels, beat_chunks[c].data(),
                    downbeat_chunks[c].data())) {
            return false;
        }
    }

    // Backwards, so an earlier chunk overwrites a later one where they overlap.
    // The reference calls this keep_first, and the direction is the whole of
    // it: written forwards, every overlap would keep the *worse* answer, the
    // one computed nearer a chunk edge.
    for (std::size_t i = starts.size(); i-- > 0;) {
        for (std::size_t j = kBorderFrames; j < kChunkFrames - kBorderFrames; ++j) {
            const long long target = starts[i] + static_cast<long long>(j);
            if (target < 0 || target >= static_cast<long long>(frames)) continue;
            beat[static_cast<std::size_t>(target)] = beat_chunks[i][j];
            downbeat[static_cast<std::size_t>(target)] = downbeat_chunks[i][j];
        }
    }

    out.beat = std::move(beat);
    out.downbeat = std::move(downbeat);
    return true;
}

}  // namespace tiktak::ml
