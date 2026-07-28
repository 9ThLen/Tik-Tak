#include "analysis/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tiktak::analysis {
namespace {

// Root-mean-square of the selected values, halved. Used twice below as a
// "convincing enough" bar: high enough to ignore the grid the DP grows into
// silence, low enough to keep a genuine beat.
double halfRms(const std::vector<double>& values, const std::vector<std::size_t>& indices) {
    if (indices.empty()) return 0.0;
    double sum = 0.0;
    for (std::size_t index : indices) sum += values[index] * values[index];
    return 0.5 * std::sqrt(sum / static_cast<double>(indices.size()));
}

}  // namespace

bool TrackerConfig::valid() const { return tightness > 0.0; }

BeatTracker::BeatTracker(const TrackerConfig& config, const TempoConfig& tempo_config)
    : config_(config), tempo_(tempo_config) {}

void BeatTracker::computeLocalScore(const double* odf, std::size_t n, double period) {
    // Smoothing over a fraction of the beat period stops the DP from latching
    // onto single-frame spikes; normalising by the standard deviation makes
    // `tightness` mean the same thing regardless of input level.
    local_score_.assign(n, 0.0);
    if (n < 2) return;

    const double mean = std::accumulate(odf, odf + n, 0.0) / static_cast<double>(n);
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = odf[i] - mean;
        variance += d * d;
    }
    variance /= static_cast<double>(n - 1);   // sample variance, as in the reference
    const double spread = std::sqrt(variance);
    if (!(spread > 0.0)) return;

    const auto half = static_cast<std::ptrdiff_t>(std::max(1.0, std::round(period)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * half + 1));
    for (std::ptrdiff_t d = -half; d <= half; ++d) {
        const double z = static_cast<double>(d) * 32.0 / period;
        kernel[static_cast<std::size_t>(d + half)] = std::exp(-0.5 * z * z);
    }

    // The kernel is symmetric, so convolution and correlation coincide; this is
    // the centred ("same") part of the full convolution, zero-padded outside.
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::ptrdiff_t d = -half; d <= half; ++d) {
            const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) + d;
            if (j < 0 || j >= static_cast<std::ptrdiff_t>(n)) continue;
            sum += odf[static_cast<std::size_t>(j)] / spread *
                   kernel[static_cast<std::size_t>(d + half)];
        }
        local_score_[i] = sum;
    }
}

void BeatTracker::forward(double period) {
    const std::size_t n = local_score_.size();
    cumulative_.assign(n, 0.0);
    backlink_.assign(n, -1);

    // Candidate gaps to the previous beat: half the period to twice it. Outside
    // that range the transition penalty dominates anyway.
    const auto lo = static_cast<std::int64_t>(std::max(1.0, std::round(period / 2.0)));
    const auto hi = std::max(lo + 1, static_cast<std::int64_t>(std::round(2.0 * period)));

    const auto gap_count = static_cast<std::size_t>(hi - lo + 1);
    std::vector<double> penalty(gap_count);
    for (std::size_t g = 0; g < gap_count; ++g) {
        const double ratio = static_cast<double>(lo + static_cast<std::int64_t>(g)) / period;
        const double log_ratio = std::log(ratio);
        penalty[g] = -config_.tightness * log_ratio * log_ratio;
    }

    for (std::size_t i = 0; i < n; ++i) {
        double best_score = 0.0;
        std::size_t best_gap = 0;
        bool have_best = false;

        for (std::size_t g = 0; g < gap_count; ++g) {
            const std::int64_t previous = static_cast<std::int64_t>(i) - (lo +
                                          static_cast<std::int64_t>(g));
            // Out-of-range predecessors keep the penalty alone, with no
            // cumulative score behind them. That is what lets a sequence
            // *start*: early frames can win without paying for a predecessor
            // that does not exist.
            double score = penalty[g];
            if (previous >= 0) score += cumulative_[static_cast<std::size_t>(previous)];

            if (!have_best || score > best_score) {
                best_score = score;
                best_gap = g;
                have_best = true;
            }
        }

        const std::int64_t previous = static_cast<std::int64_t>(i) - (lo +
                                      static_cast<std::int64_t>(best_gap));
        cumulative_[i] = local_score_[i] + best_score;
        backlink_[i] = previous >= 0 ? previous : -1;
    }
}

std::size_t BeatTracker::lastBeat() const {
    const std::size_t n = cumulative_.size();
    if (n < 3) return n - 1;

    const auto argmax = [this]() {
        return static_cast<std::size_t>(
            std::max_element(cumulative_.begin(), cumulative_.end()) - cumulative_.begin());
    };

    std::vector<std::size_t> peaks;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (cumulative_[i] >= cumulative_[i - 1] && cumulative_[i] > cumulative_[i + 1]) {
            peaks.push_back(i);
        }
    }
    if (peaks.empty()) return argmax();

    const double threshold = halfRms(cumulative_, peaks);
    for (auto it = peaks.rbegin(); it != peaks.rend(); ++it) {
        if (cumulative_[*it] >= threshold) return *it;
    }
    return argmax();
}

void BeatTracker::trim() {
    if (frames_.empty()) return;

    const double threshold = halfRms(local_score_, frames_);
    const auto keep = [&](std::size_t frame) { return local_score_[frame] >= threshold; };

    const auto first = std::find_if(frames_.begin(), frames_.end(), keep);
    if (first == frames_.end()) return;   // nothing convincing anywhere: keep it all
    const auto last = std::find_if(frames_.rbegin(), frames_.rend(), keep).base();

    frames_.assign(first, last);
}

BeatResult BeatTracker::track(const double* odf, const double* times, std::size_t n, double fps,
                              double bpm) {
    BeatResult result;
    local_score_.clear();
    frames_.clear();

    if (odf == nullptr || times == nullptr || !(fps > 0.0) || !config_.valid()) return result;

    double confidence = 1.0;
    if (!(bpm > 0.0)) {
        const TempoEstimate estimate = tempo_.estimate(odf, n, fps);
        bpm = estimate.bpm;
        confidence = estimate.confidence;
    }
    result.bpm = bpm;
    result.tempo_confidence = confidence;
    if (!(bpm > 0.0)) return result;

    if (n < 3 || !std::any_of(odf, odf + n, [](double v) { return v > 0.0; })) return result;

    const double period = 60.0 * fps / bpm;
    computeLocalScore(odf, n, period);
    if (!std::any_of(local_score_.begin(), local_score_.end(), [](double v) { return v > 0.0; })) {
        return result;
    }

    forward(period);

    for (std::int64_t cursor = static_cast<std::int64_t>(lastBeat()); cursor >= 0;
         cursor = backlink_[static_cast<std::size_t>(cursor)]) {
        frames_.push_back(static_cast<std::size_t>(cursor));
    }
    std::reverse(frames_.begin(), frames_.end());

    if (config_.trim) trim();

    result.frames = frames_;
    result.beats.reserve(frames_.size());
    for (std::size_t frame : frames_) result.beats.push_back(times[frame]);

    // Read off the sequence that was kept, not off the whole array: trim() may
    // have dropped beats from either end, and the cumulative score at the last
    // surviving frame still includes everything the backtrace passed through.
    if (!frames_.empty()) {
        result.objective_per_beat =
            cumulative_[frames_.back()] / static_cast<double>(frames_.size());
    }
    return result;
}

}  // namespace tiktak::analysis
