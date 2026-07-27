#include "dsp/matched.hpp"

#include <cmath>
#include <limits>

namespace tiktak::dsp {
namespace {

// Fits a parabola through (-1, a), (0, b), (1, c) and returns where its vertex
// sits relative to the middle point. Guarded: a flat or upward-opening triple
// has no interior maximum, and dividing by its curvature would send the offset
// somewhere arbitrary.
double refinePeak(double a, double b, double c) {
    const double curvature = a - 2.0 * b + c;
    if (!(curvature < 0.0)) return 0.0;
    const double shift = 0.5 * (a - c) / curvature;
    // A vertex further than half a sample away means the sampled maximum was
    // not the real one, which cannot happen for a correlation peak and does
    // happen for numerical noise. Keep the integer answer.
    return (shift > -0.5 && shift < 0.5) ? shift : 0.0;
}

}  // namespace

MatchResult findKnownSignal(const float* window, std::size_t window_frames,
                            const float* template_, std::size_t template_frames,
                            double min_strength) {
    MatchResult result;
    if (window == nullptr || template_ == nullptr) return result;
    if (template_frames == 0 || window_frames < template_frames) return result;

    // The template's own energy is the yardstick: correlating it against itself
    // produces exactly this, so dividing by it puts a perfect copy at 1.0 and
    // makes `min_strength` a fraction of the played level rather than a number
    // that depends on the input gain.
    double energy = 0.0;
    for (std::size_t i = 0; i < template_frames; ++i) {
        const double v = template_[i];
        energy += v * v;
    }
    if (!(energy > 0.0)) return result;

    const std::size_t positions = window_frames - template_frames + 1;
    double best = -std::numeric_limits<double>::max();
    std::size_t best_at = 0;
    double before = 0.0;
    double after = 0.0;

    for (std::size_t k = 0; k < positions; ++k) {
        double sum = 0.0;
        for (std::size_t i = 0; i < template_frames; ++i) {
            sum += static_cast<double>(window[k + i]) * static_cast<double>(template_[i]);
        }
        // Absolute value, because a speaker or an input stage may invert the
        // signal and an inverted click is still the click. The alignment is
        // what is being measured, not the polarity.
        const double score = std::fabs(sum);
        if (score > best) {
            best = score;
            best_at = k;
        }
    }

    // Re-evaluate the two neighbours of the winner rather than caching every
    // score: the loop above would otherwise have to hold the whole correlation,
    // which for a long recording is the same size again in memory for the sake
    // of two numbers.
    const auto scoreAt = [&](std::size_t k) {
        double sum = 0.0;
        for (std::size_t i = 0; i < template_frames; ++i) {
            sum += static_cast<double>(window[k + i]) * static_cast<double>(template_[i]);
        }
        return std::fabs(sum);
    };
    before = best_at > 0 ? scoreAt(best_at - 1) : best;
    after = best_at + 1 < positions ? scoreAt(best_at + 1) : best;

    const double strength = best / energy;
    if (!(strength >= min_strength)) return result;

    result.offset_samples = static_cast<double>(best_at) + refinePeak(before, best, after);
    result.strength = strength;
    return result;
}

}  // namespace tiktak::dsp
