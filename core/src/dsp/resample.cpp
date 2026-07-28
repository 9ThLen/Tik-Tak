#include "dsp/resample.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace tiktak::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Modified Bessel function of the first kind, order zero, by its series. It
// converges quickly for the arguments a Kaiser window needs (beta <= ~10), and
// bringing in a special-function library for one curve would be a poor trade
// against the core having no dependencies at all.
double besselI0(double x) {
    double term = 1.0;
    double sum = 1.0;
    const double quarter = 0.25 * x * x;
    for (int k = 1; k < 64; ++k) {
        term *= quarter / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

// Windowed sinc, normalised so the pass band has unit gain.
//
// Kaiser at beta 5.0: about 60 dB of stop band, which is past the point where
// what folds back matters against the noise floor of any real recording, and
// short enough that the filter stays affordable.
std::vector<double> lowPass(std::size_t length, double cutoff, double beta) {
    std::vector<double> h(length);
    const double centre = 0.5 * static_cast<double>(length - 1);
    const double denominator = besselI0(beta);

    for (std::size_t n = 0; n < length; ++n) {
        const double t = static_cast<double>(n) - centre;
        const double sinc = t == 0.0 ? 2.0 * cutoff
                                     : std::sin(2.0 * kPi * cutoff * t) / (kPi * t);
        const double r = 2.0 * static_cast<double>(n) / static_cast<double>(length - 1) - 1.0;
        const double window = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) /
                              denominator;
        h[n] = sinc * window;
    }

    double sum = std::accumulate(h.begin(), h.end(), 0.0);
    if (sum != 0.0) {
        for (double& v : h) v /= sum;
    }
    return h;
}

}  // namespace

Resampler::Resampler(double fromRate, double toRate) {
    assert(fromRate > 0.0 && toRate > 0.0);

    // Reduced, so 48000/22050 is 147/320 rather than a ratio with a filter
    // thousands of times longer than it needs to be.
    const auto from = static_cast<std::size_t>(std::llround(fromRate));
    const auto to = static_cast<std::size_t>(std::llround(toRate));
    const std::size_t divisor = std::gcd(from, to);
    up_ = to / divisor;
    down_ = from / divisor;

    if (up_ == 1 && down_ == 1) return;  // nothing to do; apply() copies

    const std::size_t largest = std::max(up_, down_);
    half_ = 10 * largest;
    // Cutoff is the lower of the two Nyquist limits, in units of the upsampled
    // rate: enough to keep the whole output band and to stop everything that
    // would fold into it.
    filter_ = lowPass(2 * half_ + 1, 0.5 / static_cast<double>(largest), 5.0);
    for (double& v : filter_) v *= static_cast<double>(up_);
}

std::size_t Resampler::outputLength(std::size_t count) const {
    if (up_ == down_) return count;
    return (count * up_ + down_ - 1) / down_;
}

std::vector<float> Resampler::apply(const float* samples, std::size_t count) const {
    if (samples == nullptr || count == 0) return {};
    if (up_ == 1 && down_ == 1) return std::vector<float>(samples, samples + count);

    const std::size_t frames = outputLength(count);
    std::vector<float> out(frames, 0.0f);

    // y[n] = sum_k h[n*down + half - k*up] * x[k], which is the upsample,
    // filter, decimate chain with only the samples that survive it evaluated.
    // The +half undoes the prototype's group delay, so output n lands on input
    // time n*down/up rather than half a filter later.
    for (std::size_t n = 0; n < frames; ++n) {
        const long long centre = static_cast<long long>(n) * static_cast<long long>(down_) +
                                 static_cast<long long>(half_);

        // Taps run h[centre - k*up] for k with the index inside the filter, so
        // k is bounded rather than the whole signal being swept per output.
        long long first = (centre - static_cast<long long>(filter_.size()) + 1 +
                           static_cast<long long>(up_) - 1) /
                          static_cast<long long>(up_);
        long long last = centre / static_cast<long long>(up_);
        first = std::max<long long>(first, 0);
        last = std::min<long long>(last, static_cast<long long>(count) - 1);

        double sum = 0.0;
        for (long long k = first; k <= last; ++k) {
            const long long tap = centre - k * static_cast<long long>(up_);
            sum += filter_[static_cast<std::size_t>(tap)] * static_cast<double>(samples[k]);
        }
        out[n] = static_cast<float>(sum);
    }
    return out;
}

}  // namespace tiktak::dsp
