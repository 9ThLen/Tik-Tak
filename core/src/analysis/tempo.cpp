#include "analysis/tempo.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tiktak::analysis {
namespace {

// Linear interpolation into a sequence indexed by integer position, clamping to
// the end values outside the range — the same convention as numpy's interp, so
// the reference implementation and this one stay comparable.
double interpolateAt(const std::vector<double>& values, double x) {
    if (values.empty()) return 0.0;
    if (!(x > 0.0)) return values.front();
    const auto lower = static_cast<std::size_t>(x);
    if (lower + 1 >= values.size()) return values.back();
    const double frac = x - static_cast<double>(lower);
    return values[lower] + frac * (values[lower + 1] - values[lower]);
}

}  // namespace

bool TempoConfig::valid() const {
    if (!(min_bpm > 0.0) || !(min_bpm < max_bpm)) return false;
    if (!(prior_centre_bpm > 0.0)) return false;
    if (!(prior_width_octaves > 0.0)) return false;
    if (grid_size < 8) return false;   // too coarse to resolve anything
    if (comb_harmonics < 1) return false;
    if (!(comb_weight_decay >= 0.0)) return false;
    return true;
}

TempoEstimator::TempoEstimator(const TempoConfig& config) : config_(config) {
    const auto size = static_cast<std::size_t>(config_.grid_size);
    grid_.resize(size);
    prior_.resize(size);
    posterior_.assign(size, 0.0);

    // Log-spaced grid: tempo perception is logarithmic, so equal spacing in
    // log2 gives equal resolution where it is musically meaningful instead of
    // crowding it all at the fast end.
    const double log_min = std::log10(config_.min_bpm);
    const double log_max = std::log10(config_.max_bpm);
    const double step = (log_max - log_min) / static_cast<double>(size - 1);
    for (std::size_t i = 0; i < size; ++i) {
        grid_[i] = std::pow(10.0, log_min + step * static_cast<double>(i));
    }
    grid_.front() = config_.min_bpm;
    grid_.back() = config_.max_bpm;

    for (std::size_t i = 0; i < size; ++i) {
        const double octaves = std::log2(grid_[i] / config_.prior_centre_bpm);
        const double z = octaves / config_.prior_width_octaves;
        prior_[i] = std::exp(-0.5 * z * z);
    }
}

void TempoEstimator::computeAutocorrelation(const double* values, std::size_t n) {
    acf_.assign(n, 0.0);
    if (n == 0) return;

    // Mean removal matters: the ODF is non-negative, so its DC component would
    // otherwise swamp every real periodicity.
    const double mean = std::accumulate(values, values + n, 0.0) / static_cast<double>(n);

    // Zero-pad past 2n-1 so the circular convolution the FFT computes equals
    // the linear one we want; without the padding, lag k would wrap around and
    // mix in the far end of the signal.
    const std::size_t size = dsp::Fft64::nextPowerOfTwo(n > 0 ? 2 * n - 1 : 2);
    if (fft_ == nullptr || fft_->size() != size) {
        fft_ = std::make_unique<dsp::Fft64>(size);
    }

    scratch_re_.assign(size, 0.0);
    scratch_im_.assign(size, 0.0);
    for (std::size_t i = 0; i < n; ++i) scratch_re_[i] = values[i] - mean;

    fft_->forward(scratch_re_.data(), scratch_im_.data());
    for (std::size_t i = 0; i < size; ++i) {
        const double re = scratch_re_[i];
        const double im = scratch_im_[i];
        scratch_re_[i] = re * re + im * im;   // spectrum times its conjugate
        scratch_im_[i] = 0.0;
    }
    fft_->inverse(scratch_re_.data(), scratch_im_.data());

    // Unbiased: lag k is the average of only n-k products, so without this the
    // tail sags and slow tempi are penalised for no musical reason.
    for (std::size_t k = 0; k < n; ++k) {
        acf_[k] = scratch_re_[k] / static_cast<double>(n - k);
    }
}

void TempoEstimator::computeCombScore(double fps) {
    // Scoring a period by its own autocorrelation alone picks up peaks that are
    // not metrical at all — with a kick/snare pattern the strongest peak often
    // sits at two-thirds or three-halves of the beat, where unlike events
    // happen to line up. Summing over multiples requires a candidate to be
    // supported at every level above it, which those peaks are not.
    const std::size_t size = grid_.size();
    const double limit = static_cast<double>(acf_.size()) - 1.0;

    std::vector<double> total(size, 0.0);
    std::vector<double> weight_total(size, 0.0);

    for (int k = 1; k <= config_.comb_harmonics; ++k) {
        const double weight = std::pow(static_cast<double>(k), -config_.comb_weight_decay);
        bool any_usable = false;

        for (std::size_t i = 0; i < size; ++i) {
            const double lag = 60.0 * fps / grid_[i] * static_cast<double>(k);
            if (lag < 1.0 || lag >= limit) continue;
            any_usable = true;
            total[i] += weight * std::max(interpolateAt(acf_, lag), 0.0);
            weight_total[i] += weight;
        }

        if (!any_usable) break;
    }

    // Normalise by the weight actually used: slow tempi have fewer multiples
    // inside the analysed span, and would otherwise be penalised for the length
    // of the recording rather than for anything musical.
    for (std::size_t i = 0; i < size; ++i) {
        posterior_[i] = weight_total[i] > 0.0 ? total[i] / weight_total[i] : 0.0;
    }
}

TempoEstimate TempoEstimator::estimate(const double* odf, std::size_t n, double fps) {
    posterior_.assign(grid_.size(), 0.0);

    TempoEstimate result;
    result.bpm = config_.prior_centre_bpm;
    result.confidence = 0.0;

    if (odf == nullptr || !(fps > 0.0) || !config_.valid()) return result;

    const bool any_positive = std::any_of(odf, odf + n, [](double v) { return v > 0.0; });
    computeAutocorrelation(odf, n);
    if (acf_.size() <= 2 || !any_positive) return result;

    computeCombScore(fps);
    for (std::size_t i = 0; i < posterior_.size(); ++i) posterior_[i] *= prior_[i];

    const auto peak_it = std::max_element(posterior_.begin(), posterior_.end());
    const double peak = *peak_it;
    if (!(peak > 0.0)) {
        std::fill(posterior_.begin(), posterior_.end(), 0.0);
        return result;
    }

    for (double& value : posterior_) value /= peak;

    const auto best = static_cast<std::size_t>(
        std::max_element(posterior_.begin(), posterior_.end()) - posterior_.begin());
    result.bpm = grid_[best];

    // Confidence is how periodic the signal actually is at the chosen tempo:
    // the autocorrelation there as a fraction of the autocorrelation at lag
    // zero, which is the signal's variance. 1.0 means the ODF repeats exactly
    // at that period, 0.0 means it does not repeat at all.
    //
    // This replaced a "peak sharpness against the rest of the grid" measure
    // that was measuring the wrong thing: white noise produces a sharp,
    // meaningless peak and scored higher by it than a clean beat did — exactly
    // backwards for a UI whose job is to distinguish "sure" from "guessing".
    if (acf_[0] > 0.0) {
        const double strength = interpolateAt(acf_, 60.0 * fps / result.bpm);
        result.confidence = std::clamp(strength / acf_[0], 0.0, 1.0);
    }
    return result;
}

std::size_t TempoEstimator::topCandidates(TempoCandidate* out, std::size_t count,
                                          double min_separation_octaves) const {
    if (out == nullptr || count == 0) return 0;

    std::vector<std::size_t> order(posterior_.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [this](std::size_t a, std::size_t b) {
        if (posterior_[a] != posterior_[b]) return posterior_[a] > posterior_[b];
        return a < b;
    });

    std::size_t written = 0;
    for (std::size_t index : order) {
        const double bpm = grid_[index];
        const bool too_close = std::any_of(out, out + written, [&](const TempoCandidate& kept) {
            return std::abs(std::log2(bpm / kept.bpm)) < min_separation_octaves;
        });
        if (too_close) continue;

        out[written].bpm = bpm;
        out[written].strength = posterior_[index];
        if (++written == count) break;
    }
    return written;
}

}  // namespace tiktak::analysis
