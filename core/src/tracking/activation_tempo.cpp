#include "tracking/activation_tempo.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tiktak::tracking {
namespace {

// Linear interpolation into a sequence indexed by integer position, clamping
// outside the range. The same convention as analysis/tempo.cpp and as numpy's
// interp, so the two estimators and the Python reference stay comparable.
double interpolateAt(const std::vector<double>& values, double x) {
    if (values.empty()) return 0.0;
    if (!(x > 0.0)) return values.front();
    const auto lower = static_cast<std::size_t>(x);
    if (lower + 1 >= values.size()) return values.back();
    const double frac = x - static_cast<double>(lower);
    return values[lower] + frac * (values[lower + 1] - values[lower]);
}

}  // namespace

bool ActivationTempoConfig::valid() const {
    if (!(min_bpm > 0.0) || !(min_bpm < max_bpm)) return false;
    if (!(prior_centre_bpm > 0.0)) return false;
    if (!(prior_width_octaves > 0.0)) return false;
    if (grid_size < 8) return false;
    if (!(fps > 0.0)) return false;
    if (!(window_sec > 0.0)) return false;
    if (!(min_window_sec > 0.0) || min_window_sec > window_sec) return false;
    if (!(update_interval_sec >= 0.0)) return false;
    return true;
}

ActivationTempo::ActivationTempo(const ActivationTempoConfig& config)
    : config_(config) {
    const auto size = static_cast<std::size_t>(config_.grid_size);
    grid_.resize(size);
    prior_.resize(size);
    posterior_.assign(size, 0.0);

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

    const auto frames = static_cast<std::size_t>(
        std::max(2.0, std::ceil(config_.window_sec * config_.fps)));
    ring_.assign(frames, 0.0);
    linear_.assign(frames, 0.0);
    acf_.assign(frames, 0.0);

    // Sized once, here, for the full window: the transform never sees a
    // shorter buffer, because a partly filled ring is zero-padded rather than
    // shortened. That is what makes observe() allocation-free.
    const std::size_t transform = dsp::Fft64::nextPowerOfTwo(2 * frames - 1);
    fft_ = std::make_unique<dsp::Fft64>(transform);
    scratch_re_.assign(transform, 0.0);
    scratch_im_.assign(transform, 0.0);
}

void ActivationTempo::reset() {
    std::fill(ring_.begin(), ring_.end(), 0.0);
    std::fill(posterior_.begin(), posterior_.end(), 0.0);
    head_ = 0;
    filled_ = 0;
    current_bin_ = -1;
    first_time_sec_ = 0.0;
    last_update_sec_ = 0.0;
    started_ = false;
    estimate_ = ActivationTempoEstimate{};
}

double ActivationTempo::heard_sec() const {
    return static_cast<double>(filled_) / config_.fps;
}

void ActivationTempo::observe(double time_sec, double activation) {
    if (!std::isfinite(time_sec) || !std::isfinite(activation)) return;
    if (!started_) {
        started_ = true;
        first_time_sec_ = time_sec;
        last_update_sec_ = time_sec;
        current_bin_ = -1;
    }

    const auto bin = static_cast<long long>(
        std::floor((time_sec - first_time_sec_) * config_.fps));
    if (bin < current_bin_) return;   // time went backwards; ignore rather than guess

    if (bin == current_bin_) {
        // Same bin: keep the largest. An activation is a spike a few frames
        // wide, so the mean over a bin mostly measures the bin and the latest
        // value samples the spike at whatever phase the grid happens to have.
        const std::size_t slot = (head_ + ring_.size() - 1) % ring_.size();
        ring_[slot] = std::max(ring_[slot], activation);
        return;
    }

    // Gaps are written as silence rather than skipped. The lag axis is time,
    // so closing a gap by omission would compress the history and report a
    // tempo faster than the room's — and a dropped buffer is exactly when a
    // tracker must not change its mind about the tempo.
    long long gap = bin - current_bin_ - 1;
    gap = std::min<long long>(gap, static_cast<long long>(ring_.size()));
    for (long long i = 0; i < gap; ++i) {
        ring_[head_] = 0.0;
        head_ = (head_ + 1) % ring_.size();
        if (filled_ < ring_.size()) ++filled_;
    }

    ring_[head_] = activation;
    head_ = (head_ + 1) % ring_.size();
    if (filled_ < ring_.size()) ++filled_;
    current_bin_ = bin;

    if (heard_sec() < config_.min_window_sec) return;
    if (time_sec - last_update_sec_ < config_.update_interval_sec &&
        estimate_.answered()) {
        return;
    }
    last_update_sec_ = time_sec;
    recompute();
}

void ActivationTempo::recompute() {
    const std::size_t n = filled_;
    if (n < 4) return;

    // Unwrap oldest-first. The ring is only a store; the autocorrelation needs
    // the history in order.
    const std::size_t start = (head_ + ring_.size() - n) % ring_.size();
    for (std::size_t i = 0; i < n; ++i) {
        linear_[i] = ring_[(start + i) % ring_.size()];
    }

    // Mean removal: the activation is non-negative, so its DC component would
    // otherwise swamp every real periodicity.
    const double mean =
        std::accumulate(linear_.begin(), linear_.begin() + n, 0.0) /
        static_cast<double>(n);

    const std::size_t size = fft_->size();
    std::fill(scratch_re_.begin(), scratch_re_.end(), 0.0);
    std::fill(scratch_im_.begin(), scratch_im_.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) scratch_re_[i] = linear_[i] - mean;

    fft_->forward(scratch_re_.data(), scratch_im_.data());
    for (std::size_t i = 0; i < size; ++i) {
        const double re = scratch_re_[i];
        const double im = scratch_im_[i];
        scratch_re_[i] = re * re + im * im;
        scratch_im_[i] = 0.0;
    }
    fft_->inverse(scratch_re_.data(), scratch_im_.data());

    // Unbiased, then normalised to lag zero so `confidence` is a fraction of
    // the activation's own variance and means the same thing whatever the
    // front end's scale.
    const double zero = scratch_re_[0] / static_cast<double>(n);
    if (!(zero > 0.0)) {
        estimate_ = ActivationTempoEstimate{};
        return;
    }
    for (std::size_t k = 0; k < n; ++k) {
        acf_[k] = scratch_re_[k] / static_cast<double>(n - k) / zero;
    }
    std::fill(acf_.begin() + static_cast<std::ptrdiff_t>(n), acf_.end(), 0.0);

    const double limit = static_cast<double>(n) - 1.0;
    std::size_t best = 0;
    double best_score = -1.0;
    for (std::size_t i = 0; i < grid_.size(); ++i) {
        const double lag = 60.0 * config_.fps / grid_[i];
        // A period longer than the history has no evidence for it at all: its
        // lag falls off the end of the autocorrelation, and clamping to the
        // last value would hand the slowest candidates whatever happens to be
        // there. Scoring them zero says "not measured" rather than "measured
        // and weak", which is the honest reading and keeps a fifteen-second
        // window from ever answering 40 BPM.
        const double support =
            (lag >= 1.0 && lag < limit) ? std::max(interpolateAt(acf_, lag), 0.0) : 0.0;
        posterior_[i] = support * prior_[i];
        if (posterior_[i] > best_score) {
            best_score = posterior_[i];
            best = i;
        }
    }

    if (!(best_score > 0.0)) {
        estimate_ = ActivationTempoEstimate{};
        return;
    }

    const double winner = grid_[best];

    // The runner-up, measured only outside a quarter-octave of the winner.
    // Neighbouring grid points are the same hypothesis at slightly different
    // resolution, and counting one of those as a rival would report every
    // estimate as a coin toss.
    double rival = 0.0;
    for (std::size_t i = 0; i < grid_.size(); ++i) {
        if (std::abs(std::log2(grid_[i] / winner)) < 0.25) continue;
        rival = std::max(rival, posterior_[i]);
    }

    const double lag = 60.0 * config_.fps / winner;
    estimate_.bpm = winner;
    estimate_.confidence = std::clamp(interpolateAt(acf_, lag), 0.0, 1.0);
    estimate_.octave_margin = std::clamp((best_score - rival) / best_score, 0.0, 1.0);

    for (std::size_t i = 0; i < grid_.size(); ++i) posterior_[i] /= best_score;
}

}  // namespace tiktak::tracking
