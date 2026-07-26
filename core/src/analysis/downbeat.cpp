#include "analysis/downbeat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "dsp/chroma.hpp"

namespace tiktak::analysis {
namespace {

constexpr std::size_t kChroma = dsp::ChromaFilterbank::kBins;
constexpr double kMaxDouble = std::numeric_limits<double>::max();

// An exact (apart from the final public projection back to one double)
// representation of a positive product. frexp keeps the multiplication away
// from overflow, while fma retains the rounding residual. This is portable to
// MSVC, where long double has no more precision than double.
struct ProductKey {
    bool positive = false;
    int exponent = 0;
    double high = 0.0;
    double low = 0.0;
};

ProductKey productKey(double a, double b) {
    ProductKey key;
    if (!(a > 0.0) || !(b > 0.0)) return key;

    int a_exponent = 0;
    int b_exponent = 0;
    const double a_mantissa = std::frexp(a, &a_exponent);
    const double b_mantissa = std::frexp(b, &b_exponent);
    key.high = a_mantissa * b_mantissa;
    key.low = std::fma(a_mantissa, b_mantissa, -key.high);
    key.exponent = a_exponent + b_exponent;

    // The rounded high part can equal 0.5 while the exact product lies just
    // below it. Include the residual in that boundary decision so exponent
    // comparison remains an ordering of the mathematical products.
    if (key.high < 0.5 || (key.high == 0.5 && key.low < 0.0)) {
        key.high *= 2.0;
        key.low *= 2.0;
        --key.exponent;
    }
    key.positive = true;
    return key;
}

int compareProducts(const ProductKey& a, const ProductKey& b) {
    if (a.positive != b.positive) return a.positive ? 1 : -1;
    if (!a.positive) return 0;
    if (a.exponent != b.exponent) return a.exponent > b.exponent ? 1 : -1;
    if (a.high != b.high) return a.high > b.high ? 1 : -1;
    if (a.low != b.low) return a.low > b.low ? 1 : -1;
    return 0;
}

int signOfExactSum(double a, double b) {
    const double sum = a + b;
    if (sum > 0.0) return 1;
    if (sum < 0.0) return -1;

    // TwoSum's residual distinguishes an exact tie from opposite terms whose
    // tiny remainder rounded to zero.
    const double virtual_b = sum - a;
    const double error = (a - (sum - virtual_b)) + (b - virtual_b);
    if (error > 0.0) return 1;
    if (error < 0.0) return -1;
    return 0;
}

double roundedSubnormal(const ProductKey& value) {
    // In units of DBL_TRUE_MIN, a subnormal double is an integer in
    // [0, 2^52]. Scale both exact product components into those units before
    // rounding so `high + low` cannot discard which side of a halfway point
    // the exact product lies on.
    const int shift = value.exponent + 1074;
    const double high_units = std::scalbn(value.high, shift);
    const double low_units = std::scalbn(value.low, shift);
    double rounded = std::nearbyint(high_units);
    const double delta_high = high_units - rounded;

    const int above_half =
        signOfExactSum(delta_high - 0.5, low_units);
    const int below_half =
        signOfExactSum(delta_high + 0.5, low_units);
    const bool rounded_is_even = std::fmod(rounded, 2.0) == 0.0;
    if (above_half > 0 || (above_half == 0 && !rounded_is_even)) {
        rounded += 1.0;
    } else if (below_half < 0 ||
               (below_half == 0 && !rounded_is_even)) {
        rounded -= 1.0;
    }
    return std::scalbn(rounded, -1074);
}

double saturatedValue(const ProductKey& value) {
    if (!value.positive) return 0.0;
    if (value.exponent < -1074) return 0.0;
    if (value.exponent <= -1022) return roundedSubnormal(value);

    const double projected =
        std::scalbn(value.high + value.low, value.exponent);
    return std::isfinite(projected) ? projected : kMaxDouble;
}

double saturatedProductDifference(const ProductKey& higher,
                                  const ProductKey& lower) {
    if (compareProducts(higher, lower) <= 0) return 0.0;
    if (!lower.positive) return saturatedValue(higher);

    // CompareProducts has established that higher.exponent >= lower.exponent.
    // Scale both exact two-part mantissas to the winner's exponent, subtract
    // there (where nothing can overflow), and only then project the margin to
    // the public finite double scale.
    const int shift = lower.exponent - higher.exponent;
    const double lower_high = std::scalbn(lower.high, shift);
    const double lower_low = std::scalbn(lower.low, shift);
    const double scaled =
        (higher.high - lower_high) + (higher.low - lower_low);
    if (!(scaled > 0.0)) return 0.0;  // a positive difference below double range

    const double value = std::scalbn(scaled, higher.exponent);
    return std::isfinite(value) ? value : kMaxDouble;
}

// Error-free summation of double components. This is the standard floating-
// point expansion behind recipes such as Python's math.fsum: FastTwoSum keeps
// the part rounded out of each addition instead of losing it. Products enter as
// both their rounded high part and the fma residual, so equal large terms can
// cancel before a much smaller, genuine contrast is projected back to double.
class ExpansionSum {
public:
    void add(double value) {
        if (value == 0.0) return;

        std::size_t write = 0;
        for (double part : partials_) {
            if (std::abs(value) < std::abs(part)) std::swap(value, part);
            const double high = value + part;
            const double low = part - (high - value);
            if (low != 0.0) partials_[write++] = low;
            value = high;
        }
        partials_.resize(write);
        if (value != 0.0) partials_.push_back(value);
    }

    void addProduct(double a, double b) {
        const double high = a * b;
        add(high);
        add(std::fma(a, b, -high));
    }

    double value() const {
        // The expansion is ordered from the smallest non-overlapping component
        // to the largest, which is the stable order for the final projection.
        double total = 0.0;
        for (double part : partials_) total += part;
        return total;
    }

private:
    std::vector<double> partials_;
};

struct ScaledContrast {
    int sign = 0;
    ProductKey magnitude;
};

ScaledContrast scaledQuotient(double numerator, double denominator,
                              int scale_exponent) {
    ScaledContrast result;
    if (numerator == 0.0 || !(denominator > 0.0)) return result;

    result.sign = numerator > 0.0 ? 1 : -1;
    int numerator_exponent = 0;
    const double numerator_mantissa =
        std::frexp(std::abs(numerator), &numerator_exponent);
    const double quotient = numerator_mantissa / denominator;
    int quotient_exponent = 0;
    result.magnitude.high = std::frexp(quotient, &quotient_exponent);
    result.magnitude.exponent =
        numerator_exponent + quotient_exponent + scale_exponent;
    result.magnitude.positive = true;
    return result;
}

int compareContrasts(const ScaledContrast& a, const ScaledContrast& b) {
    if (a.sign != b.sign) return a.sign > b.sign ? 1 : -1;
    if (a.sign == 0) return 0;
    const int magnitude_order =
        compareProducts(a.magnitude, b.magnitude);
    return a.sign > 0 ? magnitude_order : -magnitude_order;
}

double saturatedProductSum(ProductKey a, ProductKey b) {
    if (!a.positive) return saturatedValue(b);
    if (!b.positive) return saturatedValue(a);
    if (a.exponent < b.exponent) std::swap(a, b);

    const int shift = b.exponent - a.exponent;
    const double scaled =
        (a.high + a.low) +
        (std::scalbn(b.high, shift) + std::scalbn(b.low, shift));
    const double value = std::scalbn(scaled, a.exponent);
    return std::isfinite(value) ? value : kMaxDouble;
}

double saturatedContrastDifference(const ScaledContrast& higher,
                                   const ScaledContrast& lower) {
    if (compareContrasts(higher, lower) <= 0) return 0.0;
    if (higher.sign > 0 && lower.sign >= 0) {
        return lower.sign == 0
            ? saturatedValue(higher.magnitude)
            : saturatedProductDifference(higher.magnitude, lower.magnitude);
    }
    if (higher.sign <= 0 && lower.sign < 0) {
        return higher.sign == 0
            ? saturatedValue(lower.magnitude)
            : saturatedProductDifference(lower.magnitude, higher.magnitude);
    }
    return saturatedProductSum(higher.magnitude, lower.magnitude);
}

// Standardises a cue in place: zero mean, unit spread. A cue that never varies
// comes back all zeros, which is the right answer — a constant carries no
// information about where the bar line is, whatever its magnitude.
void standardise(std::vector<double>& v) {
    if (v.empty()) return;

    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());

    double variance = 0.0;
    for (double x : v) variance += (x - mean) * (x - mean);
    variance /= static_cast<double>(v.size());

    const double sd = std::sqrt(variance);
    if (!(sd > 0.0)) {
        std::fill(v.begin(), v.end(), 0.0);
        return;
    }
    for (double& x : v) x = (x - mean) / sd;
}

// Returns the absolute scale a salience backend supplied without changing it.
// The explicit finite check matters before min/max: NaN compares false in both
// directions and could otherwise turn a corrupt model output into an arbitrary
// phase decision.
bool salienceRange(const std::vector<double>& v, double& lowest, double& range,
                   int& weight_exponent) {
    if (v.empty() || !std::isfinite(v.front())) return false;

    lowest = v.front();
    double highest = v.front();
    for (double value : v) {
        if (!std::isfinite(value)) return false;
        lowest = std::min(lowest, value);
        highest = std::max(highest, value);
    }

    // If max - min fits, keep the ordinary path bit-for-bit unchanged. If it
    // does not, the weights below are formed after a common power-of-two scale:
    // unlike saturating each subtraction at DBL_MAX, that does not collapse
    // ordinary middle and high levels to the same ceiling. The exponent is
    // restored in diagnostics and ProductKey before any public projection.
    const bool difference_overflows =
        lowest < 0.0 && highest > kMaxDouble + lowest;
    weight_exponent = 0;
    if (difference_overflows) {
        const double largest =
            std::max(std::abs(lowest), std::abs(highest));
        (void)std::frexp(largest, &weight_exponent);
        range = kMaxDouble;
    } else {
        range = highest - lowest;
    }
    return true;
}

bool meterEligible(std::size_t beat_count, const MeterCandidate& meter,
                   int min_bars) {
    const auto beats_per_bar =
        static_cast<std::size_t>(meter.beats_per_bar);
    return static_cast<std::size_t>(min_bars) <=
           beat_count / beats_per_bar;
}

bool numericallyResolved(double range, std::size_t count,
                         const DownbeatConfig& config) {
    // A backend's calibrated thresholds define its smallest meaningful unit.
    // Refuse a span so much wider that aggregation could move that unit below
    // double precision. The count allowance covers a linear accumulation and
    // eight guard bits keep rounding noise comfortably below a decision.
    double resolution = kMaxDouble;
    bool have_resolution = false;
    std::vector<int> eligible_meter_lengths;
    for (const MeterCandidate& meter : config.meters) {
        if (meterEligible(count, meter, config.min_bars) &&
            std::find(eligible_meter_lengths.begin(),
                      eligible_meter_lengths.end(),
                      meter.beats_per_bar) == eligible_meter_lengths.end()) {
            eligible_meter_lengths.push_back(meter.beats_per_bar);
        }
    }
    if (config.min_phase_margin == 0.0 ||
        (eligible_meter_lengths.size() > 1 &&
         config.min_meter_margin == 0.0)) {
        return false;
    }

    const auto consider = [&](double candidate) {
        if (candidate > 0.0 && std::isfinite(candidate)) {
            resolution = std::min(resolution, candidate);
            have_resolution = true;
        }
    };
    consider(config.min_salience_range);
    consider(config.min_phase_margin);
    if (config.min_meter_margin > 0.0) {
        for (const MeterCandidate& meter : config.meters) {
            if (!meterEligible(count, meter, config.min_bars)) continue;
            const double candidate =
                config.min_meter_margin / meter.prior;
            if (candidate == 0.0) return false;
            consider(candidate);
        }
    }
    if (!have_resolution) return false;

    int count_exponent = 0;
    (void)std::frexp(static_cast<double>(count), &count_exponent);
    constexpr int kGuardBits = 8;
    const int headroom =
        std::numeric_limits<double>::digits - count_exponent - kGuardBits;
    if (headroom < 0) return false;
    const double largest_safe_range =
        std::scalbn(resolution, headroom);
    return range <= largest_safe_range;
}

// First frame at or after `t`, in a sorted time array.
std::size_t frameAtOrAfter(const double* times, std::size_t n, double t) {
    const auto it = std::lower_bound(times, times + n, t);
    return static_cast<std::size_t>(it - times);
}

}  // namespace

bool DownbeatConfig::valid() const {
    if (meters.empty()) return false;
    for (const MeterCandidate& m : meters) {
        if (m.beats_per_bar < 2) return false;
        if (!(m.prior > 0.0) || !std::isfinite(m.prior)) return false;
    }
    if (!(low_weight >= 0.0) || !std::isfinite(low_weight) ||
        !(accent_weight >= 0.0) || !std::isfinite(accent_weight) ||
        !(harmony_weight >= 0.0) || !std::isfinite(harmony_weight)) {
        return false;
    }
    if (!(low_weight > 0.0 || accent_weight > 0.0 || harmony_weight > 0.0)) return false;
    if (!(window_before >= 0.0) || !std::isfinite(window_before) ||
        !(window_after > 0.0) || !std::isfinite(window_after)) {
        return false;
    }
    if (window_after > 1.0) return false;
    if (min_bars < 2) return false;
    if (!(min_salience_range >= 0.0) || !std::isfinite(min_salience_range)) return false;
    if (!(min_phase_margin >= 0.0) || !std::isfinite(min_phase_margin)) return false;
    if (!(min_meter_margin >= 0.0) || !std::isfinite(min_meter_margin)) return false;
    return true;
}

std::vector<BeatFeature> beatFeatures(const BeatFeatureInput& input,
                                      const DownbeatConfig& config) {
    std::vector<BeatFeature> out;
    if (input.beats == nullptr || input.beat_count == 0) return out;
    if (input.frame_times == nullptr || input.frame_count == 0) return out;

    out.resize(input.beat_count);

    std::vector<float> chroma_now(kChroma, 0.0f);
    std::vector<float> chroma_prev(kChroma, 0.0f);
    bool have_prev = false;

    for (std::size_t i = 0; i < input.beat_count; ++i) {
        const double beat = input.beats[i];

        // The gap to the next beat sets the scale of everything below. The last
        // beat has no next one, so it borrows the previous gap; if there is
        // only one beat at all there is no meter to find anyway.
        double gap = 0.0;
        if (i + 1 < input.beat_count) {
            gap = input.beats[i + 1] - beat;
        } else if (i > 0) {
            gap = beat - input.beats[i - 1];
        }
        if (!(gap > 0.0)) gap = 0.5;

        BeatFeature& f = out[i];
        f.time_sec = beat;

        const double from = beat - config.window_before * gap;
        const double to = beat + config.window_after * gap;

        std::size_t k = frameAtOrAfter(input.frame_times, input.frame_count, from);
        for (; k < input.frame_count && input.frame_times[k] <= to; ++k) {
            // The peak, not the mean: an onset is an event, and averaging it
            // over a window mostly measures how wide the window is.
            if (input.odf_low != nullptr) f.low = std::max(f.low, input.odf_low[k]);
            if (input.odf_full != nullptr) f.accent = std::max(f.accent, input.odf_full[k]);
        }

        if (input.chroma == nullptr) continue;

        // Harmony is averaged over the whole beat, not peaked over a window:
        // a chord is a state that persists, so more of the beat is more
        // evidence, and the note that happens to be loudest is not the chord.
        std::fill(chroma_now.begin(), chroma_now.end(), 0.0f);
        std::size_t counted = 0;
        std::size_t c = frameAtOrAfter(input.frame_times, input.frame_count, beat);
        for (; c < input.frame_count && input.frame_times[c] < beat + gap; ++c) {
            const float* frame = input.chroma + c * kChroma;
            for (std::size_t b = 0; b < kChroma; ++b) chroma_now[b] += frame[b];
            ++counted;
        }

        if (counted == 0) continue;

        if (have_prev) {
            f.harmonic_change = dsp::chromaDistance(chroma_prev.data(), chroma_now.data());
        }
        std::swap(chroma_prev, chroma_now);
        have_prev = true;
    }

    return out;
}

std::vector<double> cueSalience(const std::vector<BeatFeature>& features,
                                const DownbeatConfig& config) {
    const std::size_t n = features.size();
    if (n == 0 || !config.valid()) return {};

    std::vector<double> low(n);
    std::vector<double> accent(n);
    std::vector<double> harmony(n);
    for (std::size_t i = 0; i < n; ++i) {
        low[i] = features[i].low;
        accent[i] = features[i].accent;
        // Floored, not standardised: noise below the floor is discarded and
        // what remains keeps its absolute meaning. See DownbeatConfig.
        harmony[i] = std::max(features[i].harmonic_change -
                              DownbeatConfig::kHarmonyFloor, 0.0);
    }
    // The onset cues are standardised because their units are arbitrary; the
    // harmony cue is left alone because its units are not. See the weights in
    // DownbeatConfig.
    standardise(low);
    standardise(accent);

    std::vector<double> salience(n);
    for (std::size_t i = 0; i < n; ++i) {
        salience[i] = config.low_weight * low[i] + config.accent_weight * accent[i] +
                      config.harmony_weight * DownbeatConfig::kHarmonyScale * harmony[i];
    }
    // No final normalisation: low and accent were put in this backend's chosen
    // units above, while harmony has an absolute cosine-distance scale. The
    // resolver sees that actual mixture, applies its range gate once in those
    // units, and never turns a tiny harmonic ripple into full-scale evidence.
    return salience;
}

DownbeatResult findDownbeats(const std::vector<BeatFeature>& features,
                             const DownbeatConfig& config) {
    std::vector<double> times(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) times[i] = features[i].time_sec;
    return resolveMeter(cueSalience(features, config), times, config);
}

DownbeatResult resolveMeter(const std::vector<double>& salience,
                            const std::vector<double>& beat_times,
                            const DownbeatConfig& config) {
    DownbeatResult result;
    const std::size_t n = salience.size();
    if (n == 0 || n != beat_times.size() || !config.valid()) return result;

    // Do not normalise a backend we do not own. A periodic difference of a few
    // millionths is still a few millionths of evidence, not a unit-variance bar
    // pattern. Subtracting the minimum only removes an irrelevant DC offset; it
    // preserves every contrast and its absolute scale.
    double lowest = 0.0;
    double range = 0.0;
    int weight_exponent = 0;
    if (!salienceRange(salience, lowest, range, weight_exponent) ||
        !(range >= config.min_salience_range) || !(range > 0.0)) {
        return result;
    }
    if (!numericallyResolved(range, n, config)) return result;
    std::vector<double> weight(n);
    if (weight_exponent == 0) {
        for (std::size_t i = 0; i < n; ++i) {
            weight[i] = salience[i] - lowest;
        }
    } else {
        const double scaled_lowest = std::scalbn(lowest, -weight_exponent);
        for (std::size_t i = 0; i < n; ++i) {
            weight[i] =
                std::scalbn(salience[i], -weight_exponent) - scaled_lowest;
        }
    }

    // Bound every later product and expansion component without changing the
    // backend's public scale. A power-of-two operation is exact for normal
    // values; the exponent is carried into ProductKey and all diagnostics.
    const double largest_weight =
        *std::max_element(weight.begin(), weight.end());
    int arithmetic_exponent = 0;
    (void)std::frexp(largest_weight, &arithmetic_exponent);
    for (double& value : weight) {
        value = std::scalbn(value, -arithmetic_exponent);
    }
    weight_exponent += arithmetic_exponent;

    // A single double cannot always hold the whole affine span and its finest
    // distinctions at once (for example -DBL_MAX, -1 and 0). Silently merging
    // two backend levels can manufacture a periodic contrast, so verify that
    // this representation is order-preserving on the supplied data. Returning
    // no answer is safer than claiming a meter that exists only after rounding.
    std::vector<std::pair<double, double>> levels;
    levels.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        levels.emplace_back(salience[i], weight[i]);
    }
    std::sort(levels.begin(), levels.end());
    for (std::size_t i = 1; i < levels.size(); ++i) {
        if (levels[i - 1].first < levels[i].first &&
            !(levels[i - 1].second < levels[i].second)) {
            return result;
        }
    }

    struct RankedMeter {
        MeterScore reported;
        ProductKey key;
    };
    std::vector<RankedMeter> ranked;
    ProductKey best_key;
    double best_reported_score = 0.0;
    bool have_best = false;

    for (const MeterCandidate& meter : config.meters) {
        const auto m = static_cast<std::size_t>(meter.beats_per_bar);
        if (!meterEligible(n, meter, config.min_bars)) continue;

        MeterScore entry;
        entry.beats_per_bar = meter.beats_per_bar;

        std::vector<ScaledContrast> contrast(m);
        bool scored = false;

        for (std::size_t p = 0; p < m; ++p) {
            const std::size_t in_count = 1 + (n - 1 - p) / m;
            const std::size_t out_count = n - in_count;
            if (in_count == 0 || out_count == 0) continue;

            // mean(in) - mean(out), written over the common denominator
            // in_count*out_count. Integer coefficients cancel identical large
            // levels before division; ExpansionSum retains the small residual.
            const double in_coefficient = static_cast<double>(out_count);
            const double out_coefficient = -static_cast<double>(in_count);
            ExpansionSum numerator;
            for (std::size_t i = 0; i < n; ++i) {
                numerator.addProduct(
                    weight[i], i % m == p ? in_coefficient : out_coefficient);
            }

            // Contrast, not total: with a sum, a two-beat bar would win every
            // time simply by claiming half the beats instead of a quarter.
            const double denominator =
                static_cast<double>(in_count) * static_cast<double>(out_count);
            contrast[p] =
                scaledQuotient(numerator.value(), denominator,
                               weight_exponent);
            scored = true;
        }
        if (!scored) continue;

        // Strictly greater, scanning upwards, so the earliest phase wins a tie
        // and the same audio always produces the same bar lines.
        std::size_t best_phase = 0;
        for (std::size_t p = 1; p < m; ++p) {
            if (compareContrasts(contrast[p], contrast[best_phase]) > 0) {
                best_phase = p;
            }
        }
        const ScaledContrast& best_contrast = contrast[best_phase];
        entry.phase = static_cast<int>(best_phase);

        // The rival is the best *other* place the bar line could go. Kept
        // signed: when every other phase scores well below zero the answer is
        // unambiguous, and that deserves to show up as a large margin.
        ScaledContrast runner_up = best_contrast;
        bool have_rival = false;
        for (std::size_t p = 0; p < m; ++p) {
            if (p == best_phase) continue;
            if (!have_rival ||
                compareContrasts(contrast[p], runner_up) > 0) {
                runner_up = contrast[p];
                have_rival = true;
            }
        }

        // A negative contrast says the chosen beats are quieter than the ones
        // around them. That is not a weak bar line, it is the wrong answer, and
        // scaling it by a prior would make the least likely meter look best.
        ProductKey positive_contrast;
        if (best_contrast.sign > 0) {
            positive_contrast = best_contrast.magnitude;
        }
        ProductKey entry_key =
            productKey(positive_contrast.high, meter.prior);
        if (entry_key.positive) {
            entry_key.exponent += positive_contrast.exponent;
        }
        entry.score = saturatedValue(entry_key);
        ranked.push_back({entry, entry_key});

        if (!have_best || compareProducts(entry_key, best_key) > 0) {
            have_best = true;
            best_key = entry_key;
            best_reported_score = entry.score;
            result.beats_per_bar = entry.beats_per_bar;
            result.phase = entry.phase;
            result.strength = saturatedValue(positive_contrast);
            result.phase_margin =
                saturatedContrastDifference(best_contrast, runner_up);
        }
    }

    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const RankedMeter& a, const RankedMeter& b) {
                         return compareProducts(a.key, b.key) > 0;
                     });
    result.candidates.reserve(ranked.size());
    for (const RankedMeter& candidate : ranked) {
        result.candidates.push_back(candidate.reported);
    }

    if (!have_best) return result;

    // How much better the winning meter is than the best of the others.
    //
    // This has to be a separate number from the phase margin and cannot be
    // derived from it: within one meter every rival phase has already conceded
    // the bar length, so a piece can be entirely unambiguous about where its
    // three-beat bars start while four fits it very nearly as well. Measuring
    // only the first produced confidently wrong meters — a 4/4 track read as
    // three with a large phase margin, which is the observation that put this
    // here.
    //
    // Mathematical score products rather than raw contrasts, so the prior that
    // picked the winner is the same quantity being compared. The public
    // diagnostic may saturate at DBL_MAX; ProductKey keeps the ordering and
    // difference independent of that projection. With one meter in the running
    // there is no rival to lose to, and the winner keeps its whole score.
    result.meter_margin = best_reported_score;
    for (const RankedMeter& other : ranked) {
        if (other.reported.beats_per_bar == result.beats_per_bar) continue;
        result.meter_margin = other.key.positive
            ? saturatedProductDifference(best_key, other.key)
            : best_reported_score;
        break;  // sorted best first, so the first other meter is the rival
    }

    for (std::size_t i = static_cast<std::size_t>(result.phase); i < n;
         i += static_cast<std::size_t>(result.beats_per_bar)) {
        result.downbeats.push_back(beat_times[i]);
    }
    return result;
}

}  // namespace tiktak::analysis
