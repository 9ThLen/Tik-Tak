#include "tap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tiktak::desktop {
namespace {

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 == 1) return upper;
    const auto lower = std::max_element(values.begin(), values.begin() + middle);
    return 0.5 * (*lower + upper);
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const std::size_t high = std::min(low + 1, values.size() - 1);
    const double t = position - static_cast<double>(low);
    return values[low] * (1.0 - t) + values[high] * t;
}

std::vector<double> intervals(const std::vector<double>& times) {
    std::vector<double> out;
    for (std::size_t i = 1; i < times.size(); ++i) {
        out.push_back(times[i] - times[i - 1]);
    }
    return out;
}

// The nearest beat to each tap, signed. Not a matching: a person drops taps and
// doubles them, and forcing one-to-one would turn a missed tap into an error
// about the grid, which is the opposite of what this bench is for.
std::vector<double> offsets(const std::vector<double>& taps,
                            const std::vector<double>& beats) {
    std::vector<double> out;
    out.reserve(taps.size());
    for (double tap : taps) {
        const auto after = std::lower_bound(beats.begin(), beats.end(), tap);
        double best = 0.0;
        bool have = false;
        if (after != beats.end()) {
            best = *after - tap;
            have = true;
        }
        if (after != beats.begin()) {
            const double before = *(after - 1) - tap;
            if (!have || std::abs(before) < std::abs(best)) best = before;
            have = true;
        }
        if (have) out.push_back(-best);  // tap minus beat: positive is late
    }
    return out;
}

std::size_t within(const std::vector<double>& values, double centre,
                   double tolerance) {
    std::size_t n = 0;
    for (double v : values) {
        if (std::abs(v - centre) <= tolerance) ++n;
    }
    return n;
}

}  // namespace

TapComparison compareTaps(std::vector<double> taps, std::vector<double> beats,
                          double tolerance_sec) {
    std::sort(taps.begin(), taps.end());
    std::sort(beats.begin(), beats.end());

    TapComparison out;
    out.taps = taps.size();
    if (taps.size() < 4 || beats.size() < 4) {
        out.verdict = "not enough taps to say anything";
        return out;
    }

    out.tap_period_sec = median(intervals(taps));
    out.grid_period_sec = median(intervals(beats));
    if (out.grid_period_sec > 0.0) {
        out.octave_ratio = out.tap_period_sec / out.grid_period_sec;
    }

    const std::vector<double> delta = offsets(taps, beats);
    out.median_offset_sec = median(delta);
    out.offset_spread_sec = quantile(delta, 0.75) - quantile(delta, 0.25);
    out.matched = within(delta, 0.0, tolerance_sec);
    out.matched_after_offset = delta.empty()
        ? 0.0
        : static_cast<double>(within(delta, out.median_offset_sec, tolerance_sec)) /
              static_cast<double>(delta.size());

    // Which metrical relation the two pulses stand in, if any.
    //
    // Nearest simple ratio rather than a band around 2 and 1/2: a listener who
    // taps every third beat of a waltz is in a real relation to the grid and a
    // listener whose ratio is 1.35 is in none, and a rule that only knows about
    // octaves calls the second one "the same pulse" — which is how a bench
    // starts reassuring instead of testing. 8% is the tolerance the tempo
    // statistics elsewhere in this project use for "the same tempo".
    const double ratio = out.octave_ratio;
    struct Relation { double value; const char* said; };
    static const Relation kRelations[] = {
        {1.0, nullptr},
        {2.0, "you tapped half as often as the grid — the grid is counting a "
              "faster pulse than you hear"},
        {0.5, "you tapped twice as often as the grid — the grid is counting a "
              "slower pulse than you hear"},
        {3.0, "you tapped once per bar where the grid counts three"},
        {1.0 / 3.0, "you tapped three times per grid beat"},
        {4.0, "you tapped once per bar where the grid counts four"},
        {0.25, "you tapped four times per grid beat"},
        {1.5, "you tapped in three where the grid counts two, or the reverse"},
        {2.0 / 3.0, "you tapped in two where the grid counts three, or the reverse"},
    };
    const Relation* nearest = nullptr;
    double best_distance = 0.0;
    for (const Relation& candidate : kRelations) {
        const double distance = std::abs(std::log2(ratio / candidate.value));
        if (nearest == nullptr || distance < best_distance) {
            nearest = &candidate;
            best_distance = distance;
        }
    }
    const bool related = ratio > 0.0 && best_distance <= std::log2(1.08);

    const double phase = out.grid_period_sec > 0.0
        ? std::abs(out.median_offset_sec) / out.grid_period_sec
        : 0.0;

    if (!related) {
        out.verdict = "your pulse and the grid's are not metrically related "
                      "at all";
    } else if (nearest->said != nullptr) {
        out.verdict = nearest->said;
    } else if (phase > 0.3 && phase < 0.7) {
        // Only meaningful at the same pulse: at half or double, "half a beat"
        // is a different quantity in each and the offset says nothing.
        out.verdict = "the grid sits about half a beat from where you tapped — "
                      "it is on the off-beat";
    } else if (out.matched_after_offset >= 0.7) {
        out.verdict = "the grid agrees with you";
    } else {
        out.verdict = "same pulse, but the taps scatter — listen back before "
                      "believing either side";
    }
    return out;
}

}  // namespace tiktak::desktop
