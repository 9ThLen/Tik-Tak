#include "tap.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using tiktak::desktop::compareTaps;
using tiktak::desktop::TapComparison;

constexpr double kPeriod = 0.5;   // 120 BPM
constexpr double kTolerance = 0.07;

std::vector<double> grid(std::size_t count, double period = kPeriod,
                         double start = 0.0) {
    std::vector<double> out;
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(start + static_cast<double>(i) * period);
    }
    return out;
}

// A person, not a machine: consistently early by `lead`, wobbling by `jitter`,
// and missing one tap in eight. Anything the bench concludes has to survive
// this, because this is the only kind of input it will ever get.
std::vector<double> tapping(const std::vector<double>& at, double lead,
                            double jitter, unsigned seed = 7u) {
    std::vector<double> out;
    for (std::size_t i = 0; i < at.size(); ++i) {
        if (i % 8 == 5) continue;  // missed one
        seed = seed * 1664525u + 1013904223u;
        const double wobble =
            jitter * (static_cast<double>(seed >> 8) / 8388607.0 - 1.0);
        out.push_back(at[i] - lead + wobble);
    }
    return out;
}

TEST(TapBench, AgreesWithATrackerThatIsRight) {
    const auto beats = grid(64);
    const auto result = compareTaps(tapping(beats, 0.045, 0.030), beats, kTolerance);

    EXPECT_NEAR(result.octave_ratio, 1.0, 0.05);
    EXPECT_NEAR(result.median_offset_sec, -0.045, 0.020)
        << "the listener's own anticipation should be measured, not judged";
    EXPECT_GT(result.matched_after_offset, 0.8);
    EXPECT_EQ(result.verdict, "the grid agrees with you");
}

TEST(TapBench, CallsOutAGridOnTheOffBeat) {
    // The failure that a beat F-measure of zero cannot distinguish from any
    // other failure, and the one a listener notices instantly.
    const auto beats = grid(64, kPeriod, kPeriod * 0.5);
    const auto heard = grid(64);
    const auto result = compareTaps(tapping(heard, 0.045, 0.030), beats, kTolerance);

    EXPECT_NEAR(result.octave_ratio, 1.0, 0.05) << "the pulse is the same";
    EXPECT_NE(result.verdict.find("off-beat"), std::string::npos)
        << "verdict was: " << result.verdict;
}

TEST(TapBench, CallsOutHalfAndDoubleTempoSeparately) {
    const auto slow = grid(32, kPeriod * 2.0);
    const auto fast = grid(64, kPeriod);

    const auto tapped_slow = compareTaps(tapping(slow, 0.04, 0.03), fast, kTolerance);
    EXPECT_NEAR(tapped_slow.octave_ratio, 2.0, 0.15);
    EXPECT_NE(tapped_slow.verdict.find("faster pulse"), std::string::npos)
        << "verdict was: " << tapped_slow.verdict;

    const auto tapped_fast = compareTaps(tapping(fast, 0.04, 0.03), slow, kTolerance);
    EXPECT_NEAR(tapped_fast.octave_ratio, 0.5, 0.05);
    EXPECT_NE(tapped_fast.verdict.find("slower pulse"), std::string::npos)
        << "verdict was: " << tapped_fast.verdict;
}

TEST(TapBench, AConstantOffsetIsNotCalledAFailure) {
    // Reaction time and output latency both look like this. If the bench
    // reported them as tracker error, every honest run would fail and the
    // bench would be discarded — which is how a measurement stops being used.
    const auto beats = grid(64);
    const auto result = compareTaps(tapping(beats, 0.120, 0.020), beats, kTolerance);

    EXPECT_LT(static_cast<double>(result.matched) /
                  static_cast<double>(result.taps), 0.5)
        << "raw agreement should indeed be poor at 120 ms of lead";
    EXPECT_GT(result.matched_after_offset, 0.8)
        << "and removing the listener's own constant should recover it";
    EXPECT_EQ(result.verdict, "the grid agrees with you");
}

TEST(TapBench, SaysNothingRatherThanGuessFromFourTaps) {
    const auto beats = grid(64);
    const auto result = compareTaps({1.0, 2.0}, beats, kTolerance);
    EXPECT_EQ(result.taps, 2u);
    EXPECT_EQ(result.verdict, "not enough taps to say anything");
}

TEST(TapBench, AnUnrelatedPulseIsNotDressedUpAsAnOctave) {
    const auto beats = grid(64, kPeriod);
    const auto heard = grid(64, kPeriod * 1.35);   // neither half nor double
    const auto result = compareTaps(tapping(heard, 0.04, 0.02), beats, kTolerance);
    EXPECT_NE(result.verdict.find("not metrically related"), std::string::npos)
        << "verdict was: " << result.verdict;
}

}  // namespace
