#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "dsp/matched.hpp"

using tiktak::dsp::findKnownSignal;

namespace {

constexpr double kRate = 48000.0;

// The metronome's beat click: a decaying sine, which is what the calibration
// screen actually plays.
std::vector<float> click(double freq = 1046.5, double decay = 0.060) {
    const auto n = static_cast<std::size_t>(decay * 4.0 * kRate);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / kRate;
        out[i] = static_cast<float>(std::sin(2.0 * M_PI * freq * t) * std::exp(-t / decay));
    }
    return out;
}

// A recording: silence, the click at `delay`, silence, plus noise at a chosen
// ratio to the click's own RMS.
std::vector<float> recording(const std::vector<float>& signal, std::size_t delay,
                             double snr_db, unsigned seed, double gain = 1.0) {
    std::vector<float> out(delay + signal.size() + static_cast<std::size_t>(kRate * 0.2), 0.0f);
    for (std::size_t i = 0; i < signal.size(); ++i) {
        out[delay + i] = static_cast<float>(signal[i] * gain);
    }

    double energy = 0.0;
    for (float v : signal) energy += static_cast<double>(v) * v;
    const double rms = std::sqrt(energy / static_cast<double>(signal.size())) * gain;

    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, rms / std::pow(10.0, snr_db / 20.0));
    for (float& v : out) v += static_cast<float>(noise(rng));
    return out;
}

// What the harness used to do: the first sample above thirty per cent of the
// window's own peak. Kept here so the comparison is against the real thing.
std::ptrdiff_t thresholdFind(const std::vector<float>& window) {
    double peak = 0.0;
    for (float v : window) peak = std::max(peak, std::fabs(static_cast<double>(v)));
    if (peak < 1e-3) return -1;
    for (std::size_t i = 0; i < window.size(); ++i) {
        if (std::fabs(static_cast<double>(window[i])) > 0.3 * peak) {
            return static_cast<std::ptrdiff_t>(i);
        }
    }
    return -1;
}

}  // namespace

TEST(MatchedFilter, FindsAnExactCopyExactly) {
    const std::vector<float> tmpl = click();
    for (std::size_t delay : {0u, 1u, 137u, 4096u, 20000u}) {
        const std::vector<float> window = recording(tmpl, delay, 200.0, 1);
        const auto found = findKnownSignal(window.data(), window.size(),
                                           tmpl.data(), tmpl.size());
        ASSERT_TRUE(found.found()) << delay;
        EXPECT_NEAR(found.offset_samples, static_cast<double>(delay), 0.01) << delay;
    }
}

TEST(MatchedFilter, StrengthReportsTheLevelItCameBackAt) {
    const std::vector<float> tmpl = click();
    // A perfect copy scores one; half the amplitude scores half. This is what
    // makes min_strength a fraction of the played level rather than a number
    // that has to be recalibrated per input gain.
    for (double gain : {1.0, 0.5, 0.25}) {
        const std::vector<float> window = recording(tmpl, 5000, 200.0, 2, gain);
        const auto found = findKnownSignal(window.data(), window.size(),
                                           tmpl.data(), tmpl.size(), 0.1);
        ASSERT_TRUE(found.found()) << gain;
        EXPECT_NEAR(found.strength, gain, 0.02) << gain;
    }
}

// The reason this exists. A relative threshold is exact when the click is the
// loudest thing in the window and useless when it is not, and "not" is the
// ordinary case in a room with the volume anywhere below deafening.
TEST(MatchedFilter, SurvivesNoiseThatDefeatsARelativeThreshold) {
    const std::vector<float> tmpl = click();
    const std::size_t delay = 9000;

    int matched_close = 0;
    int threshold_close = 0;
    for (unsigned seed = 0; seed < 20; ++seed) {
        const std::vector<float> window = recording(tmpl, delay, 0.0, seed);

        const auto found = findKnownSignal(window.data(), window.size(),
                                           tmpl.data(), tmpl.size());
        if (found.found() && std::fabs(found.offset_samples - static_cast<double>(delay)) < 240) {
            ++matched_close;   // within five milliseconds
        }
        const std::ptrdiff_t naive = thresholdFind(window);
        if (naive >= 0 && std::llabs(naive - static_cast<long long>(delay)) < 240) {
            ++threshold_close;
        }
    }
    EXPECT_GE(matched_close, 19);
    EXPECT_LE(threshold_close, 3) << "the threshold rule was expected to fail here";
}

TEST(MatchedFilter, RefusesARecordingThatDoesNotContainIt) {
    const std::vector<float> tmpl = click();

    // Silence: the speaker was muted, or the microphone was not the one that
    // hears it. A confident wrong latency is worse than none, since every beat
    // time downstream is corrected by it.
    const std::vector<float> quiet(48000, 0.0f);
    EXPECT_FALSE(findKnownSignal(quiet.data(), quiet.size(),
                                 tmpl.data(), tmpl.size()).found());

    // Noise alone, well above the click's level, still must not be mistaken
    // for it — being loud is not the same as being the signal.
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<float> hiss(48000);
    for (float& v : hiss) v = static_cast<float>(noise(rng));
    EXPECT_FALSE(findKnownSignal(hiss.data(), hiss.size(),
                                 tmpl.data(), tmpl.size()).found());
}

TEST(MatchedFilter, AnInvertedCopyIsStillTheClick) {
    // Some output stages invert. The alignment is being measured, not the
    // polarity, and a calibration that failed on half the hardware for this
    // reason would be very hard to diagnose from the symptom.
    std::vector<float> tmpl = click();
    std::vector<float> flipped = tmpl;
    for (float& v : flipped) v = -v;

    const std::vector<float> window = recording(flipped, 3000, 200.0, 4);
    const auto found = findKnownSignal(window.data(), window.size(),
                                       tmpl.data(), tmpl.size());
    ASSERT_TRUE(found.found());
    EXPECT_NEAR(found.offset_samples, 3000.0, 0.01);
}

TEST(MatchedFilter, LocatesThePeakBetweenSamples) {
    // A copy delayed by half a sample, made by averaging two neighbours. The
    // integer answer would be 5000 or 5001; the interpolated one has to land
    // between them, because that bias is otherwise subtracted from every beat.
    const std::vector<float> tmpl = click();
    std::vector<float> shifted(tmpl.size());
    for (std::size_t i = 1; i < tmpl.size(); ++i) {
        shifted[i] = 0.5f * (tmpl[i] + tmpl[i - 1]);
    }
    const std::vector<float> window = recording(shifted, 5000, 200.0, 5);
    const auto found = findKnownSignal(window.data(), window.size(),
                                       tmpl.data(), tmpl.size());
    ASSERT_TRUE(found.found());
    EXPECT_GT(found.offset_samples, 5000.05);
    EXPECT_LT(found.offset_samples, 5000.95);
}

TEST(MatchedFilter, DegenerateInputIsHarmless) {
    const std::vector<float> tmpl = click();
    const std::vector<float> window(1000, 0.0f);

    EXPECT_FALSE(findKnownSignal(nullptr, 10, tmpl.data(), tmpl.size()).found());
    EXPECT_FALSE(findKnownSignal(window.data(), window.size(), nullptr, 10).found());
    EXPECT_FALSE(findKnownSignal(window.data(), window.size(), tmpl.data(), 0).found());
    // A template longer than the recording has nowhere to sit.
    EXPECT_FALSE(findKnownSignal(window.data(), 100, tmpl.data(), tmpl.size()).found());
    // An all-zero template has no energy to normalise by.
    const std::vector<float> silent(64, 0.0f);
    EXPECT_FALSE(findKnownSignal(window.data(), window.size(),
                                 silent.data(), silent.size()).found());
}
