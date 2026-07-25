#include "dsp/window.hpp"

#include <gtest/gtest.h>

#include <cmath>

using tiktak::dsp::hannWindow;

TEST(HannWindow, HandlesDegenerateSizes) {
    EXPECT_TRUE(hannWindow(0).empty());
    ASSERT_EQ(hannWindow(1).size(), 1u);
    EXPECT_FLOAT_EQ(hannWindow(1)[0], 1.0f);
}

TEST(HannWindow, StartsAtZeroAndPeaksInTheMiddle) {
    const auto w = hannWindow(64);
    ASSERT_EQ(w.size(), 64u);

    EXPECT_NEAR(w.front(), 0.0f, 1e-6f);
    EXPECT_NEAR(w[32], 1.0f, 1e-6f);
    // Periodic, not symmetric: the last sample is not zero, it is one step
    // short of wrapping back to it.
    EXPECT_GT(w.back(), 0.0f);
    EXPECT_LT(w.back(), 0.01f);
}

TEST(HannWindow, IsSymmetricAboutItsPeak) {
    const auto w = hannWindow(64);
    for (std::size_t i = 1; i < 32; ++i) {
        EXPECT_NEAR(w[32 - i], w[32 + i], 1e-6f) << "at offset " << i;
    }
}

TEST(HannWindow, SatisfiesConstantOverlapAddAtHalfHop) {
    // The periodic Hann is chosen precisely so overlapping windows at hop = N/2
    // sum to a constant. If this breaks, the STFT front-end is biased.
    constexpr std::size_t n = 64;
    constexpr std::size_t hop = n / 2;
    const auto w = hannWindow(n);

    for (std::size_t i = 0; i < hop; ++i) {
        EXPECT_NEAR(w[i] + w[i + hop], 1.0f, 1e-5f) << "at index " << i;
    }
}

TEST(HannWindow, MeanIsOneHalf) {
    const auto w = hannWindow(1024);
    double sum = 0.0;
    for (float v : w) sum += v;
    EXPECT_NEAR(sum / static_cast<double>(w.size()), 0.5, 1e-6);
}
