#include "dsp/fft.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "support.hpp"

using tiktak::dsp::Fft;
using tiktak::test::kTwoPi;

TEST(Fft, RejectsNonPowerOfTwoSizes) {
    EXPECT_TRUE(Fft::isPowerOfTwo(2));
    EXPECT_TRUE(Fft::isPowerOfTwo(1024));
    EXPECT_FALSE(Fft::isPowerOfTwo(0));
    EXPECT_FALSE(Fft::isPowerOfTwo(1));
    EXPECT_FALSE(Fft::isPowerOfTwo(1000));
}

TEST(Fft, SpectrumSizeIsHalfPlusOne) {
    Fft fft(64);
    EXPECT_EQ(fft.size(), 64u);
    EXPECT_EQ(fft.spectrumSize(), 33u);
}

TEST(Fft, ImpulseGivesFlatSpectrum) {
    constexpr std::size_t n = 64;
    Fft fft(n);

    std::vector<float> input(n, 0.0f);
    input[0] = 1.0f;

    std::vector<float> mag(fft.spectrumSize());
    fft.magnitudeReal(input.data(), mag.data());

    for (float value : mag) {
        EXPECT_NEAR(value, 1.0f, 1e-5f);
    }
}

TEST(Fft, DcInputConcentratesInBinZero) {
    constexpr std::size_t n = 64;
    Fft fft(n);

    std::vector<float> input(n, 1.0f);
    std::vector<float> mag(fft.spectrumSize());
    fft.magnitudeReal(input.data(), mag.data());

    EXPECT_NEAR(mag[0], static_cast<float>(n), 1e-3f);
    for (std::size_t k = 1; k < mag.size(); ++k) {
        EXPECT_NEAR(mag[k], 0.0f, 1e-3f);
    }
}

TEST(Fft, SinePeaksAtItsOwnBin) {
    constexpr std::size_t n = 256;
    constexpr std::size_t bin = 17;
    Fft fft(n);

    // Exactly `bin` cycles across the window, so the energy lands in one bin
    // with no leakage.
    std::vector<float> input(n);
    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(
            std::sin(kTwoPi * static_cast<double>(bin) * static_cast<double>(i) /
                     static_cast<double>(n)));
    }

    std::vector<float> mag(fft.spectrumSize());
    fft.magnitudeReal(input.data(), mag.data());

    const auto peak = std::max_element(mag.begin(), mag.end());
    EXPECT_EQ(static_cast<std::size_t>(peak - mag.begin()), bin);
    EXPECT_NEAR(*peak, static_cast<float>(n) / 2.0f, 1e-2f);
}

TEST(Fft, RoundTripReproducesInput) {
    constexpr std::size_t n = 128;
    Fft fft(n);

    std::vector<float> re(n);
    std::vector<float> im(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        re[i] = static_cast<float>(std::sin(0.3 * static_cast<double>(i)) +
                                   0.25 * std::cos(1.7 * static_cast<double>(i)));
    }
    const std::vector<float> original = re;

    fft.forward(re.data(), im.data());
    fft.inverse(re.data(), im.data());

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(re[i], original[i], 1e-4f) << "at index " << i;
        EXPECT_NEAR(im[i], 0.0f, 1e-4f) << "at index " << i;
    }
}

TEST(Fft, ParsevalHolds) {
    constexpr std::size_t n = 128;
    Fft fft(n);

    std::vector<float> input(n);
    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(std::sin(0.11 * static_cast<double>(i * i)));
    }

    double timeEnergy = 0.0;
    for (float v : input) timeEnergy += static_cast<double>(v) * v;

    std::vector<float> re(n), im(n, 0.0f);
    std::copy(input.begin(), input.end(), re.begin());
    fft.forward(re.data(), im.data());

    double freqEnergy = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        freqEnergy += static_cast<double>(re[k]) * re[k] + static_cast<double>(im[k]) * im[k];
    }
    freqEnergy /= static_cast<double>(n);

    EXPECT_NEAR(timeEnergy, freqEnergy, timeEnergy * 1e-4);
}
