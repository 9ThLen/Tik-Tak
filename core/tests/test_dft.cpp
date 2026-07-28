#include "dsp/dft.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "dsp/fft.hpp"
#include "support.hpp"

using tiktak::dsp::Fft;
using tiktak::dsp::RealDft;
using tiktak::test::kTwoPi;
using tiktak::test::sine;

namespace {

// The definition, evaluated term by term. Slow and obviously correct, which is
// the only thing an oracle has to be.
std::vector<double> naiveMagnitude(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<double> out(n / 2 + 1);
    for (std::size_t k = 0; k < out.size(); ++k) {
        std::complex<double> sum{0.0, 0.0};
        for (std::size_t t = 0; t < n; ++t) {
            const double angle = -kTwoPi * static_cast<double>(k) *
                                 static_cast<double>(t) / static_cast<double>(n);
            sum += std::complex<double>(std::cos(angle), std::sin(angle)) *
                   static_cast<double>(input[t]);
        }
        out[k] = std::abs(sum);
    }
    return out;
}

}  // namespace

TEST(RealDft, MatchesTheDefinitionAtAnAwkwardLength) {
    // 1411 is BeatNet's frame size and the reason this class exists. Testing a
    // convenient length instead would prove nothing about the case that matters.
    RealDft dft(1411);
    auto input = sine(1411, 440.0, 22050.0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] += 0.3f * static_cast<float>(std::sin(0.01 * static_cast<double>(i * i)));
    }

    std::vector<float> got(dft.spectrumSize());
    dft.magnitude(input.data(), got.data());
    const auto expected = naiveMagnitude(input);

    ASSERT_EQ(got.size(), expected.size());
    double peak = 0.0;
    for (double value : expected) peak = std::max(peak, value);
    for (std::size_t k = 0; k < got.size(); ++k) {
        EXPECT_NEAR(got[k], expected[k], peak * 1e-5) << "bin " << k;
    }
}

TEST(RealDft, AgreesWithTheRadixTwoFftWhereBothApply) {
    // A power of two goes through Bluestein here and through the ordinary FFT
    // there. They are the same transform, so they must be the same numbers.
    constexpr std::size_t kSize = 512;
    RealDft dft(kSize);
    Fft fft(kSize);

    auto input = sine(kSize, 1000.0, 48000.0, 0.7f);
    for (std::size_t i = 0; i < kSize; ++i) input[i] += 0.2f * ((i % 7) - 3.0f);

    std::vector<float> viaDft(dft.spectrumSize());
    std::vector<float> viaFft(fft.spectrumSize());
    dft.magnitude(input.data(), viaDft.data());
    fft.magnitudeReal(input.data(), viaFft.data());

    ASSERT_EQ(viaDft.size(), viaFft.size());
    for (std::size_t k = 0; k < viaDft.size(); ++k) {
        EXPECT_NEAR(viaDft[k], viaFft[k], 1e-2f) << "bin " << k;
    }
}

TEST(RealDft, PutsAToneInTheBinForItsFrequency) {
    constexpr std::size_t kSize = 1411;
    constexpr double kRate = 22050.0;
    RealDft dft(kSize);

    // A frequency landing exactly on a bin centre, so there is nothing to leak.
    constexpr std::size_t kBin = 100;
    const double hz = kBin * kRate / static_cast<double>(kSize);
    const auto input = sine(kSize, hz, kRate);

    std::vector<float> spectrum(dft.spectrumSize());
    dft.magnitude(input.data(), spectrum.data());

    std::size_t loudest = 0;
    for (std::size_t k = 1; k < spectrum.size(); ++k) {
        if (spectrum[k] > spectrum[loudest]) loudest = k;
    }
    EXPECT_EQ(loudest, kBin);
}

TEST(RealDft, SilenceIsSilent) {
    RealDft dft(1411);
    const std::vector<float> input(1411, 0.0f);
    std::vector<float> spectrum(dft.spectrumSize(), 1.0f);
    dft.magnitude(input.data(), spectrum.data());
    for (float value : spectrum) EXPECT_NEAR(value, 0.0f, 1e-9f);
}

TEST(RealDft, RepeatedTransformsDoNotDriftIntoEachOther) {
    // Every buffer is reused between calls, so a scratch left dirty would show
    // up as the second answer differing from the first.
    RealDft dft(1411);
    const auto input = sine(1411, 300.0, 22050.0);

    std::vector<float> first(dft.spectrumSize());
    std::vector<float> second(dft.spectrumSize());
    dft.magnitude(input.data(), first.data());
    const auto other = sine(1411, 900.0, 22050.0);
    std::vector<float> ignored(dft.spectrumSize());
    dft.magnitude(other.data(), ignored.data());
    dft.magnitude(input.data(), second.data());

    for (std::size_t k = 0; k < first.size(); ++k) EXPECT_FLOAT_EQ(first[k], second[k]);
}
