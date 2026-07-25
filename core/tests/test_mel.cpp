#include "dsp/mel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "dsp/fft.hpp"
#include "support.hpp"

using tiktak::dsp::Fft;
using tiktak::dsp::hzToMel;
using tiktak::dsp::melToHz;
using tiktak::dsp::MelFilterbank;

TEST(MelScale, RoundTrips) {
    for (double hz : {0.0, 27.5, 100.0, 440.0, 1000.0, 4000.0, 16000.0}) {
        EXPECT_NEAR(melToHz(hzToMel(hz)), hz, 1e-6) << "at " << hz << " Hz";
    }
}

TEST(MelScale, IsMonotonic) {
    double previous = -1.0;
    for (double hz = 0.0; hz <= 20000.0; hz += 100.0) {
        const double mel = hzToMel(hz);
        EXPECT_GT(mel, previous) << "at " << hz << " Hz";
        previous = mel;
    }
}

TEST(MelScale, CompressesHighFrequencies) {
    // A fixed span in Hz covers fewer mel the higher up it sits: that is the
    // compression, and it is why the filterbank's high bands are wide.
    const double lowHundredHz = hzToMel(200.0) - hzToMel(100.0);
    const double highHundredHz = hzToMel(8100.0) - hzToMel(8000.0);
    EXPECT_GT(lowHundredHz, highHundredHz * 5.0);
}

TEST(MelScale, IsRoughlyLogarithmicWellAboveTheBreakpoint) {
    // Above ~700 Hz the scale turns logarithmic, so equal octaves span roughly
    // equal mel. Below it the scale is near-linear instead — which is why an
    // octave down low spans *less* mel, not more.
    const double octaveA = hzToMel(8000.0) - hzToMel(4000.0);
    const double octaveB = hzToMel(16000.0) - hzToMel(8000.0);
    EXPECT_NEAR(octaveA, octaveB, octaveB * 0.1);

    const double lowOctave = hzToMel(200.0) - hzToMel(100.0);
    EXPECT_LT(lowOctave, octaveB);
}

class MelFilterbankTest : public ::testing::Test {
protected:
    static constexpr std::size_t kFftSize = 2048;
    static constexpr double kSampleRate = 48000.0;
    static constexpr std::size_t kBands = 81;

    MelFilterbank bank_{kFftSize, kSampleRate, kBands, 27.5, 16000.0};
};

TEST_F(MelFilterbankTest, ReportsItsShape) {
    EXPECT_EQ(bank_.bands(), kBands);
    EXPECT_EQ(bank_.spectrumSize(), kFftSize / 2 + 1);
    EXPECT_EQ(bank_.centreFrequencies().size(), kBands);
}

TEST_F(MelFilterbankTest, CentresAreMonotonicAndInRange) {
    const auto& centres = bank_.centreFrequencies();
    for (std::size_t b = 1; b < centres.size(); ++b) {
        EXPECT_GT(centres[b], centres[b - 1]) << "at band " << b;
    }
    EXPECT_GE(centres.front(), 27.5);
    EXPECT_LE(centres.back(), 16000.0);
}

TEST_F(MelFilterbankTest, ClampsMaxToNyquist) {
    MelFilterbank narrow(512, 8000.0, 20, 50.0, 16000.0);
    EXPECT_LE(narrow.centreFrequencies().back(), 4000.0);
}

TEST_F(MelFilterbankTest, BandAtOrAboveFindsTheSplitPoints) {
    const auto& centres = bank_.centreFrequencies();

    const std::size_t low = bank_.bandAtOrAbove(200.0);
    ASSERT_LT(low, bank_.bands());
    EXPECT_GE(centres[low], 200.0);
    if (low > 0) EXPECT_LT(centres[low - 1], 200.0);

    const std::size_t high = bank_.bandAtOrAbove(4000.0);
    ASSERT_LT(high, bank_.bands());
    EXPECT_GE(centres[high], 4000.0);
    if (high > 0) EXPECT_LT(centres[high - 1], 4000.0);

    EXPECT_LT(low, high);
    EXPECT_EQ(bank_.bandAtOrAbove(1e9), bank_.bands());
}

TEST_F(MelFilterbankTest, NoBandIsSilent) {
    // Every filter must touch at least one FFT bin, otherwise part of the
    // spectrum is invisible to the onset detector.
    std::vector<float> flat(bank_.spectrumSize(), 1.0f);
    std::vector<float> out(bank_.bands(), 0.0f);
    bank_.apply(flat.data(), out.data());

    for (std::size_t b = 0; b < out.size(); ++b) {
        EXPECT_GT(out[b], 0.0f) << "band " << b << " has no FFT bins";
    }
}

TEST_F(MelFilterbankTest, ToneLandsInTheBandNearestItsFrequency) {
    Fft fft(kFftSize);

    for (double freq : {110.0, 440.0, 1000.0, 5000.0}) {
        const auto tone = tiktak::test::sine(kFftSize, freq, kSampleRate);

        std::vector<float> mag(fft.spectrumSize());
        fft.magnitudeReal(tone.data(), mag.data());

        std::vector<float> bands(bank_.bands());
        bank_.apply(mag.data(), bands.data());

        const auto peak = std::max_element(bands.begin(), bands.end());
        const std::size_t peakBand = static_cast<std::size_t>(peak - bands.begin());
        const double peakCentre = bank_.centreFrequencies()[peakBand];

        // Neighbouring mel bands overlap, so the peak may sit one band off the
        // true nearest; a factor-of-1.3 window covers that without being vacuous.
        EXPECT_LT(peakCentre, freq * 1.3) << "tone at " << freq << " Hz";
        EXPECT_GT(peakCentre, freq / 1.3) << "tone at " << freq << " Hz";
    }
}

TEST_F(MelFilterbankTest, SilentSpectrumGivesSilentBands) {
    std::vector<float> zero(bank_.spectrumSize(), 0.0f);
    std::vector<float> out(bank_.bands(), 1.0f);
    bank_.apply(zero.data(), out.data());

    for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST_F(MelFilterbankTest, IsLinearInTheInput) {
    std::vector<float> spectrum(bank_.spectrumSize());
    for (std::size_t k = 0; k < spectrum.size(); ++k) {
        spectrum[k] = static_cast<float>(0.5 + 0.5 * std::sin(0.05 * static_cast<double>(k)));
    }

    std::vector<float> once(bank_.bands());
    bank_.apply(spectrum.data(), once.data());

    for (float& v : spectrum) v *= 3.0f;
    std::vector<float> thrice(bank_.bands());
    bank_.apply(spectrum.data(), thrice.data());

    for (std::size_t b = 0; b < once.size(); ++b) {
        EXPECT_NEAR(thrice[b], once[b] * 3.0f, 1e-3f) << "at band " << b;
    }
}
