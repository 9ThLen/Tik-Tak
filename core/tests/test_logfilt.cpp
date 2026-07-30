#include "dsp/logfilt.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dsp/dft.hpp"
#include "support.hpp"

using tiktak::dsp::LogFilterbank;
using tiktak::dsp::RealDft;
using tiktak::test::sine;

namespace {

LogFilterbank beatNetBank() {
    return LogFilterbank(1411, 22050.0, 24, 30.0, 17000.0, 440.0);
}

}  // namespace

TEST(LogFilterbank, HasTheWidthTheNetworkExpects) {
    // 136 is not a preference, it is the shape of BeatNet's first layer. The
    // first attempt at this front end produced 84, by taking a default that the
    // published code overrides, and the only thing that catches that is a
    // number written down and checked.
    EXPECT_EQ(beatNetBank().bands(), 136u);
}

TEST(LogFilterbank, EveryFilterHasUnitArea) {
    const auto bank = beatNetBank();
    // A flat spectrum of ones projects each filter onto its own weight sum, so
    // unit area shows up as every band reading exactly one.
    const std::vector<float> flat(bank.spectrumSize(), 1.0f);
    std::vector<float> bands(bank.bands(), 0.0f);
    bank.apply(flat.data(), bands.data());

    for (std::size_t b = 0; b < bands.size(); ++b) {
        EXPECT_NEAR(bands[b], 1.0f, 1e-5f) << "band " << b;
    }
}

TEST(LogFilterbank, PutsAToneInTheBandTheReferenceDoes) {
    // Pinned against the reference implementation rather than derived,
    // because the obvious derivation is wrong. Twenty-four bands an octave off 440 Hz
    // would put 110 Hz in band 44 — but down there the requested centres are
    // under a hertz apart while the transform's bins are 15.6 Hz apart, so
    // whole octaves of centres round to the same bin and collapse into one
    // filter. The bottom of this bank is one filter per bin, not one per
    // twenty-fourth of an octave, and the network was trained through exactly
    // that. An index computed from the spacing would be a test that agrees with
    // a mistake.
    const auto bank = beatNetBank();
    RealDft dft(1411);

    const std::pair<double, std::size_t> expected[] = {
        {110.0, 4}, {440.0, 25}, {1760.0, 72}, {5000.0, 108},
    };
    for (const auto& [hz, band] : expected) {
        const auto tone = sine(1411, hz, 22050.0);
        std::vector<float> spectrum(dft.spectrumSize());
        dft.magnitude(tone.data(), spectrum.data());

        std::vector<float> bands(bank.bands(), 0.0f);
        bank.apply(spectrum.data(), bands.data());

        std::size_t loudest = 0;
        for (std::size_t b = 1; b < bands.size(); ++b) {
            if (bands[b] > bands[loudest]) loudest = b;
        }
        EXPECT_EQ(loudest, band) << hz << " Hz";
    }
}

TEST(LogFilterbank, TheLowestFiltersAreOneBinWide) {
    // The consequence of the collapse above, stated on its own so that a change
    // to the rounding shows up here rather than as a model that got worse.
    const auto bank = beatNetBank();
    std::vector<float> bands(bank.bands(), 0.0f);
    std::vector<float> spectrum(bank.spectrumSize(), 0.0f);

    // Bin 3 alone: only the first filter should respond, and fully, because a
    // one-bin triangle normalised to unit area is a single weight of one.
    spectrum[3] = 1.0f;
    bank.apply(spectrum.data(), bands.data());
    EXPECT_FLOAT_EQ(bands[0], 1.0f);
    EXPECT_FLOAT_EQ(bands[1], 0.0f);
}

TEST(LogFilterbank, BandsAreOrderedLowToHigh) {
    const auto bank = beatNetBank();
    RealDft dft(1411);

    // A low tone and a high one: the low one's peak band must come first.
    std::vector<float> spectrum(dft.spectrumSize());
    std::vector<float> low(bank.bands()), high(bank.bands());

    auto peakBand = [&](double hz, std::vector<float>& bands) {
        const auto tone = sine(1411, hz, 22050.0);
        dft.magnitude(tone.data(), spectrum.data());
        bank.apply(spectrum.data(), bands.data());
        std::size_t loudest = 0;
        for (std::size_t b = 1; b < bands.size(); ++b) {
            if (bands[b] > bands[loudest]) loudest = b;
        }
        return loudest;
    };

    EXPECT_LT(peakBand(200.0, low), peakBand(2000.0, high));
}

TEST(LogFilterbank, SilenceProducesNothing) {
    const auto bank = beatNetBank();
    const std::vector<float> quiet(bank.spectrumSize(), 0.0f);
    std::vector<float> bands(bank.bands(), 1.0f);
    bank.apply(quiet.data(), bands.data());
    for (float value : bands) EXPECT_FLOAT_EQ(value, 0.0f);
}

TEST(LogFilterbank, ADifferentTransformLengthStillCoversTheRange) {
    // Nothing outside BeatNet uses this yet, but a filterbank that only works
    // at one transform length is a constant wearing a constructor.
    const LogFilterbank bank(2048, 44100.0, 12, 50.0, 8000.0, 440.0);
    EXPECT_GT(bank.bands(), 0u);
    EXPECT_EQ(bank.spectrumSize(), 1025u);

    const std::vector<float> flat(bank.spectrumSize(), 1.0f);
    std::vector<float> bands(bank.bands(), 0.0f);
    bank.apply(flat.data(), bands.data());
    for (float value : bands) EXPECT_NEAR(value, 1.0f, 1e-5f);
}
