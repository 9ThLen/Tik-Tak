#include "dsp/odf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "support.hpp"

using tiktak::dsp::Odf;
using tiktak::dsp::OdfConfig;
using tiktak::dsp::OdfFrame;

namespace {

constexpr double kSampleRate = 48000.0;

OdfConfig testConfig() {
    OdfConfig cfg;
    cfg.sampleRate = kSampleRate;
    cfg.frameSize = 1024;
    cfg.hopSize = 256;
    cfg.melBands = 40;
    return cfg;
}

std::vector<OdfFrame> run(Odf& odf, const std::vector<float>& input) {
    std::vector<OdfFrame> frames;
    odf.process(input.data(), input.size(),
                [&](const OdfFrame& frame) { frames.push_back(frame); });
    return frames;
}

// Index of the frame whose timestamp is closest to `seconds`.
std::size_t frameNearest(const std::vector<OdfFrame>& frames, double seconds) {
    std::size_t best = 0;
    double bestDistance = 1e30;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const double distance = std::abs(frames[i].timeSec - seconds);
        if (distance < bestDistance) {
            bestDistance = distance;
            best = i;
        }
    }
    return best;
}

float peakBetween(const std::vector<OdfFrame>& frames, double fromSec, double toSec) {
    float peak = 0.0f;
    for (const auto& frame : frames) {
        if (frame.timeSec >= fromSec && frame.timeSec <= toSec) {
            peak = std::max(peak, frame.full);
        }
    }
    return peak;
}

}  // namespace

TEST(OdfConfig, RejectsNonsense) {
    OdfConfig cfg = testConfig();
    EXPECT_TRUE(cfg.valid());

    cfg = testConfig(); cfg.sampleRate = 0.0;      EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.frameSize = 1000;      EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.hopSize = 0;           EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.hopSize = 2048;        EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.melBands = 0;          EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.melMaxHz = 10.0;       EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.highBandHz = 100.0;    EXPECT_FALSE(cfg.valid());
    cfg = testConfig(); cfg.whiteningTau = 0.0;    EXPECT_FALSE(cfg.valid());
}

TEST(Odf, SilenceProducesNoOnsets) {
    Odf odf(testConfig());
    const auto frames = run(odf, tiktak::test::silence(48000));

    ASSERT_FALSE(frames.empty());
    for (const auto& frame : frames) {
        EXPECT_FLOAT_EQ(frame.full, 0.0f);
        EXPECT_FLOAT_EQ(frame.low, 0.0f);
        EXPECT_FLOAT_EQ(frame.high, 0.0f);
    }
}

TEST(Odf, FirstFrameHasNoFluxToReport) {
    // There is no previous spectrum to difference against, so the first frame
    // must be zero rather than a phantom onset at t=0.
    Odf odf(testConfig());
    auto input = tiktak::test::silence(8192);
    tiktak::test::addBurst(input, 0, 512, 440.0, kSampleRate);

    const auto frames = run(odf, input);
    ASSERT_FALSE(frames.empty());
    EXPECT_FLOAT_EQ(frames.front().full, 0.0f);
}

TEST(Odf, SteadyToneOnsetsOnceAtItsStart) {
    Odf odf(testConfig());

    // Half a second of silence, then a continuous tone. The flux should spike at
    // the transition and settle back down while the tone sustains.
    std::vector<float> input = tiktak::test::silence(24000);
    const auto tone = tiktak::test::sine(24000, 440.0, kSampleRate, 0.5f);
    input.insert(input.end(), tone.begin(), tone.end());

    const auto frames = run(odf, input);
    ASSERT_FALSE(frames.empty());

    const float atOnset = peakBetween(frames, 0.48, 0.56);
    const float duringSustain = peakBetween(frames, 0.75, 0.98);

    EXPECT_GT(atOnset, 0.0f);
    EXPECT_GT(atOnset, duringSustain * 5.0f)
        << "onset " << atOnset << " vs sustain " << duringSustain;
}

TEST(Odf, PeaksLineUpWithBurstTimes) {
    Odf odf(testConfig());

    const std::vector<double> onsetTimes = {0.25, 0.75, 1.25, 1.75};
    auto input = tiktak::test::silence(static_cast<std::size_t>(kSampleRate * 2.2));
    for (double t : onsetTimes) {
        tiktak::test::addBurst(input, static_cast<std::size_t>(t * kSampleRate), 2000,
                               220.0, kSampleRate, 0.8f);
    }

    const auto frames = run(odf, input);
    ASSERT_FALSE(frames.empty());

    for (double t : onsetTimes) {
        const std::size_t at = frameNearest(frames, t);
        // The reported time is the window centre, so an onset shows up within
        // roughly half a window of its true position.
        const float local = peakBetween(frames, t - 0.03, t + 0.03);
        const float elsewhere = peakBetween(frames, t + 0.15, t + 0.35);

        EXPECT_GT(local, 0.0f) << "no onset near " << t << "s";
        EXPECT_GT(local, elsewhere * 3.0f)
            << "onset at " << t << "s (frame " << at << ") did not stand out";
    }
}

namespace {

struct BandPeaks {
    float full = 0.0f;
    float low = 0.0f;
    float high = 0.0f;
};

// Peak response to a single burst at `freqHz`, played at `amplitude`.
BandPeaks burstResponse(double freqHz, double whiteningStrength, float amplitude = 0.9f) {
    OdfConfig cfg = testConfig();
    cfg.melBands = 81;
    cfg.melMinHz = 27.5;
    cfg.melMaxHz = 16000.0;
    cfg.whiteningStrength = whiteningStrength;

    Odf odf(cfg);
    auto input = tiktak::test::silence(24000);
    tiktak::test::addBurst(input, 8000, 4000, freqHz, kSampleRate, amplitude);

    BandPeaks peaks;
    for (const auto& f : run(odf, input)) {
        peaks.full = std::max(peaks.full, f.full);
        peaks.low = std::max(peaks.low, f.low);
        peaks.high = std::max(peaks.high, f.high);
    }
    return peaks;
}

}  // namespace

TEST(Odf, LowBandRespondsToBassAndHighBandToTreble) {
    const BandPeaks bass = burstResponse(60.0, testConfig().whiteningStrength);
    const BandPeaks treble = burstResponse(9000.0, testConfig().whiteningStrength);

    EXPECT_GT(bass.low, 0.0f);
    EXPECT_GT(treble.high, 0.0f);

    EXPECT_GT(bass.low, bass.high * 2.0f)
        << "a 60 Hz thump should move the low band far more than the high one ("
        << bass.low << " vs " << bass.high << ")";
    EXPECT_GT(treble.high, treble.low * 2.0f)
        << "a 9 kHz tick should move the high band far more than the low one ("
        << treble.high << " vs " << treble.low << ")";
}

TEST(Odf, WhiteningStrengthTradesBandBalanceForLevelInvariance) {
    // These two properties are in tension by construction, because both come
    // from the same per-band normalisation:
    //
    //   strength -> 1  every band is divided by its own running peak, so any
    //                  band on a rising edge lands on exactly full scale. Level
    //                  invariance is perfect and the balance between bands is
    //                  gone, which is what low/high are supposed to carry.
    //   strength -> 0  the spectrum passes through untouched, so band balance is
    //                  intact and the flux tracks absolute loudness again.
    //
    // The test pins the shape of the trade-off, not any particular operating
    // point: the default is tuned in Phase 2 against mir_eval, not here.
    const auto levelSensitivity = [](double strength) {
        const float quiet = burstResponse(440.0, strength, 0.02f).full;
        const float loud = burstResponse(440.0, strength, 1.0f).full;
        EXPECT_GT(quiet, 0.0f);
        return loud / quiet;
    };
    const auto bandDiscrimination = [](double strength) {
        const BandPeaks bass = burstResponse(60.0, strength);
        return bass.low / std::max(bass.high, 1e-6f);
    };

    const double weakLevel = levelSensitivity(0.0);
    const double midLevel = levelSensitivity(0.5);
    const double fullLevel = levelSensitivity(1.0);

    EXPECT_GT(weakLevel, midLevel);
    EXPECT_GT(midLevel, fullLevel);
    EXPECT_NEAR(fullLevel, 1.0, 0.05) << "full whitening should remove level entirely";
    EXPECT_GT(weakLevel, 3.0) << "without whitening the flux should track loudness";

    const double weakBands = bandDiscrimination(0.0);
    const double midBands = bandDiscrimination(0.5);
    const double fullBands = bandDiscrimination(1.0);

    EXPECT_GT(weakBands, midBands);
    EXPECT_GT(midBands, fullBands);
    EXPECT_NEAR(fullBands, 1.0, 0.05) << "full whitening should flatten the bands";
    EXPECT_GT(midBands, 2.0) << "the default must keep the bands usefully distinct";
}

TEST(Odf, TimestampsAreTheWindowCentre) {
    OdfConfig cfg = testConfig();
    Odf odf(cfg);

    const auto frames = run(odf, tiktak::test::silence(8192));
    ASSERT_GE(frames.size(), 2u);

    const double expectedFirst =
        static_cast<double>(cfg.frameSize) * 0.5 / cfg.sampleRate;
    EXPECT_NEAR(frames[0].timeSec, expectedFirst, 1e-9);

    const double hopSec = static_cast<double>(cfg.hopSize) / cfg.sampleRate;
    EXPECT_NEAR(frames[1].timeSec - frames[0].timeSec, hopSec, 1e-9);
}

TEST(Odf, LatencyIsHalfAWindow) {
    OdfConfig cfg = testConfig();
    Odf odf(cfg);
    EXPECT_NEAR(odf.latencySec(),
                static_cast<double>(cfg.frameSize) * 0.5 / cfg.sampleRate, 1e-12);
}

TEST(Odf, ResetClearsHistory) {
    Odf odf(testConfig());

    auto input = tiktak::test::silence(24000);
    tiktak::test::addBurst(input, 8000, 4000, 440.0, kSampleRate, 0.9f);

    const auto first = run(odf, input);
    odf.reset();
    const auto second = run(odf, input);

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_NEAR(first[i].timeSec, second[i].timeSec, 1e-12) << "at frame " << i;
        EXPECT_NEAR(first[i].full, second[i].full, 1e-6f) << "at frame " << i;
    }
}

TEST(Odf, BlockSizeDoesNotChangeTheResult) {
    auto input = tiktak::test::silence(20000);
    tiktak::test::addBurst(input, 6000, 3000, 300.0, kSampleRate, 0.7f);

    Odf reference(testConfig());
    const auto expected = run(reference, input);

    for (std::size_t block : {1u, 5u, 64u, 256u, 999u}) {
        Odf odf(testConfig());
        std::vector<OdfFrame> frames;
        for (std::size_t pos = 0; pos < input.size(); pos += block) {
            const std::size_t take = std::min(block, input.size() - pos);
            odf.process(input.data() + pos, take,
                        [&](const OdfFrame& f) { frames.push_back(f); });
        }

        ASSERT_EQ(frames.size(), expected.size()) << "block " << block;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            EXPECT_NEAR(frames[i].full, expected[i].full, 1e-5f)
                << "block " << block << ", frame " << i;
        }
    }
}
