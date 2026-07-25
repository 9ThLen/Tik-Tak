#include "analysis/tempo.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "support.hpp"

using tiktak::analysis::TempoCandidate;
using tiktak::analysis::TempoConfig;
using tiktak::analysis::TempoEstimate;
using tiktak::analysis::TempoEstimator;
using tiktak::test::impulseTrain;

namespace {

// The ODF's own frame rate at the default settings: 48 kHz over a 512 hop.
constexpr double kFps = 48000.0 / 512.0;

double spacingForBpm(double bpm) { return 60.0 * kFps / bpm; }

// The tempo ratios a listener would accept as "the same pulse, counted
// differently": octaves either way, and the triple/third relation that
// separates a bar of three from a bar of one.
constexpr double kMetricalRatios[] = {0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0,
                                      1.5,  2.0,       3.0, 4.0};

// Tolerance in cents-of-an-octave terms. The grid is 512 points from 40 to 220
// BPM, so neighbouring candidates differ by about 0.5%; anything under 2% is
// "the right grid point or its neighbour".
::testing::AssertionResult NearBpm(double actual, double expected, double tolerance = 0.02) {
    const double error = std::abs(actual - expected) / expected;
    if (error <= tolerance) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << actual << " BPM is " << error * 100.0 << "% from " << expected << " BPM";
}

}  // namespace

TEST(TempoConfigValidity, RejectsRangesThatCannotBeSearched) {
    TempoConfig cfg;
    EXPECT_TRUE(cfg.valid());

    cfg = TempoConfig{};
    cfg.min_bpm = 0.0;
    EXPECT_FALSE(cfg.valid());

    cfg = TempoConfig{};
    cfg.min_bpm = 200.0;
    cfg.max_bpm = 100.0;
    EXPECT_FALSE(cfg.valid());

    cfg = TempoConfig{};
    cfg.grid_size = 4;
    EXPECT_FALSE(cfg.valid());

    cfg = TempoConfig{};
    cfg.comb_harmonics = 0;
    EXPECT_FALSE(cfg.valid());

    cfg = TempoConfig{};
    cfg.prior_width_octaves = 0.0;
    EXPECT_FALSE(cfg.valid());
}

TEST(TempoGrid, IsLogSpacedAcrossTheConfiguredRange) {
    TempoEstimator estimator{TempoConfig{}};
    const std::vector<double>& grid = estimator.bpmGrid();

    ASSERT_EQ(grid.size(), 512u);
    EXPECT_DOUBLE_EQ(grid.front(), 40.0);
    EXPECT_DOUBLE_EQ(grid.back(), 220.0);

    // Equal ratios, not equal differences: perception of tempo is logarithmic,
    // and a linear grid would waste most of its resolution at the fast end.
    const double first = grid[1] / grid[0];
    const double last = grid[511] / grid[510];
    EXPECT_NEAR(first, last, 1e-9);
}

TEST(Tempo, FindsThePeriodOfAnImpulseTrain) {
    TempoEstimator estimator{TempoConfig{}};

    for (double bpm : {60.0, 90.0, 120.0, 140.0}) {
        const std::vector<double> odf = impulseTrain(2000, spacingForBpm(bpm));
        const TempoEstimate estimate = estimator.estimate(odf.data(), odf.size(), kFps);
        EXPECT_TRUE(NearBpm(estimate.bpm, bpm)) << "at " << bpm << " BPM";
        // An impulse train is mostly zeros, so its autocorrelation at the beat
        // period can never approach its variance the way a real onset function
        // does — 0.4-0.7 here against 0.8-0.9 for synthesised audio. The bar
        // that matters is the distance from noise, which scores 0.02.
        EXPECT_GT(estimate.confidence, 0.3) << "at " << bpm << " BPM";
    }
}

// Documents a real limitation rather than asserting it away. A bare impulse
// train is exactly as consistent with half its tempo as with its own — nothing
// in the signal distinguishes them — so at the extremes of the range the prior
// decides, and it pulls towards 120. 176 BPM comes back as 88.
//
// This is not a bug to be fixed here: choosing between 88 and 176 needs
// evidence the onset function does not carry, which is what the downbeat model
// in a later phase is for. Until then the app must show the ambiguity and offer
// the half/double toggle rather than pretend to be sure.
TEST(Tempo, HalvesAFastTempoAtTheEdgeOfTheRange) {
    TempoEstimator estimator{TempoConfig{}};
    const std::vector<double> odf = impulseTrain(2000, spacingForBpm(176.0));

    const TempoEstimate estimate = estimator.estimate(odf.data(), odf.size(), kFps);
    EXPECT_TRUE(NearBpm(estimate.bpm, 88.0));

    // The right answer is still in the posterior, and reported as a candidate.
    TempoCandidate candidates[3];
    const std::size_t written = estimator.topCandidates(candidates, 3);
    const bool offers_176 = std::any_of(candidates, candidates + written,
                                        [](const TempoCandidate& c) {
                                            return std::abs(c.bpm - 176.0) / 176.0 < 0.02;
                                        });
    EXPECT_TRUE(offers_176);
}

TEST(Tempo, IsUnaffectedByThePhaseOfTheBeat) {
    TempoEstimator estimator{TempoConfig{}};
    const double spacing = spacingForBpm(120.0);

    const std::vector<double> aligned = impulseTrain(2000, spacing, 0.0);
    const std::vector<double> shifted = impulseTrain(2000, spacing, spacing * 0.37);

    const double a = estimator.estimate(aligned.data(), aligned.size(), kFps).bpm;
    const double b = estimator.estimate(shifted.data(), shifted.size(), kFps).bpm;
    EXPECT_DOUBLE_EQ(a, b);
}

// The comb is off by default because it measured worse (see the comment on
// TempoConfig::comb_harmonics). This pins that decision so it cannot drift back
// silently, and checks the mechanism still works when asked for.
TEST(Tempo, CombIsOffByDefaultButStillFunctionsWhenEnabled) {
    EXPECT_EQ(TempoConfig{}.comb_harmonics, 1);

    const std::vector<double> odf = impulseTrain(3000, spacingForBpm(120.0), 0.0, 1.0, 0.35);

    TempoEstimator plain{TempoConfig{}};
    plain.estimate(odf.data(), odf.size(), kFps);
    const std::vector<double> without = plain.posterior();

    TempoConfig combed;
    combed.comb_harmonics = 4;
    TempoEstimator with{combed};
    const TempoEstimate estimate = with.estimate(odf.data(), odf.size(), kFps);

    // Both still find the tempo on material this clean; what the comb changes
    // is the shape of the posterior around it.
    EXPECT_TRUE(NearBpm(estimate.bpm, 120.0));
    EXPECT_NE(without, with.posterior());
}

TEST(Tempo, ReportsLowConfidenceOnNoise) {
    TempoEstimator estimator{TempoConfig{}};

    // Deterministic pseudo-noise: no periodicity, but plenty of energy.
    std::vector<double> odf(2000);
    std::uint32_t state = 12345u;
    for (double& value : odf) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
    }

    const TempoEstimate estimate = estimator.estimate(odf.data(), odf.size(), kFps);
    const std::vector<double> beats = impulseTrain(2000, spacingForBpm(120.0));
    const TempoEstimate clear = estimator.estimate(beats.data(), beats.size(), kFps);

    EXPECT_LT(estimate.confidence, clear.confidence);
}

TEST(Tempo, SilenceFallsBackToThePriorCentreWithNoConfidence) {
    TempoEstimator estimator{TempoConfig{}};
    const std::vector<double> odf(2000, 0.0);

    const TempoEstimate estimate = estimator.estimate(odf.data(), odf.size(), kFps);
    EXPECT_DOUBLE_EQ(estimate.bpm, 120.0);
    EXPECT_DOUBLE_EQ(estimate.confidence, 0.0);
}

TEST(Tempo, HandlesInputTooShortToHoldABeat) {
    TempoEstimator estimator{TempoConfig{}};
    const std::vector<double> odf{1.0, 0.0};

    const TempoEstimate estimate = estimator.estimate(odf.data(), odf.size(), kFps);
    EXPECT_DOUBLE_EQ(estimate.bpm, 120.0);
    EXPECT_DOUBLE_EQ(estimate.confidence, 0.0);

    EXPECT_DOUBLE_EQ(estimator.estimate(nullptr, 0, kFps).bpm, 120.0);
    EXPECT_DOUBLE_EQ(estimator.estimate(odf.data(), odf.size(), 0.0).bpm, 120.0);
}

TEST(Tempo, PriorBreaksTheOctaveTieTowardsItsCentre) {
    // Subdivided material — a hit on every half-beat, loud on the beat — is
    // consistent with 100 BPM and with 50 and 200 alike. Which one wins is
    // decided by the prior, and moving the prior's centre must move the answer.
    //
    // The subdivisions are what make this a fair test. A bare impulse train is
    // *not* symmetric: halving the tempo asks about a lag where the train has a
    // peak, doubling asks about a lag where nothing happens at all, so no prior
    // can pull it upwards. That is correct behaviour, not a limitation.
    const std::vector<double> odf = impulseTrain(3000, spacingForBpm(200.0), 0.0, 1.0, 0.5);

    TempoConfig low;
    low.prior_centre_bpm = 60.0;
    low.prior_width_octaves = 0.3;
    const double slow = TempoEstimator{low}.estimate(odf.data(), odf.size(), kFps).bpm;

    TempoConfig high;
    high.prior_centre_bpm = 200.0;
    high.prior_width_octaves = 0.3;
    const double fast = TempoEstimator{high}.estimate(odf.data(), odf.size(), kFps).bpm;

    // The assertion is that the prior moved the answer to its own side of the
    // beat level, not that it landed on one exact tempo: with a centre at 60 the
    // winner is the two-thirds level (66.9), which is a legitimate metrical
    // reading of the same material and closer to the prior than 50 is.
    EXPECT_LT(slow, 100.0);
    EXPECT_GT(fast, 100.0);
    EXPECT_TRUE(NearBpm(fast, 200.0, 0.05));
}

TEST(TempoCandidates, ExposeTheOctaveAmbiguityInsteadOfHidingIt) {
    TempoEstimator estimator{TempoConfig{}};
    const std::vector<double> odf = impulseTrain(3000, spacingForBpm(120.0));
    estimator.estimate(odf.data(), odf.size(), kFps);

    TempoCandidate candidates[3];
    const std::size_t written = estimator.topCandidates(candidates, 3);
    ASSERT_EQ(written, 3u);

    // Strongest first, and each far enough from the others to be a distinct
    // answer rather than the same peak reported twice.
    EXPECT_GE(candidates[0].strength, candidates[1].strength);
    EXPECT_GE(candidates[1].strength, candidates[2].strength);
    EXPECT_DOUBLE_EQ(candidates[0].strength, 1.0);

    for (std::size_t i = 1; i < written; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            EXPECT_GE(std::abs(std::log2(candidates[i].bpm / candidates[j].bpm)), 0.2);
        }
    }

    // The runner-up should be a metrical relative of the winner — a simple
    // ratio such as half, double or a third — not an unrelated tempo. That is
    // what makes the list worth showing the user: these are readings of the
    // same pulse, so a half/double toggle resolves them.
    const double ratio = candidates[1].bpm / candidates[0].bpm;
    const bool metrical = std::any_of(
        std::begin(kMetricalRatios), std::end(kMetricalRatios),
        [ratio](double simple) { return std::abs(ratio / simple - 1.0) < 0.03; });
    EXPECT_TRUE(metrical) << "runner-up is " << candidates[1].bpm << " against "
                          << candidates[0].bpm;
}

TEST(TempoCandidates, HandleDegenerateRequests) {
    TempoEstimator estimator{TempoConfig{}};
    const std::vector<double> odf = impulseTrain(2000, spacingForBpm(120.0));
    estimator.estimate(odf.data(), odf.size(), kFps);

    TempoCandidate one;
    EXPECT_EQ(estimator.topCandidates(&one, 1), 1u);
    EXPECT_EQ(estimator.topCandidates(nullptr, 4), 0u);
    EXPECT_EQ(estimator.topCandidates(&one, 0), 0u);
}

TEST(Tempo, ReusingAnEstimatorGivesTheSameAnswerAsAFreshOne) {
    // The estimator caches its transform across calls; a stale buffer would
    // show up here and nowhere else.
    const std::vector<double> a = impulseTrain(2000, spacingForBpm(90.0));
    const std::vector<double> b = impulseTrain(3000, spacingForBpm(150.0));

    TempoEstimator reused{TempoConfig{}};
    reused.estimate(a.data(), a.size(), kFps);
    const double second = reused.estimate(b.data(), b.size(), kFps).bpm;

    TempoEstimator fresh{TempoConfig{}};
    EXPECT_DOUBLE_EQ(second, fresh.estimate(b.data(), b.size(), kFps).bpm);
}
