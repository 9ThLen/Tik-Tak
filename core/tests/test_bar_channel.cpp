// What the bar channel must do before any corpus number about it means
// anything. Specified in eval/PREREGISTERED_downbeat_channel.md and written
// before the arm was measured on anything.
//
// The arm uses BeatNet's downbeat head to choose which octave of the beat gets
// anchored. Everything below is about that choice and about what it must *not*
// touch — the tempo inside the octave, and the phase.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "support.hpp"
#include "tracking/live.hpp"

using tiktak::tracking::LiveConfig;
using tiktak::tracking::LiveTracker;
using tiktak::tracking::barEndorsedOctave;

namespace {

constexpr double kRate = 48000.0;
constexpr double kFps = 50.0;
constexpr double kMargin = 0.15;

// Which octave the bar rate endorses, or 99 when it declines to choose. A
// sentinel rather than an optional so the expectations below read as one line.
int endorsed(double beat_period, double bar_period, double margin = kMargin) {
    int octave = 0;
    return barEndorsedOctave(beat_period, bar_period, margin, &octave) ? octave
                                                                      : 99;
}

}  // namespace

// ------------------------------------------------- the waltz, written first
//
// A rule that quietly assumed four beats to the bar would read as a gain on a
// corpus of 4/4 pop and as a defect on everything else. This is the test that
// catches it, so it comes before the ones that are about the arm working.

TEST(BarChannel, DoesNotPullAWaltzTowardsFour) {
    // 180 BPM in three: beat 0.333 s, bar 1.0 s, ratio exactly 3.
    //
    // The arm **declines**, and that is the correct answer rather than a
    // shortfall. Halving the beat implies six to the bar, which is also a real
    // metre, so the two candidates tie at zero and neither wins by the margin.
    // What matters is that it does not reach for four: a rule that did would
    // read as a gain on 4/4 pop and as a defect on everything else.
    EXPECT_EQ(endorsed(1.0 / 3.0, 1.0), 99);
}

TEST(BarChannel, AcceptsSixToTheBar) {
    // 6/8 at 120 BPM: beat 0.5 s, bar 3.0 s. Six against three, an octave
    // apart and both real, so again a tie and again silence — never a pull
    // towards four, which would score 1.0 here and lose to both.
    EXPECT_EQ(endorsed(0.5, 3.0), 99);
}

// ---------------------------------------------------------- the arithmetic
//
// The set of plausible bars is **closed under doubling** on (2, 4) and on
// (3, 6). So an octave shift carries one plausible bar onto another, the two
// tie, and the arm abstains. That is the shape of everything below, and it was
// not foreseen when this was pre-registered: the mechanism can only speak where
// an octave leaves the set entirely, which is the doubled case and nothing
// else. See the deviations section of the pre-registration.

TEST(BarChannel, SaysNothingWhenTheLevelIsAlreadyRight) {
    // 4/4 at 120: ratios 8, 4, 2 across the candidates. Four and two both
    // score zero, so there is no vote. Harmless — the arm is asked only to
    // improve on a wrong level, never to confirm a right one.
    EXPECT_EQ(endorsed(0.5, 2.0), 99);
}

TEST(BarChannel, RescuesADoubledTempo) {
    // The one case it can decide, and the one that matters most: Harmonix
    // doubles seven times for every one it halves.
    //
    // A tracker at twice the truth on 4/4 at 120 reports 0.25 s. Ratios are
    // 16, 8 and 4; only the last is a plausible bar, and it wins by 0.415 —
    // nearly three times the margin.
    EXPECT_EQ(endorsed(0.25, 2.0), 1);
}

TEST(BarChannel, CannotRescueAHalvedTempo) {
    // The other side of the closure, stated as a test so it cannot be
    // rediscovered as a surprise on a corpus. A tracker at half reports 1.0 s;
    // ratios are 4, 2 and 1, and four against two ties at zero.
    EXPECT_EQ(endorsed(1.0, 2.0), 99);
}

TEST(BarChannel, DeclinesWhenNothingIsAPlausibleBar) {
    // Pre-registration test 3. A ratio near five at every candidate: 10, 5 and
    // 2.5, none of which is 2, 3, 4 or 6 within a tight margin.
    EXPECT_EQ(endorsed(0.4, 2.0, 0.02), 99);
    // Seven, likewise, with 14 and 3.5 either side.
    EXPECT_EQ(endorsed(0.4, 2.8, 0.02), 99);
}

TEST(BarChannel, DeclinesWhenTwoCandidatesAreTooClose) {
    // Halfway between implying 4 and implying 2 in log space, so k=0 and k=+1
    // score identically and neither wins by the margin.
    const double beat = 0.5;
    const double bar = beat * std::sqrt(2.0) * 2.0;  // ratio 2*sqrt(2) ~ 2.83
    EXPECT_EQ(endorsed(beat, bar), 99);
}

TEST(BarChannel, RefusesNonsensePeriods) {
    EXPECT_EQ(endorsed(0.0, 2.0), 99);
    EXPECT_EQ(endorsed(0.5, 0.0), 99);
    EXPECT_EQ(endorsed(-0.5, 2.0), 99);

    int* nowhere = nullptr;
    EXPECT_FALSE(barEndorsedOctave(0.5, 2.0, kMargin, nowhere));
}

TEST(BarChannel, TheMarginGatesTheDoubledCaseToo) {
    // The one case that votes still has to clear the bar. At 0.415 the doubled
    // case passes any margin under that and fails any above it, which is the
    // whole of the arm's sensitivity to this constant.
    EXPECT_EQ(endorsed(0.25, 2.0, 0.40), 1);
    EXPECT_EQ(endorsed(0.25, 2.0, 0.45), 99);
}

// -------------------------------------------------- the arm end to end

TEST(BarChannel, IsTheBaselineWithoutADownbeatChannel) {
    // Pre-registration tests 1 and 5. The ODF front end has no downbeat head,
    // so the arm has nothing to observe and every beat must be the baseline's.
    LiveConfig with = tiktak::tracking::liveConfigFor(kRate);
    with.bar_channel = true;
    LiveConfig without = tiktak::tracking::liveConfigFor(kRate);

    LiveTracker a(with);
    LiveTracker b(without);
    const auto audio = tiktak::test::clickTrack(120.0, 20.0, kRate);

    std::vector<double> beats_a;
    std::vector<double> beats_b;
    const std::size_t block = 512;
    double time = 0.0;
    for (std::size_t i = 0; i < audio.size(); i += block) {
        const std::size_t n = std::min(block, audio.size() - i);
        a.process(time, audio.data() + i, n);
        b.process(time, audio.data() + i, n);
        time += static_cast<double>(n) / kRate;
        double beat = 0.0;
        if (a.takeBeat(time, 0.05, &beat)) beats_a.push_back(beat);
        if (b.takeBeat(time, 0.05, &beat)) beats_b.push_back(beat);
    }

    EXPECT_FALSE(a.barFromActivation().answered());
    ASSERT_EQ(beats_a.size(), beats_b.size());
    for (std::size_t i = 0; i < beats_a.size(); ++i) {
        EXPECT_DOUBLE_EQ(beats_a[i], beats_b[i]) << "beat " << i;
    }
}

TEST(BarChannel, ConfigRejectsAnImpossibleBarEstimator) {
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    ASSERT_TRUE(config.valid());

    config.bar_ratio_margin = 0.0;
    EXPECT_FALSE(config.valid());

    config = tiktak::tracking::liveConfigFor(kRate);
    config.bar_tempo.min_bpm = 0.0;
    EXPECT_FALSE(config.valid());
}

TEST(BarChannel, TheBarEstimatorCanRepresentABar) {
    // The defaults exist because the beat estimator's cannot hold the quantity:
    // a four-beat bar at 120 BPM is 30 a minute, below its min_bpm of 40. This
    // is that reason, as an assertion, so nobody restores "identical
    // configuration" from the pre-registration.
    const auto beat = tiktak::tracking::ActivationTempoConfig{};
    const auto bar = tiktak::tracking::barTempoDefaults();

    EXPECT_LT(30.0, beat.min_bpm) << "the beat estimator cannot hold a bar rate";
    EXPECT_LE(bar.min_bpm, 10.0);
    EXPECT_GE(bar.max_bpm, 120.0);
    // A partly filled ring is zero-padded and the padding reads as evidence.
    EXPECT_DOUBLE_EQ(bar.window_sec, bar.min_window_sec);
}
