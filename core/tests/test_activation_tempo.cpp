#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tracking/activation_tempo.hpp"

using tiktak::tracking::ActivationTempo;
using tiktak::tracking::ActivationTempoConfig;

namespace {

// Feeds an activation that spikes on the beat and is silent between, which is
// the shape a causal model's output actually has. Returns the time of the last
// observation fed.
double feedPulse(ActivationTempo& tempo, double bpm, double seconds,
                 double from_sec = 0.0, double height = 1.0,
                 double fps = 50.0) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * fps);
    double time = from_sec;
    for (std::size_t i = 0; i < frames; ++i) {
        time = from_sec + static_cast<double>(i) / fps;
        const double since = time - from_sec;
        const double nearest = std::round(since / period) * period;
        const bool on_beat = std::fabs(since - nearest) < 0.5 / fps;
        tempo.observe(time, on_beat ? height : 0.0);
    }
    return time;
}

}  // namespace

TEST(ActivationTempo, RejectsAConfigurationItCannotHonour) {
    ActivationTempoConfig config;
    EXPECT_TRUE(config.valid());

    config = ActivationTempoConfig{};
    config.min_window_sec = config.window_sec + 1.0;
    EXPECT_FALSE(config.valid()) << "it cannot wait for more than it stores";

    config = ActivationTempoConfig{};
    config.max_bpm = config.min_bpm;
    EXPECT_FALSE(config.valid());

    config = ActivationTempoConfig{};
    config.fps = 0.0;
    EXPECT_FALSE(config.valid());
}

TEST(ActivationTempo, SaysNothingBeforeItHasHeardEnough) {
    const ActivationTempoConfig config;
    ActivationTempo tempo{config};

    // Deliberately expressed against the configured threshold rather than
    // against a constant. Until the ring is full the rest of it is zeros, and
    // an autocorrelation reads padding as a claim about the tempo, so this is
    // not a politeness — it is why min_window_sec equals window_sec.
    feedPulse(tempo, 120.0, config.min_window_sec - 1.0);
    EXPECT_FALSE(tempo.estimate().answered())
        << "answered before the window was full; a tempo of 0 is how not "
           "knowing is reported — not a quiet guess of 120";
    EXPECT_DOUBLE_EQ(tempo.estimate().bpm, 0.0);

    feedPulse(tempo, 120.0, 3.0, config.min_window_sec - 1.0);
    EXPECT_TRUE(tempo.estimate().answered()) << "never started answering";
}

TEST(ActivationTempo, FindsASteadyPulseOnceItHas) {
    ActivationTempo tempo{ActivationTempoConfig{}};
    feedPulse(tempo, 120.0, 20.0);

    const auto estimate = tempo.estimate();
    ASSERT_TRUE(estimate.answered());
    EXPECT_NEAR(estimate.bpm, 120.0, 2.0);
    EXPECT_GT(estimate.confidence, 0.3);
}

TEST(ActivationTempo, LandsOnAMetricalLevelAcrossTheRange) {
    // What can honestly be asked of an unaccented pulse train, and no more.
    // Every beat identical means the activation repeats just as exactly at
    // half and at double the period, so the octave is not in the signal at all
    // and the prior decides it — by design, and the same way the offline
    // estimator does. Real music breaks that tie with accents; a click track
    // does not, and a test that pretended otherwise would be testing the
    // prior's centre rather than the estimator.
    //
    // The property that does hold, and that matters: the answer is always the
    // true period times a power of two. Landing at two-thirds or three-halves
    // would be a non-metrical error, which is a real failure and is what the
    // offline comb was rejected for causing.
    for (const double bpm : {75.0, 100.0, 140.0, 175.0, 200.0}) {
        ActivationTempo tempo{ActivationTempoConfig{}};
        feedPulse(tempo, bpm, 25.0);
        const auto estimate = tempo.estimate();
        ASSERT_TRUE(estimate.answered()) << bpm;

        const double octaves = std::log2(estimate.bpm / bpm);
        EXPECT_NEAR(octaves, std::round(octaves), 0.06)
            << "asked for " << bpm << ", answered " << estimate.bpm
            << " — not a metrical relative of it";
    }
}

TEST(ActivationTempo, PrefersTheLevelTheAccentsAreOn) {
    // The tie above is broken by evidence when there is any. Beats at 180 with
    // every second one weaker: the strong pulses alone would read as 90, and
    // the estimator must still prefer 180, because the weak beats are there
    // and the prior — centred at 140 — pulls the other way.
    ActivationTempoConfig config;
    ActivationTempo tempo{config};

    const double period = 60.0 / 180.0;
    const auto frames = static_cast<std::size_t>(25.0 * config.fps);
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / config.fps;
        const double nearest = std::round(time / period) * period;
        double value = 0.0;
        if (std::fabs(time - nearest) < 0.5 / config.fps) {
            value = (std::llround(time / period) % 2 == 0) ? 1.0 : 0.85;
        }
        tempo.observe(time, value);
    }

    const auto estimate = tempo.estimate();
    ASSERT_TRUE(estimate.answered());
    EXPECT_NEAR(estimate.bpm, 180.0, 8.0)
        << "took the accent pattern for the beat and halved the tempo";
}

TEST(ActivationTempo, NeverAnswersWithAPeriodLongerThanItHasHeard) {
    // A fifteen-second window cannot have evidence for 40 BPM in any
    // meaningful sense, and clamping the autocorrelation past its end would
    // let the slowest candidates collect whatever happened to be there.
    ActivationTempoConfig config;
    config.window_sec = 16.0;
    config.min_window_sec = 15.0;

    ActivationTempo tempo{config};
    feedPulse(tempo, 60.0, 16.0);
    const auto estimate = tempo.estimate();
    ASSERT_TRUE(estimate.answered());
    EXPECT_GE(60.0 / estimate.bpm, 0.0);
    EXPECT_LT(60.0 / estimate.bpm, config.window_sec)
        << "reported a period it could not have measured";
}

TEST(ActivationTempo, ReportsATiedOctaveAsATie) {
    // Every second pulse loud, the rest quiet: the activation genuinely
    // repeats at both the beat and the half-beat, and the margin is what tells
    // a caller not to commit to either.
    ActivationTempoConfig config;
    ActivationTempo alternating{config};
    const double period = 60.0 / 120.0;
    const auto frames = static_cast<std::size_t>(25.0 * config.fps);
    long long beat_index = -1;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / config.fps;
        const double nearest = std::round(time / period) * period;
        double value = 0.0;
        if (std::fabs(time - nearest) < 0.5 / config.fps) {
            beat_index = std::llround(time / period);
            value = (beat_index % 2 == 0) ? 1.0 : 0.95;
        }
        alternating.observe(time, value);
    }

    ActivationTempo plain{config};
    feedPulse(plain, 120.0, 25.0);

    ASSERT_TRUE(alternating.estimate().answered());
    ASSERT_TRUE(plain.estimate().answered());
    EXPECT_LT(alternating.estimate().octave_margin,
              plain.estimate().octave_margin)
        << "an activation supporting two levels must not look as decided as "
           "one supporting a single level";
}

TEST(ActivationTempo, ASilentGapIsHeldAsSilenceAndNotClosedUp) {
    // Dropped buffers are the case this protects. The lag axis is time, so
    // omitting a gap would compress the history and report a tempo faster than
    // the room's — and a dropped buffer is exactly when the tracker must not
    // change its mind about the tempo.
    ActivationTempoConfig config;
    ActivationTempo tempo{config};

    double time = feedPulse(tempo, 120.0, 12.0);
    const double resumed = time + 3.0;              // three seconds unheard
    feedPulse(tempo, 120.0, 12.0, resumed);

    const auto estimate = tempo.estimate();
    ASSERT_TRUE(estimate.answered());
    EXPECT_NEAR(estimate.bpm, 120.0, 4.0);
}

TEST(ActivationTempo, TimeGoingBackwardsIsIgnoredRatherThanGuessedAt) {
    ActivationTempo tempo{ActivationTempoConfig{}};
    feedPulse(tempo, 120.0, 20.0);
    const auto before = tempo.estimate();
    ASSERT_TRUE(before.answered());

    for (int i = 0; i < 50; ++i) tempo.observe(1.0, 1.0);

    EXPECT_DOUBLE_EQ(tempo.estimate().bpm, before.bpm);
}

TEST(ActivationTempo, ResetForgetsTheSongAndNotJustTheAnswer) {
    ActivationTempo tempo{ActivationTempoConfig{}};
    feedPulse(tempo, 100.0, 25.0);
    ASSERT_TRUE(tempo.estimate().answered());

    tempo.reset();
    EXPECT_FALSE(tempo.estimate().answered());
    EXPECT_DOUBLE_EQ(tempo.heard_sec(), 0.0);

    // The old song must not still be in the ring, pulling the new one toward
    // it — this is the difference between a new song and a gap in one.
    feedPulse(tempo, 160.0, 25.0);
    const auto estimate = tempo.estimate();
    ASSERT_TRUE(estimate.answered());
    EXPECT_NEAR(estimate.bpm, 160.0, 6.0);
}

TEST(ActivationTempo, SilenceIsNoAnswerRatherThanAConfidentOne) {
    ActivationTempo tempo{ActivationTempoConfig{}};
    for (int i = 0; i < 1500; ++i) tempo.observe(static_cast<double>(i) / 50.0, 0.0);
    EXPECT_FALSE(tempo.estimate().answered());
}

TEST(ActivationTempo, FollowsTheRoomWhenTheTempoActuallyChanges) {
    // The window is finite for this reason. After a full window at the new
    // tempo nothing of the old one remains, so the estimate is the new tempo
    // rather than an average of the two.
    ActivationTempoConfig config;
    config.window_sec = 20.0;
    config.min_window_sec = 15.0;

    ActivationTempo tempo{config};
    const double time = feedPulse(tempo, 100.0, 25.0, 0.0, 1.0, config.fps);
    ASSERT_NEAR(tempo.estimate().bpm, 100.0, 4.0);

    feedPulse(tempo, 150.0, 25.0, time + 1.0 / config.fps, 1.0, config.fps);
    EXPECT_NEAR(tempo.estimate().bpm, 150.0, 6.0);
}
