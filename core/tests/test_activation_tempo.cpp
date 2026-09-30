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
    // against a constant. A partly filled ring is *not* read as zero-padded —
    // the estimate uses only what was written — so this pins a measured
    // configuration, not a numerical necessity; see min_window_sec.
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

TEST(ActivationTempo, AGapKeepsItsPlaceAndIsNotClosedUp) {
    // Dropped buffers are the case this protects. The lag axis is time, so
    // omitting a gap would compress the history and report a tempo faster than
    // the room's — and a dropped buffer is exactly when the tracker must not
    // change its mind about the tempo. Asked twice: with the gap still inside
    // the window, where it can matter, and after it has left. Both ways of
    // holding a gap must pass; they differ only in what the gap is heard as.
    for (const bool mask : {true, false}) {
        ActivationTempoConfig config;
        config.mask_gaps = mask;
        ActivationTempo tempo{config};

        double time = feedPulse(tempo, 120.0, 12.0);
        const double resumed = time + 2.0;              // two seconds unheard
        time = feedPulse(tempo, 120.0, 3.0, resumed);
        ASSERT_TRUE(tempo.estimate().answered()) << "mask " << mask;
        EXPECT_NEAR(tempo.estimate().bpm, 120.0, 4.0)
            << "with the gap inside the window, mask " << mask;

        feedPulse(tempo, 120.0, 9.0, time + 1.0 / config.fps);
        ASSERT_TRUE(tempo.estimate().answered()) << "mask " << mask;
        EXPECT_NEAR(tempo.estimate().bpm, 120.0, 4.0)
            << "after the gap left the window, mask " << mask;
    }
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

TEST(ActivationTempo, AnswersTheSameWhereverTheClockStarts) {
    // BeatNet's frames arrive exactly at the grid's rate, stamped
    // origin + n*441/22050. Binning those by a bare floor() put a frame into
    // the previous bin whenever the product came out a hair below n, which
    // kept the larger of two frames and wrote a zero into the bin that was
    // skipped: 2% of frames with the clock at zero, and a quarter to a third
    // of them with the clock where a device's stream time actually starts.
    // Nothing about the music depends on where the clock starts, so nothing
    // about the answer may.
    const auto run = [](double origin) {
        ActivationTempo tempo{ActivationTempoConfig{}};
        const double period_frames = 50.0 * 60.0 / 128.0;
        for (int n = 0; n < 1500; ++n) {
            const double beats = static_cast<double>(n) / period_frames;
            const double nearest = std::round(beats);
            double value = 0.05 + 0.01 * static_cast<double>(n % 7);
            if (std::fabs(beats - nearest) * period_frames < 0.5) {
                value = (std::llround(nearest) % 4 == 0) ? 1.0 : 0.7;
            }
            tempo.observe(origin + static_cast<double>(n) * 441.0 / 22050.0, value);
        }
        return tempo;
    };

    const ActivationTempo reference = run(0.0);
    ASSERT_TRUE(reference.estimate().answered());
    for (const double origin : {1.0, 12345.678, 370000.0}) {
        const ActivationTempo shifted = run(origin);
        ASSERT_TRUE(shifted.estimate().answered()) << origin;
        // Exact, not near: the same frames in the same bins are the same
        // arithmetic, and anything short of identical means a frame moved.
        EXPECT_EQ(shifted.estimate().bpm, reference.estimate().bpm) << origin;
        EXPECT_EQ(shifted.estimate().confidence, reference.estimate().confidence) << origin;
        EXPECT_EQ(shifted.estimate().octave_margin, reference.estimate().octave_margin)
            << origin;
        EXPECT_EQ(shifted.posterior(), reference.posterior()) << origin;
    }
}

TEST(ActivationTempo, AConstantActivationIsNoAnswerRatherThanAConfidentOne) {
    // Removing the mean from a constant leaves a constant of rounding error,
    // about 1e-15 for a level of 0.2, and the autocorrelation of any constant
    // normalised to its own lag zero is exactly 1 at every lag. The posterior
    // was then the prior, and the estimator answered the prior's centre with
    // full confidence: 140.2 BPM out of a signal with no rhythm in it. A
    // level that happens to be exact in binary (0.25) left no residue and was
    // correctly no answer, which is why silence never showed it.
    for (const double level : {0.2, 0.37, 0.25, 0.9}) {
        ActivationTempo tempo{ActivationTempoConfig{}};
        for (int i = 0; i < 1500; ++i) tempo.observe(static_cast<double>(i) / 50.0, level);
        EXPECT_FALSE(tempo.estimate().answered())
            << "a flat " << level << " answered " << tempo.estimate().bpm
            << " BPM at confidence " << tempo.estimate().confidence;
    }
}

TEST(ActivationTempo, TheFramesAClickGateHidesAreNotReadAsTheTempo) {
    // What a listening metronome does to its own input: it stops hearing for a
    // few frames around every click it plays. Here the music is a pulse at 100
    // BPM over a raised floor, the way a room leaves an activation, and the
    // frames are withheld at 150 BPM — the rate a tracker that had settled on
    // the wrong tempo would be clicking at. Held as silence, the withheld
    // frames are dips at the tracker's own rate and the estimator hears its
    // own clicks as the tempo. Held as unheard, it hears the music.
    const auto run = [](bool mask) {
        ActivationTempoConfig config;
        config.mask_gaps = mask;
        ActivationTempo tempo{config};
        for (int n = 0; n < 1500; ++n) {
            if (n % 20 >= 12 && n % 20 < 16) continue;       // gated, 4 of every 20
            const double value = (n % 30 == 0) ? 1.0 : 0.5;   // 100 BPM on a floor
            tempo.observe(static_cast<double>(n) / 50.0, value);
        }
        return tempo.estimate();
    };

    const auto unheard = run(true);
    ASSERT_TRUE(unheard.answered());
    const double music = std::log2(unheard.bpm / 100.0);
    EXPECT_NEAR(music, std::round(music), 0.06)
        << "answered " << unheard.bpm << " — not a metrical relative of the music";

    // The failure the default exists to prevent, kept visible so that a change
    // which quietly stops reproducing it is noticed.
    const auto silent = run(false);
    ASSERT_TRUE(silent.answered());
    const double gate = std::log2(silent.bpm / 150.0);
    EXPECT_NEAR(gate, std::round(gate), 0.06)
        << "zeros at the gate's rate no longer win (" << silent.bpm
        << " BPM); the scenario above has stopped testing anything";
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
