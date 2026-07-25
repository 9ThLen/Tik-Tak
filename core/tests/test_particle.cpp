#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tracking/particle.hpp"

using tiktak::tracking::BeatEstimate;
using tiktak::tracking::BeatParticleFilter;
using tiktak::tracking::ParticleFilterConfig;
using tiktak::tracking::Rng;

namespace {

constexpr double kFps = 93.75;  // 48 kHz / 512, the ODF's own frame rate

// Drives the filter with an onset function that is `strength` on the frames a
// beat falls in and zero elsewhere — the input the ODF produces from a click
// track, without inheriting the ODF's own behaviour.
//
// Onsets are in the units the filter is calibrated in and LiveTracker delivers:
// a beat is about 1.0, silence 0.
double feedSteady(BeatParticleFilter& filter, double bpm, double seconds, double from_sec = 0.0,
                  double strength = 1.0, double phase_sec = 0.0) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    double time = from_sec;
    for (std::size_t i = 0; i < frames; ++i) {
        time = from_sec + static_cast<double>(i) / kFps;
        const double since = time - phase_sec;
        const double nearest = std::round(since / period) * period;
        const bool on_beat = std::fabs(since - nearest) < 0.5 / kFps;
        filter.observe(time, on_beat ? strength : 0.0);
    }
    return time;
}

double feedSilence(BeatParticleFilter& filter, double seconds, double from_sec) {
    return feedSteady(filter, 120.0, seconds, from_sec, 0.0);
}

// Distance from `time` to the nearest beat of a grid at `bpm`, seconds.
double offGrid(double time, double bpm, double phase_sec = 0.0) {
    const double period = 60.0 / bpm;
    const double since = time - phase_sec;
    return std::fabs(since - std::round(since / period) * period);
}

}  // namespace

TEST(Rng, IsReproducibleAndUniform) {
    Rng a(12345);
    Rng b(12345);
    double sum = 0.0;
    constexpr int kDraws = 100000;
    for (int i = 0; i < kDraws; ++i) {
        const double x = a.uniform();
        EXPECT_EQ(x, b.uniform());
        ASSERT_GE(x, 0.0);
        ASSERT_LT(x, 1.0);
        sum += x;
    }
    EXPECT_NEAR(sum / kDraws, 0.5, 0.01);
}

TEST(Rng, NormalPairHasUnitVariance) {
    Rng rng(7);
    double sum = 0.0;
    double squares = 0.0;
    constexpr int kDraws = 50000;
    for (int i = 0; i < kDraws; ++i) {
        double x = 0.0;
        double y = 0.0;
        rng.normalPair(&x, &y);
        sum += x + y;
        squares += x * x + y * y;
    }
    const double n = 2.0 * kDraws;
    EXPECT_NEAR(sum / n, 0.0, 0.02);
    EXPECT_NEAR(squares / n, 1.0, 0.02);
}

TEST(ParticleFilter, LocksOntoASteadyBeat) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    const double now = feedSteady(filter, 120.0, 12.0);

    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_NEAR(estimate.bpm, 120.0, 2.0);
    EXPECT_GT(estimate.confidence, 0.5);
    // The prediction is where the next beat of the input actually is.
    EXPECT_LT(offGrid(estimate.next_beat_sec, 120.0), 0.02);
    EXPECT_GT(filter.stats().resamples, 0u);
}

TEST(ParticleFilter, PredictsAheadRatherThanReporting) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    const double now = feedSteady(filter, 100.0, 12.0);

    // The whole reason for a filter rather than a detector: the answer is in
    // the future, so a click can be written into a buffer before it is due.
    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_GT(estimate.next_beat_sec, now);
    EXPECT_LE(estimate.next_beat_sec, now + 60.0 / estimate.bpm + 1e-9);
}

TEST(ParticleFilter, DoesNotRunAwayToDoubleTempo) {
    // Every real onset is on a beat of the double-tempo hypothesis too, and the
    // extra beats it predicts land in silence, which the frame-wise term is
    // deliberately blind to. Only the charge per predicted beat separates them.
    BeatParticleFilter filter{ParticleFilterConfig{}};
    const double now = feedSteady(filter, 90.0, 16.0);

    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_NEAR(estimate.bpm, 90.0, 3.0);
}

TEST(ParticleFilter, FindsThePhaseWhereverItStarts) {
    for (const double offset : {0.0, 0.13, 0.31, 0.47}) {
        BeatParticleFilter filter{ParticleFilterConfig{}};
        const double now = feedSteady(filter, 120.0, 12.0, 0.0, 1.0, offset);
        const BeatEstimate estimate = filter.estimate(now);
        EXPECT_LT(offGrid(estimate.next_beat_sec, 120.0, offset), 0.025)
            << "phase offset " << offset;
    }
}

TEST(ParticleFilter, FollowsATempoChange) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    double now = feedSteady(filter, 100.0, 12.0);
    EXPECT_NEAR(filter.estimate(now).bpm, 100.0, 3.0);

    // The phase of the new tempo continues from where the old one left off, as
    // a band speeding up would.
    now = feedSteady(filter, 132.0, 16.0, now);
    EXPECT_NEAR(filter.estimate(now).bpm, 132.0, 4.0);
}

TEST(ParticleFilter, CoastsThroughSilenceAtTheLastTempo) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    double now = feedSteady(filter, 120.0, 12.0);
    const double locked = filter.estimate(now).bpm;
    const double confident = filter.estimate(now).confidence;

    now = feedSilence(filter, 4.0, now);
    const BeatEstimate after = filter.estimate(now);

    // Silence moves no weights, so the tempo survives it — the cloud only
    // spreads. That is the difference between a metronome that keeps time
    // through a quiet bar and one that lunges at the first sound afterwards.
    EXPECT_NEAR(after.bpm, locked, 2.0);
    EXPECT_LT(after.confidence, confident);
}

TEST(ParticleFilter, NoiseDoesNotLookLikeABeat) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    Rng rng(99);
    double now = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(12.0 * kFps); ++i) {
        now = static_cast<double>(i) / kFps;
        filter.observe(now, rng.uniform() * rng.uniform());
    }
    const double noisy = filter.estimate(now).confidence;

    BeatParticleFilter clean{ParticleFilterConfig{}};
    const double clean_now = feedSteady(clean, 120.0, 12.0);

    EXPECT_LT(noisy, clean.estimate(clean_now).confidence);
    EXPECT_LT(noisy, 0.75);
}

TEST(ParticleFilter, SeedingPutsTheCloudWhereItIsTold) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    filter.seedTempo(96.0, 0.02);

    const BeatEstimate seeded = filter.estimate(0.0);
    EXPECT_NEAR(seeded.bpm, 96.0, 1.0);
    EXPECT_LT(seeded.tempo_spread_octaves, 0.05);
    // Placed, not believed: the phase is still unknown and no onset has ever
    // arrived, so nothing has been confirmed.
    EXPECT_EQ(seeded.confidence, 0.0);
}

TEST(ParticleFilter, SeedingSurvivesTheAudioAgreeing) {
    // Seeded on the truth, the filter should stay there rather than wander off
    // and come back — this is the handover from an offline analysis.
    ParticleFilterConfig config;
    BeatParticleFilter filter{config};
    filter.seedTempo(150.0, 0.02);
    const double now = feedSteady(filter, 150.0, 8.0);
    EXPECT_NEAR(filter.estimate(now).bpm, 150.0, 4.0);
}

TEST(ParticleFilter, IsDeterministic) {
    BeatParticleFilter a{ParticleFilterConfig{}};
    BeatParticleFilter b{ParticleFilterConfig{}};
    const double now_a = feedSteady(a, 128.0, 6.0);
    const double now_b = feedSteady(b, 128.0, 6.0);

    const BeatEstimate ea = a.estimate(now_a);
    const BeatEstimate eb = b.estimate(now_b);
    EXPECT_EQ(ea.bpm, eb.bpm);
    EXPECT_EQ(ea.next_beat_sec, eb.next_beat_sec);
    EXPECT_EQ(ea.confidence, eb.confidence);
}

TEST(ParticleFilter, ResetReturnsToThePrior) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    feedSteady(filter, 160.0, 8.0);
    filter.reset();

    const BeatEstimate estimate = filter.estimate(0.0);
    // Back to a cloud drawn from the prior: somewhere in range, and certain of
    // nothing. Not "back to 120" — the prior is a wide distribution, and the
    // busiest corner of a fresh draw is wherever the dice put it.
    EXPECT_GE(estimate.bpm, filter.config().min_bpm);
    EXPECT_LE(estimate.bpm, filter.config().max_bpm);
    EXPECT_EQ(estimate.confidence, 0.0);
    EXPECT_EQ(filter.stats().observations, 0u);
}

TEST(ParticleFilter, IgnoresFramesOutOfOrder) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    const double now = feedSteady(filter, 120.0, 4.0);
    const auto before = filter.stats().observations;

    filter.observe(now - 0.5, 3.0);
    EXPECT_EQ(filter.stats().observations, before);
    EXPECT_EQ(filter.stats().out_of_order, 1u);
}

TEST(ParticleFilter, ReanchorsAcrossAGapInsteadOfWindingForward) {
    BeatParticleFilter filter{ParticleFilterConfig{}};
    double now = feedSteady(filter, 120.0, 6.0);
    const double bpm = filter.estimate(now).bpm;

    now += 30.0;  // the app was suspended, the device restarted
    filter.observe(now, 0.0);
    EXPECT_EQ(filter.stats().reanchors, 1u);

    // The tempo is kept — it is the phase that a thirty-second hole destroys.
    EXPECT_NEAR(filter.estimate(now).bpm, bpm, 2.0);
    EXPECT_LT(filter.estimate(now).confidence, 0.5);
}

TEST(ParticleFilter, StaysInsideItsTempoRange) {
    ParticleFilterConfig config;
    config.min_bpm = 100.0;
    config.max_bpm = 140.0;
    BeatParticleFilter filter{config};

    // Real beats at 70, outside the range: the filter must clamp rather than
    // produce a tempo it was told is impossible.
    const double now = feedSteady(filter, 70.0, 10.0);
    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_GE(estimate.bpm, 100.0 - 1e-9);
    EXPECT_LE(estimate.bpm, 140.0 + 1e-9);
}

TEST(ParticleFilterConfig, RejectsTheImpossible) {
    EXPECT_TRUE(ParticleFilterConfig{}.valid());

    ParticleFilterConfig too_few;
    too_few.particles = 4;
    EXPECT_FALSE(too_few.valid());

    ParticleFilterConfig inverted;
    inverted.min_bpm = 200.0;
    inverted.max_bpm = 100.0;
    EXPECT_FALSE(inverted.valid());

    ParticleFilterConfig wide_window;
    wide_window.beat_window = 0.9;  // wider than half a period means no window
    EXPECT_FALSE(wide_window.valid());
}

TEST(ParticleFilter, PinningHoldsTheTempoAgainstTheAudio) {
    ParticleFilterConfig config;
    BeatParticleFilter filter{config};
    filter.pinPeriod(0.5);
    EXPECT_TRUE(filter.pinned());

    // Ten seconds of insisting on 160. The whole promise of manual mode is that
    // it does not matter.
    const double now = feedSteady(filter, 160.0, 10.0);

    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_NEAR(estimate.bpm, 120.0, 1e-9);
    // Not exactly zero: the spread is a difference of two sums of squares, and
    // on a cloud that is all one number that subtraction cancels down to the
    // rounding. A ten-millionth of an octave is the arithmetic, not a doubt.
    EXPECT_NEAR(estimate.tempo_spread_octaves, 0.0, 1e-6);
}

TEST(ParticleFilter, PinnedItStillHasThePhaseToFind) {
    ParticleFilterConfig config;
    BeatParticleFilter filter{config};
    filter.pinPeriod(0.5);

    // No phase was handed over — the cloud starts spread over the period and
    // has to work it out from the audio alone, which with the tempo already
    // known it can.
    const double now = feedSteady(filter, 120.0, 10.0, 0.0, 1.0, 0.2);

    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_LT(offGrid(estimate.next_beat_sec, 120.0, 0.2), 0.03);
    EXPECT_GT(estimate.confidence, 0.4);
}

TEST(ParticleFilter, SeedingThePhasePutsTheGridWhereItIsTold) {
    ParticleFilterConfig config;
    BeatParticleFilter filter{config};
    filter.pinPeriod(0.5);

    // Two seconds of audio only to give the filter a clock, then the grid is
    // dictated: this is the correlation handing its answer over.
    const double now = feedSteady(filter, 120.0, 2.0, 0.0, 1.0, 0.2);
    filter.seedPhase(now + 0.37);

    const BeatEstimate estimate = filter.estimate(now);
    EXPECT_NEAR(estimate.next_beat_sec, now + 0.37, 1e-9);
    EXPECT_NEAR(estimate.bpm, 120.0, 1e-9);
}

TEST(ParticleFilter, ATempoOutsideTheRangeIsStillTheUsersTempo) {
    ParticleFilterConfig config;
    config.max_bpm = 220.0;
    BeatParticleFilter filter{config};

    // 300 is outside anything the tracker would ever guess at. The range is a
    // belief about what music is likely to be, and it has no business
    // overruling a number somebody typed.
    filter.pinPeriod(60.0 / 300.0);
    const double now = feedSteady(filter, 300.0, 8.0);
    EXPECT_NEAR(filter.estimate(now).bpm, 300.0, 1e-9);
}

TEST(ParticleFilter, AResetKeepsThePin) {
    ParticleFilterConfig config;
    BeatParticleFilter filter{config};
    filter.pinPeriod(0.4);
    feedSteady(filter, 150.0, 4.0);

    // A reset forgets what was heard. The tempo the user typed was never heard.
    filter.reset();
    EXPECT_TRUE(filter.pinned());
    const double now = feedSteady(filter, 150.0, 6.0);
    EXPECT_NEAR(filter.estimate(now).bpm, 150.0, 1e-9);
}

TEST(ParticleFilter, UnpinningLetsItLookAgain) {
    ParticleFilterConfig config;
    BeatParticleFilter filter{config};
    filter.pinPeriod(0.5);
    feedSteady(filter, 100.0, 4.0);
    ASSERT_NEAR(filter.estimate(4.0).bpm, 120.0, 1e-9);

    filter.unpinPeriod();
    EXPECT_FALSE(filter.pinned());
    const double now = feedSteady(filter, 100.0, 14.0, 4.0);
    EXPECT_NEAR(filter.estimate(now).bpm, 100.0, 3.0);
}
