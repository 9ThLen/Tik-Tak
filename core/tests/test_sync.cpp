#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "tracking/sync.hpp"

using tiktak::tracking::PhaseSync;
using tiktak::tracking::SyncConfig;

namespace {

// The onset frame rate. PhaseSync never sees samples, so this is just "how
// often it is told something"; 100 a second is close to the ODF's own rate at
// 48 kHz and makes the arithmetic in these tests readable.
constexpr double kHop = 0.01;

double offGrid(double time, double period, double phase) {
    const double since = time - phase;
    return std::fabs(since - std::round(since / period) * period);
}

// Feeds onset frames: a hit on every beat of `period` starting at `phase`, and
// optionally a weaker one exactly between them. Returns the time it stopped at.
double feed(PhaseSync& sync, double from_sec, double seconds, double period, double phase,
            double strength = 1.0, double between = 0.0) {
    const auto frames = static_cast<std::size_t>(seconds / kHop);
    double t = from_sec;
    for (std::size_t i = 0; i < frames; ++i, t += kHop) {
        double onset = 0.0;
        if (offGrid(t, period, phase) < 0.5 * kHop) onset = strength;
        else if (between > 0.0 && offGrid(t, period, phase + 0.5 * period) < 0.5 * kHop)
            onset = between;
        sync.observe(t, onset);
    }
    return t;
}

// Deterministic uniform noise, so a failure is a failure and not a bad day.
struct Noise {
    std::uint64_t state = 0x243F6A8885A308D3ull;
    double next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    }
};

}  // namespace

TEST(PhaseSync, FindsThePhaseWhereverItStarts) {
    // Not a multiple of the frame period: the answer is an angle, not a frame
    // index, so nothing about it should be quantised to the front-end's grid
    // beyond where the onsets themselves land.
    for (double phase : {0.0, 0.123, 0.317, 0.481}) {
        PhaseSync sync;
        sync.setPeriod(0.5);
        const double end = feed(sync, 0.0, 6.0, 0.5, phase);

        EXPECT_TRUE(sync.ready()) << "phase " << phase;
        const double beat = sync.nextBeat(end);
        EXPECT_GT(beat, end) << "phase " << phase;
        EXPECT_LT(offGrid(beat, 0.5, phase), kHop) << "phase " << phase;
    }
}

TEST(PhaseSync, DoesNotCareHowLoudTheRoomIs) {
    // Level cancels: the answer is a ratio of the same energy to itself.
    PhaseSync loud;
    loud.setPeriod(0.5);
    const double end = feed(loud, 0.0, 6.0, 0.5, 0.2, 1.0);

    PhaseSync quiet;
    quiet.setPeriod(0.5);
    feed(quiet, 0.0, 6.0, 0.5, 0.2, 0.02);

    EXPECT_TRUE(quiet.ready());
    EXPECT_NEAR(loud.nextBeat(end), quiet.nextBeat(end), 1e-9);
    EXPECT_NEAR(loud.strength(), quiet.strength(), 1e-9);
}

TEST(PhaseSync, SubdivisionsWeakenTheAnswerWithoutMovingIt) {
    PhaseSync clean;
    clean.setPeriod(0.5);
    const double end = feed(clean, 0.0, 6.0, 0.5, 0.2);

    PhaseSync eighths;
    eighths.setPeriod(0.5);
    feed(eighths, 0.0, 6.0, 0.5, 0.2, 1.0, 0.5);

    // A hit exactly between the beats sits at the opposite angle, so it does
    // not dilute the estimate — it subtracts from it. That shows up as less
    // strength and not as a different answer, which is the behaviour wanted:
    // the phase is still right, the filter is simply told to trust it less.
    EXPECT_TRUE(eighths.ready());
    EXPECT_LT(eighths.strength(), clean.strength());
    EXPECT_LT(offGrid(eighths.nextBeat(end), 0.5, 0.2), kHop);
}

TEST(PhaseSync, NoiseHasNoPhase) {
    // The reason there is no separate "has the music started" threshold: room
    // noise has onsets in it, and no phase. One measure answers both.
    PhaseSync sync;
    sync.setPeriod(0.5);

    Noise noise;
    double t = 0.0;
    for (int i = 0; i < 800; ++i, t += kHop) sync.observe(t, noise.next());

    EXPECT_LT(sync.strength(), sync.config().acquire_strength);
    EXPECT_FALSE(sync.ready());
}

TEST(PhaseSync, SilenceIsNotAPhase) {
    PhaseSync sync;
    sync.setPeriod(0.5);
    feed(sync, 0.0, 6.0, 0.5, 0.0, 0.0);

    EXPECT_EQ(sync.strength(), 0.0);
    EXPECT_FALSE(sync.ready());
}

TEST(PhaseSync, HoldsTheAnswerThroughAThinPatch) {
    PhaseSync sync;
    sync.setPeriod(0.5);
    double t = feed(sync, 0.0, 6.0, 0.5, 0.2);
    ASSERT_TRUE(sync.ready());

    // Two seconds of nothing but noise. The strength falls, but not below the
    // release threshold, so the metronome does not switch itself off because
    // the singer took a breath.
    Noise noise;
    for (int i = 0; i < 200; ++i, t += kHop) sync.observe(t, 0.35 * noise.next());
    EXPECT_TRUE(sync.ready());
    EXPECT_LT(offGrid(sync.nextBeat(t), 0.5, 0.2), 0.02);
}

TEST(PhaseSync, FollowsThePlayerMovingWithin) {
    PhaseSync sync;
    sync.setPeriod(0.5);
    double t = feed(sync, 0.0, 6.0, 0.5, 0.2);
    ASSERT_LT(offGrid(sync.nextBeat(t), 0.5, 0.2), kHop);

    // The band shifts a fifth of a beat late and stays there. The correlation
    // is a decaying mean, so it walks across rather than jumping — and after
    // several time constants it is on the new phase.
    t = feed(sync, t, 12.0, 0.5, 0.3);
    EXPECT_LT(offGrid(sync.nextBeat(t), 0.5, 0.3), 0.02);
}

TEST(PhaseSync, ANewPeriodIsANewQuestion) {
    PhaseSync sync;
    sync.setPeriod(0.5);
    feed(sync, 0.0, 6.0, 0.5, 0.2);
    ASSERT_TRUE(sync.ready());

    // The accumulator holds angles measured against the old period. Carrying
    // them over would not be an approximation of the new answer, it would be a
    // different quantity wearing its name.
    sync.setPeriod(0.4);
    EXPECT_FALSE(sync.ready());
    EXPECT_EQ(sync.strength(), 0.0);

    const double end = feed(sync, 6.0, 6.0, 0.4, 0.1);
    EXPECT_TRUE(sync.ready());
    EXPECT_LT(offGrid(sync.nextBeat(end), 0.4, 0.1), kHop);
}

TEST(PhaseSync, SettingTheSamePeriodChangesNothing) {
    PhaseSync sync;
    sync.setPeriod(0.5);
    const double end = feed(sync, 0.0, 6.0, 0.5, 0.2);
    const double before = sync.nextBeat(end);

    sync.setPeriod(0.5);
    EXPECT_TRUE(sync.ready());
    EXPECT_EQ(sync.nextBeat(end), before);
}

TEST(PhaseSync, AGapInTheStreamIsNotAPause) {
    PhaseSync sync;
    sync.setPeriod(0.5);
    feed(sync, 0.0, 6.0, 0.5, 0.2);
    ASSERT_TRUE(sync.ready());

    // The app was suspended for five seconds. Whatever the phase was on the
    // other side of that has no bearing on this side, and decaying the
    // accumulator across the gap would be arithmetic on nothing.
    sync.observe(11.0, 1.0);
    EXPECT_FALSE(sync.ready());
}

TEST(PhaseSync, IgnoresNothing) {
    PhaseSync sync;

    // No period set: there is no question to answer yet.
    sync.observe(0.0, 1.0);
    sync.observe(0.1, 1.0);
    EXPECT_EQ(sync.strength(), 0.0);
    EXPECT_FALSE(sync.ready());
    EXPECT_EQ(sync.period(), 0.0);
    EXPECT_EQ(sync.nextBeat(3.0), 3.0);

    sync.setPeriod(0.5);
    feed(sync, 0.0, 6.0, 0.5, 0.2);
    ASSERT_TRUE(sync.ready());

    // Frames out of order are dropped, not fed backwards through the decay.
    const double before = sync.strength();
    sync.observe(1.0, 1.0);
    EXPECT_EQ(sync.strength(), before);

    sync.reset();
    EXPECT_FALSE(sync.ready());
    EXPECT_EQ(sync.strength(), 0.0);
}

TEST(SyncConfig, RejectsTheImpossible) {
    EXPECT_TRUE(SyncConfig{}.valid());

    SyncConfig forgetful;
    forgetful.tau_sec = 0.0;
    EXPECT_FALSE(forgetful.valid());

    SyncConfig backwards;
    backwards.release_strength = backwards.acquire_strength + 0.1;
    EXPECT_FALSE(backwards.valid());
}
