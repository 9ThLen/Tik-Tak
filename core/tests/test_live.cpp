#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "support.hpp"
#include "tracking/live.hpp"

using tiktak::tracking::BeatEstimate;
using tiktak::tracking::LiveConfig;
using tiktak::tracking::LiveTracker;

namespace {

constexpr double kRate = 48000.0;
constexpr std::size_t kBlock = 512;

LiveConfig liveConfig() { return tiktak::tracking::liveConfigFor(kRate); }

// Feeds a signal block by block against a virtual clock, the way a capture
// callback would. Returns the stream time one past the end.
double feed(LiveTracker& tracker, const std::vector<float>& audio, double from_sec = 0.0) {
    double time = from_sec;
    for (std::size_t i = 0; i < audio.size(); i += kBlock) {
        const std::size_t n = std::min(kBlock, audio.size() - i);
        tracker.process(time, audio.data() + i, n);
        time += static_cast<double>(n) / kRate;
    }
    return time;
}

double offGrid(double time, double bpm, double phase_sec) {
    const double period = 60.0 / bpm;
    const double since = time - phase_sec;
    return std::fabs(since - std::round(since / period) * period);
}

}  // namespace

TEST(LiveTracker, FollowsAClickTrack) {
    LiveTracker tracker{liveConfig()};
    const auto audio = tiktak::test::clickTrack(120.0, 14.0, kRate, 1.0);
    const double now = feed(tracker, audio);

    const BeatEstimate estimate = tracker.estimate(now);
    EXPECT_NEAR(estimate.bpm, 120.0, 3.0);
    EXPECT_GT(estimate.confidence, 0.4);
    EXPECT_LT(offGrid(estimate.next_beat_sec, 120.0, 1.0), 0.03);
}

TEST(LiveTracker, HandsOutEachBeatOnceAndAhead) {
    LiveTracker tracker{liveConfig()};
    const auto audio = tiktak::test::clickTrack(120.0, 14.0, kRate, 1.0);

    constexpr double kLookahead = 0.05;
    std::vector<double> beats;
    double time = 0.0;
    for (std::size_t i = 0; i < audio.size(); i += kBlock) {
        const std::size_t n = std::min(kBlock, audio.size() - i);
        tracker.process(time, audio.data() + i, n);
        time += static_cast<double>(n) / kRate;

        double beat = 0.0;
        while (tracker.takeBeat(time, kLookahead, &beat)) {
            EXPECT_GE(beat, time);
            EXPECT_LE(beat, time + kLookahead);
            beats.push_back(beat);
        }
    }

    // Roughly 26 beats are in the track; the opening seconds go to finding the
    // tempo, so more than half of them being played is the bar.
    ASSERT_GT(beats.size(), 14u);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        EXPECT_GT(beats[i], beats[i - 1]) << "beat " << i;
    }
    // Once locked, every beat handed out is one of the track's own.
    for (std::size_t i = beats.size() / 2; i < beats.size(); ++i) {
        EXPECT_LT(offGrid(beats[i], 120.0, 1.0), 0.04) << "beat " << i << " at " << beats[i];
    }
}

TEST(LiveTracker, GatingItsOwnClickStopsItTrackingItself) {
    // The failure this prevents: a metronome hears its own click, locks onto
    // it, reports full confidence and stops responding to the room. The audio
    // here is *only* our click, so a gated tracker must learn nothing at all.
    const auto own_click = tiktak::test::clickTrack(120.0, 12.0, kRate, 1.0);

    LiveTracker deaf{liveConfig()};
    double time = 0.0;
    for (std::size_t i = 0; i < own_click.size(); i += kBlock) {
        const std::size_t n = std::min(kBlock, own_click.size() - i);
        const double end = time + static_cast<double>(n) / kRate;
        // Every click we "played" is declared to the tracker as heard.
        for (double beat = 1.0; beat < 12.0; beat += 0.5) {
            if (beat >= time && beat < end) deaf.gateClick(beat);
        }
        deaf.process(time, own_click.data() + i, n);
        time = end;
    }

    LiveTracker credulous{liveConfig()};
    const double now = feed(credulous, own_click);

    EXPECT_GT(deaf.stats().gated, 0u);
    EXPECT_GT(credulous.estimate(now).confidence, 0.4);
    EXPECT_LT(deaf.estimate(time).confidence, 0.25);
}

TEST(LiveTracker, GatingLeavesRealMusicAudible) {
    // Gating must cost only the frames it covers. Here the room plays at 150
    // while our own click insists on 120: the tracker should follow the room.
    const auto room = tiktak::test::clickTrack(150.0, 16.0, kRate, 0.7);

    LiveTracker tracker{liveConfig()};
    double time = 0.0;
    for (std::size_t i = 0; i < room.size(); i += kBlock) {
        const std::size_t n = std::min(kBlock, room.size() - i);
        const double end = time + static_cast<double>(n) / kRate;
        for (double beat = 0.5; beat < 16.0; beat += 0.5) {
            if (beat >= time && beat < end) tracker.gateClick(beat);
        }
        tracker.process(time, room.data() + i, n);
        time = end;
    }

    EXPECT_NEAR(tracker.estimate(time).bpm, 150.0, 6.0);
}

TEST(LiveTracker, IsNotFooledByHowLoudTheRoomIs) {
    const auto loud = tiktak::test::clickTrack(132.0, 12.0, kRate, 0.5);
    std::vector<float> quiet(loud.size());
    for (std::size_t i = 0; i < loud.size(); ++i) quiet[i] = loud[i] * 0.02f;

    LiveTracker a{liveConfig()};
    LiveTracker b{liveConfig()};
    const double now_a = feed(a, loud);
    const double now_b = feed(b, quiet);

    // Not merely both in range: the same answer, because the onset function is
    // normalised against a running level before the filter multiplies it by a
    // fixed gain. Thirty-four decibels apart is more than any room changes by.
    EXPECT_NEAR(a.estimate(now_a).bpm, b.estimate(now_b).bpm, 4.0);
    EXPECT_NEAR(b.estimate(now_b).bpm, 132.0, 4.0);
}

TEST(LiveTracker, SilenceIsNotABeat) {
    LiveTracker tracker{liveConfig()};
    const auto audio = tiktak::test::silence(static_cast<std::size_t>(10.0 * kRate));
    const double now = feed(tracker, audio);

    EXPECT_LT(tracker.estimate(now).confidence, 0.3);
    double beat = 0.0;
    EXPECT_FALSE(tracker.takeBeat(now, 0.05, &beat));
    EXPECT_EQ(tracker.stats().beats, 0u);
}

TEST(LiveTracker, CountsABufferThatDoesNotFollowTheLast) {
    LiveTracker tracker{liveConfig()};
    const auto audio = tiktak::test::clickTrack(120.0, 4.0, kRate, 0.5);

    double time = feed(tracker, audio);
    EXPECT_EQ(tracker.stats().discontinuities, 0u);

    time += 0.25;  // the device dropped a quarter of a second
    tracker.process(time, audio.data(), kBlock);
    EXPECT_EQ(tracker.stats().discontinuities, 1u);

    // Timestamps follow the clock, so the beats it goes on to predict are in
    // the caller's time, not a quarter second behind it.
    EXPECT_GT(tracker.estimate(time).next_beat_sec, time);
}

TEST(LiveTracker, SeedingFromAnOfflineAnalysisSkipsTheSearch) {
    LiveConfig config = liveConfig();
    LiveTracker seeded{config};
    seeded.seedTempo(96.0, 0.02);

    // Two seconds is three beats: enough to confirm a tempo you were handed,
    // nowhere near enough to find one from scratch.
    const auto audio = tiktak::test::clickTrack(96.0, 2.0, kRate, 0.3);
    const double now = feed(seeded, audio);

    LiveTracker cold{config};
    feed(cold, audio);

    EXPECT_NEAR(seeded.estimate(now).bpm, 96.0, 3.0);
    EXPECT_LT(std::fabs(seeded.estimate(now).bpm - 96.0),
              std::fabs(cold.estimate(now).bpm - 96.0));
}

TEST(LiveTracker, ResetForgetsEverything) {
    LiveTracker tracker{liveConfig()};
    const auto audio = tiktak::test::clickTrack(150.0, 8.0, kRate, 0.5);
    feed(tracker, audio);
    tracker.reset();

    EXPECT_EQ(tracker.stats().frames, 0u);
    EXPECT_EQ(tracker.stats().beats, 0u);
    EXPECT_LT(tracker.estimate(0.0).confidence, 0.2);
}

TEST(LiveTracker, IgnoresNothing) {
    LiveTracker tracker{liveConfig()};
    tracker.process(0.0, nullptr, 128);
    tracker.process(0.0, tiktak::test::silence(16).data(), 0);
    EXPECT_EQ(tracker.stats().frames, 0u);
}

TEST(LiveConfig, RejectsTheImpossible) {
    EXPECT_TRUE(liveConfig().valid());

    LiveConfig backwards = liveConfig();
    backwards.release_confidence = 0.9;  // must be below lock_confidence
    EXPECT_FALSE(backwards.valid());

    LiveConfig no_level = liveConfig();
    no_level.onset_peak_tau_sec = 0.0;
    EXPECT_FALSE(no_level.valid());
}
