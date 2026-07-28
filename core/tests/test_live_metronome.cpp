#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "render/live_metronome.hpp"
#include "support.hpp"

using tiktak::render::LiveMetronome;
using tiktak::render::LiveMetronomeConfig;

namespace {

constexpr double kRate = 48000.0;
constexpr std::size_t kBlock = 512;

LiveMetronomeConfig config(double round_trip_sec = 0.0) {
    LiveMetronomeConfig cfg;
    cfg.tracker = tiktak::tracking::liveConfigFor(kRate);
    cfg.click.sample_rate = kRate;
    cfg.round_trip_sec = round_trip_sec;
    return cfg;
}

// Runs the room's audio through capture and the metronome's own output through
// process, one block at a time, the way a duplex device would. Returns the
// stream times at which a click was written into the output.
std::vector<double> run(LiveMetronome& metronome, const std::vector<float>& room,
                        double from_sec = 0.0) {
    std::vector<float> out(kBlock);
    std::vector<double> clicks;

    double time = from_sec;
    std::size_t quiet_run = 1000;
    for (std::size_t i = 0; i + kBlock <= room.size(); i += kBlock) {
        metronome.capture(time, room.data() + i, kBlock);

        std::fill(out.begin(), out.end(), 0.0f);
        metronome.process(time, out.data(), kBlock);

        for (std::size_t s = 0; s < kBlock; ++s) {
            // A sustained silence has to pass before another onset counts. A
            // click is a decaying sine and crosses zero twice a cycle, so a
            // detector that re-arms on one quiet sample finds a hundred onsets
            // in one click — and the extra ones, scattered through its decay,
            // look exactly like a metronome that is tens of milliseconds late.
            if (std::fabs(out[s]) > 1e-4f) {
                if (quiet_run >= 1000) clicks.push_back(time + static_cast<double>(s) / kRate);
                quiet_run = 0;
            } else {
                ++quiet_run;
            }
        }

        time += static_cast<double>(kBlock) / kRate;
    }

    return clicks;
}

double offGrid(double time, double bpm, double phase_sec) {
    const double period = 60.0 / bpm;
    const double since = time - phase_sec;
    return std::fabs(since - std::round(since / period) * period);
}

}  // namespace

TEST(LiveMetronome, ClicksOnTheBeatsItHearsInTheRoom) {
    LiveMetronome metronome{config()};
    metronome.start();

    // Padded with silence, because the assertion below holds every beat handed
    // out to an audible click. A beat handed out inside the last lookahead has
    // its click scheduled but not yet rendered when the stream ends — a real
    // device keeps running, so the room gets a tail for the click to sound in.
    // The tracker coasts through the tail, and any beat it hands out there is
    // heard and counted on both sides alike.
    auto room = tiktak::test::clickTrack(120.0, 16.0, kRate, 1.0);
    room.resize(room.size() + static_cast<std::size_t>(0.4 * kRate), 0.0f);
    const std::vector<double> clicks = run(metronome, room);

    ASSERT_GT(clicks.size(), 12u);
    // The opening seconds go to finding the tempo; after that every click it
    // plays belongs to the room's own grid. A click's own attack takes a few
    // samples to clear the threshold, so the tolerance is the beat tolerance
    // plus a little, not the sample-exact figure the offline path is held to.
    for (std::size_t i = clicks.size() / 2; i < clicks.size(); ++i) {
        EXPECT_LT(offGrid(clicks[i], 120.0, 1.0), 0.05) << "click " << i << " at " << clicks[i];
    }

    const LiveMetronome::Stats stats = metronome.stats();
    EXPECT_EQ(stats.beats, clicks.size());
    EXPECT_TRUE(stats.clean());
}

TEST(LiveMetronome, PutsTheClickOutEarlyByTheRoundTrip) {
    // With a 30 ms round trip the click has to leave 30 ms before the beat it
    // is meant to coincide with, or it arrives late by exactly that much.
    constexpr double kRoundTrip = 0.030;
    LiveMetronome metronome{config(kRoundTrip)};
    metronome.start();

    const auto room = tiktak::test::clickTrack(120.0, 16.0, kRate, 1.0);
    const std::vector<double> clicks = run(metronome, room);

    ASSERT_GT(clicks.size(), 12u);
    for (std::size_t i = clicks.size() / 2; i < clicks.size(); ++i) {
        // Submitted a round trip early, so it lands on the grid shifted back.
        EXPECT_LT(offGrid(clicks[i] + kRoundTrip, 120.0, 1.0), 0.05)
            << "click " << i << " at " << clicks[i];
    }
}

TEST(LiveMetronome, GatesTheClicksItPlays) {
    LiveMetronome metronome{config()};
    metronome.start();

    const auto room = tiktak::test::clickTrack(120.0, 16.0, kRate, 1.0);
    run(metronome, room);

    // Nothing has to be arranged for this: playing a click gates it, which is
    // the arrangement that stops the metronome hearing itself.
    EXPECT_GT(metronome.stats().gated, 0u);
}

TEST(LiveMetronome, PlaysNothingUntilStarted) {
    LiveMetronome metronome{config()};

    const auto room = tiktak::test::clickTrack(120.0, 12.0, kRate, 1.0);
    EXPECT_TRUE(run(metronome, room).empty());
    EXPECT_EQ(metronome.stats().beats, 0u);

    // It was listening the whole time, though, so starting it is instant.
    EXPECT_NEAR(metronome.estimate(12.0).bpm, 120.0, 4.0);
    EXPECT_GT(metronome.estimate(12.0).confidence, 0.4);
}

TEST(LiveMetronome, SilenceCutsTheClickAndStopLetsItRing) {
    LiveMetronome metronome{config()};
    metronome.start();
    const auto room = tiktak::test::clickTrack(120.0, 12.0, kRate, 1.0);
    run(metronome, room);

    metronome.stop();
    EXPECT_FALSE(metronome.running());
    metronome.silence();

    std::vector<float> out(kBlock, 0.0f);
    metronome.process(12.0, out.data(), kBlock);
    for (float sample : out) EXPECT_EQ(sample, 0.0f);
}

TEST(LiveMetronomeConfig, RejectsTheImpossible) {
    EXPECT_TRUE(config().valid());

    LiveMetronomeConfig backwards = config();
    backwards.round_trip_sec = -0.01;
    EXPECT_FALSE(backwards.valid());
}

TEST(LiveMetronome, ManualModeWaitsForTheRoomThenHoldsTheTempoItWasGiven) {
    LiveMetronome metronome{config()};
    metronome.setManualTempo(120.0);
    metronome.start();
    EXPECT_TRUE(metronome.waiting());

    // Four seconds of nothing: started, and deliberately silent. Manual mode's
    // whole point is coming in *with* the music, and there is none yet.
    const std::vector<float> quiet(static_cast<std::size_t>(4.0 * kRate), 0.0f);
    EXPECT_TRUE(run(metronome, quiet).empty());
    EXPECT_TRUE(metronome.waiting());

    // Then the band comes in, off any round number of beats. The click falls in
    // on their phase, and every click it plays is 120's spacing apart.
    const auto room = tiktak::test::clickTrack(120.0, 16.0, kRate, 1.31);
    const std::vector<double> clicks = run(metronome, room, 4.0);

    EXPECT_FALSE(metronome.waiting());
    ASSERT_GT(clicks.size(), 20u);
    for (std::size_t i = 1; i < clicks.size(); ++i) {
        EXPECT_NEAR(clicks[i] - clicks[i - 1], 0.5, 0.03) << "click " << i;
    }
    for (std::size_t i = clicks.size() / 2; i < clicks.size(); ++i) {
        const double since = clicks[i] - 1.31;
        EXPECT_LT(std::fabs(since - std::round(since / 0.5) * 0.5), 0.05) << "click " << i;
    }
    EXPECT_TRUE(metronome.stats().clean());
}
