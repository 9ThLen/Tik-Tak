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

    LiveConfig loose = liveConfig();
    loose.sync.max_drift = 0.9;  // "nudge" has to mean something
    EXPECT_FALSE(loose.valid());
}

// --------------------------------------------------------------- manual mode

namespace {

// Feeds audio and collects every beat handed out, the way a duplex shell would.
std::vector<double> collect(LiveTracker& tracker, const std::vector<float>& audio,
                            double from_sec = 0.0) {
    constexpr double kLookahead = 0.05;
    std::vector<double> beats;
    double time = from_sec;
    for (std::size_t i = 0; i < audio.size(); i += kBlock) {
        const std::size_t n = std::min(kBlock, audio.size() - i);
        tracker.process(time, audio.data() + i, n);
        time += static_cast<double>(n) / kRate;

        double beat = 0.0;
        while (tracker.takeBeat(time, kLookahead, &beat)) beats.push_back(beat);
    }
    return beats;
}

}  // namespace

TEST(LiveTracker, ManualModeWaitsForSomethingToFallInWith) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);
    EXPECT_TRUE(tracker.waiting());
    EXPECT_EQ(tracker.manualTempo(), 120.0);

    // Eight seconds of an empty room. The tempo is known and it would be easy
    // to just start clicking — but the point of the mode is to come in *with*
    // the music, and there is none yet.
    const std::vector<float> quiet(static_cast<std::size_t>(8.0 * kRate), 0.0f);
    EXPECT_TRUE(collect(tracker, quiet).empty());
    EXPECT_TRUE(tracker.waiting());
    EXPECT_EQ(tracker.syncStrength(), 0.0);
}

TEST(LiveTracker, ManualModeFallsInOnThePhaseItHears) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);

    // The room starts a second in, off any round number of beats.
    const auto audio = tiktak::test::clickTrack(120.0, 14.0, kRate, 1.23);
    const std::vector<double> beats = collect(tracker, audio);

    EXPECT_FALSE(tracker.waiting());
    ASSERT_GT(beats.size(), 12u);
    for (std::size_t i = 0; i < beats.size(); ++i) {
        EXPECT_LT(offGrid(beats[i], 120.0, 1.23), 0.04) << "beat " << i << " at " << beats[i];
    }
}

TEST(LiveTracker, ManualModeKeepsTheUsersTempoAndNotTheRooms) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);

    // A half-time groove: the room puts a hit down every second, and the user
    // has asked to be counted in eighths of it. In auto mode the tracker's job
    // would be to find 60; here its job is to say where the beat falls and let
    // the user's number decide how often to click.
    const auto audio = tiktak::test::clickTrack(60.0, 16.0, kRate, 1.0);
    const std::vector<double> beats = collect(tracker, audio);

    EXPECT_NEAR(tracker.estimate(16.0).bpm, 120.0, 1e-9);
    ASSERT_GT(beats.size(), 12u);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        EXPECT_NEAR(beats[i] - beats[i - 1], 0.5, 0.02) << "beat " << i;
    }
    for (std::size_t i = 0; i < beats.size(); ++i) {
        EXPECT_LT(offGrid(beats[i], 120.0, 1.0), 0.05) << "beat " << i << " at " << beats[i];
    }
}

TEST(LiveTracker, ManualModeWillNotFallInWithARoomThatHasNoSuchBeat) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);

    // The room is at 150 and the dial says 120. There is no 120 phase in that
    // room to find — the two grids slide past each other twice a second — and
    // saying so is better than clicking somewhere and calling it synchronised.
    // A shell shows this as "listening...", which is exactly what it is doing.
    const auto audio = tiktak::test::clickTrack(150.0, 16.0, kRate, 1.0);
    EXPECT_TRUE(collect(tracker, audio).empty());
    EXPECT_TRUE(tracker.waiting());
}

TEST(LiveTracker, ManualModeDoesNotStopWhenTheRoomDoes) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);

    const auto audio = tiktak::test::clickTrack(120.0, 8.0, kRate, 1.0);
    const std::vector<double> heard = collect(tracker, audio);
    ASSERT_GT(heard.size(), 6u);
    ASSERT_FALSE(tracker.waiting());

    // Now the band stops for eight seconds. The tempo was never theirs to take
    // away, so the click carries on — and it carries on *on the same grid*,
    // because with the period pinned and the observation zero-mean, silence
    // moves no weights at all.
    const std::vector<float> quiet(static_cast<std::size_t>(8.0 * kRate), 0.0f);
    const std::vector<double> alone = collect(tracker, quiet, 8.0);

    ASSERT_GT(alone.size(), 12u);
    for (std::size_t i = 0; i < alone.size(); ++i) {
        EXPECT_LT(offGrid(alone[i], 120.0, 1.0), 0.05) << "beat " << i << " at " << alone[i];
    }
}

TEST(LiveTracker, MovingTheSliderDoesNotSilenceTheClick) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);

    const auto audio = tiktak::test::clickTrack(120.0, 8.0, kRate, 1.0);
    ASSERT_GT(collect(tracker, audio).size(), 6u);
    ASSERT_FALSE(tracker.waiting());

    // A different spacing between clicks is what was asked for, not a fresh
    // start: the click keeps playing and simply changes rate.
    tracker.setManualTempo(90.0);
    EXPECT_FALSE(tracker.waiting());

    const std::vector<float> quiet(static_cast<std::size_t>(6.0 * kRate), 0.0f);
    const std::vector<double> after = collect(tracker, quiet, 8.0);
    ASSERT_GT(after.size(), 6u);
    for (std::size_t i = 1; i < after.size(); ++i) {
        EXPECT_NEAR(after[i] - after[i - 1], 60.0 / 90.0, 0.02) << "beat " << i;
    }
}

TEST(LiveTracker, ATempoOutsideTheRangeIsStillTheUsersTempo) {
    LiveConfig config = liveConfig();
    config.filter.max_bpm = 220.0;
    LiveTracker tracker{config};

    tracker.setManualTempo(240.0);
    const auto audio = tiktak::test::clickTrack(240.0, 10.0, kRate, 1.0);
    ASSERT_GT(collect(tracker, audio).size(), 20u);
    EXPECT_NEAR(tracker.estimate(10.0).bpm, 240.0, 1e-9);
}

TEST(LiveTracker, GoingBackToAutoLooksForTheTempoAgain) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);
    const auto wrong = tiktak::test::clickTrack(100.0, 6.0, kRate, 1.0);
    collect(tracker, wrong);
    ASSERT_NEAR(tracker.estimate(6.0).bpm, 120.0, 1e-9);

    tracker.setManualTempo(0.0);
    EXPECT_EQ(tracker.manualTempo(), 0.0);
    EXPECT_FALSE(tracker.waiting());

    const auto audio = tiktak::test::clickTrack(100.0, 16.0, kRate, 1.0);
    const double now = feed(tracker, audio, 6.0);
    EXPECT_NEAR(tracker.estimate(now).bpm, 100.0, 4.0);
}

TEST(LiveTracker, ResettingManualModeGoesBackToWaiting) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);
    const auto audio = tiktak::test::clickTrack(120.0, 8.0, kRate, 1.0);
    ASSERT_GT(collect(tracker, audio).size(), 6u);

    tracker.reset();
    // The audio is forgotten; the number the user typed is not.
    EXPECT_EQ(tracker.manualTempo(), 120.0);
    EXPECT_TRUE(tracker.waiting());
    EXPECT_EQ(tracker.syncStrength(), 0.0);
}

TEST(LiveTracker, ManualModeFollowsAPlayerDriftingInsideItsPullIn) {
    LiveTracker tracker{liveConfig()};
    tracker.setManualTempo(120.0);

    // The band is at 121 — well inside the ±2% the click is allowed to be
    // nudged by, so it should stay with them rather than sliding a sixth of a
    // second away over the twenty seconds this lasts.
    const auto audio = tiktak::test::clickTrack(121.0, 20.0, kRate, 1.0);
    const std::vector<double> beats = collect(tracker, audio);

    ASSERT_GT(beats.size(), 30u);
    for (std::size_t i = beats.size() / 2; i < beats.size(); ++i) {
        EXPECT_LT(offGrid(beats[i], 121.0, 1.0), 0.06) << "beat " << i << " at " << beats[i];
    }
}

// The seam a learned front end arrives through. What matters is that swapping
// the evidence changes only the evidence: the gating, the hysteresis and the
// publishing rules have to behave identically, or a measurement taken through
// observe() would not describe the tracker that ships.
TEST(LiveTracker, TracksAnActivationHandedToItDirectly) {
    LiveTracker tracker{tiktak::tracking::liveConfigFor(48000.0)};

    // A model's output for a 120 BPM piece: near one on the beat, near zero
    // between. No level normalisation is applied to this, and none is needed —
    // it is already a probability.
    constexpr double kFps = 50.0;
    constexpr double kPeriod = 0.5;
    double now = 0.0;
    for (int i = 0; i < static_cast<int>(20.0 * kFps); ++i) {
        now = i / kFps;
        const double since = now - std::round(now / kPeriod) * kPeriod;
        tracker.observe(now, std::fabs(since) < 0.011 ? 0.95 : 0.02);
    }

    const tiktak::tracking::BeatEstimate estimate = tracker.estimate(now);
    EXPECT_NEAR(estimate.bpm, 120.0, 3.0);
    EXPECT_GT(estimate.confidence, 0.5);
}

TEST(LiveTracker, GatesAnActivationTheSameWayItGatesAudio) {
    // A learned front end hears our own click too, and rather better than
    // spectral flux does, so the gate has to apply to it as well. Without
    // this the tracker would lock onto itself through the new seam while
    // remaining safe through the old one.
    LiveTracker tracker{tiktak::tracking::liveConfigFor(48000.0)};
    tracker.gateClick(1.0);

    for (int i = 0; i < 100; ++i) tracker.observe(i / 50.0, 1.0);
    EXPECT_GT(tracker.stats().gated, 0u);
}

TEST(LiveTracker, AnActivationIsNotRenormalisedByItsOwnLoudest) {
    // The difference from process(), and the reason observe() is not simply a
    // shortcut into it. A quiet stretch that the model correctly reports as
    // uncertain must stay uncertain: dividing by a running peak would scale
    // that doubt back up into confidence, which is exactly what the ODF path
    // has to do and a probability must not.
    LiveTracker loud{tiktak::tracking::liveConfigFor(48000.0)};
    LiveTracker quiet{tiktak::tracking::liveConfigFor(48000.0)};

    constexpr double kFps = 50.0;
    double now = 0.0;
    for (int i = 0; i < static_cast<int>(20.0 * kFps); ++i) {
        now = i / kFps;
        const double since = now - std::round(now / 0.5) * 0.5;
        const bool on_beat = std::fabs(since) < 0.011;
        loud.observe(now, on_beat ? 0.95 : 0.02);
        quiet.observe(now, on_beat ? 0.20 : 0.02);   // the model is unsure
    }

    EXPECT_GT(loud.estimate(now).confidence, quiet.estimate(now).confidence);
}

// -------------------------------------------- the learned front end, wired --
//
// The seam above proved the filter can be driven by an activation. These prove
// the tracker can produce one itself, from audio, through the same process()
// call a shell already makes — which is what "in the product" means.
//
// The weights are made up, as in test_beatnet.cpp. What is checked here is the
// wiring, and wiring is exactly what a synthetic model still exercises.
namespace {

std::vector<unsigned char> stubWeightFile() {
    using tiktak::ml::BeatNetWeights;
    std::vector<unsigned char> out(BeatNetWeights::kFileBytes);
    std::memcpy(out.data(), "TTBN", 4);

    const std::uint32_t header[7] = {
        1, BeatNetWeights::kFeatures, BeatNetWeights::kConvChannels,
        BeatNetWeights::kKernel, BeatNetWeights::kHidden, BeatNetWeights::kLayers,
        BeatNetWeights::kClasses,
    };
    for (std::size_t i = 0; i < 7; ++i) {
        for (std::size_t b = 0; b < 4; ++b) {
            out[4 + i * 4 + b] = static_cast<unsigned char>((header[i] >> (8 * b)) & 0xFF);
        }
    }
    for (std::size_t i = 0; i < BeatNetWeights::kParameters; ++i) {
        const float value = 0.05f * std::sin(0.7 * static_cast<double>(i));
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (std::size_t b = 0; b < 4; ++b) {
            out[BeatNetWeights::kHeaderBytes + i * 4 + b] =
                static_cast<unsigned char>((bits >> (8 * b)) & 0xFF);
        }
    }
    return out;
}

}  // namespace

TEST(LiveTracker, RunsTheModelFromTheSameProcessCall) {
    const auto blob = stubWeightFile();
    tiktak::ml::BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    LiveTracker tracker{liveConfig(), weights};
    EXPECT_TRUE(tracker.usingModel());

    const auto audio = tiktak::test::clickTrack(120.0, 4.0, kRate);
    feed(tracker, audio);

    // Fifty a second, not the ODF's rate: the model's front end has replaced
    // spectral flux rather than running alongside it.
    EXPECT_NEAR(static_cast<double>(tracker.stats().frames), 200.0, 3.0);
}

TEST(LiveTracker, WithoutWeightsItIsTheTrackerItAlwaysWas) {
    // The default has not moved. Spectral flux costs a few hundred kFLOP a
    // second and the model tens of MFLOP plus 1.6 MB of weights, and which of
    // those a device should spend is not a question a workstation can answer.
    LiveTracker tracker{liveConfig()};
    EXPECT_FALSE(tracker.usingModel());

    tiktak::ml::BeatNetWeights empty;
    LiveTracker refused{liveConfig(), empty};
    EXPECT_FALSE(refused.usingModel()) << "weights that never loaded must not engage";
}

TEST(LiveTracker, GatesItsOwnClickThroughTheModelToo) {
    const auto blob = stubWeightFile();
    tiktak::ml::BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    LiveTracker tracker{liveConfig(), weights};
    for (double t = 0.5; t < 3.0; t += 0.5) tracker.gateClick(t);

    const auto audio = tiktak::test::clickTrack(120.0, 3.0, kRate);
    feed(tracker, audio);

    // The gate is measured against the model's own 64 ms window, not the ODF's
    // frame — the two differ, and a gate sized for the wrong one either lets
    // the click through or blinds the tracker either side of it.
    EXPECT_GT(tracker.stats().gated, 0u);
    EXPECT_LT(tracker.stats().gated, tracker.stats().frames);
}

TEST(LiveTracker, ResettingForgetsWhatTheModelHeard) {
    const auto blob = stubWeightFile();
    tiktak::ml::BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    LiveTracker tracker{liveConfig(), weights};
    const auto audio = tiktak::test::clickTrack(120.0, 2.0, kRate);

    std::vector<double> first;
    feed(tracker, audio);
    const auto before = tracker.stats().frames;

    tracker.reset();
    feed(tracker, audio);

    // Same audio from a clean start gives the same frame count, and would not
    // if the recurrent state had survived the reset with the frame clock.
    EXPECT_EQ(tracker.stats().frames, before);
}

TEST(LiveTracker, TheModelPathIsBlockSizeAgnostic) {
    const auto blob = stubWeightFile();
    tiktak::ml::BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    const auto audio = tiktak::test::clickTrack(120.0, 3.0, kRate);

    auto run = [&](std::size_t block) {
        LiveTracker tracker{liveConfig(), weights};
        double time = 0.0;
        for (std::size_t i = 0; i < audio.size(); i += block) {
            const std::size_t n = std::min(block, audio.size() - i);
            tracker.process(time, audio.data() + i, n);
            time += static_cast<double>(n) / kRate;
        }
        return tracker.estimate(time);
    };

    const BeatEstimate reference = run(512);
    for (std::size_t block : {64u, 137u, 1024u}) {
        const BeatEstimate other = run(block);
        EXPECT_NEAR(other.bpm, reference.bpm, 1e-6) << "block " << block;
    }
}

TEST(LiveTracker, ADipInConfidenceDoesNotLetTwoBeatsOutTogether) {
    // Found by porting the model and then comparing the ported path against the
    // research harness that had measured it. The guard against handing the same
    // beat out twice used to be skipped whenever confidence had dipped below
    // the release threshold — and a dip is not rare, it is what a hard passage
    // looks like. Every time the tracker came back it was free to place a beat
    // however close to the last one it liked, which on a piece with a 0.9 s
    // beat produced clicks 30 ms apart. Not a fast tempo: a stutter.
    LiveTracker tracker{liveConfig()};

    constexpr double kFps = 50.0;
    constexpr double kPeriod = 0.5;
    double now = 0.0;
    std::vector<double> beats;

    const auto pump = [&](double from, double to, bool musical) {
        for (double t = from; t < to; t += 1.0 / kFps) {
            const double since = t - std::round(t / kPeriod) * kPeriod;
            // Noise rather than silence during the gap: silence moves no
            // weights at all, and what has to be reproduced is a confidence
            // that falls and recovers, not a filter that pauses.
            const double level = musical
                                     ? (std::fabs(since) < 0.011 ? 0.95 : 0.02)
                                     : (0.3 + 0.2 * std::sin(37.0 * t));
            tracker.observe(t, level);
            now = t;
            double beat = 0.0;
            while (tracker.takeBeat(now, 0.05, &beat)) beats.push_back(beat);
        }
    };

    // Lock, lose it, get it back — twice, so a single recovery cannot pass by
    // luck.
    pump(0.0, 20.0, true);
    pump(20.0, 26.0, false);
    pump(26.0, 40.0, true);
    pump(40.0, 46.0, false);
    pump(46.0, 60.0, true);

    ASSERT_GT(beats.size(), 20u);
    double closest = 1e9;
    for (std::size_t i = 1; i < beats.size(); ++i) {
        closest = std::min(closest, beats[i] - beats[i - 1]);
    }
    // Half a beat is the floor the guard promises. Anything under it is two
    // clicks where the user asked for one.
    EXPECT_GT(closest, 0.5 * kPeriod * 0.99)
        << "closest pair of beats was " << closest << " s apart";
}

// --------------------------------------------------------- octave anchoring
//
// The corpora cannot decide how tightly the anchor should hold. Ballroom and
// GTZAN are steady-tempo music, so on them a held metrical level and a held
// period are indistinguishable, and the sweep simply rewards whichever holds
// harder — over all 698 and 999 recordings, F against anchor width:
//
//     width      ballroom          GTZAN
//     free       0.700  0.584      0.632  0.508
//     0.20       0.737  0.633      0.648  0.531
//     0.10       0.757  0.653      0.658  0.544
//     0.05       0.772  0.671      0.666  0.552
//     hard pin   0.778  0.782      0.697  0.637
//
// Monotone the whole way to a pin, on both. Every point on that curve is a
// corpus asking for the one thing it is constitutionally unable to be charged
// for, and the tests below are the bill.

TEST(LiveTracker, AnAnchoredTrackerAlwaysFollowsARealTempoChangeEventually) {
    // The anchor is re-measured from a sliding window rather than fixed once,
    // which is the whole difference between it and pinPeriod: a pinned period
    // is wrong until the recording ends, an anchored one is wrong until the
    // window turns over. So the tracker always gets there. How long it takes
    // is the open question, and it is not settled by a width.
    //
    // Over six tempo changes, seconds until the estimate is within 3% of the
    // new tempo -- mean, and worst of the six:
    //
    //     window   width 0.05     width 0.10     width 0.20
    //     10 s     6.5 /  9.6     6.4 / 10.4     3.5 /  6.6
    //     15 s     9.3 / 13.2     5.2 / 10.7     3.8 / 15.4
    //     20 s    11.5 / 17.4     6.0 / 15.3     3.5 / 13.6
    //     30 s    16.2 / 24.4     6.9 / 20.6     5.1 / 23.1
    //
    // Read down the columns, not across: the window bounds the worst case and
    // the width barely does. A single tempo change is not evidence about
    // either -- the first pair measured at 30 s / 0.20 came back at 0.6 s and
    // the worst of the same six is 23.1 s.
    //
    // The corpora want the opposite of what this does: ballroom F rises from
    // 0.737 at width 0.20 to 0.772 at 0.05, GTZAN from 0.648 to 0.666, because
    // neither corpus changes tempo and so neither is ever billed. Which is why
    // the anchor is off, and why the number to settle is the window on
    // material that does change tempo -- not the width on material that does
    // not.
    constexpr double kFps = 50.0;
    const double pairs[][2] = {{100, 132}, {120, 90}, {84, 112},
                               {150, 108}, {96, 144}, {132, 100}};

    for (const auto& pair : pairs) {
        LiveConfig config = liveConfig();
        config.anchor_tempo = true;
        LiveTracker tracker{config};

        double now = 0.0;
        double phase = 0.0;
        double lag_sec = -1.0;
        const auto play = [&](double bpm, double seconds, bool watch) {
            const double period = 60.0 / bpm;
            for (int i = 0; i < static_cast<int>(seconds * kFps); ++i) {
                now += 1.0 / kFps;
                phase += 1.0 / kFps;
                double activation = 0.02;
                if (phase >= period) {
                    phase -= period;
                    activation = 0.95;
                }
                tracker.observe(now, activation);
                if (watch && lag_sec < 0.0 &&
                    std::fabs(tracker.estimate(now).bpm - bpm) < 0.03 * bpm) {
                    lag_sec = static_cast<double>(i) / kFps;
                }
            }
        };

        play(pair[0], 45.0, false);
        ASSERT_NEAR(tracker.estimate(now).bpm, pair[0], 0.05 * pair[0])
            << "did not settle on " << pair[0] << " at all";

        play(pair[1], 60.0, true);
        EXPECT_GE(lag_sec, 0.0)
            << "never followed " << pair[0] << " -> " << pair[1]
            << ": the anchor is behaving as a pin";
        // Deliberately loose, and named for what it is: this asserts that the
        // anchor is leaveable, not that it is left quickly. A tight bound here
        // would be asserting the lucky pair.
        EXPECT_LT(lag_sec, 30.0)
            << pair[0] << " -> " << pair[1] << " took " << lag_sec << " s";
    }
}

TEST(LiveTracker, TheAnchorIsOffUntilTheCorpusThatCanJudgeItExists) {
    // Not a preference, and the reason is written down so that turning it on
    // later is a decision someone makes rather than a default someone
    // inherits. The estimator behind the anchor is measured and better at the
    // octave than the filter is; the soft *use* of it is measured only on
    // material that cannot distinguish it from the hard use, which is
    // separately measured and rejected.
    EXPECT_FALSE(liveConfig().anchor_tempo);
    EXPECT_TRUE(liveConfig().valid());
}

