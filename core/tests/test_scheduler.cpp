#include "schedule/scheduler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using tiktak::schedule::BeatKind;
using tiktak::schedule::Channel;
using tiktak::schedule::Event;
using tiktak::schedule::kChannelCount;
using tiktak::schedule::Scheduler;
using tiktak::schedule::SchedulerConfig;

namespace {

SchedulerConfig audioOnly(double bpm = 120.0, int beatsPerBar = 4, int subdivisions = 1) {
    SchedulerConfig config;
    config.bpm = bpm;
    config.beats_per_bar = beatsPerBar;
    config.subdivisions = subdivisions;
    config.lookahead_sec = 0.25;
    config.channel_enabled = {{true, false, false}};
    return config;
}

// Drives the scheduler the way a host would: repeated polls on a fixed period,
// collecting everything it hands out.
std::vector<Event> drive(Scheduler& scheduler, double from_sec, double to_sec,
                         double poll_sec = 0.01, std::size_t capacity = 64) {
    std::vector<Event> collected;
    std::vector<Event> buffer(capacity);

    for (double now = from_sec; now < to_sec; now += poll_sec) {
        const std::size_t count = scheduler.pull(now, buffer.data(), buffer.size());
        collected.insert(collected.end(), buffer.begin(), buffer.begin() + count);
    }
    return collected;
}

std::vector<Event> channelOf(const std::vector<Event>& events, Channel channel) {
    std::vector<Event> filtered;
    for (const auto& event : events) {
        if (event.channel == channel) filtered.push_back(event);
    }
    return filtered;
}

}  // namespace

TEST(SchedulerConfig, RejectsNonsense) {
    EXPECT_TRUE(audioOnly().valid());

    SchedulerConfig config = audioOnly();
    config.bpm = 0.0;
    EXPECT_FALSE(config.valid());

    config = audioOnly();
    config.beats_per_bar = 0;
    EXPECT_FALSE(config.valid());

    config = audioOnly();
    config.subdivisions = 0;
    EXPECT_FALSE(config.valid());

    config = audioOnly();
    config.latency_sec[0] = -0.01;
    EXPECT_FALSE(config.valid());
}

TEST(Scheduler, EmitsNothingBeforeStart) {
    Scheduler scheduler{audioOnly()};
    EXPECT_FALSE(scheduler.running());

    Event event;
    EXPECT_EQ(scheduler.pull(0.0, &event, 1), 0u);
}

TEST(Scheduler, BeatsLandOnTheGrid) {
    Scheduler scheduler{audioOnly(120.0)};
    scheduler.start(10.0);

    const auto events = drive(scheduler, 10.0, 14.0);
    ASSERT_GE(events.size(), 8u);

    for (std::size_t i = 0; i < events.size(); ++i) {
        // 120 BPM = one beat every 0.5 s, starting at t=10.
        EXPECT_NEAR(events[i].beat_time_sec, 10.0 + 0.5 * static_cast<double>(i), 1e-12)
            << "at beat " << i;
    }
}

TEST(Scheduler, PollRateDoesNotChangeTheResult) {
    // The host polls on whatever period its audio callback happens to use. The
    // grid must be identical regardless.
    std::vector<double> reference;
    for (double poll : {0.001, 0.005, 0.02, 0.05, 0.11}) {
        Scheduler scheduler{audioOnly(140.0)};
        scheduler.start(3.0);

        std::vector<double> times;
        for (const auto& event : drive(scheduler, 3.0, 9.0, poll)) {
            times.push_back(event.beat_time_sec);
        }

        if (reference.empty()) {
            reference = times;
            ASSERT_GE(reference.size(), 10u);
        } else {
            ASSERT_EQ(times.size(), reference.size()) << "poll " << poll;
            for (std::size_t i = 0; i < times.size(); ++i) {
                EXPECT_NEAR(times[i], reference[i], 1e-12) << "poll " << poll << ", beat " << i;
            }
        }
    }
}

TEST(Scheduler, NeverEmitsTheSameStepTwice) {
    Scheduler scheduler{audioOnly()};
    scheduler.start(0.0);

    const auto events = drive(scheduler, 0.0, 20.0, 0.003);
    ASSERT_FALSE(events.empty());

    for (std::size_t i = 1; i < events.size(); ++i) {
        EXPECT_GT(events[i].step, events[i - 1].step) << "at index " << i;
    }
}

TEST(Scheduler, NeverEmitsAnEventInThePast) {
    // The whole point is to schedule ahead. An event handed over with a
    // timestamp already gone cannot be placed by the device.
    SchedulerConfig config = audioOnly();
    config.channel_enabled = {{true, true, true}};
    config.latency_sec = {{0.02, 0.01, 0.005}};

    Scheduler scheduler{config};
    scheduler.start(1.0);

    std::vector<Event> buffer(64);
    for (double now = 1.0; now < 10.0; now += 0.007) {
        const std::size_t count = scheduler.pull(now, buffer.data(), buffer.size());
        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_GE(buffer[i].time_sec, now) << "handed over an event already due";
        }
    }
}

TEST(Scheduler, EventsArriveBeforeTheyAreDue) {
    Scheduler scheduler{audioOnly()};
    scheduler.start(0.0);

    std::vector<Event> buffer(64);
    for (double now = 0.0; now < 8.0; now += 0.01) {
        const std::size_t count = scheduler.pull(now, buffer.data(), buffer.size());
        for (std::size_t i = 0; i < count; ++i) {
            // Within the lookahead horizon, never later.
            EXPECT_LE(buffer[i].time_sec, now + 0.25 + 1e-9);
        }
    }
}

TEST(Scheduler, LatencyIsCompensatedPerChannel) {
    SchedulerConfig config = audioOnly();
    config.channel_enabled = {{true, true, true}};
    config.latency_sec = {{0.030, 0.015, 0.008}};
    config.lookahead_sec = 0.5;

    Scheduler scheduler{config};
    scheduler.start(0.0);

    const auto events = drive(scheduler, 0.0, 4.0);
    const auto audio = channelOf(events, Channel::Audio);
    const auto haptic = channelOf(events, Channel::Haptic);
    const auto visual = channelOf(events, Channel::Visual);

    ASSERT_FALSE(audio.empty());
    ASSERT_EQ(audio.size(), haptic.size());
    ASSERT_EQ(audio.size(), visual.size());

    for (std::size_t i = 0; i < audio.size(); ++i) {
        // Same musical instant, three different hand-over times.
        EXPECT_NEAR(audio[i].beat_time_sec, haptic[i].beat_time_sec, 1e-12);
        EXPECT_NEAR(audio[i].beat_time_sec, visual[i].beat_time_sec, 1e-12);

        EXPECT_NEAR(audio[i].time_sec, audio[i].beat_time_sec - 0.030, 1e-12);
        EXPECT_NEAR(haptic[i].time_sec, haptic[i].beat_time_sec - 0.015, 1e-12);
        EXPECT_NEAR(visual[i].time_sec, visual[i].beat_time_sec - 0.008, 1e-12);
    }
}

TEST(Scheduler, ClassifiesDownbeatsAndSubdivisions) {
    Scheduler scheduler{audioOnly(120.0, /*beatsPerBar=*/4, /*subdivisions=*/2)};
    scheduler.start(0.0);

    const auto events = drive(scheduler, 0.0, 5.0);
    ASSERT_GE(events.size(), 16u);

    for (std::size_t i = 0; i < 16; ++i) {
        const auto& event = events[i];
        EXPECT_EQ(event.step, static_cast<std::int64_t>(i));
        EXPECT_EQ(event.subdivision, static_cast<int>(i % 2));
        EXPECT_EQ(event.beat_in_bar, static_cast<int>((i / 2) % 4));
        EXPECT_EQ(event.bar, static_cast<std::int64_t>(i / 8));

        if (i % 2 == 1) {
            EXPECT_EQ(event.kind, BeatKind::Subdivision) << "at step " << i;
        } else if ((i / 2) % 4 == 0) {
            EXPECT_EQ(event.kind, BeatKind::Downbeat) << "at step " << i;
        } else {
            EXPECT_EQ(event.kind, BeatKind::Beat) << "at step " << i;
        }
    }
}

TEST(Scheduler, SubdivisionsSplitTheBeatEvenly) {
    Scheduler scheduler{audioOnly(120.0, 4, /*subdivisions=*/3)};
    scheduler.start(0.0);

    const auto events = drive(scheduler, 0.0, 3.0);
    ASSERT_GE(events.size(), 9u);

    // Triplets at 120 BPM: one every 0.5/3 s.
    for (std::size_t i = 1; i < 9; ++i) {
        EXPECT_NEAR(events[i].beat_time_sec - events[i - 1].beat_time_sec, 0.5 / 3.0, 1e-12);
    }
}

TEST(Scheduler, TempoChangeDoesNotDisturbCommittedBeats) {
    // A beat already handed to the device cannot be recalled. Re-anchoring on
    // "now" instead of on that beat would shift it, and the player hears a
    // stumble exactly at the moment the tempo changes.
    Scheduler scheduler{audioOnly(120.0)};
    scheduler.start(0.0);

    std::vector<Event> buffer(64);
    std::vector<Event> collected;

    for (double now = 0.0; now < 2.0; now += 0.01) {
        const std::size_t count = scheduler.pull(now, buffer.data(), buffer.size());
        collected.insert(collected.end(), buffer.begin(), buffer.begin() + count);
    }
    const std::size_t before = collected.size();
    ASSERT_GE(before, 4u);
    const double last_committed = collected.back().beat_time_sec;

    scheduler.set_tempo(180.0);

    for (double now = 2.0; now < 6.0; now += 0.01) {
        const std::size_t count = scheduler.pull(now, buffer.data(), buffer.size());
        collected.insert(collected.end(), buffer.begin(), buffer.begin() + count);
    }
    ASSERT_GT(collected.size(), before + 4);

    // Beats emitted before the change keep their times, and the grid continues
    // from the last committed one at the new tempo.
    EXPECT_NEAR(collected[before - 1].beat_time_sec, last_committed, 1e-12);
    for (std::size_t i = before + 1; i < collected.size(); ++i) {
        EXPECT_NEAR(collected[i].beat_time_sec - collected[i - 1].beat_time_sec,
                    60.0 / 180.0, 1e-9)
            << "at index " << i;
    }
}

TEST(Scheduler, TimesStayMonotonicAcrossTempoChanges) {
    Scheduler scheduler{audioOnly(100.0)};
    scheduler.start(0.0);

    std::vector<Event> buffer(64);
    std::vector<Event> collected;
    double bpm = 100.0;

    for (double now = 0.0; now < 12.0; now += 0.01) {
        const std::size_t count = scheduler.pull(now, buffer.data(), buffer.size());
        collected.insert(collected.end(), buffer.begin(), buffer.begin() + count);

        // Nudge the tempo the way a live tracker would.
        if (std::fmod(now, 1.0) < 0.005) {
            bpm = 90.0 + 40.0 * std::fabs(std::sin(now));
            scheduler.set_tempo(bpm);
        }
    }

    ASSERT_GT(collected.size(), 10u);
    for (std::size_t i = 1; i < collected.size(); ++i) {
        EXPECT_GT(collected[i].beat_time_sec, collected[i - 1].beat_time_sec)
            << "grid went backwards at index " << i;
    }
}

TEST(Scheduler, AlignToShiftsThePhaseWithoutChangingTempo) {
    Scheduler scheduler{audioOnly(120.0)};
    scheduler.start(0.0);

    // Music is heard to start at 10.13 s; the grid should snap onto it.
    scheduler.align_to(10.13, 10.0);

    const auto events = drive(scheduler, 10.0, 14.0);
    ASSERT_GE(events.size(), 4u);

    for (const auto& event : events) {
        const double offset = event.beat_time_sec - 10.13;
        const double beats = offset / 0.5;
        EXPECT_NEAR(beats, std::round(beats), 1e-9)
            << "beat at " << event.beat_time_sec << " is off the aligned grid";
    }
    for (std::size_t i = 1; i < events.size(); ++i) {
        EXPECT_NEAR(events[i].beat_time_sec - events[i - 1].beat_time_sec, 0.5, 1e-12);
    }
}

TEST(Scheduler, AlignToDoesNotResurrectPastBeats) {
    Scheduler scheduler{audioOnly(120.0)};
    scheduler.start(0.0);

    const auto before = drive(scheduler, 0.0, 5.0);
    ASSERT_FALSE(before.empty());
    const double last = before.back().beat_time_sec;

    // Align to something well in the past — the grid must move on, not rewind.
    scheduler.align_to(1.02, 5.0);

    const auto after = drive(scheduler, 5.0, 8.0);
    for (const auto& event : after) {
        EXPECT_GT(event.beat_time_sec, last);
        EXPECT_GE(event.time_sec, 5.0);
    }
}

TEST(Scheduler, DropsLateEventsRatherThanPlayingThem) {
    // A click that arrives after its beat actively misleads the player, so a
    // gap is the lesser evil — but it must be reported, not hidden.
    SchedulerConfig config = audioOnly();
    config.latency_sec = {{0.05, 0.0, 0.0}};

    Scheduler scheduler{config};
    scheduler.start(0.0);

    std::vector<Event> buffer(64);
    std::size_t dropped = 0;

    // First poll arrives well after the grid began: the earliest beats are gone.
    const std::size_t count = scheduler.pull(1.7, buffer.data(), buffer.size(), &dropped);

    EXPECT_GT(dropped, 0u);
    EXPECT_EQ(scheduler.late_count(), dropped);
    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_GE(buffer[i].time_sec, 1.7);
    }
}

TEST(Scheduler, RecoversFromALongStall) {
    Scheduler scheduler{audioOnly()};
    scheduler.start(0.0);

    std::vector<Event> buffer(8);
    scheduler.pull(0.0, buffer.data(), buffer.size());

    // The app was suspended for a minute. The scheduler must not try to walk
    // every missed beat.
    std::size_t dropped = 0;
    const std::size_t count = scheduler.pull(60.0, buffer.data(), buffer.size(), &dropped);

    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_GE(buffer[i].time_sec, 60.0);
        EXPECT_LE(buffer[i].time_sec, 60.0 + 0.25 + 1e-9);
    }
}

TEST(Scheduler, SmallBufferNeverSplitsABeat) {
    SchedulerConfig config = audioOnly();
    config.channel_enabled = {{true, true, true}};
    config.lookahead_sec = 2.0;  // deliberately more than a 4-event buffer holds

    Scheduler scheduler{config};
    scheduler.start(0.0);

    std::vector<Event> buffer(4);
    const std::size_t count = scheduler.pull(0.0, buffer.data(), buffer.size());

    // Three channels per step: only whole steps fit in a buffer of four.
    EXPECT_EQ(count % 3u, 0u);
    EXPECT_LE(count, 4u);
}

TEST(Scheduler, DisabledChannelsProduceNothing) {
    SchedulerConfig config = audioOnly();
    config.channel_enabled = {{false, true, false}};

    Scheduler scheduler{config};
    scheduler.start(0.0);

    const auto events = drive(scheduler, 0.0, 3.0);
    ASSERT_FALSE(events.empty());
    for (const auto& event : events) {
        EXPECT_EQ(event.channel, Channel::Haptic);
    }
}

TEST(Scheduler, StopHaltsOutput) {
    Scheduler scheduler{audioOnly()};
    scheduler.start(0.0);
    ASSERT_FALSE(drive(scheduler, 0.0, 2.0).empty());

    scheduler.stop();
    EXPECT_FALSE(scheduler.running());
    EXPECT_TRUE(drive(scheduler, 2.0, 4.0).empty());
}

TEST(Scheduler, RestartResetsTheGrid) {
    Scheduler scheduler{audioOnly()};
    scheduler.start(0.0);
    drive(scheduler, 0.0, 3.0);

    scheduler.start(100.0);
    const auto events = drive(scheduler, 100.0, 102.0);

    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.front().step, 0);
    EXPECT_NEAR(events.front().beat_time_sec, 100.0, 1e-12);
    EXPECT_EQ(scheduler.late_count(), 0u);
}
