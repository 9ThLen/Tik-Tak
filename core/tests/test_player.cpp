#include "render/player.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using tiktak::render::PlayerConfig;
using tiktak::render::TrackPlayer;
using tiktak::schedule::BeatKind;
using tiktak::schedule::Channel;
using tiktak::schedule::Event;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr std::size_t kBuffer = 512;

PlayerConfig testConfig() {
    PlayerConfig cfg;
    cfg.sample_rate = kSampleRate;
    cfg.click.sample_rate = kSampleRate;
    return cfg;
}

// A track whose every sample states its own frame index (scaled to stay in
// float's exact-integer range), so "which part of the track is playing" can be
// read straight off the output.
std::vector<float> rampTrack(double seconds) {
    std::vector<float> track(static_cast<std::size_t>(seconds * kSampleRate));
    for (std::size_t i = 0; i < track.size(); ++i) {
        track[i] = static_cast<float>(i) * 1e-7f;
    }
    return track;
}

std::size_t frameOf(float sample) {
    return static_cast<std::size_t>(std::lround(static_cast<double>(sample) * 1e7));
}

std::vector<double> regularGrid(double bpm, double first_sec, std::size_t count) {
    std::vector<double> grid(count);
    for (std::size_t i = 0; i < count; ++i) {
        grid[i] = first_sec + static_cast<double>(i) * 60.0 / bpm;
    }
    return grid;
}

// Runs the player from stream time zero and concatenates the output.
std::vector<float> run(TrackPlayer& player, double seconds,
                       std::vector<Event>* cues_out = nullptr) {
    std::vector<float> out;
    Event cues[64];
    const std::size_t buffers =
        static_cast<std::size_t>(seconds * kSampleRate) / kBuffer;
    for (std::size_t b = 0; b < buffers; ++b) {
        float buffer[kBuffer] = {0.0f};
        const double t = static_cast<double>(b * kBuffer) / kSampleRate;
        std::size_t cue_count = 0;
        player.process(t, buffer, kBuffer, cues, 64, &cue_count);
        for (std::size_t i = 0; i < cue_count; ++i) {
            if (cues_out) cues_out->push_back(cues[i]);
        }
        out.insert(out.end(), buffer, buffer + kBuffer);
    }
    return out;
}

// Click onsets in an otherwise silent output: the first sample above the
// threshold after at least one silent sample. The click's very first sample is
// zero (the oscillator starts at phase zero), so the beat lands one sample
// before the detected rise — the same convention the click tests use.
std::vector<std::size_t> clickOnsets(const std::vector<float>& out) {
    std::vector<std::size_t> onsets;
    // Re-arm only after a sustained stretch of silence: a sounding click
    // crosses zero twice a period, and a single sub-threshold sample there is
    // the oscillator passing through zero, not the gap between clicks.
    std::size_t quiet_run = 1000;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (std::fabs(out[i]) > 1e-4f) {
            if (quiet_run >= 1000) onsets.push_back(i - 1);
            quiet_run = 0;
        } else {
            ++quiet_run;
        }
    }
    return onsets;
}

}  // namespace

TEST(PlayerConfig, RejectsWhatItCannotHonour) {
    PlayerConfig cfg = testConfig();
    EXPECT_TRUE(cfg.valid());

    cfg.click.sample_rate = 44100.0;  // track and click in different clocks
    EXPECT_FALSE(cfg.valid());

    cfg = testConfig();
    cfg.beats_per_bar = 0;
    EXPECT_FALSE(cfg.valid());

    cfg = testConfig();
    cfg.count_in_beats = -1;
    EXPECT_FALSE(cfg.valid());
}

TEST(Player, PlaysTheTrackFromTheStartBeatSampleForSample) {
    PlayerConfig cfg = testConfig();
    cfg.channel_enabled = {{false, false, false}};  // the track alone

    const std::vector<float> track = rampTrack(6.0);
    const std::vector<double> grid = regularGrid(120.0, 0.5, 10);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    const std::vector<float> out = run(player, 2.0);

    // Entry at grid[0] = 0.5 s → frame 24000, and every later sample follows.
    ASSERT_EQ(frameOf(out[0]), 24000u);
    for (std::size_t i = 0; i < out.size(); i += 997) {
        EXPECT_EQ(frameOf(out[i]), 24000u + i) << "at sample " << i;
    }
}

TEST(Player, PutsAClickOnEveryBeatOfAnIrregularGrid) {
    PlayerConfig cfg = testConfig();

    // Deliberately uneven: the player follows the analysed grid, not a tempo
    // formula, and rubato is exactly where the two differ.
    const std::vector<double> grid = {0.5, 1.0, 1.55, 2.03, 2.5};
    const std::vector<float> track(static_cast<std::size_t>(4.0 * kSampleRate), 0.0f);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    const std::vector<float> out = run(player, 3.0);
    const std::vector<std::size_t> onsets = clickOnsets(out);

    ASSERT_EQ(onsets.size(), grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const double expected = (grid[i] - grid[0]) * kSampleRate;
        EXPECT_NEAR(static_cast<double>(onsets[i]), expected, 1.0)
            << "beat " << i;
    }
    EXPECT_TRUE(player.stats().clean());
    EXPECT_EQ(player.stats().beats, grid.size());
}

TEST(Player, CountsInBeforeTheMusicStarts) {
    PlayerConfig cfg = testConfig();
    cfg.count_in_beats = 4;

    const std::vector<float> track = rampTrack(6.0);
    const std::vector<double> grid = regularGrid(120.0, 1.0, 8);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    const std::vector<float> out = run(player, 3.0);

    // Four count-in clicks half a second apart, then the track enters at
    // timeline 2.0 s exactly, at its own frame 48000 (grid[0] = 1.0 s).
    const std::size_t entry = static_cast<std::size_t>(2.0 * kSampleRate);
    for (std::size_t i = 0; i < entry; i += 512) {
        // Before the entry the only signal is clicks; the ramp values in this
        // region of the track are far larger than a click's first samples, so
        // any track leak would trip this.
        EXPECT_LT(std::fabs(out[i]), 1.1f);
    }
    EXPECT_EQ(frameOf(out[entry] - 0.0f), 48000u);

    const std::vector<std::size_t> onsets = clickOnsets(out);
    ASSERT_GE(onsets.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(static_cast<double>(onsets[i]), 0.5 * kSampleRate * i, 1.0)
            << "count-in click " << i;
    }
}

TEST(Player, LoopsBarsSampleExactly) {
    PlayerConfig cfg = testConfig();
    cfg.channel_enabled = {{false, false, false}};

    const std::vector<float> track = rampTrack(8.0);
    // 120 BPM, beat 0 at 0.0: bar 1 spans beats 4..8 → 2.0 s to 4.0 s.
    const std::vector<double> grid = regularGrid(120.0, 0.0, 12);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.setLoop(1, 2));
    ASSERT_TRUE(player.start(0.0, 1));

    const std::vector<float> out = run(player, 5.0);

    const std::size_t loop_frames = static_cast<std::size_t>(2.0 * kSampleRate);
    const std::size_t loop_start = static_cast<std::size_t>(2.0 * kSampleRate);
    for (std::size_t i = 0; i < out.size(); i += 499) {
        const std::size_t expected = loop_start + (i % loop_frames);
        EXPECT_EQ(frameOf(out[i]), expected) << "at sample " << i;
    }
    EXPECT_GE(player.stats().loops, 2u);

    // Position stays inside the loop however long it plays.
    EXPECT_GE(player.positionSec(), 2.0);
    EXPECT_LT(player.positionSec(), 4.0);
}

TEST(Player, TheClickFollowsTheTrackThroughTheLoop) {
    PlayerConfig cfg = testConfig();

    const std::vector<float> track(static_cast<std::size_t>(8.0 * kSampleRate), 0.0f);
    const std::vector<double> grid = regularGrid(120.0, 0.0, 12);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.setLoop(1, 2));
    ASSERT_TRUE(player.start(0.0, 1));

    const std::vector<float> out = run(player, 5.0);
    const std::vector<std::size_t> onsets = clickOnsets(out);

    // Four beats per two-second loop, clicking for five seconds: ten clicks,
    // each on a half-second boundary — including the first beat of every
    // iteration, which is the click that proves the wrap re-arms the cursor.
    ASSERT_EQ(onsets.size(), 10u);
    for (std::size_t i = 0; i < onsets.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(onsets[i]), 0.5 * kSampleRate * i, 1.0)
            << "click " << i;
    }
    EXPECT_TRUE(player.stats().clean());
}

TEST(Player, HandsCuesCompensatedAgainstTheHeardBeat) {
    PlayerConfig cfg = testConfig();
    cfg.channel_enabled = {{true, true, true}};
    cfg.latency_sec = {{0.020, 0.015, 0.008}};  // audio, haptic, visual

    const std::vector<float> track(static_cast<std::size_t>(4.0 * kSampleRate), 0.0f);
    const std::vector<double> grid = regularGrid(120.0, 0.5, 6);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    std::vector<Event> cues;
    run(player, 2.0, &cues);
    ASSERT_GE(cues.size(), 4u);

    // Beat 0 plays at stream time 0 (entry beat), is heard one audio latency
    // later, and each channel fires its own latency before that moment.
    EXPECT_EQ(cues[0].channel, Channel::Haptic);
    EXPECT_NEAR(cues[0].beat_time_sec, 0.020, 1e-9);
    EXPECT_NEAR(cues[0].time_sec, 0.020 - 0.015, 1e-9);
    EXPECT_EQ(cues[0].kind, BeatKind::Downbeat);
    EXPECT_EQ(cues[0].bar, 0);
    EXPECT_EQ(cues[0].beat_in_bar, 0);

    EXPECT_EQ(cues[1].channel, Channel::Visual);
    EXPECT_NEAR(cues[1].time_sec, 0.020 - 0.008, 1e-9);
    EXPECT_NEAR(cues[1].beat_time_sec, cues[0].beat_time_sec, 1e-12);

    // The second beat, half a second on, kind Beat.
    EXPECT_EQ(cues[2].kind, BeatKind::Beat);
    EXPECT_NEAR(cues[2].beat_time_sec, 0.5 + 0.020, 1e-9);
    EXPECT_EQ(cues[2].beat_in_bar, 1);
}

TEST(Player, CountInCuesAreMarkedAsSuch) {
    PlayerConfig cfg = testConfig();
    cfg.count_in_beats = 2;
    cfg.channel_enabled = {{true, false, true}};

    const std::vector<float> track = rampTrack(4.0);
    const std::vector<double> grid = regularGrid(120.0, 0.5, 6);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    std::vector<Event> cues;
    run(player, 2.0, &cues);
    ASSERT_GE(cues.size(), 3u);

    // Two count-in cues on bar -1 with negative steps, then the real beat 0.
    EXPECT_EQ(cues[0].bar, -1);
    EXPECT_EQ(cues[0].step, -2);
    EXPECT_EQ(cues[0].kind, BeatKind::Beat);
    EXPECT_EQ(cues[1].bar, -1);
    EXPECT_EQ(cues[1].step, -1);
    EXPECT_EQ(cues[2].bar, 0);
    EXPECT_EQ(cues[2].step, 0);
    EXPECT_EQ(cues[2].kind, BeatKind::Downbeat);
}

TEST(Player, EndsAtTheEndOfTheTrack) {
    PlayerConfig cfg = testConfig();
    cfg.channel_enabled = {{false, false, false}};

    const std::vector<float> track = rampTrack(1.0);
    const std::vector<double> grid = regularGrid(120.0, 0.0, 2);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));
    EXPECT_TRUE(player.running());

    const std::vector<float> out = run(player, 2.0);

    EXPECT_FALSE(player.running());
    // Past the end the player mixes nothing rather than repeating or reading
    // out of bounds.
    for (std::size_t i = static_cast<std::size_t>(1.1 * kSampleRate); i < out.size();
         ++i) {
        ASSERT_EQ(out[i], 0.0f) << "at sample " << i;
    }
}

TEST(Player, RefusesWhatItCannotPlay) {
    PlayerConfig cfg = testConfig();
    const std::vector<float> track = rampTrack(2.0);
    const std::vector<double> one_beat = {0.5};

    TrackPlayer no_track(cfg);
    EXPECT_FALSE(no_track.start(0.0));

    // A count-in needs a beat interval, and one beat does not define one.
    PlayerConfig counted = cfg;
    counted.count_in_beats = 4;
    TrackPlayer lone(counted);
    lone.setTrack(track.data(), track.size());
    lone.setGrid(one_beat.data(), one_beat.size());
    EXPECT_FALSE(lone.start(0.0));

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    const std::vector<double> grid = regularGrid(120.0, 0.0, 4);  // one bar
    player.setGrid(grid.data(), grid.size());
    EXPECT_FALSE(player.setLoop(1, 1));   // empty range
    EXPECT_FALSE(player.setLoop(2, 3));   // bars beyond the grid
    EXPECT_FALSE(player.setLoop(-1, 1));  // before the grid
    EXPECT_TRUE(player.setLoop(0, 1));
    EXPECT_FALSE(player.start(0.0, 3));   // starting past the grid
}

TEST(Player, AGridlessTrackJustPlays) {
    PlayerConfig cfg = testConfig();
    const std::vector<float> track = rampTrack(1.0);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    ASSERT_TRUE(player.start(0.0));

    const std::vector<float> out = run(player, 0.5);
    for (std::size_t i = 0; i < out.size(); i += 313) {
        EXPECT_EQ(frameOf(out[i]), i);
    }
    EXPECT_EQ(player.stats().beats, 0u);
}

TEST(Player, RestartIsAsGoodAsTheFirstStart) {
    PlayerConfig cfg = testConfig();

    const std::vector<float> track(static_cast<std::size_t>(4.0 * kSampleRate), 0.0f);
    const std::vector<double> grid = regularGrid(120.0, 0.0, 6);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());

    ASSERT_TRUE(player.start(0.0));
    const std::vector<float> first = run(player, 2.0);
    player.silence();

    ASSERT_TRUE(player.start(0.0));
    const std::vector<float> second = run(player, 2.0);

    EXPECT_EQ(first, second);
    EXPECT_TRUE(player.stats().clean());
}

TEST(Player, MixesIntoTheBufferRatherThanClearingIt) {
    PlayerConfig cfg = testConfig();
    cfg.channel_enabled = {{false, false, false}};

    const std::vector<float> track = rampTrack(1.0);
    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    ASSERT_TRUE(player.start(0.0));

    float buffer[kBuffer];
    for (std::size_t i = 0; i < kBuffer; ++i) buffer[i] = 1.0f;
    player.process(0.0, buffer, kBuffer);

    for (std::size_t i = 0; i < kBuffer; i += 100) {
        EXPECT_NEAR(buffer[i], 1.0f + track[i], 1e-6f);
    }
}

// The accent is a claim about the music, and when nothing supports it the
// honest output is an even click. This is the behaviour the `track` command
// relies on when the analysis is not confident, and it was possible to report
// "too close to call" while still accenting every fourth beat — the message and
// the sound disagreeing is worse than either alone.
TEST(Player, WithTheAccentOffEveryBeatSoundsTheSame) {
    PlayerConfig cfg = testConfig();
    cfg.accent_downbeats = false;
    cfg.channel_enabled = {{true, true, false}};

    const std::vector<float> track(static_cast<std::size_t>(4.0 * kSampleRate), 0.0f);
    const std::vector<double> grid = regularGrid(120.0, 0.0, 8);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    std::vector<Event> cues;
    run(player, 3.0, &cues);
    ASSERT_GE(cues.size(), 4u);
    for (const Event& cue : cues) {
        EXPECT_EQ(cue.kind, BeatKind::Beat) << "at " << cue.beat_time_sec;
    }
}

// Bars still exist with the accent off — looping and --from-bar are bar-based
// and must keep working. Only the sound stops distinguishing them.
TEST(Player, TheAccentIsSilencedWithoutLosingTheBarCount) {
    PlayerConfig cfg = testConfig();
    cfg.accent_downbeats = false;
    cfg.beats_per_bar = 4;
    cfg.channel_enabled = {{true, true, false}};

    const std::vector<float> track(static_cast<std::size_t>(6.0 * kSampleRate), 0.0f);
    const std::vector<double> grid = regularGrid(120.0, 0.0, 12);

    TrackPlayer player(cfg);
    player.setTrack(track.data(), track.size());
    player.setGrid(grid.data(), grid.size());
    ASSERT_TRUE(player.start(0.0));

    std::vector<Event> cues;
    run(player, 3.0, &cues);
    ASSERT_GE(cues.size(), 5u);
    EXPECT_EQ(cues[0].bar, 0);
    EXPECT_EQ(cues[0].beat_in_bar, 0);
    EXPECT_EQ(cues[4].bar, 1);
    EXPECT_EQ(cues[4].beat_in_bar, 0);
}

// The accented and unaccented renders must differ in the audio itself, not only
// in the cue metadata — the click is rendered in the callback, so a test that
// only checked cues would pass even if the downbeat click were still sounding.
TEST(Player, TheAccentIsAudiblyAbsentAndNotJustUnreported) {
    const std::vector<float> track(static_cast<std::size_t>(3.0 * kSampleRate), 0.0f);
    const std::vector<double> grid = regularGrid(120.0, 0.0, 6);

    const auto render = [&](bool accent) {
        PlayerConfig cfg = testConfig();
        cfg.accent_downbeats = accent;
        cfg.beats_per_bar = 4;
        cfg.channel_enabled = {{true, false, false}};

        TrackPlayer player(cfg);
        player.setTrack(track.data(), track.size());
        player.setGrid(grid.data(), grid.size());
        EXPECT_TRUE(player.start(0.0));

        std::vector<float> out(static_cast<std::size_t>(2.5 * kSampleRate), 0.0f);
        constexpr std::size_t kBlock = 128;
        for (std::size_t at = 0; at + kBlock <= out.size(); at += kBlock) {
            player.process(static_cast<double>(at) / kSampleRate, out.data() + at, kBlock);
        }
        return out;
    };

    const std::vector<float> accented = render(true);
    const std::vector<float> even = render(false);
    ASSERT_EQ(accented.size(), even.size());

    // The first beat is the downbeat, so the two renders differ there.
    const std::size_t first_beat_end = static_cast<std::size_t>(0.25 * kSampleRate);
    double difference = 0.0;
    for (std::size_t i = 0; i < first_beat_end; ++i) {
        difference = std::max(difference, static_cast<double>(std::fabs(accented[i] - even[i])));
    }
    EXPECT_GT(difference, 1e-4) << "the downbeat click sounds the same either way";

    // Beat one, half a second in, is an ordinary beat in both and must be
    // identical — switching the accent off changes the downbeat, nothing else.
    const std::size_t second = static_cast<std::size_t>(0.5 * kSampleRate);
    for (std::size_t i = second; i < second + first_beat_end; ++i) {
        EXPECT_FLOAT_EQ(accented[i], even[i]) << "at sample " << i;
    }
}
