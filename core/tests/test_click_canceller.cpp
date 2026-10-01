#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "render/click_canceller.hpp"
#include "render/live_metronome.hpp"
#include "support.hpp"

using tiktak::render::ClickCanceller;
using tiktak::render::ClickCancellerConfig;
using tiktak::render::ClickConfig;
using tiktak::render::ClickRenderer;
using tiktak::render::LiveMetronome;
using tiktak::render::LiveMetronomeConfig;

namespace {

constexpr double kRate = 48000.0;
constexpr std::size_t kBlock = 480;
constexpr double kPeriod = 0.5;   // 120 BPM

ClickConfig click(double gain = 1.0) {
    ClickConfig cfg;
    cfg.sample_rate = kRate;
    cfg.beat.gain *= gain;
    return cfg;
}

ClickCancellerConfig cancellerAt(double round_trip_sec) {
    ClickCancellerConfig cfg;
    cfg.sample_rate = kRate;
    cfg.round_trip_sec = round_trip_sec;
    return cfg;
}

// One reflection of a path: this many samples after the first arrival, at this
// gain.
using Path = std::vector<std::pair<std::int64_t, double>>;

struct Room {
    std::vector<float> played;   // the click alone, as it left
    std::vector<float> echo;     // what of it reaches the microphone
    std::vector<std::size_t> starts;
};

// Clicks every half second from 0.5 s, played and brought back `delay` samples
// later through `path`. From `change_at` on, `later` is the path instead.
Room play(double seconds, std::int64_t delay, const Path& path, double change_at = 1e9,
          const Path& later = {}) {
    const auto total = static_cast<std::size_t>(seconds * kRate);
    Room room;
    room.played.assign(total, 0.0f);
    room.echo.assign(total, 0.0f);

    ClickRenderer renderer(click());
    for (std::size_t at = 0; at < total; at += kBlock) {
        const double time = static_cast<double>(at) / kRate;
        // Scheduled a little ahead, as a metronome does, and never more than
        // the renderer's queue can hold.
        for (double beat = 0.5; beat < seconds - 0.3; beat += kPeriod) {
            if (beat >= time && beat < time + static_cast<double>(kBlock) / kRate) {
                renderer.schedule(beat, tiktak::schedule::BeatKind::Beat);
                room.starts.push_back(static_cast<std::size_t>(std::llround(beat * kRate)));
            }
        }
        renderer.mix(time, room.played.data() + at, std::min(kBlock, total - at));
    }

    for (std::size_t n = 0; n < total; ++n) {
        const Path& now = static_cast<double>(n) / kRate >= change_at ? later : path;
        for (const auto& [lag, gain] : now) {
            const std::int64_t from = static_cast<std::int64_t>(n) - delay - lag;
            if (from >= 0) room.echo[n] += static_cast<float>(gain * room.played[from]);
        }
    }
    return room;
}

// Both streams through the canceller a block at a time, capture first, the way
// a duplex callback delivers them. Returns what was left of `heard`.
std::vector<float> run(ClickCanceller& canceller, const std::vector<float>& played,
                       const std::vector<float>& heard) {
    std::vector<float> out(heard.size(), 0.0f);
    for (std::size_t at = 0; at + kBlock <= heard.size(); at += kBlock) {
        canceller.heard(static_cast<std::int64_t>(at), heard.data() + at, out.data() + at,
                        kBlock);
        canceller.played(static_cast<std::int64_t>(at), played.data() + at, kBlock);
    }
    return out;
}

double energy(const std::vector<float>& a, const std::vector<float>& b, std::size_t from,
              std::size_t to) {
    double sum = 0.0;
    for (std::size_t n = from; n < to && n < a.size(); ++n) {
        const double v = static_cast<double>(a[n]) - (b.empty() ? 0.0 : static_cast<double>(b[n]));
        sum += v * v;
    }
    return sum;
}

// How far down the click is in what was left, between two times. `music` is
// taken out of both sides first, so this is about the click alone.
double suppressionDb(const Room& room, const std::vector<float>& left,
                     const std::vector<float>& music, double from_sec, double to_sec) {
    const auto from = static_cast<std::size_t>(from_sec * kRate);
    const auto to = static_cast<std::size_t>(to_sec * kRate);
    const double before = energy(room.echo, {}, from, to);
    const double after = energy(left, music, from, to);
    return 10.0 * std::log10(before / after);
}

std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (std::size_t n = 0; n < a.size(); ++n) out[n] = a[n] + b[n];
    return out;
}

}  // namespace

TEST(ClickCanceller, LeavesTheRoomAloneWhileNothingIsPlayed) {
    ClickCanceller canceller(cancellerAt(0.040), click());
    ASSERT_TRUE(canceller.enabled());

    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<float> heard(static_cast<std::size_t>(4 * kRate));
    for (float& v : heard) v = noise(rng);

    const std::vector<float> silence(heard.size(), 0.0f);
    const std::vector<float> left = run(canceller, silence, heard);
    for (std::size_t n = 0; n + kBlock <= heard.size(); ++n) ASSERT_EQ(left[n], heard[n]) << n;
    EXPECT_EQ(canceller.stats().clicks, 0u);
}

TEST(ClickCanceller, RemovesAClickThatComesStraightBack) {
    constexpr std::int64_t kDelay = 1920;   // 40 ms
    const Room room = play(8.0, kDelay, {{0, 0.6}});
    ClickCanceller canceller(cancellerAt(0.040), click());

    const std::vector<float> left = run(canceller, room.played, room.echo);

    // The first click is heard whole: there was nothing to predict it from.
    const std::size_t first = room.starts.front() + kDelay;
    for (std::size_t n = first; n < first + 2000; ++n) ASSERT_EQ(left[n], room.echo[n]) << n;

    // Every one after it is gone, bar what the ridge leaves of its edges.
    EXPECT_GT(suppressionDb(room, left, {}, 3.5, 7.5), 40.0);
    EXPECT_GE(canceller.stats().clicks, 10u);
    EXPECT_EQ(canceller.stats().skipped, 0u);
}

TEST(ClickCanceller, RemovesAClickThroughADeviceADeskAndAWrongRoundTrip) {
    // The round trip given is 1.5 ms short of the real one, the first arrival
    // is not the loudest, and three reflections follow within the span.
    constexpr std::int64_t kDelay = 1992;   // 41.5 ms
    const Path path = {{0, 0.5}, {37, -0.3}, {190, 0.2}, {700, 0.1}};
    const Room room = play(10.0, kDelay, path);
    ClickCanceller canceller(cancellerAt(0.040), click());

    const std::vector<float> left = run(canceller, room.played, room.echo);
    EXPECT_GT(suppressionDb(room, left, {}, 5.5, 9.5), 30.0);
}

TEST(ClickCanceller, FindsTheClickUnderMusicLouderThanItIs) {
    constexpr std::int64_t kDelay = 1920;
    const Room room = play(30.0, kDelay, {{0, 0.3}, {120, 0.1}});

    // Broadband music with no pause in it, 18 dB above the click's power over
    // a beat: every one of the path's taps has the music in it, and averaging
    // across clicks is all there is to get it out again.
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<float> music(room.echo.size());
    for (float& v : music) v = noise(rng);
    const std::vector<float> heard = add(music, room.echo);

    ClickCanceller canceller(cancellerAt(0.040), click());
    const std::vector<float> left = run(canceller, room.played, heard);

    // The click goes down by a useful amount with the music on top of it ...
    EXPECT_GT(suppressionDb(room, left, music, 24.5, 29.5), 10.0);

    // ... and between clicks the music is not touched at all.
    const std::size_t window = static_cast<std::size_t>(0.25 * kRate);
    for (std::size_t k = 4; k + 1 < room.starts.size(); ++k) {
        const std::size_t from = room.starts[k] + kDelay + window;
        const std::size_t to = room.starts[k + 1] + kDelay - 300;
        for (std::size_t n = from; n < to; ++n) ASSERT_EQ(left[n], heard[n]) << n;
    }
}

TEST(ClickCanceller, FollowsThePathWhenItChanges) {
    constexpr std::int64_t kDelay = 1920;
    // The phone is picked up half way: a different gain, and a new reflection.
    const Room room = play(28.0, kDelay, {{0, 0.6}}, 8.0, {{0, -0.4}, {90, 0.2}});
    // Quicker than it ships, so that the forgetting fits in a test: the shipped
    // figure takes three times as long over the same curve.
    ClickCancellerConfig cfg = cancellerAt(0.040);
    cfg.update = 0.15;
    ClickCanceller canceller(cfg, click());

    const std::vector<float> left = run(canceller, room.played, room.echo);
    // Wrong at once, and by more than the click itself: the old path is being
    // subtracted from the new one.
    EXPECT_LT(suppressionDb(room, left, {}, 8.0, 9.0), 0.0);
    // Then it is forgotten at `update` a click.
    EXPECT_GT(suppressionDb(room, left, {}, 24.5, 27.5), 30.0);
}

TEST(ClickCanceller, OneClickInAPauseSettlesThePath) {
    constexpr std::int64_t kDelay = 1920;
    const Room room = play(20.0, kDelay, {{0, 0.3}, {120, 0.1}});

    // The same loud music as above, except that it stops from 8 s to 11 s.
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<float> music(room.echo.size());
    for (std::size_t n = 0; n < music.size(); ++n) {
        const double time = static_cast<double>(n) / kRate;
        const float value = noise(rng);
        music[n] = time >= 8.0 && time < 11.0 ? 0.0f : value;
    }
    const std::vector<float> heard = add(music, room.echo);

    ClickCanceller canceller(cancellerAt(0.040), click());
    const std::vector<float> left = run(canceller, room.played, heard);

    const double before = suppressionDb(room, left, music, 4.0, 8.0);
    const double during = suppressionDb(room, left, music, 10.0, 11.0);
    const double after = suppressionDb(room, left, music, 11.5, 15.5);
    // Under the music the estimate is as good as the music lets it be. Once the
    // room has died away a click is heard clean, and the ones after it vanish ...
    EXPECT_GT(during, 35.0);
    // ... and what was learned there is kept when the music comes back.
    EXPECT_GT(after, before + 10.0);
}

TEST(ClickCanceller, KnowsWhenTheClickHasTheRoomToItself) {
    constexpr std::int64_t kDelay = 1920;
    const Room room = play(16.0, kDelay, {{0, 0.3}, {120, 0.1}});

    // Music for the first eight seconds, then nothing but the click.
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<float> heard(room.echo.size());
    for (std::size_t n = 0; n < heard.size(); ++n) {
        const float value = noise(rng);
        heard[n] = room.echo[n] + (static_cast<double>(n) / kRate < 8.0 ? value : 0.0f);
    }

    ClickCanceller canceller(cancellerAt(0.040), click());
    EXPECT_FALSE(canceller.alone());   // nothing played yet: nothing to be alone

    std::vector<float> out(kBlock);
    double last_under_music = -1.0;
    double first_alone = -1.0;
    for (std::size_t at = 0; at + kBlock <= heard.size(); at += kBlock) {
        canceller.heard(static_cast<std::int64_t>(at), heard.data() + at, out.data(), kBlock);
        canceller.played(static_cast<std::int64_t>(at), room.played.data() + at, kBlock);
        const double time = static_cast<double>(at) / kRate;
        if (canceller.alone()) {
            if (time < 8.0) last_under_music = time;
            if (time >= 8.0 && first_alone < 0.0) first_alone = time;
        }
    }
    // Never while the music plays, and within a beat of its stopping.
    EXPECT_LT(last_under_music, 0.0);
    ASSERT_GT(first_alone, 8.0);
    EXPECT_LT(first_alone, 8.6);
}

TEST(ClickCanceller, ASilentClickLeavesItOff) {
    ClickCanceller canceller(cancellerAt(0.040), click(0.0));
    EXPECT_FALSE(canceller.enabled());

    const std::vector<float> heard = {0.25f, -0.5f, 0.125f};
    std::vector<float> out(heard.size(), 0.0f);
    canceller.heard(0, heard.data(), out.data(), heard.size());
    EXPECT_EQ(out, heard);
}

TEST(ClickCanceller, CannotPredictAClickBeforeItHasBeenPlayed) {
    // No round trip at all: each click is heard in the very buffer that plays
    // it, and capture comes first, so its reference does not exist yet. Such a
    // click is passed through and learned from never, not half-removed.
    const Room room = play(6.0, 0, {{0, 0.6}});
    ClickCanceller canceller(cancellerAt(0.0), click());

    const std::vector<float> left = run(canceller, room.played, room.echo);
    EXPECT_EQ(canceller.stats().clicks, 0u);
    EXPECT_GT(canceller.stats().skipped, 5u);
    for (std::size_t n = 0; n + kBlock <= left.size(); ++n) ASSERT_EQ(left[n], room.echo[n]) << n;
}

TEST(LiveMetronome, SubtractsItsOwnClickAndStillFollowsTheRoom) {
    constexpr double kRoundTrip = 0.040;
    const auto delay = static_cast<std::size_t>(kRoundTrip * kRate);

    LiveMetronomeConfig cfg;
    cfg.tracker = tiktak::tracking::liveConfigFor(kRate);
    cfg.click.sample_rate = kRate;
    cfg.round_trip_sec = kRoundTrip;
    cfg.gate_own_clicks = false;
    cfg.subtract_own_clicks = true;
    ASSERT_TRUE(cfg.valid());
    LiveMetronome metronome{cfg};
    metronome.start();

    // The room plays a click track, and the microphone hears it together with
    // everything the metronome itself played one round trip earlier.
    const auto room = tiktak::test::clickTrack(120.0, 20.0, kRate, 1.0);
    std::vector<float> played(room.size(), 0.0f);
    std::vector<float> heard(kBlock);
    std::vector<double> clicks;
    std::size_t quiet = 1000;
    for (std::size_t at = 0; at + kBlock <= room.size(); at += kBlock) {
        const double time = static_cast<double>(at) / kRate;
        for (std::size_t i = 0; i < kBlock; ++i) {
            const std::size_t n = at + i;
            heard[i] = room[n] + (n >= delay ? 0.8f * played[n - delay] : 0.0f);
        }
        metronome.capture(time, heard.data(), kBlock);
        metronome.process(time, played.data() + at, kBlock);
        for (std::size_t i = 0; i < kBlock; ++i) {
            if (std::fabs(played[at + i]) > 1e-4f) {
                if (quiet >= 1000) clicks.push_back(time + static_cast<double>(i) / kRate);
                quiet = 0;
            } else {
                ++quiet;
            }
        }
    }

    ASSERT_GT(clicks.size(), 20u);
    // This room only sounds on the beat, which is under our own click every
    // time: between clicks it is silent, and it still must not count as empty.
    EXPECT_EQ(metronome.stats().gated, 0u);
    EXPECT_GT(metronome.subtraction().clicks, 15u);
    // Sent a round trip early, so each lands on the room's grid when heard.
    for (std::size_t i = clicks.size() / 2; i < clicks.size(); ++i) {
        const double heard_at = clicks[i] + kRoundTrip - 1.0;
        const double off = std::fabs(heard_at - std::round(heard_at / 0.5) * 0.5);
        EXPECT_LT(off, 0.05) << "click " << i << " at " << clicks[i];
    }
}

TEST(LiveMetronome, GatesAClickLeftAloneInTheRoomAndNotOneUnderMusic) {
    constexpr double kRoundTrip = 0.040;
    const auto delay = static_cast<std::size_t>(kRoundTrip * kRate);

    LiveMetronomeConfig cfg;
    cfg.tracker = tiktak::tracking::liveConfigFor(kRate);
    cfg.click.sample_rate = kRate;
    cfg.round_trip_sec = kRoundTrip;
    cfg.gate_own_clicks = false;
    cfg.subtract_own_clicks = true;
    LiveMetronome metronome{cfg};
    metronome.start();

    // Twelve seconds of a room playing, then the room stops and the metronome
    // is the only thing left sounding. The room hums between its beats: one
    // that fell silent between them would be as silent before every click as
    // after it stopped, and that is the case the test above this one covers.
    auto room = tiktak::test::clickTrack(120.0, 12.0, kRate, 1.0);
    std::mt19937 rng(3);
    std::normal_distribution<float> hum(0.0f, 0.02f);
    for (float& v : room) v += hum(rng);
    room.resize(static_cast<std::size_t>(20.0 * kRate), 0.0f);
    std::vector<float> played(room.size(), 0.0f);
    std::vector<float> heard(kBlock);
    std::size_t alone_while_playing = 0;
    for (std::size_t at = 0; at + kBlock <= room.size(); at += kBlock) {
        const double time = static_cast<double>(at) / kRate;
        for (std::size_t i = 0; i < kBlock; ++i) {
            const std::size_t n = at + i;
            heard[i] = room[n] + (n >= delay ? 0.8f * played[n - delay] : 0.0f);
        }
        metronome.capture(time, heard.data(), kBlock);
        metronome.process(time, played.data() + at, kBlock);
        if (time < 11.5) alone_while_playing = metronome.stats().clicks_alone;
    }

    EXPECT_EQ(alone_while_playing, 0u);
    EXPECT_GT(metronome.stats().clicks_alone, 0u);
    EXPECT_GT(metronome.stats().gated, 0u);
}

TEST(LiveMetronomeConfig, SubtractionNeedsCaptureAndClickAtOneRate) {
    LiveMetronomeConfig cfg;
    cfg.tracker = tiktak::tracking::liveConfigFor(44100.0);
    cfg.click.sample_rate = 48000.0;
    EXPECT_TRUE(cfg.valid());
    cfg.subtract_own_clicks = true;
    EXPECT_FALSE(cfg.valid());
}
