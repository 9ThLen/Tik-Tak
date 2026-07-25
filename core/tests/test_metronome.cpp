// The beat grid and the click renderer are separately correct in
// test_scheduler.cpp and test_click.cpp. These drive them together, because
// what has to be right is the composition: an exact grid and an exact synth
// still make a metronome that drifts if the two clock domains are wired up
// wrongly, and no unit test of either piece can see that.
#include "render/metronome.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using tiktak::render::ClickConfig;
using tiktak::render::Metronome;
using tiktak::render::MetronomeConfig;
using tiktak::schedule::Channel;
using tiktak::schedule::Event;

namespace {

constexpr double kRate = 48000.0;

MetronomeConfig makeConfig(double bpm, double audio_latency_sec = 0.0,
                           int subdivisions = 1, double sample_rate = kRate) {
    MetronomeConfig cfg;
    cfg.grid.bpm = bpm;
    cfg.grid.beats_per_bar = 4;
    cfg.grid.subdivisions = subdivisions;
    cfg.grid.lookahead_sec = 0.25;
    cfg.grid.channel_enabled = {{true, false, false}};
    cfg.grid.latency_sec = {{audio_latency_sec, 0.0, 0.0}};
    cfg.click.sample_rate = sample_rate;
    return cfg;
}

// Every click onset in the buffer.
//
// A click's own first sample is exactly zero — it starts from silence so as not
// to put a step into the signal — so the first sample above the floor is the
// one after it. `skip` has to clear the longest click, or a click's own tail is
// reported as the next beat.
std::vector<std::ptrdiff_t> onsets(const std::vector<float>& buffer,
                                   std::size_t skip = 4000, float threshold = 1e-4f) {
    std::vector<std::ptrdiff_t> found;
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        if (std::fabs(buffer[i]) <= threshold) continue;
        found.push_back(static_cast<std::ptrdiff_t>(i) - 1);
        i += skip;
    }
    return found;
}

// The audio device, simulated: fixed buffers back to back, from stream time
// zero. The buffer is cleared first because a real device hands over whatever
// was in it, and process() mixes rather than fills.
std::vector<float> run(Metronome& metronome, double duration_sec,
                       std::size_t block = 256, double sample_rate = kRate) {
    const auto total = static_cast<std::size_t>(duration_sec * sample_rate);
    std::vector<float> out(total, 0.0f);

    for (std::size_t i = 0; i < total; i += block) {
        const std::size_t n = std::min(block, total - i);
        metronome.process(static_cast<double>(i) / sample_rate, out.data() + i, n);
    }
    return out;
}

}  // namespace

TEST(Metronome, PutsEveryBeatOnItsExactSample) {
    Metronome metronome(makeConfig(120.0));

    // Started a second into the stream so the first beat is not already in the
    // past by the time the first buffer is filled.
    metronome.start(1.0);
    const std::vector<float> out = run(metronome, 3.0);

    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_EQ(found.size(), 4u);   // 1.0, 1.5, 2.0, 2.5

    for (std::size_t i = 0; i < found.size(); ++i) {
        const double expected_sec = 1.0 + 0.5 * static_cast<double>(i);
        EXPECT_EQ(found[i], static_cast<std::ptrdiff_t>(expected_sec * kRate)) << "beat " << i;
    }
    EXPECT_TRUE(metronome.stats().clean());
}

TEST(Metronome, CompensatesTheOutputLatencyByPlayingEarly) {
    // The whole point of the design. The click is written one latency *before*
    // the musical instant, so that after the device's delay it arrives on it.
    const double latency = 0.005;

    Metronome metronome(makeConfig(120.0, latency));
    metronome.start(1.0);

    const std::vector<float> out = run(metronome, 2.0);
    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_EQ(found.size(), 3u);

    for (std::size_t i = 0; i < found.size(); ++i) {
        const double beat_sec = 1.0 + 0.5 * static_cast<double>(i);
        EXPECT_EQ(found[i], static_cast<std::ptrdiff_t>((beat_sec - latency) * kRate))
            << "beat " << i;
    }
}

TEST(Metronome, DoesNotDriftOverAMinute) {
    // A metronome that gains a millisecond a minute is useless, and advancing
    // beat by beat instead of computing each one from the grid is exactly how
    // that happens. The 120th beat must still be on its own sample.
    Metronome metronome(makeConfig(120.0));
    metronome.start(0.5);

    const std::vector<float> out = run(metronome, 60.5);
    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_EQ(found.size(), 120u);

    for (std::size_t i = 0; i < found.size(); ++i) {
        const double expected_sec = 0.5 + 0.5 * static_cast<double>(i);
        ASSERT_EQ(found[i], static_cast<std::ptrdiff_t>(std::floor(expected_sec * kRate + 0.5)))
            << "beat " << i;
    }
    EXPECT_TRUE(metronome.stats().clean());
    // One more than sounded: the grid runs a lookahead ahead of the audio, so
    // the beat just past the end of the buffer has already been handed over.
    EXPECT_EQ(metronome.stats().beats, 121u);
}

TEST(Metronome, DriftsNoMoreAtATempoThatDoesNotDivideTheSampleRate) {
    // 137 BPM puts every beat between two samples. Rounding is unavoidable;
    // accumulating it is not, and this is the case that tells them apart.
    Metronome metronome(makeConfig(137.0));
    metronome.start(0.5);

    const std::vector<float> out = run(metronome, 60.5);
    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_GT(found.size(), 100u);

    const double beat_sec = 60.0 / 137.0;
    for (std::size_t i = 0; i < found.size(); ++i) {
        const double expected = (0.5 + beat_sec * static_cast<double>(i)) * kRate;
        // Half a sample, ten microseconds — the rounding itself, with nothing
        // added to it by the hundredth beat.
        ASSERT_NEAR(static_cast<double>(found[i]), expected, 0.5) << "beat " << i;
    }
}

TEST(Metronome, MarksTheDownbeatOfEveryBar) {
    Metronome metronome(makeConfig(120.0));
    metronome.start(0.5);

    const std::vector<float> out = run(metronome, 5.0);
    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_GE(found.size(), 8u);

    std::vector<double> peaks;
    for (std::ptrdiff_t at : found) {
        double m = 0.0;
        const auto from = static_cast<std::size_t>(at);
        for (std::size_t i = from; i < std::min(from + 3000, out.size()); ++i) {
            m = std::max(m, std::fabs(static_cast<double>(out[i])));
        }
        peaks.push_back(m);
    }

    for (std::size_t i = 0; i < peaks.size(); ++i) {
        if (i % 4 == 0) {
            EXPECT_GT(peaks[i], peaks[1]) << "downbeat " << i << " should stand out";
        } else {
            EXPECT_NEAR(peaks[i], peaks[1], 1e-6) << "beat " << i;
        }
    }
}

TEST(Metronome, SubdivisionsLandBetweenTheBeats) {
    Metronome metronome(makeConfig(120.0, 0.0, 2));   // eighths
    metronome.start(0.5);

    const std::vector<float> out = run(metronome, 2.0);
    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_EQ(found.size(), 6u);   // 0.5, 0.75, 1.0, 1.25, 1.5, 1.75

    for (std::size_t i = 0; i < found.size(); ++i) {
        const double expected_sec = 0.5 + 0.25 * static_cast<double>(i);
        EXPECT_EQ(found[i], static_cast<std::ptrdiff_t>(expected_sec * kRate)) << "step " << i;
    }
}

TEST(Metronome, FollowsATempoChangeWithoutStumbling) {
    Metronome metronome(makeConfig(120.0));
    metronome.start(0.5);

    std::vector<float> out(static_cast<std::size_t>(4.0 * kRate), 0.0f);
    bool changed = false;

    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        const double t = static_cast<double>(i) / kRate;

        if (!changed && t >= 1.2) {
            metronome.set_tempo(240.0);
            changed = true;
        }
        metronome.process(t, out.data() + i, n);
    }

    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_GE(found.size(), 6u);

    // Beats already handed to the device keep their times; the new tempo takes
    // over from the last one committed. So every gap is one tempo or the other,
    // never something in between, which is what a stumble would look like.
    for (std::size_t i = 1; i < found.size(); ++i) {
        const double gap = static_cast<double>(found[i] - found[i - 1]) / kRate;
        EXPECT_TRUE(std::fabs(gap - 0.5) < 1e-3 || std::fabs(gap - 0.25) < 1e-3)
            << "gap " << i << " was " << gap;
    }
    EXPECT_NEAR(static_cast<double>(found.back() - found[found.size() - 2]) / kRate, 0.25, 1e-3);
}

TEST(Metronome, AlignsItsPhaseWithoutChangingTheTempo) {
    // Manual mode: the player sets the tempo, and the app only has to find where
    // the beat falls.
    Metronome metronome(makeConfig(120.0));
    metronome.start(0.5);

    std::vector<float> out(static_cast<std::size_t>(4.0 * kRate), 0.0f);
    bool aligned = false;

    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        const double t = static_cast<double>(i) / kRate;

        if (!aligned && t >= 1.5) {
            metronome.align_to(2.1, t);   // a beat belongs on 2.1, not on 2.0
            aligned = true;
        }
        metronome.process(t, out.data() + i, n);
    }

    const std::vector<std::ptrdiff_t> found = onsets(out);
    ASSERT_GE(found.size(), 5u);

    // Everything after the alignment sits on the new phase, still half a second
    // apart: the tempo was not touched.
    bool saw_new_phase = false;
    for (std::ptrdiff_t at : found) {
        const double sec = static_cast<double>(at) / kRate;
        if (sec < 2.0) continue;
        const double offset = std::fmod(sec - 2.1 + 5.0, 0.5);
        EXPECT_TRUE(offset < 1e-3 || offset > 0.5 - 1e-3) << "beat at " << sec;
        saw_new_phase = true;
    }
    EXPECT_TRUE(saw_new_phase);
}

TEST(Metronome, HandsHapticAndVisualCuesBackToTheShell) {
    // They cannot be rendered here — they are not audio — so they come back with
    // their own compensated times, which differ from the audio one because the
    // taptic engine and the next display frame are not the same delay.
    MetronomeConfig cfg = makeConfig(120.0, 0.005);
    cfg.grid.channel_enabled = {{true, true, true}};
    cfg.grid.latency_sec = {{0.005, 0.030, 0.016}};

    Metronome metronome(cfg);
    metronome.start(1.0);

    std::vector<float> out(static_cast<std::size_t>(2.0 * kRate), 0.0f);
    std::vector<Event> cues;
    Event batch[16];

    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        const double t = static_cast<double>(i) / kRate;

        std::size_t written = 0;
        metronome.process(t, out.data() + i, n, batch, 16, &written);
        cues.insert(cues.end(), batch, batch + written);
    }

    ASSERT_FALSE(cues.empty());
    for (const Event& cue : cues) {
        EXPECT_NE(cue.channel, Channel::Audio);
        const double latency = cue.channel == Channel::Haptic ? 0.030 : 0.016;
        EXPECT_NEAR(cue.time_sec, cue.beat_time_sec - latency, 1e-12);
    }
    EXPECT_EQ(metronome.stats().cues_dropped, 0u);
}

TEST(Metronome, CountsCuesItHadNowhereToPutRatherThanLosingThem) {
    MetronomeConfig cfg = makeConfig(120.0);
    cfg.grid.channel_enabled = {{true, true, false}};

    Metronome metronome(cfg);
    metronome.start(1.0);

    std::vector<float> out(static_cast<std::size_t>(2.0 * kRate), 0.0f);
    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        metronome.process(static_cast<double>(i) / kRate, out.data() + i, n);
    }

    EXPECT_GT(metronome.stats().cues_dropped, 0u);
    EXPECT_FALSE(metronome.stats().clean());
}

TEST(Metronome, MixesIntoTheBufferRatherThanClearingIt) {
    // Phase 4 plays the click over a backing track. A process() that filled
    // would silently delete it.
    Metronome metronome(makeConfig(120.0));
    metronome.start(1.0);

    std::vector<float> out(static_cast<std::size_t>(2.0 * kRate), 0.25f);
    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        metronome.process(static_cast<double>(i) / kRate, out.data() + i, n);
    }

    EXPECT_FLOAT_EQ(out[0], 0.25f);
    EXPECT_FLOAT_EQ(out[out.size() - 1], 0.25f);
}

TEST(Metronome, StopLetsTheLastClickRingOutAndSilenceCutsIt) {
    MetronomeConfig cfg = makeConfig(120.0);
    cfg.click.beat.length_sec = 0.2;
    cfg.click.downbeat.length_sec = 0.2;

    Metronome ringing(cfg);
    ringing.start(0.1);

    std::vector<float> out(static_cast<std::size_t>(0.5 * kRate), 0.0f);
    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        const double t = static_cast<double>(i) / kRate;
        if (t >= 0.12) ringing.stop();
        ringing.process(t, out.data() + i, n);
    }
    // Stopped 20 ms into a 200 ms click: it finishes, the way letting go of a
    // real metronome sounds.
    EXPECT_GT(std::fabs(out[static_cast<std::size_t>(0.25 * kRate)]), 1e-4f);

    Metronome cut(cfg);
    cut.start(0.1);
    std::fill(out.begin(), out.end(), 0.0f);
    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        const double t = static_cast<double>(i) / kRate;
        if (t >= 0.12) cut.silence();
        cut.process(t, out.data() + i, n);
    }
    for (std::size_t i = static_cast<std::size_t>(0.2 * kRate); i < out.size(); ++i) {
        ASSERT_EQ(out[i], 0.0f) << "sample " << i;
    }
}

TEST(Metronome, StatsNoticeADeviceThatSkipsABuffer) {
    // A device that drops or repeats a buffer breaks the assumption a sounding
    // click makes about time being continuous. Noticing is the difference
    // between the harness reporting a glitch and quietly mistiming.
    Metronome metronome(makeConfig(120.0));
    metronome.start(1.0);

    std::vector<float> buffer(256, 0.0f);
    metronome.process(0.0, buffer.data(), buffer.size());
    metronome.process(256.0 / kRate, buffer.data(), buffer.size());
    EXPECT_EQ(metronome.stats().discontinuities, 0u);

    metronome.process(1.0, buffer.data(), buffer.size());   // a jump
    EXPECT_EQ(metronome.stats().discontinuities, 1u);
    EXPECT_FALSE(metronome.stats().clean());
}

TEST(MetronomeConfig, RejectsWhatEitherHalfWouldReject) {
    EXPECT_TRUE(makeConfig(120.0).valid());

    MetronomeConfig bad_grid = makeConfig(120.0);
    bad_grid.grid.bpm = 0.0;
    EXPECT_FALSE(bad_grid.valid());

    MetronomeConfig bad_click = makeConfig(120.0);
    bad_click.click.sample_rate = 0.0;
    EXPECT_FALSE(bad_click.valid());
}
