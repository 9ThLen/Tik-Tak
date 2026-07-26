#include "analysis/offline.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "support.hpp"

using tiktak::analysis::OfflineAnalyzer;
using tiktak::analysis::OfflineConfig;
using tiktak::analysis::OfflineResult;
using tiktak::analysis::analyseOffline;
using tiktak::test::bandTrack;
using tiktak::test::clickTrack;

namespace {

constexpr double kSampleRate = 48000.0;

OfflineConfig testConfig() {
    OfflineConfig cfg;
    cfg.odf.sampleRate = kSampleRate;
    return cfg;
}

// Fraction of true beats matched within `tolerance`, the recall half of the
// mir_eval F-measure. 70 ms is the standard beat-tracking tolerance.
double recall(const std::vector<double>& truth, const std::vector<double>& found,
              double tolerance = 0.07) {
    if (truth.empty()) return 0.0;
    std::size_t hits = 0;
    for (double expected : truth) {
        const bool matched = std::any_of(found.begin(), found.end(), [&](double beat) {
            return std::abs(beat - expected) <= tolerance;
        });
        if (matched) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(truth.size());
}

std::vector<double> beatTimes(double bpm, double durationSec, double leadSec = 0.0) {
    std::vector<double> beats;
    for (double t = leadSec; t < durationSec; t += 60.0 / bpm) beats.push_back(t);
    return beats;
}

}  // namespace

TEST(Offline, FindsTheTempoAndBeatsOfAClickTrack) {
    constexpr double kBpm = 120.0;
    constexpr double kDuration = 20.0;
    const std::vector<float> audio = clickTrack(kBpm, kDuration, kSampleRate);

    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

    EXPECT_NEAR(result.bpm, kBpm, kBpm * 0.03);
    EXPECT_GT(result.tempo_confidence, 0.3);
    EXPECT_GT(recall(beatTimes(kBpm, kDuration), result.beats), 0.9);
}

TEST(Offline, HandlesARangeOfTempi) {
    constexpr double kDuration = 20.0;
    for (double bpm : {90.0, 110.0, 120.0, 132.0, 150.0}) {
        const std::vector<float> audio = clickTrack(bpm, kDuration, kSampleRate);
        const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

        EXPECT_NEAR(result.bpm, bpm, bpm * 0.03) << "at " << bpm << " BPM";
        EXPECT_GT(recall(beatTimes(bpm, kDuration), result.beats), 0.9) << "at " << bpm << " BPM";
    }
}

TEST(Offline, LocksOntoMusicThatStartsAfterSomeSilence) {
    // An imported backing track usually has a moment of silence before the
    // first note. The grid must start with the music, not with the file.
    constexpr double kBpm = 120.0;
    constexpr double kDuration = 22.0;
    constexpr double kLead = 3.0;
    const std::vector<float> audio = clickTrack(kBpm, kDuration, kSampleRate, kLead);

    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

    EXPECT_NEAR(result.bpm, kBpm, kBpm * 0.03);
    EXPECT_GT(recall(beatTimes(kBpm, kDuration, kLead), result.beats), 0.9);

    // No clicking through the silence before the first hit.
    const auto early = std::count_if(result.beats.begin(), result.beats.end(),
                                     [](double t) { return t < kLead - 0.1; });
    EXPECT_LE(early, 1);
}

// The reason audio is fed in blocks at all: a five-minute track becomes a few
// hundred kilobytes of onset frames instead of tens of megabytes of samples.
// The block boundaries must not be visible in the result.
TEST(Offline, BlockSizeDoesNotChangeTheAnswer) {
    const std::vector<float> audio = clickTrack(128.0, 15.0, kSampleRate);

    const OfflineResult whole = analyseOffline(audio.data(), audio.size(), testConfig());

    // 997 is deliberately not a multiple of the 512-sample hop, so hops land at
    // every possible offset within a block.
    OfflineAnalyzer chunked{testConfig()};
    for (std::size_t at = 0; at < audio.size(); at += 997) {
        const std::size_t n = std::min<std::size_t>(997, audio.size() - at);
        chunked.feed(audio.data() + at, n);
    }
    const OfflineResult streamed = chunked.finish();

    EXPECT_DOUBLE_EQ(streamed.bpm, whole.bpm);
    EXPECT_EQ(streamed.frame_count, whole.frame_count);
    ASSERT_EQ(streamed.beats.size(), whole.beats.size());
    for (std::size_t i = 0; i < whole.beats.size(); ++i) {
        EXPECT_DOUBLE_EQ(streamed.beats[i], whole.beats[i]) << "at beat " << i;
    }
}

TEST(Offline, AFixedTempoIsUsedInsteadOfEstimating) {
    constexpr double kBpm = 120.0;
    const std::vector<float> audio = clickTrack(kBpm, 15.0, kSampleRate);

    OfflineConfig cfg = testConfig();
    cfg.bpm_hint = 60.0;   // the user says half time

    const OfflineResult result = analyseOffline(audio.data(), audio.size(), cfg);
    EXPECT_DOUBLE_EQ(result.bpm, 60.0);

    // Half time means every other click, so the grid should be roughly half as
    // dense as the material.
    const double expected = 15.0 / (60.0 / 60.0);
    EXPECT_LT(static_cast<double>(result.beats.size()), expected * 1.3);
}

TEST(Offline, ResetClearsEverythingForTheNextFile) {
    const std::vector<float> first = clickTrack(100.0, 15.0, kSampleRate);
    const std::vector<float> second = clickTrack(140.0, 15.0, kSampleRate);

    OfflineAnalyzer analyzer{testConfig()};
    analyzer.feed(first.data(), first.size());
    analyzer.finish();

    analyzer.reset();
    EXPECT_TRUE(analyzer.odfValues().empty());

    analyzer.feed(second.data(), second.size());
    const OfflineResult reused = analyzer.finish();
    const OfflineResult fresh = analyseOffline(second.data(), second.size(), testConfig());

    EXPECT_DOUBLE_EQ(reused.bpm, fresh.bpm);
    EXPECT_EQ(reused.frame_count, fresh.frame_count);
}

TEST(Offline, SilenceProducesNoBeats) {
    const std::vector<float> audio(static_cast<std::size_t>(kSampleRate * 10.0), 0.0f);
    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

    EXPECT_TRUE(result.beats.empty());
    EXPECT_DOUBLE_EQ(result.tempo_confidence, 0.0);
    EXPECT_GT(result.frame_count, 0u);
}

TEST(Offline, EmptyInputIsHandled) {
    OfflineAnalyzer analyzer{testConfig()};
    analyzer.feed(nullptr, 0);

    const OfflineResult result = analyzer.finish();
    EXPECT_TRUE(result.beats.empty());
    EXPECT_EQ(result.frame_count, 0u);
}

// ---------------------------------------------------------------- downbeats --

TEST(Offline, FindsTheBarLinesOfAFourFourTrack) {
    const std::vector<float> audio = bandTrack(120.0, 8, 4, kSampleRate);

    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

    EXPECT_EQ(result.beats_per_bar, 4);
    ASSERT_FALSE(result.downbeats.empty());
    EXPECT_GT(result.downbeat_strength, 0.5);
    EXPECT_GT(result.downbeat_phase_margin, 0.3);

    // Every bar line must sit on a bar line of the material, two seconds apart
    // at 120 BPM in four.
    for (double t : result.downbeats) {
        const double bars = t / 2.0;
        EXPECT_LT(std::abs(bars - std::round(bars)), 0.05) << "bar line at " << t;
    }
}

TEST(Offline, FindsTheBarLinesOfAWaltz) {
    const std::vector<float> audio = bandTrack(150.0, 10, 3, kSampleRate);

    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

    EXPECT_EQ(result.beats_per_bar, 3);
    EXPECT_GT(result.downbeat_phase_margin, 0.3);
}

TEST(Offline, BarLinesAreNotAskedForWhenTheyAreNotWanted) {
    OfflineConfig cfg = testConfig();
    cfg.find_downbeats = false;

    const std::vector<float> audio = bandTrack(120.0, 8, 4, kSampleRate);
    const OfflineResult result = analyseOffline(audio.data(), audio.size(), cfg);

    EXPECT_FALSE(result.beats.empty());
    EXPECT_TRUE(result.downbeats.empty());
    EXPECT_EQ(result.beats_per_bar, 0);
}

TEST(Offline, ATrackTooShortToRepeatGetsNoBarLines) {
    const std::vector<float> audio = bandTrack(120.0, 1, 4, kSampleRate);
    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
}

TEST(Offline, ResetClearsTheDownbeatCuesToo) {
    OfflineAnalyzer analyzer(testConfig());

    const std::vector<float> waltz = bandTrack(150.0, 10, 3, kSampleRate);
    analyzer.feed(waltz.data(), waltz.size());
    ASSERT_EQ(analyzer.finish().beats_per_bar, 3);

    analyzer.reset();
    const std::vector<float> four = bandTrack(120.0, 8, 4, kSampleRate);
    analyzer.feed(four.data(), four.size());
    EXPECT_EQ(analyzer.finish().beats_per_bar, 4);
}

TEST(Offline, FindsTheBarLineWhenTheRecordingStartsMidBar) {
    // Two beats of pickup before the first bar line. Placing the accent on
    // beat three is the one downbeat error a listener notices immediately.
    const std::vector<float> audio = bandTrack(120.0, 8, 4, kSampleRate, 2);

    const OfflineResult result = analyseOffline(audio.data(), audio.size(), testConfig());
    ASSERT_EQ(result.beats_per_bar, 4);
    ASSERT_FALSE(result.downbeats.empty());

    // Bar lines now fall on odd seconds: two beats of 0.5 s, then every 2 s.
    for (double t : result.downbeats) {
        const double bars = (t - 1.0) / 2.0;
        EXPECT_LT(std::abs(bars - std::round(bars)), 0.05) << "bar line at " << t;
    }
}

TEST(Offline, TheHarmonyAloneFindsTheBarWithNoDrumsToHelp) {
    // Worth proving separately, because it is the cue most easily lost: the
    // front end's window cannot resolve a semitone below roughly 800 Hz, so
    // this only works at all if a chord change moves the upper partials enough
    // to see. With the rhythm cues switched off there is nothing else left.
    OfflineConfig cfg = testConfig();
    cfg.downbeat.low_weight = 0.0;
    cfg.downbeat.accent_weight = 0.0;

    const std::vector<float> audio = bandTrack(120.0, 8, 4, kSampleRate, 2);
    const OfflineResult result = analyseOffline(audio.data(), audio.size(), cfg);

    EXPECT_EQ(result.beats_per_bar, 4);
    ASSERT_FALSE(result.downbeats.empty());
    for (double t : result.downbeats) {
        const double bars = (t - 1.0) / 2.0;
        EXPECT_LT(std::abs(bars - std::round(bars)), 0.05) << "bar line at " << t;
    }
}

TEST(Offline, TheDrumsAloneFindTheBarWithNoHarmonyToHelp) {
    OfflineConfig cfg = testConfig();
    cfg.downbeat.harmony_weight = 0.0;

    const std::vector<float> audio = bandTrack(120.0, 8, 4, kSampleRate, 2);
    const OfflineResult result = analyseOffline(audio.data(), audio.size(), cfg);

    EXPECT_EQ(result.beats_per_bar, 4);
    ASSERT_FALSE(result.downbeats.empty());
    for (double t : result.downbeats) {
        const double bars = (t - 1.0) / 2.0;
        EXPECT_LT(std::abs(bars - std::round(bars)), 0.05) << "bar line at " << t;
    }
}
