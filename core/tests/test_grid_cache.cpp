#include "analysis/grid_cache.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "support.hpp"

using tiktak::analysis::OfflineConfig;
using tiktak::analysis::OfflineResult;
using tiktak::analysis::analyseOffline;
using tiktak::analysis::deserializeGrid;
using tiktak::analysis::gridCacheKey;
using tiktak::analysis::serializeGrid;
using tiktak::test::clickTrack;

namespace {

constexpr double kSampleRate = 48000.0;

OfflineConfig testConfig() {
    OfflineConfig cfg;
    cfg.odf.sampleRate = kSampleRate;
    return cfg;
}

OfflineResult analysedTrack() {
    const std::vector<float> audio = clickTrack(120.0, 10.0, kSampleRate);
    return analyseOffline(audio.data(), audio.size(), testConfig());
}

}  // namespace

// The whole point of the cache: what comes back is the analysis, not an
// approximation of it. Beat times compare with == on purpose — serialisation
// stores the exact bits, so anything short of identity is a format bug.
TEST(GridCache, RoundTripsTheAnalysisExactly) {
    const OfflineConfig config = testConfig();
    const OfflineResult original = analysedTrack();
    ASSERT_FALSE(original.beats.empty());

    const std::vector<std::uint8_t> blob = serializeGrid(original, config);

    OfflineResult restored;
    ASSERT_TRUE(deserializeGrid(blob.data(), blob.size(), config, &restored));

    EXPECT_EQ(restored.beats, original.beats);
    EXPECT_EQ(restored.bpm, original.bpm);
    EXPECT_EQ(restored.tempo_confidence, original.tempo_confidence);
    EXPECT_EQ(restored.estimated_bpm, original.estimated_bpm);
    EXPECT_EQ(restored.frame_count, original.frame_count);
}

TEST(GridCache, RoundTripsAGridWithNoBeats) {
    const OfflineConfig config = testConfig();
    OfflineResult empty;
    empty.estimated_bpm = 97.0;

    const std::vector<std::uint8_t> blob = serializeGrid(empty, config);

    OfflineResult restored;
    restored.beats.push_back(1.0);  // must be overwritten, not appended to
    ASSERT_TRUE(deserializeGrid(blob.data(), blob.size(), config, &restored));
    EXPECT_TRUE(restored.beats.empty());
    EXPECT_EQ(restored.estimated_bpm, 97.0);
}

// SHA-256 of "abc", from FIPS 180-4's own worked example. Pins the
// implementation to the real algorithm: a hash that is merely deterministic
// would pass every other test here while producing keys nothing else on earth
// agrees with.
TEST(GridCache, KeyMatchesTheKnownSha256TestVector) {
    EXPECT_EQ(gridCacheKey("abc", 3),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(gridCacheKey("", 0),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(GridCache, KeySeesEveryByte) {
    // Longer than one SHA block, so the multi-block path is exercised too.
    std::vector<std::uint8_t> file(200, 0x5a);
    const std::string key = gridCacheKey(file.data(), file.size());
    EXPECT_EQ(key.size(), 64u);

    file[150] ^= 0x01;
    EXPECT_NE(gridCacheKey(file.data(), file.size()), key);
}

// The config is part of the grid's identity. The same track analysed with a
// manual-mode tempo hint is a different grid, and serving the auto-mode answer
// for it would be the classic stale-cache bug — beats in the wrong place with
// nothing obviously broken.
TEST(GridCache, RefusesAGridAnalysedUnderAnotherConfig) {
    const OfflineConfig config = testConfig();
    const std::vector<std::uint8_t> blob = serializeGrid(analysedTrack(), config);

    OfflineConfig hinted = config;
    hinted.bpm_hint = 120.0;
    OfflineConfig tighter = config;
    tighter.tracker.tightness = 50.0;

    OfflineResult restored;
    EXPECT_FALSE(deserializeGrid(blob.data(), blob.size(), hinted, &restored));
    EXPECT_FALSE(deserializeGrid(blob.data(), blob.size(), tighter, &restored));
    EXPECT_TRUE(deserializeGrid(blob.data(), blob.size(), config, &restored));
}

TEST(GridCache, RefusesBytesThatAreNotAGrid) {
    const OfflineConfig config = testConfig();
    OfflineResult restored;

    const std::vector<std::uint8_t> noise(128, 0xab);
    EXPECT_FALSE(deserializeGrid(noise.data(), noise.size(), config, &restored));
    EXPECT_FALSE(deserializeGrid(noise.data(), 0, config, &restored));
    EXPECT_FALSE(deserializeGrid(nullptr, 128, config, &restored));
}

// A cache file that lost its tail must read as a miss, not as a shorter grid:
// a track that silently loses its last minute of beats looks like the analyser
// failing, and nobody would think to suspect the cache.
TEST(GridCache, RefusesATruncatedGrid) {
    const OfflineConfig config = testConfig();
    const std::vector<std::uint8_t> blob = serializeGrid(analysedTrack(), config);

    OfflineResult restored;
    for (const std::size_t cut : {std::size_t{1}, std::size_t{8}, blob.size() / 2}) {
        EXPECT_FALSE(deserializeGrid(blob.data(), blob.size() - cut, config, &restored))
            << "accepted a grid missing its last " << cut << " bytes";
    }
}

TEST(GridCache, RefusesACorruptedGrid) {
    const OfflineConfig config = testConfig();
    std::vector<std::uint8_t> blob = serializeGrid(analysedTrack(), config);

    // Flip one bit in the middle of the beat data — the header still parses,
    // only the checksum can notice.
    blob[blob.size() / 2] ^= 0x10;

    OfflineResult restored;
    EXPECT_FALSE(deserializeGrid(blob.data(), blob.size(), config, &restored));
}

// Bar lines are part of the analysis, so they are part of what the cache owes
// the caller. A grid that came back without them would make the second import
// of a track quietly worse than the first.
TEST(GridCache, RoundTripsTheBarLines) {
    OfflineConfig config = testConfig();
    OfflineResult original = analysedTrack();
    // The click track has no harmony and an alternating accent, so rather than
    // depend on what it detects, state a grid and require it back untouched.
    original.beats_per_bar = 3;
    original.downbeat_strength = 1.25;
    original.downbeat_phase_margin = 0.75;
    original.downbeats = {0.5, 2.0, 3.5, 5.0};

    const std::vector<std::uint8_t> blob = serializeGrid(original, config);

    OfflineResult restored;
    ASSERT_TRUE(deserializeGrid(blob.data(), blob.size(), config, &restored));

    EXPECT_EQ(restored.beats_per_bar, 3);
    EXPECT_DOUBLE_EQ(restored.downbeat_strength, 1.25);
    EXPECT_DOUBLE_EQ(restored.downbeat_phase_margin, 0.75);
    ASSERT_EQ(restored.downbeats.size(), original.downbeats.size());
    for (std::size_t i = 0; i < original.downbeats.size(); ++i) {
        EXPECT_DOUBLE_EQ(restored.downbeats[i], original.downbeats[i]) << "at bar " << i;
    }
    // And the beats still survive alongside them.
    ASSERT_EQ(restored.beats.size(), original.beats.size());
    EXPECT_DOUBLE_EQ(restored.beats.front(), original.beats.front());
}

TEST(GridCache, RefusesAGridAnalysedWithADifferentIdeaOfBars) {
    OfflineConfig config = testConfig();
    const OfflineResult original = analysedTrack();
    const std::vector<std::uint8_t> blob = serializeGrid(original, config);

    // Same audio, same tracker — but a caller who weighs harmony differently
    // gets different bar lines, and serving the old ones would be a stale
    // cache the user experiences as the accent in the wrong place.
    OfflineConfig other = config;
    other.downbeat.harmony_weight = 0.25;

    OfflineResult restored;
    EXPECT_FALSE(deserializeGrid(blob.data(), blob.size(), other, &restored));

    OfflineConfig fewer_meters = config;
    fewer_meters.downbeat.meters = {{4, 1.0}};
    EXPECT_FALSE(deserializeGrid(blob.data(), blob.size(), fewer_meters, &restored));
}
