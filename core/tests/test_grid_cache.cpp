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
