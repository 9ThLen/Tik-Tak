// Exercises the public C API exactly as a platform shell would — through the
// header alone, with no access to the C++ internals.
#include "tiktak/tiktak.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "support.hpp"

namespace {

constexpr double kSampleRate = 48000.0;

struct OdfHandle {
    tt_odf* ptr = nullptr;
    ~OdfHandle() { tt_odf_destroy(ptr); }
};

}  // namespace

TEST(Api, ReportsItsVersion) {
    ASSERT_NE(tt_version(), nullptr);
    EXPECT_EQ(std::string(tt_version()), "0.1.0");
}

TEST(Api, StatusStringsAreAlwaysPresent) {
    for (tt_status s : {TT_OK, TT_ERR_INVALID_ARG, TT_ERR_OUT_OF_MEMORY, TT_ERR_UNSUPPORTED}) {
        ASSERT_NE(tt_status_string(s), nullptr);
        EXPECT_FALSE(std::string(tt_status_string(s)).empty());
    }
    EXPECT_NE(tt_status_string(static_cast<tt_status>(9999)), nullptr);
}

TEST(Api, DefaultsAreUsable) {
    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);

    EXPECT_DOUBLE_EQ(cfg.sample_rate, kSampleRate);
    EXPECT_EQ(cfg.frame_size, 2048u);
    EXPECT_EQ(cfg.hop_size, 512u);
    EXPECT_EQ(cfg.mel_bands, 81u);
    EXPECT_EQ(cfg.whitening, 1);

    tt_status status = TT_ERR_UNSUPPORTED;
    OdfHandle odf{tt_odf_create(&cfg, &status)};
    EXPECT_EQ(status, TT_OK);
    ASSERT_NE(odf.ptr, nullptr);
}

TEST(Api, ZeroedConfigFallsBackToDefaults) {
    // A caller should be able to memset the struct, set the sample rate, and get
    // a working analyser.
    tt_odf_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = kSampleRate;

    tt_status status = TT_ERR_UNSUPPORTED;
    OdfHandle odf{tt_odf_create(&cfg, &status)};
    EXPECT_EQ(status, TT_OK);
    ASSERT_NE(odf.ptr, nullptr);

    EXPECT_NEAR(tt_odf_latency_sec(odf.ptr), 2048.0 * 0.5 / kSampleRate, 1e-12);
}

TEST(Api, RejectsBadConfigs) {
    tt_status status = TT_OK;
    EXPECT_EQ(tt_odf_create(nullptr, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);
    cfg.frame_size = 1000;  // not a power of two
    status = TT_OK;
    EXPECT_EQ(tt_odf_create(&cfg, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    tt_odf_config_defaults(&cfg, -1.0);
    status = TT_OK;
    EXPECT_EQ(tt_odf_create(&cfg, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);
}

TEST(Api, AcceptsNullStatusOut) {
    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);
    OdfHandle odf{tt_odf_create(&cfg, nullptr)};
    EXPECT_NE(odf.ptr, nullptr);
    EXPECT_EQ(tt_odf_create(nullptr, nullptr), nullptr);
}

TEST(Api, NullHandleIsHarmless) {
    tt_odf_destroy(nullptr);
    tt_odf_reset(nullptr);
    EXPECT_EQ(tt_odf_frames_available(nullptr, 1024), 0u);
    EXPECT_DOUBLE_EQ(tt_odf_latency_sec(nullptr), 0.0);

    tt_odf_frame frame;
    size_t dropped = 123;
    EXPECT_EQ(tt_odf_process(nullptr, nullptr, 0, &frame, 1, &dropped), 0u);
    EXPECT_EQ(dropped, 0u);
}

TEST(Api, ProducesFramesAtTheAdvertisedRate) {
    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);
    OdfHandle odf{tt_odf_create(&cfg, nullptr)};
    ASSERT_NE(odf.ptr, nullptr);

    const auto input = tiktak::test::sine(48000, 440.0, kSampleRate, 0.5f);
    const size_t predicted = tt_odf_frames_available(odf.ptr, input.size());

    std::vector<tt_odf_frame> frames(predicted + 8);
    size_t dropped = 0;
    const size_t written = tt_odf_process(odf.ptr, input.data(), input.size(),
                                          frames.data(), frames.size(), &dropped);

    EXPECT_EQ(written, predicted);
    EXPECT_EQ(dropped, 0u);
    ASSERT_GT(written, 0u);

    const double hopSec = 512.0 / kSampleRate;
    for (size_t i = 1; i < written; ++i) {
        EXPECT_NEAR(frames[i].time_sec - frames[i - 1].time_sec, hopSec, 1e-9)
            << "at frame " << i;
    }
}

TEST(Api, ReportsDroppedFramesWhenTheBufferIsTooSmall) {
    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);
    OdfHandle odf{tt_odf_create(&cfg, nullptr)};
    ASSERT_NE(odf.ptr, nullptr);

    const auto input = tiktak::test::silence(48000);
    const size_t expected = tt_odf_frames_available(odf.ptr, input.size());
    ASSERT_GT(expected, 3u);

    tt_odf_frame frames[3];
    size_t dropped = 0;
    const size_t written =
        tt_odf_process(odf.ptr, input.data(), input.size(), frames, 3, &dropped);

    EXPECT_EQ(written, 3u);
    EXPECT_EQ(dropped, expected - 3);
}

TEST(Api, SurvivesArbitraryBlockSizes) {
    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);
    cfg.frame_size = 1024;
    cfg.hop_size = 256;

    OdfHandle odf{tt_odf_create(&cfg, nullptr)};
    ASSERT_NE(odf.ptr, nullptr);

    auto input = tiktak::test::silence(20000);
    tiktak::test::addBurst(input, 6000, 3000, 300.0, kSampleRate, 0.7f);

    std::vector<tt_odf_frame> collected;
    const size_t block = 137;  // deliberately not a multiple of the hop
    for (size_t pos = 0; pos < input.size(); pos += block) {
        const size_t take = std::min(block, input.size() - pos);

        tt_odf_frame out[4];
        size_t dropped = 0;
        const size_t written =
            tt_odf_process(odf.ptr, input.data() + pos, take, out, 4, &dropped);

        EXPECT_EQ(dropped, 0u) << "a 4-frame buffer should cover a " << block
                               << "-sample block";
        collected.insert(collected.end(), out, out + written);
    }

    ASSERT_FALSE(collected.empty());

    float peak = 0.0f;
    double peakTime = 0.0;
    for (const auto& frame : collected) {
        if (frame.full > peak) {
            peak = frame.full;
            peakTime = frame.time_sec;
        }
    }

    EXPECT_GT(peak, 0.0f);
    EXPECT_NEAR(peakTime, 6000.0 / kSampleRate, 0.03);
}

TEST(Api, ResetRestartsTheStreamClock) {
    tt_odf_config cfg;
    tt_odf_config_defaults(&cfg, kSampleRate);
    OdfHandle odf{tt_odf_create(&cfg, nullptr)};
    ASSERT_NE(odf.ptr, nullptr);

    const auto input = tiktak::test::silence(48000);
    std::vector<tt_odf_frame> frames(256);

    const size_t first =
        tt_odf_process(odf.ptr, input.data(), input.size(), frames.data(), frames.size(), nullptr);
    ASSERT_GT(first, 1u);
    const double lastTime = frames[first - 1].time_sec;
    EXPECT_GT(lastTime, 0.5);

    tt_odf_reset(odf.ptr);

    const size_t second =
        tt_odf_process(odf.ptr, input.data(), input.size(), frames.data(), frames.size(), nullptr);
    EXPECT_EQ(second, first);
    EXPECT_LT(frames[0].time_sec, 0.05);
}
