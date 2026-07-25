// Exercises the public C API exactly as a platform shell would — through the
// header alone, with no access to the C++ internals.
#include "tiktak/tiktak.h"

#include <gtest/gtest.h>

#include <cmath>
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

/* -------------------------------------------------------------- scheduler -- */

namespace {

struct SchedulerHandle {
    tt_scheduler* ptr = nullptr;
    ~SchedulerHandle() { tt_scheduler_destroy(ptr); }
};

tt_scheduler_config beatsOnly(double bpm = 120.0) {
    tt_scheduler_config cfg;
    tt_scheduler_config_defaults(&cfg);
    cfg.bpm = bpm;
    cfg.channel_enabled[TT_CHANNEL_HAPTIC] = 0;
    cfg.channel_enabled[TT_CHANNEL_VISUAL] = 0;
    return cfg;
}

}  // namespace

TEST(SchedulerApi, DefaultsAreUsable) {
    tt_scheduler_config cfg;
    tt_scheduler_config_defaults(&cfg);
    EXPECT_DOUBLE_EQ(cfg.bpm, 120.0);
    EXPECT_EQ(cfg.beats_per_bar, 4);
    EXPECT_EQ(cfg.subdivisions, 1);

    tt_status status = TT_ERR_UNSUPPORTED;
    SchedulerHandle scheduler{tt_scheduler_create(&cfg, &status)};
    EXPECT_EQ(status, TT_OK);
    ASSERT_NE(scheduler.ptr, nullptr);
    EXPECT_EQ(tt_scheduler_running(scheduler.ptr), 0);
}

TEST(SchedulerApi, ZeroedConfigFallsBackToDefaults) {
    tt_scheduler_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.channel_enabled[TT_CHANNEL_AUDIO] = 1;

    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);

    tt_scheduler_start(scheduler.ptr, 0.0);
    EXPECT_NEAR(tt_scheduler_step_time(scheduler.ptr, 2), 1.0, 1e-12);  // 120 BPM
}

TEST(SchedulerApi, NullHandleIsHarmless) {
    tt_scheduler_destroy(nullptr);
    tt_scheduler_start(nullptr, 0.0);
    tt_scheduler_stop(nullptr);
    tt_scheduler_set_tempo(nullptr, 120.0);
    tt_scheduler_align_to(nullptr, 1.0, 1.0);

    EXPECT_EQ(tt_scheduler_running(nullptr), 0);
    EXPECT_EQ(tt_scheduler_late_count(nullptr), 0u);
    EXPECT_DOUBLE_EQ(tt_scheduler_step_time(nullptr, 3), 0.0);

    tt_event event;
    size_t dropped = 99;
    EXPECT_EQ(tt_scheduler_pull(nullptr, 0.0, &event, 1, &dropped), 0u);
    EXPECT_EQ(dropped, 0u);
}

TEST(SchedulerApi, RejectsBadConfig) {
    tt_status status = TT_OK;
    EXPECT_EQ(tt_scheduler_create(nullptr, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    tt_scheduler_config cfg = beatsOnly();
    cfg.latency_sec[TT_CHANNEL_AUDIO] = 100.0;  // absurd
    status = TT_OK;
    EXPECT_EQ(tt_scheduler_create(&cfg, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);
}

TEST(SchedulerApi, DeliversBeatsOnTheGrid) {
    tt_scheduler_config cfg = beatsOnly(120.0);
    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);

    tt_scheduler_start(scheduler.ptr, 5.0);
    EXPECT_EQ(tt_scheduler_running(scheduler.ptr), 1);

    std::vector<tt_event> collected;
    tt_event buffer[64];
    for (double now = 5.0; now < 9.0; now += 0.01) {
        const size_t count = tt_scheduler_pull(scheduler.ptr, now, buffer, 64, nullptr);
        collected.insert(collected.end(), buffer, buffer + count);
    }

    ASSERT_GE(collected.size(), 8u);
    for (size_t i = 0; i < collected.size(); ++i) {
        EXPECT_NEAR(collected[i].beat_time_sec, 5.0 + 0.5 * static_cast<double>(i), 1e-12);
        EXPECT_EQ(collected[i].channel, TT_CHANNEL_AUDIO);
        EXPECT_EQ(collected[i].kind, i % 4 == 0 ? TT_BEAT_DOWNBEAT : TT_BEAT_BEAT);
        EXPECT_EQ(collected[i].beat_in_bar, static_cast<int>(i % 4));
    }
}

TEST(SchedulerApi, PullBeyondTheStagingBatchIsContiguous) {
    // The C shim stages through a fixed stack buffer; a request larger than one
    // batch must still come back as one unbroken run of events.
    tt_scheduler_config cfg = beatsOnly(600.0);  // fast, so many fit in the horizon
    cfg.lookahead_sec = 8.0;

    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);
    tt_scheduler_start(scheduler.ptr, 0.0);

    std::vector<tt_event> buffer(100);
    const size_t count = tt_scheduler_pull(scheduler.ptr, 0.0, buffer.data(), buffer.size(),
                                           nullptr);

    ASSERT_GT(count, 40u) << "expected more events than one staging batch";
    for (size_t i = 1; i < count; ++i) {
        EXPECT_EQ(buffer[i].step, buffer[i - 1].step + 1) << "gap at index " << i;
        EXPECT_GT(buffer[i].beat_time_sec, buffer[i - 1].beat_time_sec);
    }
}

TEST(SchedulerApi, LatencyIsCompensatedPerChannel) {
    tt_scheduler_config cfg;
    tt_scheduler_config_defaults(&cfg);
    cfg.lookahead_sec = 0.5;
    cfg.latency_sec[TT_CHANNEL_AUDIO] = 0.030;
    cfg.latency_sec[TT_CHANNEL_HAPTIC] = 0.015;
    cfg.latency_sec[TT_CHANNEL_VISUAL] = 0.008;

    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);
    tt_scheduler_start(scheduler.ptr, 0.0);

    tt_event buffer[64];
    const size_t count = tt_scheduler_pull(scheduler.ptr, 0.0, buffer, 64, nullptr);
    ASSERT_GE(count, 3u);

    const double expected[TT_CHANNEL_COUNT] = {0.030, 0.015, 0.008};
    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(buffer[i].time_sec,
                    buffer[i].beat_time_sec - expected[buffer[i].channel], 1e-12);
    }
}

TEST(SchedulerApi, ReportsLateDrops) {
    tt_scheduler_config cfg = beatsOnly(120.0);
    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);

    tt_scheduler_start(scheduler.ptr, 0.0);

    tt_event buffer[64];
    size_t dropped = 0;
    tt_scheduler_pull(scheduler.ptr, 30.0, buffer, 64, &dropped);

    EXPECT_GT(dropped, 0u);
    EXPECT_EQ(tt_scheduler_late_count(scheduler.ptr), dropped);
}

TEST(SchedulerApi, TempoChangeKeepsTheGridMonotonic) {
    tt_scheduler_config cfg = beatsOnly(100.0);
    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);
    tt_scheduler_start(scheduler.ptr, 0.0);

    std::vector<tt_event> collected;
    tt_event buffer[64];
    for (double now = 0.0; now < 8.0; now += 0.01) {
        const size_t count = tt_scheduler_pull(scheduler.ptr, now, buffer, 64, nullptr);
        collected.insert(collected.end(), buffer, buffer + count);
        if (now > 3.0 && now < 3.011) tt_scheduler_set_tempo(scheduler.ptr, 165.0);
    }

    ASSERT_GT(collected.size(), 10u);
    for (size_t i = 1; i < collected.size(); ++i) {
        EXPECT_GT(collected[i].beat_time_sec, collected[i - 1].beat_time_sec);
    }
}

TEST(SchedulerApi, AlignToSnapsThePhase) {
    tt_scheduler_config cfg = beatsOnly(120.0);
    SchedulerHandle scheduler{tt_scheduler_create(&cfg, nullptr)};
    ASSERT_NE(scheduler.ptr, nullptr);

    tt_scheduler_start(scheduler.ptr, 0.0);
    tt_scheduler_align_to(scheduler.ptr, 4.37, 4.0);

    tt_event buffer[64];
    std::vector<tt_event> collected;
    for (double now = 4.0; now < 7.0; now += 0.01) {
        const size_t count = tt_scheduler_pull(scheduler.ptr, now, buffer, 64, nullptr);
        collected.insert(collected.end(), buffer, buffer + count);
    }

    ASSERT_GE(collected.size(), 4u);
    for (const auto& event : collected) {
        const double beats = (event.beat_time_sec - 4.37) / 0.5;
        EXPECT_NEAR(beats, std::round(beats), 1e-9);
    }
}

/* --------------------------------------------------------------- offline -- */

namespace {

using tiktak::test::clickTrack;

constexpr double kOfflineRate = 48000.0;

// RAII around the C handle so a failing assertion cannot leak it.
struct Offline {
    explicit Offline(const tt_offline_config& cfg) {
        handle = tt_offline_create(&cfg, &status);
    }
    ~Offline() { tt_offline_destroy(handle); }
    Offline(const Offline&) = delete;
    Offline& operator=(const Offline&) = delete;

    tt_offline* handle = nullptr;
    tt_status status = TT_OK;
};

tt_offline_config offlineDefaults() {
    tt_offline_config cfg;
    tt_offline_config_defaults(&cfg, kOfflineRate);
    return cfg;
}

std::vector<double> readBeats(const tt_offline* offline) {
    std::vector<double> beats(tt_offline_beat_count(offline));
    if (!beats.empty()) tt_offline_beats(offline, beats.data(), beats.size());
    return beats;
}

}  // namespace

TEST(OfflineApi, AnalysesAClickTrackEndToEnd) {
    const std::vector<float> audio = clickTrack(120.0, 20.0, kOfflineRate);

    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);
    EXPECT_EQ(offline.status, TT_OK);

    EXPECT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);
    EXPECT_EQ(tt_offline_finish(offline.handle), TT_OK);

    EXPECT_NEAR(tt_offline_bpm(offline.handle), 120.0, 4.0);
    EXPECT_GT(tt_offline_confidence(offline.handle), 0.3);
    EXPECT_GT(tt_offline_frame_count(offline.handle), 0u);

    const std::vector<double> beats = readBeats(offline.handle);
    ASSERT_GT(beats.size(), 30u);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        EXPECT_GT(beats[i], beats[i - 1]) << "at beat " << i;
    }
}

TEST(OfflineApi, ZeroedConfigFallsBackToDefaults) {
    tt_offline_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.odf.sample_rate = kOfflineRate;

    Offline offline{cfg};
    ASSERT_NE(offline.handle, nullptr);

    const std::vector<float> audio = clickTrack(120.0, 15.0, kOfflineRate);
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);
    EXPECT_NEAR(tt_offline_bpm(offline.handle), 120.0, 4.0);
}

TEST(OfflineApi, ResultsAreUnavailableUntilFinish) {
    const std::vector<float> audio = clickTrack(120.0, 15.0, kOfflineRate);

    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);

    // Reading before finishing must return nothing rather than a stale or
    // half-computed grid.
    EXPECT_DOUBLE_EQ(tt_offline_bpm(offline.handle), 0.0);
    EXPECT_EQ(tt_offline_beat_count(offline.handle), 0u);

    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);
    EXPECT_GT(tt_offline_beat_count(offline.handle), 0u);

    // Feeding more invalidates the answer again.
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), 1024), TT_OK);
    EXPECT_EQ(tt_offline_beat_count(offline.handle), 0u);
}

TEST(OfflineApi, BeatCopyRespectsCapacity) {
    const std::vector<float> audio = clickTrack(120.0, 20.0, kOfflineRate);

    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);

    const std::size_t total = tt_offline_beat_count(offline.handle);
    ASSERT_GT(total, 5u);

    double few[5];
    EXPECT_EQ(tt_offline_beats(offline.handle, few, 5), 5u);

    std::vector<double> many(total + 10, -1.0);
    EXPECT_EQ(tt_offline_beats(offline.handle, many.data(), many.size()), total);
    EXPECT_DOUBLE_EQ(many[total], -1.0);   // nothing written past the end

    for (std::size_t i = 0; i < 5; ++i) EXPECT_DOUBLE_EQ(few[i], many[i]);
}

TEST(OfflineApi, AHintIsUsedButTheAudioIsStillMeasured) {
    const std::vector<float> audio = clickTrack(120.0, 20.0, kOfflineRate);

    tt_offline_config cfg = offlineDefaults();
    cfg.bpm_hint = 75.0;

    Offline offline{cfg};
    ASSERT_NE(offline.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);

    EXPECT_DOUBLE_EQ(tt_offline_bpm(offline.handle), 75.0);
    // The disagreement is what lets manual mode warn instead of tracking badly.
    EXPECT_NEAR(tt_offline_estimated_bpm(offline.handle), 120.0, 4.0);
}

TEST(OfflineApi, OffersAlternativeTempoReadings) {
    const std::vector<float> audio = clickTrack(120.0, 20.0, kOfflineRate);

    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);

    tt_tempo_candidate candidates[3];
    const std::size_t written = tt_offline_tempo_candidates(offline.handle, candidates, 3);
    ASSERT_EQ(written, 3u);

    EXPECT_DOUBLE_EQ(candidates[0].strength, 1.0);
    EXPECT_NEAR(candidates[0].bpm, tt_offline_bpm(offline.handle), 1e-9);
    EXPECT_GE(candidates[0].strength, candidates[1].strength);
    EXPECT_GE(candidates[1].strength, candidates[2].strength);
}

TEST(OfflineApi, ResetAllowsAnotherFile) {
    const std::vector<float> slow = clickTrack(90.0, 20.0, kOfflineRate);
    const std::vector<float> fast = clickTrack(140.0, 20.0, kOfflineRate);

    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);

    ASSERT_EQ(tt_offline_feed(offline.handle, slow.data(), slow.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);
    EXPECT_NEAR(tt_offline_bpm(offline.handle), 90.0, 4.0);

    tt_offline_reset(offline.handle);
    EXPECT_EQ(tt_offline_frame_count(offline.handle), 0u);
    EXPECT_DOUBLE_EQ(tt_offline_bpm(offline.handle), 0.0);

    ASSERT_EQ(tt_offline_feed(offline.handle, fast.data(), fast.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);
    EXPECT_NEAR(tt_offline_bpm(offline.handle), 140.0, 5.0);
}

TEST(OfflineApi, RejectsBadConfig) {
    tt_offline_config cfg = offlineDefaults();
    cfg.odf.sample_rate = 0.0;
    tt_status status = TT_OK;
    EXPECT_EQ(tt_offline_create(&cfg, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    cfg = offlineDefaults();
    cfg.min_bpm = 300.0;   // above max_bpm
    EXPECT_EQ(tt_offline_create(&cfg, nullptr), nullptr);

    EXPECT_EQ(tt_offline_create(nullptr, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);
}

TEST(OfflineApi, NullHandleIsHarmless) {
    double beats[4];
    tt_tempo_candidate candidates[2];

    EXPECT_EQ(tt_offline_feed(nullptr, nullptr, 0), TT_ERR_INVALID_ARG);
    EXPECT_EQ(tt_offline_finish(nullptr), TT_ERR_INVALID_ARG);
    EXPECT_DOUBLE_EQ(tt_offline_bpm(nullptr), 0.0);
    EXPECT_DOUBLE_EQ(tt_offline_estimated_bpm(nullptr), 0.0);
    EXPECT_DOUBLE_EQ(tt_offline_confidence(nullptr), 0.0);
    EXPECT_EQ(tt_offline_beat_count(nullptr), 0u);
    EXPECT_EQ(tt_offline_beats(nullptr, beats, 4), 0u);
    EXPECT_EQ(tt_offline_tempo_candidates(nullptr, candidates, 2), 0u);
    EXPECT_EQ(tt_offline_frame_count(nullptr), 0u);

    tt_offline_reset(nullptr);
    tt_offline_destroy(nullptr);
    tt_offline_config_defaults(nullptr, 48000.0);
}

TEST(OfflineApi, FeedRejectsANullBufferWithSamples) {
    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);

    EXPECT_EQ(tt_offline_feed(offline.handle, nullptr, 128), TT_ERR_INVALID_ARG);
    EXPECT_EQ(tt_offline_feed(offline.handle, nullptr, 0), TT_OK);
}

/* ------------------------------------------------------------------ click -- */

namespace {

struct Click {
    explicit Click(const tt_click_config& cfg) {
        handle = tt_click_create(&cfg, &status);
    }
    ~Click() { tt_click_destroy(handle); }

    Click(const Click&) = delete;
    Click& operator=(const Click&) = delete;

    tt_click* handle = nullptr;
    tt_status status = TT_OK;
};

tt_click_config clickDefaults(double sample_rate = 48000.0) {
    tt_click_config cfg{};
    tt_click_config_defaults(&cfg, sample_rate);
    return cfg;
}

}  // namespace

TEST(ClickApi, DefaultsProduceAWorkingRenderer) {
    Click click{clickDefaults()};
    ASSERT_NE(click.handle, nullptr);
    EXPECT_EQ(click.status, TT_OK);

    EXPECT_EQ(tt_click_schedule(click.handle, 0.01, TT_BEAT_DOWNBEAT), 1);
    EXPECT_EQ(tt_click_pending(click.handle), 1u);

    // Shorter than the click, so it is still sounding at the end of the buffer.
    std::vector<float> out(2400, 0.0f);
    tt_click_mix(click.handle, 0.0, out.data(), out.size());

    EXPECT_EQ(tt_click_pending(click.handle), 0u);
    EXPECT_EQ(tt_click_active_voices(click.handle), 1u);

    double peak = 0.0;
    for (float v : out) peak = std::max(peak, std::fabs(static_cast<double>(v)));
    EXPECT_GT(peak, 0.1);
}

TEST(ClickApi, ZeroMeansDefaultForEveryField) {
    // A caller that memsets the struct and sets only the sample rate must get a
    // usable metronome, the same convention the rest of the API follows.
    tt_click_config cfg{};
    cfg.sample_rate = 48000.0;

    Click click{cfg};
    ASSERT_NE(click.handle, nullptr);

    tt_click_schedule(click.handle, 0.0, TT_BEAT_BEAT);
    std::vector<float> out(4800, 0.0f);
    tt_click_mix(click.handle, 0.0, out.data(), out.size());

    double peak = 0.0;
    for (float v : out) peak = std::max(peak, std::fabs(static_cast<double>(v)));
    EXPECT_GT(peak, 0.1);
}

TEST(ClickApi, RejectsAConfigItCannotHonour) {
    tt_click_config cfg = clickDefaults();
    cfg.sample_rate = 0.0;
    tt_status status = TT_OK;
    EXPECT_EQ(tt_click_create(&cfg, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    // A tone past Nyquist would come back as a lower one, so it is refused
    // rather than quietly mirrored.
    cfg = clickDefaults(8000.0);
    cfg.beat.frequency_hz = 5000.0;
    EXPECT_EQ(tt_click_create(&cfg, nullptr), nullptr);

    EXPECT_EQ(tt_click_create(nullptr, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);
}

TEST(ClickApi, RefusesABeatKindItDoesNotKnow) {
    Click click{clickDefaults()};
    ASSERT_NE(click.handle, nullptr);

    EXPECT_EQ(tt_click_schedule(click.handle, 0.1, -1), 0);
    EXPECT_EQ(tt_click_schedule(click.handle, 0.1, 99), 0);
    EXPECT_EQ(tt_click_pending(click.handle), 0u);
}

TEST(ClickApi, TakesTheSchedulersEventsDirectly) {
    // The two are meant to be wired together without the shell converting
    // anything: tt_event.time_sec and tt_event.kind go straight in.
    tt_scheduler_config sched_cfg;
    tt_scheduler_config_defaults(&sched_cfg);
    sched_cfg.bpm = 120.0;
    sched_cfg.channel_enabled[TT_CHANNEL_HAPTIC] = 0;
    sched_cfg.channel_enabled[TT_CHANNEL_VISUAL] = 0;

    tt_scheduler* scheduler = tt_scheduler_create(&sched_cfg, nullptr);
    ASSERT_NE(scheduler, nullptr);

    Click click{clickDefaults()};
    ASSERT_NE(click.handle, nullptr);

    tt_scheduler_start(scheduler, 1.0);

    std::vector<float> out(static_cast<std::size_t>(2.0 * 48000), 0.0f);
    tt_event events[32];

    for (std::size_t i = 0; i < out.size(); i += 256) {
        const std::size_t n = std::min<std::size_t>(256, out.size() - i);
        const double t = static_cast<double>(i) / 48000.0;

        const size_t count = tt_scheduler_pull(scheduler, t, events, 32, nullptr);
        for (size_t e = 0; e < count; ++e) {
            tt_click_schedule(click.handle, events[e].time_sec, events[e].kind);
        }
        tt_click_mix(click.handle, t, out.data() + i, n);
    }

    // Beats at 1.0 and 1.5, each landing on its own sample.
    for (double beat_sec : {1.0, 1.5}) {
        const auto at = static_cast<std::size_t>(beat_sec * 48000);
        EXPECT_NEAR(out[at], 0.0f, 1e-6f) << "beat at " << beat_sec << " starts from silence";
        EXPECT_GT(std::fabs(out[at + 1]), 1e-4f) << "beat at " << beat_sec;
        EXPECT_EQ(out[at - 1], 0.0f) << "nothing before the beat at " << beat_sec;
    }

    EXPECT_EQ(tt_click_dropped_late(click.handle), 0u);
    EXPECT_EQ(tt_click_dropped_overflow(click.handle), 0u);
    EXPECT_EQ(tt_click_discontinuities(click.handle), 0u);

    tt_scheduler_destroy(scheduler);
}

TEST(ClickApi, NullHandleIsHarmless) {
    float buffer[16] = {0.0f};

    EXPECT_EQ(tt_click_schedule(nullptr, 0.0, TT_BEAT_BEAT), 0);
    EXPECT_EQ(tt_click_pending(nullptr), 0u);
    EXPECT_EQ(tt_click_active_voices(nullptr), 0u);
    EXPECT_EQ(tt_click_dropped_late(nullptr), 0u);
    EXPECT_EQ(tt_click_dropped_overflow(nullptr), 0u);
    EXPECT_EQ(tt_click_stolen(nullptr), 0u);
    EXPECT_EQ(tt_click_discontinuities(nullptr), 0u);

    tt_click_mix(nullptr, 0.0, buffer, 16);
    tt_click_reset(nullptr);
    tt_click_destroy(nullptr);
    tt_click_config_defaults(nullptr, 48000.0);
}

/* ------------------------------------------------------------- grid cache -- */

TEST(GridApi, SavesAndRestoresAnAnalysisAcrossHandles) {
    const std::vector<float> audio = clickTrack(120.0, 10.0, kOfflineRate);

    Offline analysed{offlineDefaults()};
    ASSERT_NE(analysed.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(analysed.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(analysed.handle), TT_OK);

    const size_t size = tt_offline_grid_size(analysed.handle);
    ASSERT_GT(size, 0u);
    std::vector<unsigned char> blob(size);
    ASSERT_EQ(tt_offline_grid_serialize(analysed.handle, blob.data(), blob.size()), size);

    // A fresh handle with the same config, no audio ever fed — the cache-hit
    // path a shell takes on the second import of the same track.
    Offline restored{offlineDefaults()};
    ASSERT_NE(restored.handle, nullptr);
    ASSERT_EQ(tt_offline_grid_restore(restored.handle, blob.data(), blob.size()), TT_OK);

    EXPECT_EQ(readBeats(restored.handle), readBeats(analysed.handle));
    EXPECT_EQ(tt_offline_bpm(restored.handle), tt_offline_bpm(analysed.handle));
    EXPECT_EQ(tt_offline_confidence(restored.handle),
              tt_offline_confidence(analysed.handle));
    EXPECT_EQ(tt_offline_estimated_bpm(restored.handle),
              tt_offline_estimated_bpm(analysed.handle));
}

TEST(GridApi, NothingToSaveBeforeFinish) {
    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);

    unsigned char buffer[256];
    EXPECT_EQ(tt_offline_grid_size(offline.handle), 0u);
    EXPECT_EQ(tt_offline_grid_serialize(offline.handle, buffer, sizeof(buffer)), 0u);
}

TEST(GridApi, SerializeRefusesATooSmallBuffer) {
    const std::vector<float> audio = clickTrack(120.0, 10.0, kOfflineRate);

    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(offline.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(offline.handle), TT_OK);

    const size_t size = tt_offline_grid_size(offline.handle);
    ASSERT_GT(size, 1u);
    std::vector<unsigned char> blob(size - 1);
    EXPECT_EQ(tt_offline_grid_serialize(offline.handle, blob.data(), blob.size()), 0u);
}

TEST(GridApi, RestoreRefusesAGridFromAnotherConfig) {
    const std::vector<float> audio = clickTrack(120.0, 10.0, kOfflineRate);

    Offline analysed{offlineDefaults()};
    ASSERT_NE(analysed.handle, nullptr);
    ASSERT_EQ(tt_offline_feed(analysed.handle, audio.data(), audio.size()), TT_OK);
    ASSERT_EQ(tt_offline_finish(analysed.handle), TT_OK);

    std::vector<unsigned char> blob(tt_offline_grid_size(analysed.handle));
    ASSERT_EQ(tt_offline_grid_serialize(analysed.handle, blob.data(), blob.size()),
              blob.size());

    tt_offline_config hinted = offlineDefaults();
    hinted.bpm_hint = 120.0;
    Offline other{hinted};
    ASSERT_NE(other.handle, nullptr);

    EXPECT_EQ(tt_offline_grid_restore(other.handle, blob.data(), blob.size()),
              TT_ERR_UNSUPPORTED);
    // The refusal must leave the handle empty, not half-restored.
    EXPECT_EQ(tt_offline_beat_count(other.handle), 0u);
}

TEST(GridApi, RestoreRefusesGarbage) {
    Offline offline{offlineDefaults()};
    ASSERT_NE(offline.handle, nullptr);

    const unsigned char noise[64] = {0};
    EXPECT_EQ(tt_offline_grid_restore(offline.handle, noise, sizeof(noise)),
              TT_ERR_UNSUPPORTED);
    EXPECT_EQ(tt_offline_beat_count(offline.handle), 0u);
}

TEST(GridApi, KeyMatchesTheKnownSha256TestVector) {
    char key[TT_GRID_KEY_HEX + 1];
    ASSERT_EQ(tt_grid_key("abc", 3, key, sizeof(key)), TT_OK);
    EXPECT_EQ(std::string(key),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(GridApi, KeyRejectsATooSmallBufferInsteadOfTruncating) {
    char key[TT_GRID_KEY_HEX];  // one byte short of key + terminator
    EXPECT_EQ(tt_grid_key("abc", 3, key, sizeof(key)), TT_ERR_INVALID_ARG);
}

TEST(GridApi, NullHandleIsHarmless) {
    unsigned char buffer[64] = {0};
    char key[TT_GRID_KEY_HEX + 1];

    EXPECT_EQ(tt_offline_grid_size(nullptr), 0u);
    EXPECT_EQ(tt_offline_grid_serialize(nullptr, buffer, sizeof(buffer)), 0u);
    EXPECT_EQ(tt_offline_grid_restore(nullptr, buffer, sizeof(buffer)),
              TT_ERR_INVALID_ARG);
    EXPECT_EQ(tt_grid_key(nullptr, 3, key, sizeof(key)), TT_ERR_INVALID_ARG);
    EXPECT_EQ(tt_grid_key(nullptr, 0, key, sizeof(key)), TT_OK);
    EXPECT_EQ(tt_grid_key("abc", 3, nullptr, 65), TT_ERR_INVALID_ARG);
}
