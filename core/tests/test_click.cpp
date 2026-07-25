#include "render/click.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using tiktak::render::ClickConfig;
using tiktak::render::ClickRenderer;
using tiktak::render::ClickTone;
using tiktak::schedule::BeatKind;

namespace {

constexpr double kRate = 48000.0;

ClickConfig config(double sample_rate = kRate) {
    ClickConfig cfg;
    cfg.sample_rate = sample_rate;
    return cfg;
}

// The sample a click started on, or -1 if nothing sounded.
//
// A click's own first sample is exactly zero — it starts from silence so as not
// to put a step into the signal — so the first sample above the noise floor is
// the one after it. The threshold sits well above the noise of the mix and well
// below a click's peak, so this does not depend on the envelope's exact shape.
std::ptrdiff_t onsetSample(const std::vector<float>& buffer, float threshold = 1e-4f) {
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        if (std::fabs(buffer[i]) > threshold) return static_cast<std::ptrdiff_t>(i) - 1;
    }
    return -1;
}

double peak(const std::vector<float>& buffer) {
    double m = 0.0;
    for (float v : buffer) m = std::max(m, std::fabs(static_cast<double>(v)));
    return m;
}

// Drives the renderer the way an audio device would: fixed-size buffers, back
// to back, starting at `start_sec`.
std::vector<float> render(ClickRenderer& renderer, double start_sec, double duration_sec,
                          std::size_t block = 256, double sample_rate = kRate) {
    const auto total = static_cast<std::size_t>(duration_sec * sample_rate);
    std::vector<float> out(total, 0.0f);

    for (std::size_t i = 0; i < total; i += block) {
        const std::size_t n = std::min(block, total - i);
        renderer.mix(start_sec + static_cast<double>(i) / sample_rate, out.data() + i, n);
    }
    return out;
}

}  // namespace

TEST(Click, PlacesAClickOnTheSampleItWasScheduledFor) {
    ClickRenderer renderer(config());

    // Deliberately not a multiple of the block size: the click must land on its
    // own sample, not at the start of the buffer that happens to contain it.
    const double at = 0.5 + 137.0 / kRate;
    renderer.schedule(at, BeatKind::Beat);

    const std::vector<float> out = render(renderer, 0.5, 0.2);

    EXPECT_EQ(onsetSample(out), 137);
}

TEST(Click, PlacementDoesNotDependOnTheBlockSize) {
    // The device picks the block size and changes it between runs; if that
    // moved the click, every timing measurement taken on the harness would be
    // measuring the device's buffering instead of the metronome.
    std::vector<float> reference;

    for (std::size_t block : {32u, 64u, 128u, 480u, 1024u}) {
        ClickRenderer renderer(config());
        renderer.schedule(0.011, BeatKind::Downbeat);
        renderer.schedule(0.0517, BeatKind::Beat);

        const std::vector<float> out = render(renderer, 0.0, 0.15, block);

        if (reference.empty()) {
            reference = out;
            continue;
        }
        ASSERT_EQ(out.size(), reference.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_FLOAT_EQ(out[i], reference[i]) << "block " << block << ", sample " << i;
        }
    }
}

TEST(Click, ContinuesAClickThatStartedInThePreviousBuffer) {
    ClickConfig cfg = config();
    cfg.beat.length_sec = 0.05;   // 2400 samples, far longer than a block
    ClickRenderer renderer(cfg);

    // Ten samples before the end of the first 256-sample block.
    renderer.schedule(246.0 / kRate, BeatKind::Beat);

    const std::vector<float> out = render(renderer, 0.0, 0.1, 256);

    // The tail must be continuous across the boundary, not restarted or cut.
    EXPECT_GT(std::fabs(out[255]), 0.0f);
    EXPECT_GT(std::fabs(out[256]), 0.0f);
    EXPECT_GT(std::fabs(out[1000]), 0.0f);

    // And it must stop where the tone says it does, not run on.
    const auto ends = static_cast<std::size_t>(246 + 0.05 * kRate);
    for (std::size_t i = ends + 1; i < out.size(); ++i) {
        ASSERT_EQ(out[i], 0.0f) << "sample " << i;
    }
}

TEST(Click, MixesIntoTheBufferInsteadOfClearingIt) {
    // Phase 4 plays the click over the backing track. A renderer that filled
    // rather than mixed would silently delete the music.
    ClickRenderer renderer(config());
    renderer.schedule(0.01, BeatKind::Beat);

    std::vector<float> out(4800, 0.25f);
    renderer.mix(0.0, out.data(), out.size());

    // Untouched before the click, and still carrying the backing signal after.
    EXPECT_FLOAT_EQ(out[0], 0.25f);
    EXPECT_NE(out[500], 0.25f);
    EXPECT_GT(out[4799], 0.0f);
}

TEST(Click, TellsTheDownbeatApartFromTheOtherBeats) {
    ClickConfig cfg = config();
    ClickRenderer renderer(cfg);

    renderer.schedule(0.0, BeatKind::Downbeat);
    const std::vector<float> down = render(renderer, 0.0, 0.2);

    renderer.reset();
    renderer.schedule(0.0, BeatKind::Beat);
    const std::vector<float> beat = render(renderer, 0.0, 0.2);

    renderer.reset();
    renderer.schedule(0.0, BeatKind::Subdivision);
    const std::vector<float> sub = render(renderer, 0.0, 0.2);

    // Loud to quiet, so the bar reads without the player having to concentrate.
    EXPECT_GT(peak(down), peak(beat));
    EXPECT_GT(peak(beat), peak(sub));

    // And each is a different length, which is the other half of telling them
    // apart when the volume is low.
    EXPECT_NE(onsetSample(down, 1e-3f), -1);
}

TEST(Click, StartsFromSilenceRatherThanWithAStep) {
    // A click that begins at full amplitude puts a discontinuity into the
    // signal, heard as a thump underneath it.
    ClickRenderer renderer(config());
    renderer.schedule(0.0, BeatKind::Downbeat);

    std::vector<float> out(2048, 0.0f);
    renderer.mix(0.0, out.data(), out.size());

    EXPECT_NEAR(out[0], 0.0f, 1e-6f);
}

TEST(Click, DecaysToSilenceByTheEndOfItsLength) {
    ClickConfig cfg = config();
    cfg.beat.length_sec = 0.04;
    cfg.beat.gain = 1.0;
    ClickRenderer renderer(cfg);
    renderer.schedule(0.0, BeatKind::Beat);

    const auto samples = static_cast<std::size_t>(0.04 * kRate);
    std::vector<float> out(samples + 100, 0.0f);
    renderer.mix(0.0, out.data(), out.size());

    // The cut at the end is a thousandth of the peak: inaudible under a click,
    // which is what lets the voice simply stop instead of fading forever.
    double tail = 0.0;
    for (std::size_t i = samples - 20; i < samples; ++i) {
        tail = std::max(tail, std::fabs(static_cast<double>(out[i])));
    }
    EXPECT_LT(tail, 2e-3);
    EXPECT_EQ(renderer.active_voice_count(), 0u);
}

TEST(Click, HoldsItsPitch) {
    // The oscillator is a rotating vector rather than a sine call per sample.
    // That is only acceptable if it has not drifted by the end of a click.
    ClickConfig cfg = config();
    cfg.beat.frequency_hz = 1000.0;
    cfg.beat.length_sec = 0.2;   // far longer than a real click, to expose drift
    ClickRenderer renderer(cfg);
    renderer.schedule(0.0, BeatKind::Beat);

    const auto samples = static_cast<std::size_t>(0.2 * kRate);
    std::vector<float> out(samples, 0.0f);
    renderer.mix(0.0, out.data(), out.size());

    // Count zero crossings over the middle of the click, where the envelope is
    // still well above the noise floor: 1 kHz means 2000 crossings a second.
    const std::size_t from = samples / 4;
    const std::size_t to = samples / 2;
    std::size_t crossings = 0;
    for (std::size_t i = from + 1; i < to; ++i) {
        if ((out[i - 1] < 0.0f) != (out[i] < 0.0f)) ++crossings;
    }
    const double seconds = static_cast<double>(to - from) / kRate;
    EXPECT_NEAR(static_cast<double>(crossings) / seconds, 2000.0, 20.0);
}

TEST(Click, OverlappingClicksAddUp) {
    ClickConfig cfg = config();
    cfg.beat.length_sec = 0.1;
    ClickRenderer renderer(cfg);

    renderer.schedule(0.0, BeatKind::Beat);
    renderer.schedule(0.01, BeatKind::Beat);
    renderer.schedule(0.02, BeatKind::Beat);

    // Half the click length, so all three are still sounding at the end of it.
    std::vector<float> out(2400, 0.0f);
    renderer.mix(0.0, out.data(), out.size());

    EXPECT_EQ(renderer.active_voice_count(), 3u);
    EXPECT_EQ(renderer.stolen(), 0u);
}

TEST(Click, StealsAVoiceInsteadOfAllocatingWhenTheyRunOut) {
    ClickConfig cfg = config();
    cfg.max_voices = 2;
    cfg.beat.length_sec = 0.5;   // long enough that they all overlap
    ClickRenderer renderer(cfg);

    for (int i = 0; i < 5; ++i) {
        renderer.schedule(0.001 * i, BeatKind::Beat);
    }

    std::vector<float> out(4800, 0.0f);
    renderer.mix(0.0, out.data(), out.size());

    EXPECT_EQ(renderer.active_voice_count(), 2u);
    EXPECT_EQ(renderer.stolen(), 3u);
    // Still sounding: stealing degrades the sound, it does not stop it.
    EXPECT_GT(peak(out), 0.1);
}

TEST(Click, NudgesASlightlyLateClickInsteadOfDroppingIt) {
    ClickConfig cfg = config();
    cfg.late_tolerance_sec = 0.002;
    ClickRenderer renderer(cfg);

    // Scheduled half a millisecond before the buffer it is handed to: the host
    // polled a hair late. Half a millisecond of error is inaudible; a missing
    // beat is not.
    renderer.schedule(0.9995, BeatKind::Beat);

    std::vector<float> out(4800, 0.0f);
    renderer.mix(1.0, out.data(), out.size());

    EXPECT_EQ(onsetSample(out), 0);
    EXPECT_EQ(renderer.dropped_late(), 0u);
}

TEST(Click, DropsAClickThatIsTrulyLateRatherThanPlayingItWrong) {
    ClickConfig cfg = config();
    cfg.late_tolerance_sec = 0.002;
    ClickRenderer renderer(cfg);

    renderer.schedule(0.9, BeatKind::Beat);   // 100 ms late: grossly audible

    std::vector<float> out(4800, 0.0f);
    renderer.mix(1.0, out.data(), out.size());

    EXPECT_EQ(peak(out), 0.0);
    EXPECT_EQ(renderer.dropped_late(), 1u);
    EXPECT_EQ(renderer.pending_count(), 0u);
}

TEST(Click, KeepsAClickQueuedUntilItsBufferArrives) {
    ClickRenderer renderer(config());
    renderer.schedule(1.0, BeatKind::Beat);

    std::vector<float> out(480, 0.0f);
    renderer.mix(0.0, out.data(), out.size());

    EXPECT_EQ(peak(out), 0.0);
    EXPECT_EQ(renderer.pending_count(), 1u);
    EXPECT_EQ(renderer.dropped_late(), 0u);
}

TEST(Click, RefusesMoreThanItCanHoldInsteadOfGrowing) {
    ClickConfig cfg = config();
    cfg.max_pending = 4;
    ClickRenderer renderer(cfg);

    for (int i = 0; i < 4; ++i) EXPECT_TRUE(renderer.schedule(10.0 + i, BeatKind::Beat));
    EXPECT_FALSE(renderer.schedule(20.0, BeatKind::Beat));
    EXPECT_EQ(renderer.dropped_overflow(), 1u);
    EXPECT_EQ(renderer.pending_count(), 4u);
}

TEST(Click, RefusesANaNTimeRatherThanQueueingSomethingThatNeverLeaves) {
    ClickRenderer renderer(config());
    EXPECT_FALSE(renderer.schedule(std::nan(""), BeatKind::Beat));
    EXPECT_EQ(renderer.pending_count(), 0u);
}

TEST(Click, CountsBuffersThatDoNotFollowOnFromTheLast) {
    // A device that drops or repeats a buffer breaks the assumption a sounding
    // click makes about time being continuous. Counting it is what lets the
    // desktop harness report a glitch instead of quietly mistiming.
    ClickRenderer renderer(config());

    std::vector<float> out(480, 0.0f);
    renderer.mix(0.0, out.data(), out.size());
    renderer.mix(0.01, out.data(), out.size());      // contiguous
    EXPECT_EQ(renderer.discontinuities(), 0u);

    renderer.mix(0.05, out.data(), out.size());      // a gap
    EXPECT_EQ(renderer.discontinuities(), 1u);
}

TEST(Click, ResetSilencesEverythingSoundingAndQueued) {
    ClickConfig cfg = config();
    cfg.beat.length_sec = 0.5;
    ClickRenderer renderer(cfg);

    renderer.schedule(0.0, BeatKind::Beat);
    renderer.schedule(5.0, BeatKind::Beat);

    std::vector<float> out(480, 0.0f);
    renderer.mix(0.0, out.data(), out.size());
    ASSERT_EQ(renderer.active_voice_count(), 1u);

    renderer.reset();
    EXPECT_EQ(renderer.active_voice_count(), 0u);
    EXPECT_EQ(renderer.pending_count(), 0u);

    std::fill(out.begin(), out.end(), 0.0f);
    renderer.mix(0.01, out.data(), out.size());
    EXPECT_EQ(peak(out), 0.0);
}

TEST(ClickConfig, RejectsATonePastNyquist) {
    // Past Nyquist a tone does not become a higher click, it reappears as a
    // lower one at a mirrored frequency. Refusing it is what stops a config
    // that sounds right at 48 kHz from sounding wrong at 8.
    ClickConfig cfg = config(8000.0);
    ASSERT_TRUE(cfg.valid()) << "the default tones fit under 4 kHz";

    cfg.downbeat.frequency_hz = 5000.0;
    EXPECT_FALSE(cfg.valid());

    cfg.downbeat.frequency_hz = 3000.0;
    EXPECT_TRUE(cfg.valid());
}

TEST(ClickConfig, DefaultsAreUsable) {
    EXPECT_TRUE(config().valid());
    EXPECT_TRUE(config(44100.0).valid());
}

TEST(Click, WorksAtEveryRateADeviceMightPick) {
    for (double rate : {44100.0, 48000.0, 96000.0}) {
        ClickConfig cfg = config(rate);
        ClickRenderer renderer(cfg);
        renderer.schedule(0.02, BeatKind::Downbeat);

        const std::vector<float> out = render(renderer, 0.0, 0.2, 256, rate);

        EXPECT_EQ(onsetSample(out), static_cast<std::ptrdiff_t>(0.02 * rate))
            << "rate " << rate;
        EXPECT_GT(peak(out), 0.1) << "rate " << rate;
    }
}

TEST(Click, PutsAClickInTheLastHalfSampleIntoTheNextBuffer) {
    // Regression. A click whose time falls inside a buffer can still round to
    // the *next* buffer's first sample, and forcing it into the current one put
    // it a whole sample early. It only shows at tempos that do not divide the
    // sample rate, which is most of them.
    ClickRenderer renderer(config());

    // 0.3 samples past the end of a 256-sample buffer's last sample: inside the
    // buffer by time, but nearest to sample 256, which the buffer does not have.
    renderer.schedule(255.7 / kRate, BeatKind::Beat);

    std::vector<float> first(256, 0.0f);
    renderer.mix(0.0, first.data(), first.size());
    EXPECT_EQ(peak(first), 0.0) << "nothing belongs in this buffer";
    EXPECT_EQ(renderer.pending_count(), 1u);
    EXPECT_EQ(renderer.dropped_late(), 0u);

    std::vector<float> second(256, 0.0f);
    renderer.mix(256.0 / kRate, second.data(), second.size());
    EXPECT_EQ(onsetSample(second), 0);
}
