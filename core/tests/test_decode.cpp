#include "tiktak/tiktak.h"
#include "tiktak/tiktak_decode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "support.hpp"

namespace {

constexpr double kToneHz = 440.0;
constexpr unsigned kRate = 22050;
constexpr unsigned long long kFrames = 11025;   // half a second

std::string dataPath(const char* name) {
    return std::string(TIKTAK_TEST_DATA_DIR) + "/" + name;
}

std::vector<unsigned char> readFile(const char* name) {
    const std::string path = dataPath(name);
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return {};

    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    std::vector<unsigned char> bytes(static_cast<std::size_t>(std::max(0L, size)));
    if (!bytes.empty() && std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
        bytes.clear();
    }
    std::fclose(file);
    return bytes;
}

// RAII around the C handle so a failing assertion cannot leak it.
struct Decoder {
    explicit Decoder(const char* name) {
        handle = tt_decoder_open_file(dataPath(name).c_str(), &status);
    }
    Decoder(const void* data, std::size_t bytes) {
        handle = tt_decoder_open_memory(data, bytes, &status);
    }
    ~Decoder() { tt_decoder_close(handle); }
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    tt_decoder* handle = nullptr;
    tt_status status = TT_OK;
};

std::vector<float> decodeAll(tt_decoder* decoder, std::size_t chunk = 4096) {
    std::vector<float> all;
    std::vector<float> buffer(chunk);
    for (;;) {
        const std::size_t got = tt_decoder_read(decoder, buffer.data(), chunk);
        if (got == 0) break;
        all.insert(all.end(), buffer.begin(),
                   buffer.begin() + static_cast<std::ptrdiff_t>(got));
        if (got < chunk) break;
    }
    return all;
}

double rms(const std::vector<float>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (float v : values) sum += static_cast<double>(v) * v;
    return std::sqrt(sum / static_cast<double>(values.size()));
}

// Energy at `hz` relative to the total, by a single-bin Goertzel. Enough to say
// "this is still the tone we encoded" without caring about encoder delay.
double toneFraction(const std::vector<float>& values, double hz, double rate) {
    if (values.size() < 64) return 0.0;

    const double omega = 2.0 * M_PI * hz / rate;
    const double coeff = 2.0 * std::cos(omega);
    double s1 = 0.0;
    double s2 = 0.0;
    double energy = 0.0;
    for (float v : values) {
        const double s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
        energy += static_cast<double>(v) * v;
    }
    const double power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    if (energy <= 0.0) return 0.0;
    return power / (energy * static_cast<double>(values.size()) * 0.5);
}

}  // namespace

TEST(Sniff, IdentifiesEachSupportedFormat) {
    for (const auto& [name, expected] : {std::pair{"tone_mono.wav", TT_FORMAT_WAV},
                                         std::pair{"tone_mono.flac", TT_FORMAT_FLAC},
                                         std::pair{"tone_mono.mp3", TT_FORMAT_MP3}}) {
        const std::vector<unsigned char> bytes = readFile(name);
        ASSERT_FALSE(bytes.empty()) << name;
        EXPECT_EQ(tt_sniff_format(bytes.data(), bytes.size()), expected) << name;
        // A handful of bytes is enough; a caller should not have to read the
        // whole file to find out whether it can be opened.
        EXPECT_EQ(tt_sniff_format(bytes.data(), 32), expected) << name;
    }
}

// Refusing is the correct answer, not a failure to try harder: handed arbitrary
// bytes, an MP3 decoder produces noise rather than an error, and silently
// analysing noise is worse than reporting an unsupported file.
TEST(Sniff, RefusesWhatItDoesNotRecognise) {
    const char text[] = "this is not audio, it is a text file with words in it";
    EXPECT_EQ(tt_sniff_format(text, sizeof(text)), TT_FORMAT_UNKNOWN);

    const unsigned char zeros[64] = {};
    EXPECT_EQ(tt_sniff_format(zeros, sizeof(zeros)), TT_FORMAT_UNKNOWN);

    // Eleven set bits are the MP3 sync word, but they also occur constantly in
    // ordinary binary data, so the layer and bitrate fields have to agree too.
    const unsigned char false_sync[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    EXPECT_EQ(tt_sniff_format(false_sync, sizeof(false_sync)), TT_FORMAT_UNKNOWN);

    EXPECT_EQ(tt_sniff_format(nullptr, 16), TT_FORMAT_UNKNOWN);
    EXPECT_EQ(tt_sniff_format(text, 0), TT_FORMAT_UNKNOWN);
    EXPECT_EQ(tt_sniff_format(text, 2), TT_FORMAT_UNKNOWN);   // truncated
}

TEST(Decode, ReportsTheStreamsProperties) {
    Decoder decoder{"tone_mono.wav"};
    ASSERT_NE(decoder.handle, nullptr);
    EXPECT_EQ(decoder.status, TT_OK);

    const tt_audio_info info = tt_decoder_info(decoder.handle);
    EXPECT_DOUBLE_EQ(info.sample_rate, kRate);
    EXPECT_EQ(info.channels, 1u);
    EXPECT_EQ(info.frames, kFrames);
    EXPECT_NEAR(info.duration_sec, 0.5, 1e-9);
    EXPECT_EQ(info.format, TT_FORMAT_WAV);
    EXPECT_STREQ(tt_format_name(info.format), "wav");
}

// WAV and FLAC are lossless encodings of the same 16-bit source, so anything
// other than an exact match means one of the two decoders is wrong.
TEST(Decode, LosslessFormatsAgreeExactly) {
    Decoder wav{"tone_mono.wav"};
    Decoder flac{"tone_mono.flac"};
    ASSERT_NE(wav.handle, nullptr);
    ASSERT_NE(flac.handle, nullptr);

    const std::vector<float> from_wav = decodeAll(wav.handle);
    const std::vector<float> from_flac = decodeAll(flac.handle);

    ASSERT_EQ(from_wav.size(), kFrames);
    ASSERT_EQ(from_flac.size(), kFrames);
    for (std::size_t i = 0; i < from_wav.size(); ++i) {
        EXPECT_FLOAT_EQ(from_wav[i], from_flac[i]) << "at sample " << i;
    }
}

TEST(Decode, RecoversTheToneFromEachFormat) {
    for (const char* name : {"tone_mono.wav", "tone_mono.flac", "tone_mono.mp3"}) {
        Decoder decoder{name};
        ASSERT_NE(decoder.handle, nullptr) << name;

        const std::vector<float> samples = decodeAll(decoder.handle);
        ASSERT_GT(samples.size(), kFrames / 2) << name;

        // MP3 is lossy and carries encoder delay, so the comparison is on
        // content rather than sample by sample: is this still a 440 Hz tone at
        // the level we encoded?
        EXPECT_GT(toneFraction(samples, kToneHz, kRate), 0.9) << name;
        EXPECT_NEAR(rms(samples), 0.5 / std::sqrt(2.0), 0.05) << name;
    }
}

// Averaging, not summing. A sum would clip anything already near full scale,
// and the onset function cares about the shape of an attack, which clipping
// flattens. Silence in one channel is the cleanest way to see which was done.
TEST(Decode, DownmixesStereoByAveraging) {
    Decoder mono{"tone_mono.wav"};
    Decoder stereo{"tone_stereo.wav"};
    ASSERT_NE(mono.handle, nullptr);
    ASSERT_NE(stereo.handle, nullptr);

    EXPECT_EQ(tt_decoder_info(stereo.handle).channels, 2u);

    const std::vector<float> one = decodeAll(mono.handle);
    const std::vector<float> two = decodeAll(stereo.handle);

    ASSERT_EQ(one.size(), two.size());
    for (std::size_t i = 0; i < one.size(); ++i) {
        EXPECT_FLOAT_EQ(two[i], one[i] * 0.5f) << "at sample " << i;
    }
}

TEST(Decode, ReadSizeDoesNotChangeTheOutput) {
    Decoder whole{"tone_mono.flac"};
    Decoder dribbled{"tone_mono.flac"};
    ASSERT_NE(whole.handle, nullptr);
    ASSERT_NE(dribbled.handle, nullptr);

    const std::vector<float> big = decodeAll(whole.handle, 8192);
    // 173 is deliberately awkward: not a power of two, not a multiple of the
    // decoder's internal chunk, smaller than one FLAC block.
    const std::vector<float> small = decodeAll(dribbled.handle, 173);

    ASSERT_EQ(big.size(), small.size());
    for (std::size_t i = 0; i < big.size(); ++i) {
        EXPECT_FLOAT_EQ(big[i], small[i]) << "at sample " << i;
    }
}

TEST(Decode, MemoryAndFileGiveTheSameResult) {
    const std::vector<unsigned char> bytes = readFile("tone_mono.flac");
    ASSERT_FALSE(bytes.empty());

    Decoder from_file{"tone_mono.flac"};
    Decoder from_memory{bytes.data(), bytes.size()};
    ASSERT_NE(from_file.handle, nullptr);
    ASSERT_NE(from_memory.handle, nullptr);

    const std::vector<float> a = decodeAll(from_file.handle);
    const std::vector<float> b = decodeAll(from_memory.handle);

    ASSERT_EQ(a.size(), b.size());
    EXPECT_TRUE(std::equal(a.begin(), a.end(), b.begin()));
}

TEST(Decode, SeeksBackToTheStart) {
    Decoder decoder{"tone_mono.wav"};
    ASSERT_NE(decoder.handle, nullptr);

    const std::vector<float> first = decodeAll(decoder.handle);
    ASSERT_FALSE(first.empty());

    ASSERT_NE(tt_decoder_seek(decoder.handle, 0), 0);
    const std::vector<float> again = decodeAll(decoder.handle);

    ASSERT_EQ(again.size(), first.size());
    EXPECT_TRUE(std::equal(first.begin(), first.end(), again.begin()));
}

TEST(Decode, SeeksToAnOffset) {
    Decoder decoder{"tone_mono.wav"};
    ASSERT_NE(decoder.handle, nullptr);
    const std::vector<float> all = decodeAll(decoder.handle);

    constexpr unsigned long long kOffset = 1000;
    ASSERT_NE(tt_decoder_seek(decoder.handle, kOffset), 0);

    std::vector<float> tail(256);
    const std::size_t got = tt_decoder_read(decoder.handle, tail.data(), tail.size());
    ASSERT_EQ(got, tail.size());

    for (std::size_t i = 0; i < got; ++i) {
        EXPECT_FLOAT_EQ(tail[i], all[kOffset + i]) << "at sample " << i;
    }
}

TEST(Decode, RefusesUnsupportedInput) {
    const char text[] = "definitely not audio";
    tt_status status = TT_OK;

    EXPECT_EQ(tt_decoder_open_memory(text, sizeof(text), &status), nullptr);
    EXPECT_EQ(status, TT_ERR_UNSUPPORTED);

    EXPECT_EQ(tt_decoder_open_memory(nullptr, 16, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    EXPECT_EQ(tt_decoder_open_memory(text, 0, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    EXPECT_EQ(tt_decoder_open_file("/no/such/file.wav", &status), nullptr);
    EXPECT_EQ(status, TT_ERR_UNSUPPORTED);

    EXPECT_EQ(tt_decoder_open_file(nullptr, &status), nullptr);
    EXPECT_EQ(status, TT_ERR_INVALID_ARG);

    // A null status out-parameter must be as usable as a real one.
    EXPECT_EQ(tt_decoder_open_memory(text, sizeof(text), nullptr), nullptr);
}

// A truncated file is the common real-world corruption: an interrupted download
// or a copy that ran out of disk. It must not read past the end of the buffer,
// which is what the sanitizer build is checking here.
TEST(Decode, SurvivesTruncatedFiles) {
    for (const char* name : {"tone_mono.wav", "tone_mono.flac", "tone_mono.mp3"}) {
        const std::vector<unsigned char> bytes = readFile(name);
        ASSERT_FALSE(bytes.empty()) << name;

        for (double fraction : {0.1, 0.5, 0.9}) {
            const auto cut = static_cast<std::size_t>(
                static_cast<double>(bytes.size()) * fraction);
            Decoder decoder{bytes.data(), cut};
            if (decoder.handle == nullptr) continue;   // refusing is fine

            const std::vector<float> samples = decodeAll(decoder.handle);
            EXPECT_LE(samples.size(), kFrames + 4096) << name << " at " << fraction;
        }
    }
}

TEST(Decode, NullHandleIsHarmless) {
    float buffer[16];

    EXPECT_EQ(tt_decoder_read(nullptr, buffer, 16), 0u);
    EXPECT_EQ(tt_decoder_seek(nullptr, 0), 0);

    const tt_audio_info info = tt_decoder_info(nullptr);
    EXPECT_EQ(info.format, TT_FORMAT_UNKNOWN);
    EXPECT_DOUBLE_EQ(info.sample_rate, 0.0);

    tt_decoder_close(nullptr);

    Decoder decoder{"tone_mono.wav"};
    ASSERT_NE(decoder.handle, nullptr);
    EXPECT_EQ(tt_decoder_read(decoder.handle, nullptr, 16), 0u);
    EXPECT_EQ(tt_decoder_read(decoder.handle, buffer, 0), 0u);
}

/* ------------------------------------------------------------ end to end -- */

// What the whole file path exists to do: a real encoded file goes in, a beat
// grid comes out. Everything up to here tests a piece in isolation; this is the
// only test that would notice if two correct pieces were wired together wrongly
// — a sample rate not carried through, a mono downmix that halved the level
// below the analyser's threshold, frame times measured against the wrong clock.
//
// MP3 deliberately: it is what users import, and it is the format most likely
// to break onset detection, because its pre-echo smears energy backwards across
// exactly the transients the onset function looks for.
TEST(DecodeAndAnalyse, FindsTheBeatsOfAnEncodedClickTrack) {
    Decoder decoder{"click_120.mp3"};
    ASSERT_NE(decoder.handle, nullptr);

    const tt_audio_info info = tt_decoder_info(decoder.handle);
    ASSERT_GT(info.sample_rate, 0.0);
    EXPECT_NEAR(info.duration_sec, 10.0, 0.1);

    tt_offline_config config;
    tt_offline_config_defaults(&config, info.sample_rate);

    tt_status status = TT_OK;
    tt_offline* analysis = tt_offline_create(&config, &status);
    ASSERT_NE(analysis, nullptr) << tt_status_string(status);

    // Decode straight into the analyser, a block at a time — the shape the app
    // uses, so a long track never sits in memory as samples.
    std::vector<float> block(4096);
    for (;;) {
        const std::size_t got = tt_decoder_read(decoder.handle, block.data(), block.size());
        if (got == 0) break;
        ASSERT_EQ(tt_offline_feed(analysis, block.data(), got), TT_OK);
        if (got < block.size()) break;
    }
    ASSERT_EQ(tt_offline_finish(analysis), TT_OK);

    EXPECT_NEAR(tt_offline_bpm(analysis), 120.0, 4.0);
    EXPECT_GT(tt_offline_confidence(analysis), 0.3);

    std::vector<double> beats(tt_offline_beat_count(analysis));
    ASSERT_GE(beats.size(), 15u);
    tt_offline_beats(analysis, beats.data(), beats.size());

    // The clip has a beat every 0.5 s from zero. Each true beat must have an
    // estimated one within the standard 70 ms tolerance.
    std::size_t hits = 0;
    for (double expected = 0.0; expected < 10.0; expected += 0.5) {
        const bool matched = std::any_of(beats.begin(), beats.end(), [&](double beat) {
            return std::abs(beat - expected) <= 0.07;
        });
        if (matched) ++hits;
    }
    EXPECT_GE(hits, 18u) << "matched " << hits << " of 20 beats";

    tt_offline_destroy(analysis);
}

TEST(DecodeAndTrackLive, FollowsAnEncodedClickTrackThroughTheMicrophonePath) {
    // The same file the offline test analyses, driven through the *online*
    // tracker in capture-sized blocks against a virtual clock. The two paths
    // share only the ODF, so this is what would catch the live one being wired
    // up correctly to a front-end it reads wrongly.
    Decoder decoder{"click_120.mp3"};
    ASSERT_NE(decoder.handle, nullptr);

    const tt_audio_info info = tt_decoder_info(decoder.handle);
    ASSERT_GT(info.sample_rate, 0.0);

    tt_live_config config;
    tt_live_config_defaults(&config, info.sample_rate);

    tt_status status = TT_OK;
    tt_live* live = tt_live_create(&config, &status);
    ASSERT_NE(live, nullptr) << tt_status_string(status);

    constexpr std::size_t kBlock = 256;
    std::vector<float> block(kBlock);
    std::vector<double> beats;
    double time = 0.0;
    for (;;) {
        const std::size_t got = tt_decoder_read(decoder.handle, block.data(), block.size());
        if (got == 0) break;
        tt_live_process(live, time, block.data(), got);
        time += static_cast<double>(got) / info.sample_rate;

        double beat = 0.0;
        while (tt_live_take_beat(live, time, 0.05, &beat)) beats.push_back(beat);
        if (got < block.size()) break;
    }

    tt_live_estimate estimate;
    tt_live_estimate_get(live, time, &estimate);
    EXPECT_NEAR(estimate.bpm, 120.0, 6.0);
    EXPECT_GT(estimate.confidence, 0.3);

    // The clip has a beat every 0.5 s from zero, with eighth-note hits between
    // them — the case that tempts a tracker into reading the subdivision as the
    // beat. It takes this one about five of the clip's ten seconds to be sure,
    // which is why only a handful of beats are expected: what matters is that
    // every beat it does commit to lands on the real grid, within the standard
    // 70 ms tolerance. (The offline path is both faster and more accurate here,
    // and that is the point of having two — a file is never tracked live.)
    ASSERT_GE(beats.size(), 4u);
    for (std::size_t i = 0; i < beats.size(); ++i) {
        const double off = std::fabs(beats[i] - std::round(beats[i] / 0.5) * 0.5);
        EXPECT_LE(off, 0.07) << "beat " << i << " at " << beats[i];
    }

    tt_live_destroy(live);
}

TEST(DecodeAndSyncLive, FallsInWithARealFileAtATempoItIsToldAndRefusesOneItIsNot) {
    // Manual + sync on real audio, which is where the mode has to earn its
    // keep: the same file the tracker above takes half its length to be sure
    // of, with the tempo simply given.
    const tt_audio_info probe = [] {
        Decoder decoder{"click_120.mp3"};
        EXPECT_NE(decoder.handle, nullptr);
        return tt_decoder_info(decoder.handle);
    }();
    ASSERT_GT(probe.sample_rate, 0.0);

    const auto play = [&](double manual_bpm, std::vector<double>* beats) {
        Decoder decoder{"click_120.mp3"};
        EXPECT_NE(decoder.handle, nullptr);

        tt_live_config config;
        tt_live_config_defaults(&config, probe.sample_rate);
        tt_live* live = tt_live_create(&config, nullptr);
        EXPECT_NE(live, nullptr);
        tt_live_set_manual_tempo(live, manual_bpm);

        constexpr std::size_t kBlock = 256;
        std::vector<float> block(kBlock);
        double time = 0.0;
        for (;;) {
            const std::size_t got = tt_decoder_read(decoder.handle, block.data(), block.size());
            if (got == 0) break;
            tt_live_process(live, time, block.data(), got);
            time += static_cast<double>(got) / probe.sample_rate;

            double beat = 0.0;
            while (tt_live_take_beat(live, time, 0.05, &beat)) beats->push_back(beat);
            if (got < block.size()) break;
        }
        const int waiting = tt_live_waiting(live);
        tt_live_destroy(live);
        return waiting;
    };

    // Told 120, which the clip is: it falls in and every click lands on the
    // clip's own grid. Nothing had to be discovered but the offset.
    std::vector<double> beats;
    EXPECT_EQ(play(120.0, &beats), 0);
    ASSERT_GE(beats.size(), 10u);
    for (std::size_t i = 0; i < beats.size(); ++i) {
        const double off = std::fabs(beats[i] - std::round(beats[i] / 0.5) * 0.5);
        EXPECT_LE(off, 0.05) << "beat " << i << " at " << beats[i];
    }

    // Told 137, which it is not. There is no 137 phase in this room to find,
    // and refusing is the whole reason the mode can be trusted at 120: a
    // synchroniser that always synchronises has said nothing.
    std::vector<double> none;
    EXPECT_EQ(play(137.0, &none), 1);
    EXPECT_TRUE(none.empty());
}

// Phase 7 on a real encoded file: the bar lines, not just the beats.
//
// The clip was generated in four with the first bar starting at zero, and the
// pattern that says so — kick on the one, snare on the three — survives an MP3
// encode. Worth checking here rather than only on synthetic buffers, because
// the harmony cue is silent on this material and the meter has to come from
// the drums alone.
TEST(DecodeAndAnalyse, FindsTheBarLinesOfAnEncodedClickTrack) {
    Decoder decoder{"click_120.mp3"};
    ASSERT_NE(decoder.handle, nullptr);

    const tt_audio_info info = tt_decoder_info(decoder.handle);
    ASSERT_GT(info.sample_rate, 0.0);

    tt_offline_config config;
    tt_offline_config_defaults(&config, info.sample_rate);

    tt_status status = TT_OK;
    tt_offline* analysis = tt_offline_create(&config, &status);
    ASSERT_NE(analysis, nullptr) << tt_status_string(status);

    std::vector<float> block(4096);
    for (;;) {
        const std::size_t got = tt_decoder_read(decoder.handle, block.data(), block.size());
        if (got == 0) break;
        ASSERT_EQ(tt_offline_feed(analysis, block.data(), got), TT_OK);
        if (got < block.size()) break;
    }
    ASSERT_EQ(tt_offline_finish(analysis), TT_OK);

    EXPECT_EQ(tt_offline_beats_per_bar(analysis), 4);
    EXPECT_GT(tt_offline_downbeat_strength(analysis), 0.5);
    EXPECT_GT(tt_offline_downbeat_phase_margin(analysis), 0.25);

    const std::size_t count = tt_offline_downbeat_count(analysis);
    ASSERT_GE(count, 4u);
    std::vector<double> bars(count);
    ASSERT_EQ(tt_offline_downbeats(analysis, bars.data(), bars.size()), count);

    // At 120 BPM in four from zero, every bar line is on an even second.
    for (double t : bars) {
        const double n = t / 2.0;
        EXPECT_LT(std::abs(n - std::round(n)), 0.07) << "bar line at " << t;
    }

    tt_offline_destroy(analysis);
}
