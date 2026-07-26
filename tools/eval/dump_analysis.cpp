// Dumps one file's offline analysis as JSON, for research/eval to score.
//
// Why a tool and not a Python port: the thing being measured has to be the
// thing that ships. A reimplementation of the downbeat scorer in Python would
// produce numbers about the reimplementation, and the first time the two drifted
// the evaluation would be measuring the wrong program without saying so. This
// runs the same core the app runs and prints what it concluded.
//
//   dump_analysis <song.mp3>                 decode WAV, FLAC or MP3
//   dump_analysis <clip.f32> <sample_rate>   raw 32-bit float mono, native order
//
// The second form exists so the synthetic clips in research/tiktak/synth.py can
// be scored without inventing a file format between here and there — they are
// already float arrays in memory.
//
// Output is one JSON object on stdout. Times are printed at full double
// precision because this is a machine format: a diff between two runs should
// show a change in behaviour, never a change in rounding.
//
// The grid cache is deliberately not consulted or written. An evaluation that
// silently scored a blob analysed under an older configuration would be the
// worst possible failure here, and the run is cheap enough not to need it.
#include "tiktak/tiktak.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(TIKTAK_HAVE_DECODE)
#include "decode/decoder.hpp"
#endif

namespace {

std::vector<float> readRaw(const char* path) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return {};
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (bytes <= 0) {
        std::fclose(file);
        return {};
    }
    std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
    if (std::fread(samples.data(), sizeof(float), samples.size(), file) != samples.size()) {
        std::fclose(file);
        return {};
    }
    std::fclose(file);
    return samples;
}

// JSON has no way to say "not a number", and a reader that meets NaN either
// throws or silently invents null. Analysis of silence legitimately produces
// none of these values, so they are reported as 0 with the empty beat list
// alongside saying why.
double finite(double value) {
    return (value == value && value > -1e308 && value < 1e308) ? value : 0.0;
}

void printTimes(const char* name, const std::vector<double>& times, bool last) {
    std::printf("  \"%s\": [", name);
    for (std::size_t i = 0; i < times.size(); ++i) {
        std::printf("%s%.17g", i ? ", " : "", times[i]);
    }
    std::printf("]%s\n", last ? "" : ",");
}

// The bare minimum of JSON string escaping — enough for a file path, which is
// the only string this prints. A path with a quote or a backslash in it is
// ordinary on Windows and must not produce a broken document.
std::string escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr,
                     "usage: %s <song.mp3|song.wav|song.flac>\n"
                     "       %s <clip.f32> <sample_rate>\n",
                     argv[0], argv[0]);
        return 2;
    }

    const std::string path = argv[1];
    const bool raw = argc == 3;

    std::vector<float> samples;
    double rate = 0.0;

    if (raw) {
        rate = std::atof(argv[2]);
        if (rate <= 0.0) {
            std::fprintf(stderr, "bad sample rate\n");
            return 2;
        }
        samples = readRaw(path.c_str());
        if (samples.empty()) {
            std::fprintf(stderr, "cannot read %s as raw float32\n", path.c_str());
            return 1;
        }
    } else {
#if defined(TIKTAK_HAVE_DECODE)
        auto decoder = tiktak::decode::Decoder::open(path.c_str());
        if (!decoder) {
            std::fprintf(stderr, "%s is not a WAV, FLAC or MP3 file\n", path.c_str());
            return 1;
        }
        rate = decoder->info().sample_rate;
        samples.reserve(static_cast<std::size_t>(decoder->info().frames));
        std::vector<float> block(65536);
        for (;;) {
            const std::size_t got = decoder->readMono(block.data(), block.size());
            if (got == 0) break;
            samples.insert(samples.end(), block.begin(), block.begin() + got);
        }
        if (samples.empty()) {
            std::fprintf(stderr, "%s decoded to nothing\n", path.c_str());
            return 1;
        }
#else
        std::fprintf(stderr,
                     "this build has no decoder — rebuild with -DTIKTAK_BUILD_DECODE=ON, "
                     "or pass a raw .f32 file and its sample rate\n");
        return 1;
#endif
    }

    tt_offline_config config;
    tt_offline_config_defaults(&config, rate);

    tt_status status = TT_OK;
    tt_offline* offline = tt_offline_create(&config, &status);
    if (!offline) {
        std::fprintf(stderr, "tt_offline_create failed: %s\n", tt_status_string(status));
        return 1;
    }

    // Fed in blocks that are not a multiple of the hop, for the same reason
    // dump_beats does it: a decoder hands over whatever size it likes and the
    // framing must not change the answer.
    constexpr std::size_t kBlock = 4099;
    for (std::size_t pos = 0; pos < samples.size(); pos += kBlock) {
        const std::size_t take = std::min(kBlock, samples.size() - pos);
        if (tt_offline_feed(offline, samples.data() + pos, take) != TT_OK) {
            std::fprintf(stderr, "tt_offline_feed failed\n");
            tt_offline_destroy(offline);
            return 1;
        }
    }
    if (tt_offline_finish(offline) != TT_OK) {
        std::fprintf(stderr, "tt_offline_finish failed\n");
        tt_offline_destroy(offline);
        return 1;
    }

    std::vector<double> beats(tt_offline_beat_count(offline));
    if (!beats.empty()) tt_offline_beats(offline, beats.data(), beats.size());

    std::vector<double> downbeats(tt_offline_downbeat_count(offline));
    if (!downbeats.empty()) tt_offline_downbeats(offline, downbeats.data(), downbeats.size());

    std::printf("{\n");
    std::printf("  \"path\": \"%s\",\n", escape(path).c_str());
    std::printf("  \"sample_rate\": %.17g,\n", rate);
    std::printf("  \"duration_sec\": %.17g,\n", static_cast<double>(samples.size()) / rate);
    std::printf("  \"bpm\": %.17g,\n", finite(tt_offline_bpm(offline)));
    std::printf("  \"confidence\": %.17g,\n", finite(tt_offline_confidence(offline)));
    std::printf("  \"beats_per_bar\": %d,\n", tt_offline_beats_per_bar(offline));
    std::printf("  \"downbeat_strength\": %.17g,\n",
                finite(tt_offline_downbeat_strength(offline)));
    std::printf("  \"downbeat_phase_margin\": %.17g,\n",
                finite(tt_offline_downbeat_phase_margin(offline)));
    std::printf("  \"downbeat_meter_margin\": %.17g,\n",
                finite(tt_offline_downbeat_meter_margin(offline)));
    std::printf("  \"downbeat_confident\": %s,\n",
                tt_offline_downbeat_confident(offline) ? "true" : "false");
    printTimes("beats", beats, false);
    printTimes("downbeats", downbeats, true);
    std::printf("}\n");

    tt_offline_destroy(offline);
    return 0;
}
