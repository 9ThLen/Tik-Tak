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
//   dump_analysis <audio> [rate] --salience <file>
//                    [--salience-min-range <value>]
//                                            replace the built-in cues with a
//                                            per-beat salience read from a file
//
// The second form exists so the synthetic clips in research/tiktak/synth.py can
// be scored without inventing a file format between here and there — they are
// already float arrays in memory.
//
// --salience is the seam in analysis/downbeat.hpp made reachable from outside:
// one finite number per beat, whitespace-separated, `#` starts a comment. The
// values keep their original scale: the resolver will not turn an almost-flat
// model output into unit-variance evidence. The beat grid still comes from the
// core's own tracker, and the bar length and phase still come from the core's
// own resolver — only the per-beat scorer is swapped. That is exactly the
// substitution an ONNX model will make, which is what lets a model be *scored*
// through the shipping resolver before a line of it is ported: run once to get
// the beats, sample the model's activation at those beat times, run again with
// the file. The count must match the beat count exactly; a mismatch is an
// error, not an alignment guess. --salience-min-range supplies the evidence
// gate in that backend's own units; it is part of a backend's calibration and
// deliberately has no universal model-independent value.
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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Internal, on purpose: resolveMeter is the seam itself, and going through the
// public C API instead would mean growing that API for a research need. The
// parity tools set the precedent.
#include "analysis/downbeat.hpp"

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

// One value per beat, whitespace-separated, `#` to end of line. Text rather
// than binary because the writer is a numpy one-liner and the file is worth
// being able to look at when a result surprises.
bool readSalience(const char* path, std::vector<double>& out,
                  std::string& error) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) {
        error = "cannot open salience file";
        return false;
    }

    std::string text;
    char block[4096];
    std::size_t got;
    while ((got = std::fread(block, 1, sizeof(block), file)) > 0) {
        text.append(block, got);
    }
    std::fclose(file);

    std::size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '#') {
            while (i < text.size() && text[i] != '\n') ++i;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else {
            char* end = nullptr;
            const double value = std::strtod(text.c_str() + i, &end);
            const std::size_t consumed = static_cast<std::size_t>(end - (text.c_str() + i));
            if (consumed == 0) {
                error = "invalid salience token at byte " + std::to_string(i);
                return false;
            }
            if (!std::isfinite(value)) {
                error = "non-finite salience value " +
                        std::to_string(out.size() + 1) + " at byte " +
                        std::to_string(i);
                return false;
            }
            out.push_back(value);
            i += consumed;
        }
    }
    return true;
}

// JSON has no way to say "not a number", and a reader that meets NaN either
// throws or silently invents null. Analysis of silence legitimately produces
// none of these values, so they are reported as 0 with the empty beat list
// alongside saying why.
double finite(double value) {
    return std::isfinite(value) ? value : 0.0;
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

bool nonnegativeFinite(const char* text, double& value) {
    char* end = nullptr;
    value = std::strtod(text, &end);
    return end != text && *end == '\0' && value >= 0.0 && std::isfinite(value);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    std::string salience_path;
    double salience_min_range = 0.0;
    bool salience_min_range_given = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--salience") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--salience needs a file\n");
                return 2;
            }
            salience_path = argv[++i];
        } else if (std::strcmp(argv[i], "--salience-min-range") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--salience-min-range needs a value\n");
                return 2;
            }
            if (!nonnegativeFinite(argv[++i], salience_min_range)) {
                std::fprintf(stderr,
                             "--salience-min-range must be a finite, non-negative number\n");
                return 2;
            }
            salience_min_range_given = true;
        } else {
            positional.push_back(argv[i]);
        }
    }

    if (salience_min_range_given && salience_path.empty()) {
        std::fprintf(stderr, "--salience-min-range only applies with --salience\n");
        return 2;
    }

    if (positional.empty() || positional.size() > 2) {
        std::fprintf(stderr,
                     "usage: %s <song.mp3|song.wav|song.flac> "
                     "[--salience <file> [--salience-min-range <value>]]\n"
                     "       %s <clip.f32> <sample_rate> "
                     "[--salience <file> [--salience-min-range <value>]]\n",
                     argv[0], argv[0]);
        return 2;
    }

    const std::string path = positional[0];
    const bool raw = positional.size() == 2;

    std::vector<float> samples;
    double rate = 0.0;

    if (raw) {
        rate = std::atof(positional[1].c_str());
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

    int beats_per_bar = tt_offline_beats_per_bar(offline);
    double strength = tt_offline_downbeat_strength(offline);
    double phase_margin = tt_offline_downbeat_phase_margin(offline);
    double meter_margin = tt_offline_downbeat_meter_margin(offline);
    bool confident = tt_offline_downbeat_confident(offline) != 0;

    if (!salience_path.empty()) {
        std::vector<double> salience;
        std::string salience_error;
        if (!readSalience(salience_path.c_str(), salience, salience_error)) {
            std::fprintf(stderr, "%s: %s\n", salience_path.c_str(),
                         salience_error.c_str());
            tt_offline_destroy(offline);
            return 1;
        }
        if (salience.size() != beats.size()) {
            std::fprintf(stderr,
                         "%s holds %zu value(s) but the analysis found %zu beat(s) — "
                         "one number per beat, in beat order\n",
                         salience_path.c_str(), salience.size(), beats.size());
            tt_offline_destroy(offline);
            return 1;
        }
        // The C API offers no way to override the downbeat configuration, so
        // this research seam reaches the resolver directly. The range gate is
        // backend-specific and can be supplied explicitly; the margin values
        // remain in the same backend's units and must be calibrated with it
        // before downbeat_confident is a product claim.
        tiktak::analysis::DownbeatConfig db_config;
        if (salience_min_range_given) {
            db_config.min_salience_range = salience_min_range;
        }
        const tiktak::analysis::DownbeatResult resolved =
            tiktak::analysis::resolveMeter(salience, beats, db_config);
        downbeats = resolved.downbeats;
        beats_per_bar = resolved.beats_per_bar;
        strength = resolved.strength;
        phase_margin = resolved.phase_margin;
        meter_margin = resolved.meter_margin;
        confident = resolved.confident(db_config.min_phase_margin,
                                       db_config.min_meter_margin);
    }

    std::printf("{\n");
    std::printf("  \"path\": \"%s\",\n", escape(path).c_str());
    std::printf("  \"salience_source\": \"%s\",\n",
                salience_path.empty() ? "cues" : "file");
    std::printf("  \"sample_rate\": %.17g,\n", rate);
    std::printf("  \"duration_sec\": %.17g,\n", static_cast<double>(samples.size()) / rate);
    std::printf("  \"bpm\": %.17g,\n", finite(tt_offline_bpm(offline)));
    std::printf("  \"confidence\": %.17g,\n", finite(tt_offline_confidence(offline)));
    std::printf("  \"beats_per_bar\": %d,\n", beats_per_bar);
    std::printf("  \"downbeat_strength\": %.17g,\n", finite(strength));
    std::printf("  \"downbeat_phase_margin\": %.17g,\n", finite(phase_margin));
    std::printf("  \"downbeat_meter_margin\": %.17g,\n", finite(meter_margin));
    std::printf("  \"downbeat_confident\": %s,\n", confident ? "true" : "false");
    printTimes("beats", beats, false);
    printTimes("downbeats", downbeats, true);
    std::printf("}\n");

    tt_offline_destroy(offline);
    return 0;
}
