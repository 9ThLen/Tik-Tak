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
//   dump_analysis <audio> [rate] --salience <file> [calibration]
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
// error, not an alignment guess.
//
// A backend's calibration is three numbers — --salience-min-range,
// --salience-min-phase-margin, --salience-min-meter-margin — and they are
// passed together or not at all. Since the resolver no longer rescales
// arbitrary backend output, all three live in that backend's own units and
// none of them has a universal model-independent value. Supplying a range gate
// while inheriting the cue backend's margins would report `downbeat_confident`
// judged by numbers belonging to a different scorer, so it is refused.
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
#include <iterator>
#include <string>
#include <vector>

// Internal, on purpose: resolveMeter is the seam itself, and going through the
// public C API instead would mean growing that API for a research need. The
// parity tools set the precedent.
#include "analysis/downbeat.hpp"
#include "dsp/resample.hpp"
#include "ml/beatnet.hpp"
#if TIKTAK_HAVE_ML
#include "ml/beat_this.hpp"
#include "ml/beat_this_session.hpp"
#endif
#include "analysis/offline.hpp"
#include "tracking/live.hpp"
#include "tracking/particle.hpp"

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
// The model's weights, as bytes. The core does no I/O and takes the blob.
bool readBytes(const char* path, std::vector<unsigned char>& out) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    out.resize(static_cast<std::size_t>(bytes));
    const bool ok = std::fread(out.data(), 1, out.size(), file) == out.size();
    std::fclose(file);
    return ok;
}

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
double finiteOrZero(double value) {
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

using tiktak::analysis::BeatFeature;

std::vector<double> cueColumn(const std::vector<BeatFeature>& features,
                              double BeatFeature::*member) {
    std::vector<double> out;
    out.reserve(features.size());
    for (const BeatFeature& f : features) out.push_back(f.*member);
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
    // A beat grid supplied from outside, so that a bad grid and a bad scorer
    // can be told apart. The bar-line stage takes the beats as given, and when
    // the tracker is an octave out every bar line is wrong for a reason that
    // has nothing to do with the downbeat cues. Replacing the grid isolates
    // the resolver; it is a research seam and never a product path.
    std::string beats_path;
    // The three numbers a backend calibrates as one set. Defaulted from the
    // built-in cue backend and overridden together — see the loop below for
    // why passing only some of them is refused.
    const tiktak::analysis::DownbeatConfig cue_defaults;
    double salience_min_range = cue_defaults.min_salience_range;
    double salience_min_phase = cue_defaults.min_phase_margin;
    double salience_min_meter = cue_defaults.min_meter_margin;
    int calibration_given = 0;
    // Fixes the tempo instead of estimating it, so an evaluation can separate
    // "the tempo hypothesis was wrong" from "the phase drifted at the right
    // tempo" — two failures that look identical in a beat grid and need
    // opposite fixes.
    double bpm_hint = 0.0;
    // Runs the causal microphone tracker over the file instead of the offline
    // analyser. Everything measured so far has been the offline path; the live
    // one ships in the microphone mode and had never been scored on real music
    // at all, which this exists to fix.
    bool live = false;
    // Seeds the causal tracker with the tempo the offline analyser found, which
    // is what the backing-track case can actually do: the app is playing the
    // file, so it has already analysed it and the microphone does not have to
    // rediscover the tempo from a dense mix.
    bool live_seed = false;
    // Prints the onset function itself, for experiments on the statistics the
    // live tracker derives from it — measuring a proposed confidence change in
    // Python first is a rebuild per idea cheaper.
    bool dump_odf = false;
    // Overrides for the live tracker's lock/release hysteresis, so a threshold
    // can be chosen on one batch and validated on another without a rebuild.
    double live_lock = 0.0;
    double live_release = 0.0;
    // Drives the particle filter from an activation computed elsewhere instead
    // of from the built-in onset function. The same seam as --salience on the
    // offline resolver, and for the same reason: the observation model is what
    // the live path's confidence was measured down to, and swapping it is the
    // only way to find out whether that diagnosis was right.
    std::string activation_path;
    double activation_fps = 50.0;
    // The same swap, but made by the core itself rather than handed to it: the
    // tracker computes the activation from the audio through its own front end.
    // --live-activation answers "would a better observation help"; this answers
    // "does the thing we would ship actually do it", which is a different
    // question and the one that matters now.
    std::string model_path;
    // The offline path's replacement for a beat tracker from 2007. Decodes,
    // resamples to the model's rate, runs the network and picks the peaks —
    // the same code an app would run, not a research approximation of it.
    std::string beat_this_path;
    // The tempo posterior's shape, so the octave choice can be swept over a real
    // annotated corpus without a rebuild per point. Zero means "leave the
    // shipped default alone", the same convention --live-lock already uses.
    // Deliberately not part of the calibration table below: those three are one
    // indivisible backend calibration, and these are independent knobs.
    double tempo_prior_centre = 0.0;
    double tempo_prior_width = 0.0;
    double tempo_comb_harmonics = 0.0;
    double tempo_comb_decay = 0.0;
    // The same two numbers for the causal path, which carries its own copy of
    // the belief in ParticleFilterConfig. They are separate flags rather than
    // one, because the two paths were calibrated apart and a sweep that moved
    // both at once could not say which one paid.
    double live_prior_centre = 0.0;
    double live_prior_width = 0.0;

    struct Threshold {
        const char* flag;
        double* target;
    };
    const Threshold thresholds[] = {
        {"--salience-min-range", &salience_min_range},
        {"--salience-min-phase-margin", &salience_min_phase},
        {"--salience-min-meter-margin", &salience_min_meter},
    };
    const Threshold tempo_knobs[] = {
        {"--tempo-prior-centre", &tempo_prior_centre},
        {"--tempo-prior-width", &tempo_prior_width},
        {"--tempo-comb-harmonics", &tempo_comb_harmonics},
        {"--tempo-comb-decay", &tempo_comb_decay},
        {"--live-prior-centre", &live_prior_centre},
        {"--live-prior-width", &live_prior_width},
    };

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--salience") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--salience needs a file\n");
                return 2;
            }
            salience_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--dump-odf") == 0) {
            dump_odf = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live") == 0) {
            live = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-activation") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-activation needs a file\n"); return 2; }
            activation_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--beat-this") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--beat-this needs a model\n"); return 2; }
            beat_this_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--live-model") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-model needs a file\n"); return 2; }
            model_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--activation-fps") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--activation-fps needs a value\n"); return 2; }
            activation_fps = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-lock") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-lock needs a value\n"); return 2; }
            live_lock = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-release") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-release needs a value\n"); return 2; }
            live_release = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-seeded") == 0) {
            live = true;
            live_seed = true;
            continue;
        }
        if (std::strcmp(argv[i], "--bpm") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--bpm needs a value\n");
                return 2;
            }
            bpm_hint = std::atof(argv[++i]);
            if (!(bpm_hint > 0.0)) {
                std::fprintf(stderr, "--bpm needs a positive value\n");
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--beats") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--beats needs a file\n");
                return 2;
            }
            beats_path = argv[++i];
            continue;
        }

        bool tempo_matched = false;
        for (const Threshold& knob : tempo_knobs) {
            if (std::strcmp(argv[i], knob.flag) != 0) continue;
            tempo_matched = true;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", knob.flag);
                return 2;
            }
            if (!nonnegativeFinite(argv[++i], *knob.target)) {
                std::fprintf(stderr, "%s must be a finite, non-negative number\n",
                             knob.flag);
                return 2;
            }
            break;
        }
        if (tempo_matched) continue;

        bool matched = false;
        for (const Threshold& threshold : thresholds) {
            if (std::strcmp(argv[i], threshold.flag) != 0) continue;
            matched = true;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", threshold.flag);
                return 2;
            }
            if (!nonnegativeFinite(argv[++i], *threshold.target)) {
                std::fprintf(stderr,
                             "%s must be a finite, non-negative number\n",
                             threshold.flag);
                return 2;
            }
            ++calibration_given;
            break;
        }
        if (!matched) positional.push_back(argv[i]);
    }

    const int kCalibrationSize = static_cast<int>(std::size(thresholds));

    // A foreign grid without a foreign scorer would be a lie: the built-in cues
    // are gathered from ODF frames against the tracker's own beats and cannot
    // be recomputed here for someone else's grid, so the metre and phase would
    // still be the original analysis's while the beats printed beside them came
    // from the file.
    if (!beats_path.empty() && salience_path.empty()) {
        std::fprintf(stderr, "--beats replaces the grid the scorer runs on, so it "
                             "needs --salience for that grid too\n");
        return 2;
    }

    if (calibration_given > 0 && salience_path.empty()) {
        std::fprintf(stderr,
                     "the calibration flags only apply with --salience\n");
        return 2;
    }
    // All three or none. They are one calibration: a backend that supplied a
    // range gate in its own units while silently inheriting the cue backend's
    // margins would be judged confident by numbers belonging to a different
    // scorer, and `downbeat_confident` would be a claim about nothing. Half a
    // calibration is the failure this refuses to let anyone make quietly.
    if (calibration_given > 0 && calibration_given < kCalibrationSize) {
        std::fprintf(stderr,
                     "a backend calibration is all %d of --salience-min-range, "
                     "--salience-min-phase-margin and --salience-min-meter-margin, "
                     "or none of them — %d given\n",
                     kCalibrationSize, calibration_given);
        return 2;
    }

    if (positional.empty() || positional.size() > 2) {
        std::fprintf(stderr,
                     "usage: %s <song.mp3|song.wav|song.flac> [--salience <file>"
                     " [calibration]]\n"
                     "       %s <clip.f32> <sample_rate> [--salience <file>"
                     " [calibration]]\n"
                     "  --bpm <value>      track at this tempo instead of estimating it\n"
                     "  --live             use the causal microphone tracker, not the offline one\n"
                     "  --live-seeded      the same, seeded with the offline tempo\n"
                     "  --live-activation <file> [--activation-fps N]\n"
                     "                     drive the particle filter from this activation\n"
                     "  calibration: --salience-min-range <v> "
                     "--salience-min-phase-margin <v> "
                     "--salience-min-meter-margin <v>\n"
                     "               (all three together, in the backend's own units)\n",
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

    // The C++ analyser rather than the C API, so the per-beat cues and the
    // runner-up tempos are reachable. Growing the product API to expose either
    // would be paying for a research need in a header that ships; the parity
    // tools set this precedent already.
    tiktak::analysis::OfflineConfig config;
    config.odf.sampleRate = rate;
    // The same clamp tt_odf_config_defaults applies. Without it a file below
    // 32 kHz gets mel bands above its own Nyquist and a different onset
    // function from the one the app would compute — the tool would be
    // measuring a configuration that never ships.
    config.odf.melMaxHz = std::min(16000.0, rate * 0.5);
    config.find_downbeats = true;
    config.bpm_hint = bpm_hint;
    if (tempo_prior_centre > 0.0) config.tempo.prior_centre_bpm = tempo_prior_centre;
    if (tempo_prior_width > 0.0) config.tempo.prior_width_octaves = tempo_prior_width;
    if (tempo_comb_harmonics > 0.0) {
        config.tempo.comb_harmonics = static_cast<int>(tempo_comb_harmonics);
    }
    if (tempo_comb_decay > 0.0) config.tempo.comb_weight_decay = tempo_comb_decay;
    tiktak::analysis::OfflineAnalyzer analyzer(config);

    // Fed in blocks that are not a multiple of the hop, for the same reason
    // dump_beats does it: a decoder hands over whatever size it likes and the
    // framing must not change the answer.
    constexpr std::size_t kBlock = 4099;
    for (std::size_t pos = 0; pos < samples.size(); pos += kBlock) {
        const std::size_t take = std::min(kBlock, samples.size() - pos);
        analyzer.feed(samples.data() + pos, take);
    }
    const tiktak::analysis::OfflineResult analysis = analyzer.finish();
    std::vector<double> beats = analysis.beats;

    // The causal path, driven over the same file. Fed in the same odd-sized
    // blocks, and read the way the shell reads it: ask for the next beat as
    // soon as it comes within the lookahead, and never revise one already
    // handed out. Scoring anything else would be scoring a tracker that does
    // not exist.
    std::vector<double> live_beats;
    double live_confidence = 0.0;
    tiktak::tracking::LiveTracker::Stats live_stats;
    std::vector<double> live_share, live_agreement, live_coincidence;

    // The whole live path, driven by an activation from outside instead of by
    // the built-in onset function. The same tracker, the same hysteresis and
    // the same publishing rules — only the evidence differs, so the grids that
    // come out are comparable with --live's.
    std::vector<double> activation;
    if (!activation_path.empty()) {
        std::string complaint;
        if (!readSalience(activation_path.c_str(), activation, complaint)) {
            std::fprintf(stderr, "%s\n", complaint.c_str());
            return 1;
        }
        live = true;
    }
    std::vector<double> model_downbeats;
    tiktak::ml::BeatNetWeights model_weights;
    if (!model_path.empty()) {
        std::vector<unsigned char> blob;
        if (!readBytes(model_path.c_str(), blob)) {
            std::fprintf(stderr, "cannot read %s\n", model_path.c_str());
            return 1;
        }
        if (!model_weights.load(blob.data(), blob.size())) {
            std::fprintf(stderr, "%s is not a weight file this build can run\n",
                         model_path.c_str());
            return 1;
        }
        live = true;
    }
    if (live) {
        tiktak::tracking::LiveConfig live_config;
        live_config.odf = config.odf;
        if (live_lock > 0.0) live_config.lock_confidence = live_lock;
        if (live_release > 0.0) live_config.release_confidence = live_release;
        if (live_prior_centre > 0.0) {
            live_config.filter.prior_centre_bpm = live_prior_centre;
        }
        if (live_prior_width > 0.0) {
            live_config.filter.prior_width_octaves = live_prior_width;
        }
        tiktak::tracking::LiveTracker tracker =
            model_weights.valid()
                ? tiktak::tracking::LiveTracker(live_config, model_weights)
                : tiktak::tracking::LiveTracker(live_config);
        if (live_seed && analysis.bpm > 0.0) {
            tracker.seedTempo(analysis.bpm);
        }

        // A device-sized buffer, not the odd block above. takeBeat only hands
        // over a beat once it is within the lookahead of now, so the polling
        // rate is part of the algorithm: poll every 4099 samples and most
        // beats fall between checks and are simply never played. A real shell
        // polls once per audio callback, and scoring anything slower would be
        // measuring the harness rather than the tracker.
        constexpr std::size_t kLiveBlock = 512;
        constexpr double kLookahead = 0.05;
        double now = 0.0;
        double next_sample = 1.0;
        const auto poll = [&]() {
            double beat = 0.0;
            while (tracker.takeBeat(now, kLookahead, &beat)) live_beats.push_back(beat);
            if (now >= next_sample) {
                // Once a second: which of confidence's three factors is low is
                // the diagnosis, and the product alone cannot say.
                const tiktak::tracking::BeatEstimate e = tracker.estimate(now);
                live_share.push_back(e.cluster_share);
                live_agreement.push_back(e.phase_agreement);
                live_coincidence.push_back(e.onset_coincidence);
                next_sample += 1.0;
            }
        };

        if (!activation.empty()) {
            // Driven by a block clock, not by the frame index, so that this
            // path and the audio path below ask takeBeat the same question at
            // the same moments and differ only in where the evidence came from.
            //
            // The first version of this loop polled once per activation frame
            // and let `now` jump a frame at a time. That is a different
            // cadence from a device's, and takeBeat is sensitive to it: while
            // coasting it hands out one beat per call until it catches up with
            // now + lookahead, so a coarser clock flushes out several beats
            // where a device would have taken one. It inflated the beat count
            // roughly threefold — visible afterwards as beats 50 ms apart,
            // which is the lookahead, not a tempo — without much changing
            // which beats were right. The accuracy measured through it stands;
            // the count of beats emitted through it did not.
            const double frame_sec = 1.0 / activation_fps;
            const double block_sec = static_cast<double>(kLiveBlock) / rate;
            const double last_sec = static_cast<double>(activation.size()) * frame_sec;
            std::size_t next_frame = 0;
            for (double clock = 0.0; clock < last_sec; clock += block_sec) {
                while (next_frame < activation.size() &&
                       static_cast<double>(next_frame) * frame_sec <= clock) {
                    tracker.observe(static_cast<double>(next_frame) * frame_sec,
                                    activation[next_frame]);
                    ++next_frame;
                }
                now = clock;
                poll();
            }
        } else {
            for (std::size_t pos = 0; pos < samples.size(); pos += kLiveBlock) {
                const std::size_t take = std::min(kLiveBlock, samples.size() - pos);
                tracker.process(now, samples.data() + pos, take);
                now += static_cast<double>(take) / rate;
                poll();
            }
        }
        live_confidence = tracker.estimate(now).confidence;
        live_stats = tracker.stats();
        beats = live_beats;
    }

    bool beats_replaced = false;
    const char* beats_source = "tracker";

#if TIKTAK_HAVE_ML
    if (!beat_this_path.empty()) {
        tiktak::ml::BeatThisSession session;
        if (!session.open(beat_this_path)) {
            std::fprintf(stderr, "%s\n", session.reason().c_str());
            return 1;
        }

        const tiktak::dsp::Resampler resampler(rate, tiktak::ml::BeatThisFeatures::kModelRate);
        const std::vector<float> model_audio = resampler.apply(samples.data(), samples.size());

        tiktak::ml::BeatThisFeatures features;
        const std::vector<float> mel =
            features.compute(model_audio.data(), model_audio.size());
        const std::size_t mels = tiktak::ml::BeatThisFeatures::kMels;
        const auto activations = session.run(mel.data(), mel.size() / mels, mels);
        const auto grid = tiktak::ml::pickBeats(activations.beat.data(),
                                                activations.downbeat.data(),
                                                activations.beat.size());
        if (grid.beats.size() < 2) {
            std::fprintf(stderr, "the model found %zu beat(s); a grid needs at least 2\n",
                         grid.beats.size());
            return 1;
        }
        beats = grid.beats;
        model_downbeats = grid.downbeats;
        beats_replaced = true;
        beats_source = "beat_this";
    }
#endif

    if (!beats_path.empty()) {
        std::vector<double> supplied;
        std::string complaint;
        if (!readSalience(beats_path.c_str(), supplied, complaint)) {
            std::fprintf(stderr, "%s: %s\n", beats_path.c_str(), complaint.c_str());
            return 1;
        }
        if (supplied.size() < 2) {
            std::fprintf(stderr, "%s holds %zu beat time(s); a grid needs at least 2\n",
                         beats_path.c_str(), supplied.size());
            return 1;
        }
        // Sorted and strictly increasing, because everything downstream indexes
        // bars by position in this list. An unsorted grid would not fail, it
        // would silently answer about a different piece of music.
        for (std::size_t i = 1; i < supplied.size(); ++i) {
            if (!(supplied[i] > supplied[i - 1])) {
                std::fprintf(stderr,
                             "%s: beat times must strictly increase (%zu: %.17g <= %.17g)\n",
                             beats_path.c_str(), i, supplied[i], supplied[i - 1]);
                return 1;
            }
        }
        beats = supplied;
        beats_replaced = true;
        beats_source = "file";
    }

    std::vector<double> downbeats = analysis.downbeats;
    // The model has its own opinion about bar lines, and when it was the one
    // that found the beats it is the one to ask. The resolver still runs — its
    // metre and margins are what the rest of this tool reports — but the bar
    // lines themselves come from the head that was trained to find them.
    if (!model_downbeats.empty()) downbeats = model_downbeats;
    int beats_per_bar = analysis.beats_per_bar;
    double strength = analysis.downbeat_strength;
    double phase_margin = analysis.downbeat_phase_margin;
    double meter_margin = analysis.downbeat_meter_margin;
    bool confident = analysis.downbeat_confident;

    if (!salience_path.empty()) {
        std::vector<double> salience;
        std::string salience_error;
        if (!readSalience(salience_path.c_str(), salience, salience_error)) {
            std::fprintf(stderr, "%s: %s\n", salience_path.c_str(),
                         salience_error.c_str());
            return 1;
        }
        if (salience.size() != beats.size()) {
            std::fprintf(stderr,
                         "%s holds %zu value(s) but the analysis found %zu beat(s) — "
                         "one number per beat, in beat order\n",
                         salience_path.c_str(), salience.size(), beats.size());
            return 1;
        }
        // The C API offers no way to override the downbeat configuration, so
        // this research seam reaches the resolver directly. All three
        // calibration numbers travel together: the resolver no longer rescales
        // arbitrary backend output, so margins are in the backend's own units
        // and `downbeat_confident` means nothing unless judged by that
        // backend's own thresholds.
        tiktak::analysis::DownbeatConfig db_config;
        db_config.min_salience_range = salience_min_range;
        db_config.min_phase_margin = salience_min_phase;
        db_config.min_meter_margin = salience_min_meter;
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
    std::printf("  \"beats_source\": \"%s\",\n",
                beats_source);
    std::printf("  \"sample_rate\": %.17g,\n", rate);
    std::printf("  \"duration_sec\": %.17g,\n", static_cast<double>(samples.size()) / rate);
    std::printf("  \"bpm\": %.17g,\n", finiteOrZero(analysis.bpm));
    std::printf("  \"confidence\": %.17g,\n",
                finiteOrZero(analysis.tempo_confidence));
    std::printf("  \"beat_objective_per_beat\": %.9g,\n",
                finiteOrZero(analysis.beat_objective_per_beat));
    std::printf("  \"beats_causal\": %s,\n", live ? "true" : "false");
    if (dump_odf) {
        const std::vector<double>& odf_values = analyzer.odfValues();
        const std::vector<double>& odf_times = analyzer.frameTimes();
        std::printf("  \"odf\": [");
        for (std::size_t i = 0; i < odf_values.size(); ++i) {
            std::printf("%s%.6g", i == 0 ? "" : ",", odf_values[i]);
        }
        std::printf("],\n  \"odf_times\": [");
        for (std::size_t i = 0; i < odf_times.size(); ++i) {
            std::printf("%s%.6f", i == 0 ? "" : ",", odf_times[i]);
        }
        std::printf("],\n");
    }
    if (live) {
        std::printf("  \"live_confidence\": %.9g,\n", finiteOrZero(live_confidence));
        // Beats the tracker decided on but could not play, because by the time
        // it was asked the beat was already behind the clock. Free to report
        // and the only way to tell a tracker that lost the grid from one that
        // found it too late — which look identical in the beat list.
        std::printf("  \"live_frames\": %zu,\n", live_stats.frames);
        std::printf("  \"live_gated\": %zu,\n", live_stats.gated);
        std::printf("  \"live_beats\": %zu,\n", live_stats.beats);
        std::printf("  \"live_beats_late\": %zu,\n", live_stats.beats_late);
        std::printf("  \"live_seeded\": %s,\n", live_seed ? "true" : "false");
        const auto column = [](const char* name, const std::vector<double>& values) {
            std::printf("  \"%s\": [", name);
            for (std::size_t i = 0; i < values.size(); ++i) {
                std::printf("%s%.4g", i == 0 ? "" : ",", values[i]);
            }
            std::printf("],\n");
        };
        column("live_share", live_share);
        column("live_agreement", live_agreement);
        column("live_coincidence", live_coincidence);
    }
    std::printf("  \"beats_per_bar\": %d,\n", beats_per_bar);
    std::printf("  \"downbeat_strength\": %.17g,\n", finiteOrZero(strength));
    std::printf("  \"downbeat_phase_margin\": %.17g,\n",
                finiteOrZero(phase_margin));
    std::printf("  \"downbeat_meter_margin\": %.17g,\n",
                finiteOrZero(meter_margin));
    std::printf("  \"downbeat_confident\": %s,\n", confident ? "true" : "false");
    // The three cues, one value per beat, in the order the beats are printed.
    // Measured on real music the metre is usually right and the phase usually
    // wrong, which is a claim about these numbers; printing them is what lets
    // that be investigated, and what lets cue weights be swept outside the core
    // by recombining them and feeding the result back through --salience.
    printTimes("cue_low", cueColumn(analysis.beat_features, &BeatFeature::low), false);
    printTimes("cue_accent", cueColumn(analysis.beat_features, &BeatFeature::accent), false);
    printTimes("cue_harmony",
               cueColumn(analysis.beat_features, &BeatFeature::harmonic_change), false);

    // Runner-up tempos, strongest first. An octave-away runner-up with a
    // similar strength is the classic ambiguity, and it is the difference
    // between a tracker that is wrong and one that was handed a coin toss.
    std::printf("  \"tempo_candidates\": [");
    tiktak::analysis::TempoCandidate candidates[8];
    const std::size_t found = analyzer.tempoCandidates(candidates, 8);
    for (std::size_t i = 0; i < found; ++i) {
        std::printf("%s{\"bpm\": %.17g, \"strength\": %.17g}", i ? ", " : "",
                    finiteOrZero(candidates[i].bpm),
                    finiteOrZero(candidates[i].strength));
    }
    std::printf("],\n");

    printTimes("beats", beats, false);
    printTimes("downbeats", downbeats, true);
    std::printf("}\n");

    return 0;
}
