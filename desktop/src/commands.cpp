#include "commands.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "device.hpp"
#include "keys.hpp"
#include "tap.hpp"
#include "dsp/matched.hpp"
#include "ml/beatnet.hpp"
#include "tracking/live.hpp"
#include "render/click.hpp"
#include "render/metronome.hpp"
#include "render/live_metronome.hpp"
#include "render/player.hpp"
#include "wav.hpp"

#if defined(TIKTAK_HAVE_DECODE)
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "analysis/grid_cache.hpp"
#include "analysis/offline.hpp"
#include "decode/decoder.hpp"
#endif

namespace tiktak::desktop {
namespace {

using tiktak::render::Metronome;
using tiktak::render::MetronomeConfig;

// The grid is started this far into the stream. It has to clear the device's
// own buffering, or the first beat is already in the past by the time the
// callback that would play it runs, and gets dropped for lateness.
constexpr double kStartDelaySec = 0.5;

MetronomeConfig makeConfig(const Options& options, double sample_rate) {
    MetronomeConfig cfg;
    cfg.grid.bpm = options.bpm;
    cfg.grid.beats_per_bar = options.beats_per_bar;
    cfg.grid.subdivisions = options.subdivisions;
    cfg.grid.lookahead_sec = options.lookahead_sec;
    cfg.grid.channel_enabled = {{true, false, false}};
    cfg.grid.latency_sec = {{options.output_latency_sec, 0.0, 0.0}};
    cfg.click.sample_rate = sample_rate;
    return cfg;
}

void reportStats(const Metronome::Stats& stats) {
    std::printf("  beats scheduled     %zu\n", stats.beats);
    if (stats.clean()) {
        std::printf("  nothing dropped, nothing stolen, no gaps in the stream\n");
        return;
    }
    std::printf("  ** the run was not clean **\n");
    if (stats.grid_late) std::printf("  beats given up as late  %zu\n", stats.grid_late);
    if (stats.clicks_late) std::printf("  clicks past their buffer %zu\n", stats.clicks_late);
    if (stats.clicks_overflowed)
        std::printf("  clicks refused, queue full %zu\n", stats.clicks_overflowed);
    if (stats.voices_stolen) std::printf("  clicks cut short        %zu\n", stats.voices_stolen);
    if (stats.discontinuities)
        std::printf("  buffers out of sequence %zu  (the device glitched)\n",
                    stats.discontinuities);
}

// ---------------------------------------------------------------- callbacks --

struct PlayState {
    Metronome* metronome = nullptr;
};

void playCallback(void* user, double stream_time_sec, const float* input, float* output,
                  std::size_t frames) {
    (void)input;
    auto* state = static_cast<PlayState*>(user);
    state->metronome->process(stream_time_sec, output, frames);
}

#if defined(TIKTAK_HAVE_DECODE)

struct TrackState {
    tiktak::render::TrackPlayer* player = nullptr;
};

void trackCallback(void* user, double stream_time_sec, const float* input, float* output,
                   std::size_t frames) {
    (void)input;
    auto* state = static_cast<TrackState*>(user);
    state->player->process(stream_time_sec, output, frames);
}

void reportPlayerStats(const tiktak::render::TrackPlayer::Stats& stats) {
    std::printf("  clicks scheduled    %zu\n", stats.beats);
    if (stats.loops) std::printf("  loops completed     %zu\n", stats.loops);
    if (stats.clean()) {
        std::printf("  nothing dropped, nothing stolen, no gaps in the stream\n");
        return;
    }
    std::printf("  ** the run was not clean **\n");
    if (stats.clicks_late) std::printf("  clicks past their buffer %zu\n", stats.clicks_late);
    if (stats.clicks_overflowed)
        std::printf("  clicks refused, queue full %zu\n", stats.clicks_overflowed);
    if (stats.voices_stolen) std::printf("  clicks cut short        %zu\n", stats.voices_stolen);
    if (stats.discontinuities)
        std::printf("  buffers out of sequence %zu  (the device glitched)\n",
                    stats.discontinuities);
    if (stats.cues_dropped) std::printf("  cues with no room       %zu\n", stats.cues_dropped);
}

#endif  // TIKTAK_HAVE_DECODE

struct ListenState {
    tiktak::render::LiveMetronome* metronome = nullptr;
};

// The one callback of a duplex device: the room comes in, the click goes out,
// and both are stamped with the same clock — which is the whole reason the
// tracker can put a click on a beat it heard.
void listenCallback(void* user, double stream_time_sec, const float* input, float* output,
                    std::size_t frames) {
    auto* state = static_cast<ListenState*>(user);
    if (input != nullptr) state->metronome->capture(stream_time_sec, input, frames);
    state->metronome->process(stream_time_sec, output, frames);
}

// The causal tracker fed straight from a capture device, with the beats it
// hands out collected as they come.
//
// LiveMetronome is not used here even though it wraps the same tracker,
// because it also plays a click, and a click is what this bench must not do:
// a listener tapping along to our own metronome confirms whatever grid it has.
// So the tracker is driven directly and nothing goes to the speaker.
struct TapMicState {
    tiktak::tracking::LiveTracker* tracker = nullptr;
    std::vector<double>* beats = nullptr;
    std::atomic<double>* clock = nullptr;
    double lookahead_sec = 0.25;
};

void tapMicCallback(void* user, double stream_time_sec, const float* input, float* output,
                    std::size_t frames) {
    auto* state = static_cast<TapMicState*>(user);
    if (output != nullptr) std::fill(output, output + frames, 0.0f);
    if (input == nullptr) return;
    state->tracker->process(stream_time_sec, input, frames);
    state->clock->store(stream_time_sec, std::memory_order_relaxed);
    // Drained here rather than from the main thread: `takeBeat` hands out each
    // beat exactly once, and asking from two threads would lose some of them
    // to whichever asked first.
    double beat = 0.0;
    while (state->tracker->takeBeat(stream_time_sec, state->lookahead_sec, &beat)) {
        state->beats->push_back(beat);
    }
}

int cmdTapMic(const Options& options) {
    std::vector<unsigned char> blob;
    if (!options.model_path.empty()) {
        std::ifstream file(options.model_path, std::ios::binary);
        if (!file) {
            std::fprintf(stderr, "tiktak: cannot read %s\n", options.model_path.c_str());
            return 1;
        }
        blob.assign(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>());
    }
    tiktak::ml::BeatNetWeights weights;
    if (!blob.empty() && !weights.load(blob.data(), blob.size())) {
        std::fprintf(stderr, "tiktak: %s is not a BeatNet weight file\n",
                     options.model_path.c_str());
        return 1;
    }

    Device probe;
    const double rate = options.sample_rate > 0.0 ? options.sample_rate : 48000.0;
    tiktak::tracking::LiveConfig config = tiktak::tracking::liveConfigFor(rate);
    tiktak::tracking::LiveTracker tracker =
        weights.valid() ? tiktak::tracking::LiveTracker(config, weights)
                        : tiktak::tracking::LiveTracker(config);

    std::vector<double> beats;
    beats.reserve(4096);
    std::atomic<double> clock{0.0};
    TapMicState state;
    state.tracker = &tracker;
    state.beats = &beats;
    state.clock = &clock;
    state.lookahead_sec = options.lookahead_sec;

    Device device;
    if (!device.start(tapMicCallback, &state, rate, true, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }
    std::printf("%s via %s — %.0f Hz, %zu-frame periods, front end: %s\n",
                device.name().c_str(), device.backend().c_str(),
                device.sample_rate(), device.period_frames(),
                tracker.usingModel() ? "BeatNet" : "spectral flux");
    std::printf("\nplay something in the room and tap the beat. q or Enter to stop.\n"
                "nothing is played back — the point is that you are not tapping\n"
                "along with us.\n\n");

    std::vector<double> taps;
    KeyReader keys;
    const auto started = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
               .count() < options.seconds) {
        const int key = keys.poll();
        if (key < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (key == 'q' || key == 'Q' || key == '\r' || key == '\n') break;
        taps.push_back(clock.load(std::memory_order_relaxed));
        std::printf("\r%zu taps, %zu beats heard  ", taps.size(), beats.size());
        std::fflush(stdout);
    }
    device.stop();
    std::printf("\n\n");

    // The tracker's beats are predictions of when a beat *will* fall, so they
    // run ahead of the audio the listener is reacting to; the constant that
    // removes is the same one that removes their reaction time, and the report
    // separates it out already.
    const TapComparison result =
        compareTaps(taps, beats, options.tap_tolerance_sec);
    std::printf("taps                 %zu\n", result.taps);
    std::printf("beats the tracker emitted  %zu\n", beats.size());
    if (result.taps < 4 || beats.size() < 4) {
        std::printf("%s\n", result.verdict.c_str());
        return 1;
    }
    std::printf("your pulse           %.1f BPM\n", 60.0 / result.tap_period_sec);
    std::printf("the tracker's pulse  %.1f BPM   (ratio %.2f)\n",
                60.0 / result.grid_period_sec, result.octave_ratio);
    std::printf("you tapped           %+.0f ms from its nearest beat "
                "(spread %.0f ms)\n",
                result.median_offset_sec * 1000.0,
                result.offset_spread_sec * 1000.0);
    std::printf("within %.0f ms       %zu of %zu raw, %.0f%% once your own "
                "offset is removed\n",
                options.tap_tolerance_sec * 1000.0, result.matched, result.taps,
                result.matched_after_offset * 100.0);
    std::printf("\n%s\n", result.verdict.c_str());
    return 0;
}

void reportListen(const tiktak::render::LiveMetronome& metronome, double now_sec) {
    const tiktak::tracking::BeatEstimate estimate = metronome.estimate(now_sec);
    const tiktak::render::LiveMetronome::Stats stats = metronome.stats();

    if (metronome.manualTempo() > 0.0) {
        // In manual mode the tempo is not a finding, so reporting it as one
        // would be reporting the dial back to whoever set it. What was actually
        // worked out is the phase, and whether it was found at all.
        std::printf("live tracker: manual %.1f BPM, %s\n", metronome.manualTempo(),
                    metronome.waiting() ? "still listening for a beat to fall in with"
                                        : "synchronised to the room");
    } else {
        std::printf("live tracker: %.1f BPM (confidence %.2f, tempo spread %.3f octaves)\n",
                    estimate.bpm, estimate.confidence, estimate.tempo_spread_octaves);
    }
    std::printf("  beats played        %zu\n", stats.beats);
    std::printf("  frames gated as ours %zu\n", stats.gated);
    if (stats.clean()) {
        std::printf("  nothing dropped, nothing stolen, no gaps in the stream\n");
        return;
    }
    std::printf("  ** the run was not clean **\n");
    if (stats.beats_late) std::printf("  beats predicted too late %zu\n", stats.beats_late);
    if (stats.clicks_late) std::printf("  clicks past their buffer %zu\n", stats.clicks_late);
    if (stats.clicks_overflowed)
        std::printf("  clicks refused, queue full %zu\n", stats.clicks_overflowed);
    if (stats.voices_stolen) std::printf("  clicks cut short        %zu\n", stats.voices_stolen);
    if (stats.discontinuities)
        std::printf("  buffers out of sequence %zu  (the device glitched)\n",
                    stats.discontinuities);
    if (stats.capture_discontinuities)
        std::printf("  capture buffers out of sequence %zu\n", stats.capture_discontinuities);
}

struct MeasureState {
    Metronome* metronome = nullptr;
    float* recording = nullptr;
    std::size_t capacity = 0;
    std::atomic<std::size_t> recorded{0};
};

void measureCallback(void* user, double stream_time_sec, const float* input, float* output,
                     std::size_t frames) {
    auto* state = static_cast<MeasureState*>(user);
    state->metronome->process(stream_time_sec, output, frames);

    // Input frame i lines up with output frame i inside one duplex callback, so
    // recording straight into the same index means the offset measured later is
    // the whole round trip and nothing else.
    const std::size_t at = state->recorded.load(std::memory_order_relaxed);
    if (!input || at >= state->capacity) return;
    const std::size_t n = std::min(frames, state->capacity - at);
    for (std::size_t i = 0; i < n; ++i) state->recording[at + i] = input[i];
    state->recorded.store(at + n, std::memory_order_relaxed);
}

// ------------------------------------------------------------- measurement --

// Where a click starts inside `window`, or -1.
//
// Deliberately not the general onset detector from the core: this looks for one
// known burst in a short window around a time we already know, and a
// purpose-built detector that can be reasoned about is worth more here than a
// general one whose failures would be mistaken for jitter.
// The clicks this run is playing, rendered once so the recording can be
// correlated against the actual waveform rather than against a guess at it.
// One per kind, because the three differ in pitch and length and the wrong one
// correlates weakly enough to be refused.
std::vector<std::vector<float>> clickTemplates(const render::ClickConfig& config) {
    std::vector<std::vector<float>> out;
    for (schedule::BeatKind kind : {schedule::BeatKind::Downbeat, schedule::BeatKind::Beat,
                                    schedule::BeatKind::Subdivision}) {
        render::ClickRenderer renderer(config);
        if (!renderer.schedule(0.0, kind)) continue;
        // Long enough for the longest tone to decay by 60 dB, which is what
        // ClickTone::length_sec is defined as.
        std::vector<float> buffer(static_cast<std::size_t>(config.sample_rate * 0.4), 0.0f);
        renderer.mix(0.0, buffer.data(), buffer.size());
        out.push_back(std::move(buffer));
    }
    return out;
}

// Where a click starts inside `window`, in samples, or -1.
//
// Correlation against the known click rather than a level test. The rule this
// replaced took the first sample above thirty per cent of the window's own
// peak, which is exact down a loopback cable and unusable in a room: once the
// click is no longer the loudest thing in its window, the peak is set by
// whatever else is and the first crossing lands on that instead. Measured on
// synthetic mixtures at 0 dB SNR, the old rule landed within five milliseconds
// on at most 3 windows in 20 and this lands within five on at least 19.
//
// See dsp/matched.hpp for what this does not fix — in a live room the
// correlation peak sits late by an amount the room decides.
double findClick(const float* window, std::size_t frames,
                 const std::vector<std::vector<float>>& templates) {
    double best_offset = -1.0;
    double best_strength = 0.0;
    for (const std::vector<float>& tmpl : templates) {
        const dsp::MatchResult match =
            dsp::findKnownSignal(window, frames, tmpl.data(), tmpl.size());
        if (match.found() && match.strength > best_strength) {
            best_strength = match.strength;
            best_offset = match.offset_samples;
        }
    }
    return best_offset;
}

// Which beat of the grid a bar line falls on. The downbeats are a subset of the
// beats by construction, so this is a lookup and not a nearest-match: the small
// slack absorbs the round trip through the cache, where both went through the
// same decimal conversion but not necessarily the same arithmetic.
int beatIndexOf(const std::vector<double>& beats, double downbeat) {
    const auto at = std::lower_bound(beats.begin(), beats.end(), downbeat - 1e-9);
    return static_cast<int>(at - beats.begin());
}

}  // namespace

// ---------------------------------------------------------------- arguments --

bool parseOptions(const std::vector<std::string>& args, Options& options, std::string& error) {
    const auto number = [&error](const std::string& text, double& out) {
        char* end = nullptr;
        const double value = std::strtod(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0') {
            error = "'" + text + "' is not a number";
            return false;
        }
        out = value;
        return true;
    };

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        const bool has_value = i + 1 < args.size();

        const auto need = [&](double& out) {
            if (!has_value) {
                error = arg + " needs a value";
                return false;
            }
            return number(args[++i], out);
        };

        double value = 0.0;
        if (arg == "--bpm") {
            if (!need(value)) return false;
            options.bpm = value;
        } else if (arg == "--beats") {
            if (!need(value)) return false;
            options.beats_per_bar = static_cast<int>(value);
            options.beats_per_bar_given = true;
        } else if (arg == "--sub") {
            if (!need(value)) return false;
            options.subdivisions = static_cast<int>(value);
        } else if (arg == "--seconds") {
            if (!need(value)) return false;
            options.seconds = value;
        } else if (arg == "--rate") {
            if (!need(value)) return false;
            options.sample_rate = value;
        } else if (arg == "--latency-ms") {
            if (!need(value)) return false;
            options.output_latency_sec = value / 1000.0;
        } else if (arg == "--lookahead-ms") {
            if (!need(value)) return false;
            options.lookahead_sec = value / 1000.0;
        } else if (arg == "--device") {
            if (!has_value) {
                error = "--device needs a name";
                return false;
            }
            options.device_name = args[++i];
        } else if (arg == "-o" || arg == "--out") {
            if (!has_value) {
                error = arg + " needs a path";
                return false;
            }
            options.output_path = args[++i];
        } else if (arg == "--count-in") {
            if (!need(value)) return false;
            options.count_in = static_cast<int>(value);
        } else if (arg == "--from") {
            if (!need(value)) return false;
            options.from_bar = static_cast<long long>(value);
        } else if (arg == "--loop") {
            if (!has_value) {
                error = "--loop needs a range, as in --loop 4:8";
                return false;
            }
            const std::string range = args[++i];
            const std::size_t colon = range.find(':');
            bool ok = colon != std::string::npos && colon > 0 && colon + 1 < range.size();
            if (ok) {
                char* end = nullptr;
                options.loop_from = std::strtoll(range.c_str(), &end, 10);
                ok = end == range.c_str() + colon;
                if (ok) {
                    options.loop_to = std::strtoll(range.c_str() + colon + 1, &end, 10);
                    ok = *end == '\0';
                }
            }
            if (!ok) {
                error = "'" + range + "' is not a bar range like 4:8";
                return false;
            }
        } else if (arg == "--hint") {
            if (!need(value)) return false;
            options.hint_bpm = value;
        } else if (arg == "--manual") {
            if (!need(value)) return false;
            options.manual_bpm = value;
        } else if (arg == "--no-click") {
            options.no_click = true;
        } else if (arg == "--click") {
            options.click = true;
        } else if (arg == "--mic") {
            options.tap_mic = true;
        } else if (arg == "--model") {
            if (!has_value) {
                error = arg + " needs a path";
                return false;
            }
            options.model_path = args[++i];
        } else if (arg == "--tolerance-ms") {
            if (!need(value)) return false;
            options.tap_tolerance_sec = value / 1000.0;
        } else if (arg == "--no-cache") {
            options.no_cache = true;
        } else if (!arg.empty() && arg[0] != '-') {
            if (!options.track_path.empty()) {
                error = "only one file at a time — got '" + options.track_path +
                        "' and '" + arg + "'";
                return false;
            }
            options.track_path = arg;
        } else {
            error = "unknown option '" + arg + "'";
            return false;
        }
    }

    if (!(options.bpm > 0.0)) {
        error = "--bpm must be positive";
        return false;
    }
    if (options.beats_per_bar < 1) {
        error = "--beats must be at least 1";
        return false;
    }
    if (options.subdivisions < 1) {
        error = "--sub must be at least 1";
        return false;
    }
    if (!(options.seconds > 0.0)) {
        error = "--seconds must be positive";
        return false;
    }
    return true;
}

void printUsage() {
    std::printf(
        "tiktak — the desktop harness for the tik-tak core\n"
        "\n"
        "  tiktak devices                    list the audio devices\n"
        "  tiktak render -o out.wav          render the metronome to a file\n"
        "  tiktak play                       play it on a real device\n"
        "  tiktak measure                    measure the round trip and the jitter\n"
        "  tiktak track FILE                 play a file with the click on its own beats\n"
        "  tiktak listen                     click on the beat of what the microphone hears\n"
        "  tiktak tap FILE                   play a file, you tap along, and it compares\n"
        "\n"
        "Options:\n"
        "  --bpm N            tempo (120)\n"
        "  --beats N          beats per bar (4; `track` reads it off the audio)\n"
        "  --sub N            subdivisions per beat, 1 = beats only (1)\n"
        "  --seconds N        how long to run (10)\n"
        "  --rate N           sample rate; 0 takes the device's own (0)\n"
        "  --latency-ms N     output latency to compensate (0) — measure it first\n"
        "  --lookahead-ms N   how far ahead beats are handed out (250)\n"
        "  --device NAME      playback device, as printed by `devices`\n"
        "  -o, --out PATH     where to write (render, measure, track)\n"
        "\n"
        "Track options:\n"
        "  --count-in N       count-in beats before the music (4)\n"
        "  --from N           start at bar N (0)\n"
        "  --loop A:B         loop bars A to B, end exclusive\n"
        "  --hint N           manual-mode tempo hint in BPM; 0 estimates (0)\n"
        "\n"
        "`track` finds the bar lines as well as the beats, and accents the one.\n"
        "It reports separate phase and metre margins and accents only when both\n"
        "are convincing, because an accent on the wrong beat is harder to play\n"
        "to than none.\n"
        "Passing --beats overrides the meter it found — the number you type is\n"
        "an assertion about the music, the same way --hint is. It does not invent\n"
        "which beat starts the bar when the audio cannot say.\n"
        "  --no-click         the track alone, no metronome\n"
        "  --no-cache         re-analyse even when the beat grid is cached\n"
        "\n"
        "Tap options:\n"
        "  --mic              compare against the live tracker listening to the\n"
        "                     room, not against an offline analysis of a file\n"
        "  --model PATH       BeatNet weights for --mic; without it the live\n"
        "                     tracker runs on spectral flux\n"
        "  --click            play our click too — only for checking the bench\n"
        "                     itself. Tapping along to our own click confirms\n"
        "                     whatever grid we found, including a wrong one.\n"
        "  --tolerance-ms N   how close a tap counts as on the beat (70)\n"
        "\n"
        "`tap FILE` plays the file and compares your taps against the *offline*\n"
        "grid — the one `track` would click on. `tap --mic` opens the microphone\n"
        "and compares them against the *causal* tracker, which is a different\n"
        "program with different numbers; take care which one a verdict is about.\n"
        "Neither plays a click, because tapping along to our own click confirms\n"
        "any grid at all.\n"
        "\n"
        "It cannot measure accuracy — people tap early and unevenly — but it sees\n"
        "the three failures a corpus average hides: a grid on the off-beat, a\n"
        "grid at half or double the pulse you hear, and a grid that drifts.\n"
        "\n"
        "Listen options:\n"
        "  FILE               drive the tracker from a file instead of a microphone\n"
        "  --hint N           tempo to start from, in BPM; 0 searches (0)\n"
        "  --manual N         manual + sync: hold N BPM, take only the phase from\n"
        "                     the room; plays nothing until it finds one (0)\n"
        "  --no-click         listen and report, play nothing\n"
        "\n"
        "`listen` follows the room: the tracker predicts each beat and the click\n"
        "goes out early by the round trip, so it is heard on the beat. Pass the\n"
        "figure `measure` reports as --latency-ms, or the click is late by it.\n"
        "\n"
        "With --manual the tempo stops being a question: the click holds the BPM\n"
        "given, waits for the room to start, falls in on its phase, and then keeps\n"
        "going whether the room does or not. Finding a phase at a known tempo is a\n"
        "far smaller problem than finding a tempo, so this works on material the\n"
        "automatic mode cannot follow — but it refuses to fall in with a room whose\n"
        "beat is not the one asked for, rather than clicking somewhere and calling\n"
        "it synchronised.\n"
        "\n"
        "`track` analyses the file once and caches the beat grid next to it\n"
        "(.tiktak/<content-hash>.grid), so the second start is instant. With -o\n"
        "it renders to a WAV instead of a device — same callback, virtual clock.\n"
        "\n"
        "`render` needs no sound card, which is what makes the timing testable in\n"
        "CI and on any machine. `measure` needs the output to reach the input —\n"
        "a loopback cable, or speakers and a microphone in a quiet room.\n");
}

// ----------------------------------------------------------------- commands --

int cmdDevices() {
    const DeviceList list = listDevices();
    if (!list.ok) {
        std::fprintf(stderr, "tiktak: %s\n", list.error.c_str());
        return 1;
    }

    std::printf("backend: %s\n\nplayback:\n", list.backend.c_str());
    if (list.playback.empty()) std::printf("  (none)\n");
    for (const DeviceInfo& info : list.playback) {
        std::printf("  %s%s\n", info.name.c_str(), info.is_default ? "  [default]" : "");
    }

    std::printf("\ncapture:\n");
    if (list.capture.empty()) std::printf("  (none)\n");
    for (const DeviceInfo& info : list.capture) {
        std::printf("  %s%s\n", info.name.c_str(), info.is_default ? "  [default]" : "");
    }
    return 0;
}

int cmdRender(const Options& options) {
    if (options.output_path.empty()) {
        std::fprintf(stderr, "tiktak: render needs -o PATH\n");
        return 2;
    }

    // No device, and the same callback the device would drive. That is the
    // point: the timing this writes out is the timing a speaker would get, so
    // it can be checked on a machine with no sound card at all.
    const double rate = options.sample_rate > 0.0 ? options.sample_rate : 48000.0;

    MetronomeConfig cfg = makeConfig(options, rate);
    if (!cfg.valid()) {
        std::fprintf(stderr, "tiktak: those settings do not make a metronome\n");
        return 2;
    }

    Metronome metronome(cfg);
    metronome.start(kStartDelaySec);

    constexpr std::size_t kBlock = 256;
    const auto total = static_cast<std::size_t>(options.seconds * rate);
    std::vector<float> out(total, 0.0f);

    for (std::size_t i = 0; i < total; i += kBlock) {
        const std::size_t n = std::min(kBlock, total - i);
        metronome.process(static_cast<double>(i) / rate, out.data() + i, n);
    }

    if (!writeWav(options.output_path, out, rate)) {
        std::fprintf(stderr, "tiktak: could not write %s\n", options.output_path.c_str());
        return 1;
    }

    std::printf("wrote %s — %.1f s at %.0f Hz, %g BPM\n", options.output_path.c_str(),
                options.seconds, rate, options.bpm);
    reportStats(metronome.stats());
    return metronome.stats().clean() ? 0 : 1;
}

int cmdPlay(const Options& options) {
    Device device;
    PlayState state;

    // The device picks the rate, so the metronome cannot be built until it is
    // open — which is why the callback is armed with a null metronome and the
    // device is started only after it exists.
    MetronomeConfig cfg = makeConfig(options, 48000.0);
    Metronome placeholder(cfg);
    state.metronome = &placeholder;

    if (!device.start(playCallback, &state, options.sample_rate, false, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }
    device.stop();

    // Reopened at the rate the first attempt reported. Two opens is a small
    // price for never running the click through a resampler.
    cfg = makeConfig(options, device.sample_rate() > 0.0 ? device.sample_rate() : 48000.0);
    if (!cfg.valid()) {
        std::fprintf(stderr, "tiktak: those settings do not make a metronome\n");
        return 2;
    }
    Metronome metronome(cfg);
    state.metronome = &metronome;

    if (!device.start(playCallback, &state, cfg.click.sample_rate, false,
                      options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }

    metronome.start(kStartDelaySec);
    std::printf("%s via %s — %.0f Hz, %zu-frame periods, driver claims %.1f ms out\n",
                device.name().c_str(), device.backend().c_str(), device.sample_rate(),
                device.period_frames(), device.reported_output_latency() * 1000.0);
    std::printf("%g BPM, %d beats per bar, %d per beat, compensating %.1f ms. %g s.\n",
                options.bpm, options.beats_per_bar, options.subdivisions,
                options.output_latency_sec * 1000.0, options.seconds);

    std::this_thread::sleep_for(std::chrono::duration<double>(options.seconds));
    metronome.stop();
    device.stop();

    reportStats(metronome.stats());
    return metronome.stats().clean() ? 0 : 1;
}

int cmdMeasure(const Options& options) {
    Device device;
    MeasureState state;

    MetronomeConfig probe = makeConfig(options, 48000.0);
    Metronome placeholder(probe);
    state.metronome = &placeholder;

    if (!device.start(measureCallback, &state, options.sample_rate, true, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }
    const double rate = device.sample_rate();
    device.stop();

    // The point of the measurement is the raw round trip, so nothing is
    // compensated: whatever offset comes back *is* the latency, and its spread
    // is the jitter.
    Options raw = options;
    raw.output_latency_sec = 0.0;
    MetronomeConfig cfg = makeConfig(raw, rate);
    if (!cfg.valid()) {
        std::fprintf(stderr, "tiktak: those settings do not make a metronome\n");
        return 2;
    }
    Metronome metronome(cfg);

    const auto capacity = static_cast<std::size_t>((options.seconds + 1.0) * rate);
    std::vector<float> recording(capacity, 0.0f);
    state.metronome = &metronome;
    state.recording = recording.data();
    state.capacity = capacity;
    state.recorded.store(0, std::memory_order_relaxed);

    if (!device.start(measureCallback, &state, rate, true, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }

    metronome.start(kStartDelaySec);
    std::printf("%s via %s — %.0f Hz, %zu-frame periods\n", device.name().c_str(),
                device.backend().c_str(), rate, device.period_frames());
    std::printf("driver claims %.1f ms out + %.1f ms in = %.1f ms round trip\n",
                device.reported_output_latency() * 1000.0,
                device.reported_input_latency() * 1000.0,
                (device.reported_output_latency() + device.reported_input_latency()) * 1000.0);
    std::printf("listening for %g s at %g BPM...\n", options.seconds, options.bpm);

    std::this_thread::sleep_for(std::chrono::duration<double>(options.seconds));
    metronome.stop();
    device.stop();

    const std::size_t recorded = state.recorded.load(std::memory_order_relaxed);
    recording.resize(recorded);

    reportStats(metronome.stats());

    if (!options.output_path.empty()) {
        writeWav(options.output_path, recording, rate);
        std::printf("  recording written to %s\n", options.output_path.c_str());
    }

    // Each beat is looked for in its own window, starting at the moment its
    // click was submitted: a click cannot come back before it went out, and
    // half a second of slack is far more round trip than any real device has.
    const double beat_sec = 60.0 / (options.bpm * options.subdivisions);
    const auto window_frames = static_cast<std::size_t>(std::min(0.5, beat_sec * 0.9) * rate);

    const std::vector<std::vector<float>> templates = clickTemplates(cfg.click);

    std::vector<double> offsets;
    for (int k = 0;; ++k) {
        const double submitted = kStartDelaySec + beat_sec * k;
        const auto at = static_cast<std::size_t>(submitted * rate);
        if (at + window_frames >= recorded) break;

        const double found = findClick(recording.data() + at, window_frames, templates);
        if (found < 0.0) continue;
        offsets.push_back(found / rate);
    }

    if (offsets.size() < 4) {
        std::printf(
            "\nfound %zu clicks in the recording — not enough to measure.\n"
            "The output has to reach the input: a loopback cable, or speakers and a\n"
            "microphone in a quiet room with the volume up.\n",
            offsets.size());
        return 1;
    }

    double sum = 0.0;
    for (double v : offsets) sum += v;
    const double mean = sum / static_cast<double>(offsets.size());

    double variance = 0.0;
    for (double v : offsets) variance += (v - mean) * (v - mean);
    variance /= static_cast<double>(offsets.size());

    const double lowest = *std::min_element(offsets.begin(), offsets.end());
    const double highest = *std::max_element(offsets.begin(), offsets.end());

    std::printf("\n%zu clicks measured\n", offsets.size());
    std::printf("  round trip   %.2f ms   (this is the number to pass to --latency-ms,\n"
                "                          less whatever the microphone path adds)\n",
                mean * 1000.0);
    std::printf("  jitter       %.2f ms rms, %.2f ms peak to peak\n", std::sqrt(variance) * 1000.0,
                (highest - lowest) * 1000.0);
    std::printf("  earliest     %.2f ms\n  latest       %.2f ms\n", lowest * 1000.0,
                highest * 1000.0);
    std::printf(
        "\nThe spread is what matters. The mean is latency and can be compensated;\n"
        "the spread cannot, and it is what a player hears as a metronome that\n"
        "will not sit still.\n");
    return 0;
}

// -------------------------------------------------------------------- track --


// ------------------------------------------------------------------ listen --

int cmdListen(const Options& options) {
    using tiktak::render::LiveMetronome;
    using tiktak::render::LiveMetronomeConfig;

    // A file was named: drive the tracker from it instead of a microphone,
    // against a virtual clock. Not a lesser mode — it is what makes the
    // microphone path testable on a machine with no microphone, which is every
    // CI runner, and it is the only way to run the same input twice.
    const bool from_file = !options.track_path.empty();

    std::vector<float> room;
    double rate = options.sample_rate > 0.0 ? options.sample_rate : 48000.0;

    if (from_file) {
#if defined(TIKTAK_HAVE_DECODE)
        auto decoder = tiktak::decode::Decoder::open(options.track_path.c_str());
        if (!decoder) {
            std::fprintf(stderr, "tiktak: %s is not a WAV, FLAC or MP3 file\n",
                         options.track_path.c_str());
            return 1;
        }
        rate = decoder->info().sample_rate;
        float block[65536];
        for (;;) {
            const std::size_t got = decoder->readMono(block, 65536);
            if (got == 0) break;
            room.insert(room.end(), block, block + got);
        }
        if (room.empty()) {
            std::fprintf(stderr, "tiktak: %s decoded to nothing\n", options.track_path.c_str());
            return 1;
        }
#else
        std::fprintf(stderr,
                     "tiktak: this build has no decoder — rebuild with "
                     "-DTIKTAK_BUILD_DECODE=ON, or run `listen` with a microphone\n");
        return 2;
#endif
    }

    LiveMetronomeConfig cfg;
    cfg.tracker = tiktak::tracking::liveConfigFor(rate);
    cfg.click.sample_rate = rate;
    // For `listen` the latency that matters is the *round trip*: the tracker's
    // clock is the capture stream's, so the click has to leave early by the
    // whole way out and back. That is the number `measure` reports.
    cfg.round_trip_sec = options.output_latency_sec;
    if (!cfg.valid()) {
        std::fprintf(stderr, "tiktak: those settings do not make a live metronome\n");
        return 2;
    }

    LiveMetronome metronome(cfg);
    if (options.manual_bpm > 0.0) {
        // Manual + sync: the tempo is not up for discussion, and the room is
        // asked only where the beat falls. Nothing plays until it answers.
        metronome.setManualTempo(options.manual_bpm);
        std::printf("manual %.1f BPM — listening for a beat to fall in with\n",
                    options.manual_bpm);
    } else if (options.hint_bpm > 0.0) {
        metronome.seedTempo(options.hint_bpm);
        std::printf("starting from %.1f BPM\n", options.hint_bpm);
    }
    if (!options.no_click) metronome.start();

    if (from_file) {
        constexpr std::size_t kBlock = 256;
        const auto total =
            std::min(room.size(), static_cast<std::size_t>(options.seconds * rate));
        std::vector<float> out(total, 0.0f);

        std::printf("%s — %.1f s at %.0f Hz, through the microphone path\n",
                    options.track_path.c_str(), static_cast<double>(total) / rate, rate);

        for (std::size_t i = 0; i < total; i += kBlock) {
            const std::size_t n = std::min(kBlock, total - i);
            const double time = static_cast<double>(i) / rate;
            metronome.capture(time, room.data() + i, n);

            // The track is written under the click so the result can be
            // listened to: a click that is off the beat is obvious in a second
            // and invisible in any number of counters.
            std::copy(room.begin() + static_cast<std::ptrdiff_t>(i),
                      room.begin() + static_cast<std::ptrdiff_t>(i + n),
                      out.begin() + static_cast<std::ptrdiff_t>(i));
            metronome.process(time, out.data() + i, n);
        }

        if (!options.output_path.empty()) {
            if (!writeWav(options.output_path, out, rate)) {
                std::fprintf(stderr, "tiktak: could not write %s\n", options.output_path.c_str());
                return 1;
            }
            std::printf("wrote %s — the room and the click it played over it\n",
                        options.output_path.c_str());
        }

        reportListen(metronome, static_cast<double>(total) / rate);
        return metronome.stats().clean() ? 0 : 1;
    }

    Device device;
    ListenState state;
    state.metronome = &metronome;

    // Duplex: the microphone is the whole point, so a machine that cannot
    // capture cannot run this at all.
    if (!device.start(listenCallback, &state, options.sample_rate, true, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }

    std::printf("%s via %s — %.0f Hz, %zu-frame periods\n", device.name().c_str(),
                device.backend().c_str(), device.sample_rate(), device.period_frames());
    if (cfg.round_trip_sec <= 0.0) {
        std::printf(
            "no round trip given: the click will be late by whatever the device's is.\n"
            "  measure it with `tiktak measure` and pass it as --latency-ms\n");
    }
    std::printf("listening for %g s...\n", options.seconds);

    const auto started = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    while (elapsed < options.seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        // The tracker's clock is the stream's, and the harness only knows wall
        // time, so this line is a progress report and not a measurement.
        const auto estimate = metronome.estimate(elapsed);
        if (options.manual_bpm > 0.0) {
            std::printf("  %5.1f s   %6.1f BPM   %s   phase %.2f\n", elapsed, estimate.bpm,
                        metronome.waiting() ? "listening" : "in sync  ",
                        metronome.syncStrength());
        } else {
            std::printf("  %5.1f s   %6.1f BPM   confidence %.2f\n", elapsed, estimate.bpm,
                        estimate.confidence);
        }
        std::fflush(stdout);
    }

    metronome.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // let the tail ring
    device.stop();

    reportListen(metronome, elapsed);
    return metronome.stats().clean() ? 0 : 1;
}

#if defined(TIKTAK_HAVE_DECODE)

// Decode a file and get its beat grid, cache and all.
//
// Factored out rather than copied because `tap` needs exactly what `track`
// needs: the same decode, the same analysis config, the same content-addressed
// cache. A second copy would drift, and the first symptom would be the bench
// comparing a person against a grid the player never used.
struct LoadedTrack {
    std::vector<float> samples;
    double rate = 0.0;
    tiktak::analysis::OfflineConfig analysis;
    tiktak::analysis::OfflineResult grid;
    bool from_cache = false;
};

bool loadTrack(const Options& options, const char* command, LoadedTrack& out) {
    namespace fs = std::filesystem;

    if (options.track_path.empty()) {
        std::fprintf(stderr, "tiktak: %s needs a file, as in `tiktak %s song.mp3`\n",
                     command, command);
        return false;
    }

    // The encoded bytes are read once and used twice: hashed for the cache key
    // before anything is decoded, then decoded — the same flow a shell takes.
    std::ifstream file(fs::path(options.track_path), std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "tiktak: cannot read %s\n", options.track_path.c_str());
        return false;
    }
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(file),
                                           std::istreambuf_iterator<char>()};
    file.close();

    auto decoder = tiktak::decode::Decoder::openMemory(bytes.data(), bytes.size());
    if (!decoder) {
        std::fprintf(stderr, "tiktak: %s is not a WAV, FLAC or MP3 file\n",
                     options.track_path.c_str());
        return false;
    }
    out.rate = decoder->info().sample_rate;
    out.samples.reserve(static_cast<std::size_t>(decoder->info().frames));
    {
        float block[65536];
        for (;;) {
            const std::size_t got = decoder->readMono(block, 65536);
            if (got == 0) break;
            out.samples.insert(out.samples.end(), block, block + got);
        }
    }
    if (out.samples.empty()) {
        std::fprintf(stderr, "tiktak: %s decoded to nothing\n",
                     options.track_path.c_str());
        return false;
    }

    out.analysis.odf.sampleRate = out.rate;
    out.analysis.bpm_hint = options.hint_bpm;

    std::string cache_name = tiktak::analysis::gridCacheKey(bytes.data(), bytes.size());
    if (options.hint_bpm > 0.0) {
        char hint[32];
        std::snprintf(hint, sizeof(hint), "-hint%g", options.hint_bpm);
        cache_name += hint;
    }
    const fs::path cache_path =
        fs::path(options.track_path).parent_path() / ".tiktak" / (cache_name + ".grid");

    if (!options.no_cache) {
        std::ifstream cached(cache_path, std::ios::binary);
        if (cached) {
            const std::vector<std::uint8_t> blob{std::istreambuf_iterator<char>(cached),
                                                 std::istreambuf_iterator<char>()};
            out.from_cache = tiktak::analysis::deserializeGrid(
                blob.data(), blob.size(), out.analysis, &out.grid);
        }
    }
    if (!out.from_cache) {
        out.grid = tiktak::analysis::analyseOffline(out.samples.data(),
                                                    out.samples.size(), out.analysis);
        if (!options.no_cache) {
            std::error_code ec;
            fs::create_directories(cache_path.parent_path(), ec);
            std::ofstream write_out(cache_path, std::ios::binary);
            if (write_out) {
                const std::vector<std::uint8_t> blob =
                    tiktak::analysis::serializeGrid(out.grid, out.analysis);
                write_out.write(reinterpret_cast<const char*>(blob.data()),
                                static_cast<std::streamsize>(blob.size()));
            }
        }
    }
    return true;
}

int cmdTrack(const Options& options) {
    namespace fs = std::filesystem;
    using tiktak::analysis::OfflineConfig;
    using tiktak::analysis::OfflineResult;
    using tiktak::render::PlayerConfig;
    using tiktak::render::TrackPlayer;

    if (options.track_path.empty()) {
        std::fprintf(stderr, "tiktak: track needs a file, as in `tiktak track song.mp3`\n");
        return 2;
    }

    // The encoded bytes are read once and used twice: hashed for the cache key
    // before anything is decoded, then decoded — the same flow a shell takes.
    std::ifstream file(fs::path(options.track_path), std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "tiktak: cannot read %s\n", options.track_path.c_str());
        return 1;
    }
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(file),
                                           std::istreambuf_iterator<char>()};
    file.close();

    auto decoder = tiktak::decode::Decoder::openMemory(bytes.data(), bytes.size());
    if (!decoder) {
        std::fprintf(stderr, "tiktak: %s is not a WAV, FLAC or MP3 file\n",
                     options.track_path.c_str());
        return 1;
    }
    const double rate = decoder->info().sample_rate;

    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(decoder->info().frames));
    float block[65536];
    for (;;) {
        const std::size_t got = decoder->readMono(block, 65536);
        if (got == 0) break;
        samples.insert(samples.end(), block, block + got);
    }
    if (samples.empty()) {
        std::fprintf(stderr, "tiktak: %s decoded to nothing\n", options.track_path.c_str());
        return 1;
    }

    OfflineConfig analysis;
    analysis.odf.sampleRate = rate;
    analysis.bpm_hint = options.hint_bpm;

    // Content-addressed cache next to the file: the key is the hash of the
    // encoded bytes, so a renamed file hits and a re-encoded one misses. The
    // tempo hint goes into the name too — a hinted grid is a different grid,
    // and one name per track would make the two modes overwrite each other on
    // every switch. (The blob itself refuses a config it was not analysed
    // under; the name only keeps both alive side by side.)
    std::string cache_name = tiktak::analysis::gridCacheKey(bytes.data(), bytes.size());
    if (options.hint_bpm > 0.0) {
        char hint[32];
        std::snprintf(hint, sizeof(hint), "-hint%g", options.hint_bpm);
        cache_name += hint;
    }
    const fs::path cache_path =
        fs::path(options.track_path).parent_path() / ".tiktak" / (cache_name + ".grid");

    OfflineResult grid;
    bool from_cache = false;
    if (!options.no_cache) {
        std::ifstream cached(cache_path, std::ios::binary);
        if (cached) {
            const std::vector<std::uint8_t> blob{std::istreambuf_iterator<char>(cached),
                                                 std::istreambuf_iterator<char>()};
            from_cache =
                tiktak::analysis::deserializeGrid(blob.data(), blob.size(), analysis, &grid);
        }
    }

    if (!from_cache) {
        grid = tiktak::analysis::analyseOffline(samples.data(), samples.size(), analysis);
        if (!options.no_cache) {
            std::error_code ec;
            fs::create_directories(cache_path.parent_path(), ec);
            std::ofstream out(cache_path, std::ios::binary);
            if (out) {
                const std::vector<std::uint8_t> blob =
                    tiktak::analysis::serializeGrid(grid, analysis);
                out.write(reinterpret_cast<const char*>(blob.data()),
                          static_cast<std::streamsize>(blob.size()));
            }
        }
    }

    std::printf("%s — %.1f s at %.0f Hz\n", options.track_path.c_str(),
                static_cast<double>(samples.size()) / rate, rate);
    std::printf("beat grid: %s — %zu beats at %.1f BPM (confidence %.2f)\n",
                from_cache ? "cache hit" : "analysed", grid.beats.size(), grid.bpm,
                grid.tempo_confidence);
    if (grid.beats.empty()) {
        std::printf("no beats found — playing the track without a click\n");
    }

    // Where the bar starts, from the audio rather than from a convention.
    //
    // Two doubts have to clear before the accent is used at all: the phase
    // margin says which beat starts the bar is settled, and the meter margin
    // says no other bar length fits nearly as well. Either one alone is not
    // enough — a piece read in three can be perfectly settled about where its
    // bars start while four fits it just as well, and the phase margin cannot
    // see that because every rival it weighs has already accepted three.
    //
    // Both thresholds live in the analysis config, not here, so that
    // research/eval sweeps the same numbers this uses rather than a copy. They
    // are placeholders; see research/eval/README.md.
    int beats_per_bar = options.beats_per_bar;
    int downbeat_offset = 0;
    bool accent = false;

    if (!grid.beats.empty() && grid.beats_per_bar > 0) {
        std::printf("bar lines: %d beats to the bar "
                    "(strength %.2f, phase margin %.2f, metre margin %.2f)%s\n",
                    grid.beats_per_bar, grid.downbeat_strength,
                    grid.downbeat_phase_margin, grid.downbeat_meter_margin,
                    grid.downbeat_confident ? "" : " — too close to call");
    } else if (!grid.beats.empty()) {
        std::printf("bar lines: none found — not enough repeated bar-level evidence\n");
    }

    if (options.beats_per_bar_given) {
        // An explicit --beats is the user's assertion about the music and
        // outranks the analysis, exactly as an explicit --bpm does. It asserts
        // the bar *length* though, and says nothing about which beat starts the
        // bar. Use the phase only when the analysis independently supports it;
        // otherwise an even click is the only answer that does not invent one.
        if (grid.beats_per_bar == options.beats_per_bar && !grid.downbeats.empty() &&
            grid.downbeat_phase_margin >= analysis.downbeat.min_phase_margin) {
            downbeat_offset = beatIndexOf(grid.beats, grid.downbeats.front());
            accent = true;
            std::printf("bar starts on beat %d, from the audio\n", downbeat_offset + 1);
        } else if (grid.beats_per_bar > 0 && grid.beats_per_bar != options.beats_per_bar) {
            std::printf("using --beats %d over the %d the audio suggests"
                        " — phase unknown, every beat clicks the same\n",
                        options.beats_per_bar, grid.beats_per_bar);
        } else {
            std::printf("using --beats %d — phase unknown, every beat clicks the same\n",
                        options.beats_per_bar);
        }
    } else if (grid.downbeat_confident) {
        beats_per_bar = grid.beats_per_bar;
        downbeat_offset = beatIndexOf(grid.beats, grid.downbeats.front());
        accent = true;
    } else if (!grid.beats.empty()) {
        // Nothing was detected and nothing was asserted. Counting fours from the
        // first beat would be an arbitrary accent worn with the same confidence
        // as a real one, so the click stays even and says so.
        std::printf("no accent — every beat clicks the same\n");
    }

    PlayerConfig cfg;
    cfg.sample_rate = rate;
    cfg.click.sample_rate = rate;
    cfg.beats_per_bar = beats_per_bar;
    cfg.downbeat_offset = downbeat_offset;
    cfg.accent_downbeats = accent;
    cfg.count_in_beats = grid.beats.empty() ? 0 : options.count_in;
    cfg.channel_enabled = {{!options.no_click && !grid.beats.empty(), false, false}};
    if (!cfg.valid()) {
        std::fprintf(stderr, "tiktak: those settings do not make a player\n");
        return 2;
    }

    TrackPlayer player(cfg);
    player.setTrack(samples.data(), samples.size());
    player.setGrid(grid.beats.data(), grid.beats.size());

    if (options.loop_from >= 0) {
        if (!player.setLoop(options.loop_from, options.loop_to)) {
            std::fprintf(stderr, "tiktak: the grid has no bars %lld:%lld to loop\n",
                         options.loop_from, options.loop_to);
            return 2;
        }
        std::printf("looping bars %lld:%lld\n", options.loop_from, options.loop_to);
    }

    const long long from_bar = grid.beats.empty() ? 0 : options.from_bar;

    // Render mode: the same callback against a virtual clock, into a file —
    // how the player's timing is checked without a sound card.
    if (!options.output_path.empty()) {
        if (!player.start(0.0, from_bar)) {
            std::fprintf(stderr, "tiktak: bar %lld is not in the grid\n", from_bar);
            return 2;
        }

        constexpr std::size_t kBlock = 256;
        const auto total = static_cast<std::size_t>(options.seconds * rate);
        std::vector<float> out(total, 0.0f);
        for (std::size_t i = 0; i < total; i += kBlock) {
            const std::size_t n = std::min(kBlock, total - i);
            player.process(static_cast<double>(i) / rate, out.data() + i, n);
        }

        if (!writeWav(options.output_path, out, rate)) {
            std::fprintf(stderr, "tiktak: could not write %s\n", options.output_path.c_str());
            return 1;
        }
        std::printf("wrote %s — %.1f s\n", options.output_path.c_str(), options.seconds);
        reportPlayerStats(player.stats());
        return player.stats().clean() ? 0 : 1;
    }

    Device device;
    TrackState state;
    state.player = &player;

    // The track fixes the rate; the device is asked for it and miniaudio
    // converts if the hardware insists on another. The click stays bound to
    // the track either way — they share the buffer.
    if (!device.start(trackCallback, &state, rate, false, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }

    if (!player.start(kStartDelaySec, from_bar)) {
        device.stop();
        std::fprintf(stderr, "tiktak: bar %lld is not in the grid\n", from_bar);
        return 2;
    }

    std::printf("%s via %s — %.0f Hz, %zu-frame periods\n", device.name().c_str(),
                device.backend().c_str(), device.sample_rate(), device.period_frames());

    const double run_for = options.seconds;
    const auto started = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                   .count() < run_for &&
           player.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    player.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // let the tail ring
    device.stop();

    std::printf("stopped at %.1f s into the track\n", player.positionSec());
    reportPlayerStats(player.stats());
    return player.stats().clean() ? 0 : 1;
}

int cmdTap(const Options& options) {
    using tiktak::render::PlayerConfig;
    using tiktak::render::TrackPlayer;

    if (options.tap_mic) return cmdTapMic(options);

    LoadedTrack track;
    if (!loadTrack(options, "tap", track)) return 1;

    std::printf("%s — %.1f s at %.0f Hz\n", options.track_path.c_str(),
                static_cast<double>(track.samples.size()) / track.rate, track.rate);
    std::printf("beat grid: %s — %zu beats at %.1f BPM\n",
                track.from_cache ? "cache hit" : "analysed",
                track.grid.beats.size(), track.grid.bpm);
    if (track.grid.beats.size() < 4) {
        std::fprintf(stderr, "tiktak: this file has no grid to compare against\n");
        return 1;
    }

    PlayerConfig cfg;
    cfg.sample_rate = track.rate;
    cfg.click.sample_rate = track.rate;
    cfg.beats_per_bar = track.grid.beats_per_bar > 0 ? track.grid.beats_per_bar : 4;
    cfg.accent_downbeats = false;
    // No count-in. A count-in is our own click telling the listener where the
    // beat is before they have heard the music, which is the same leak the
    // silent click exists to avoid.
    cfg.count_in_beats = 0;
    cfg.channel_enabled = {{options.click, false, false}};
    if (!cfg.valid()) {
        std::fprintf(stderr, "tiktak: those settings do not make a player\n");
        return 2;
    }
    if (options.click) {
        std::printf("** the click is ON — you will tap along with our grid, so "
                    "this run\n   checks the bench, not the tracker **\n");
    }

    TrackPlayer player(cfg);
    player.setTrack(track.samples.data(), track.samples.size());
    player.setGrid(track.grid.beats.data(), track.grid.beats.size());

    Device device;
    TrackState state;
    state.player = &player;
    if (!device.start(trackCallback, &state, track.rate, false, options.device_name)) {
        std::fprintf(stderr, "tiktak: %s\n", device.error().c_str());
        return 1;
    }
    if (!player.start(kStartDelaySec, 0)) {
        device.stop();
        std::fprintf(stderr, "tiktak: could not start the player\n");
        return 2;
    }

    std::printf("\ntap any key on the beat. q or Enter to stop.\n\n");

    // The player's own position, not the wall clock: it is the same number the
    // grid is expressed in, so a tap and a beat are directly comparable and no
    // device latency has to be guessed at. What it does include is the
    // listener's reaction time, which is why the report removes a constant
    // offset before judging anything.
    std::vector<double> taps;
    KeyReader keys;
    const auto started = std::chrono::steady_clock::now();
    while (player.running() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                   .count() < options.seconds) {
        const int key = keys.poll();
        if (key < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (key == 'q' || key == 'Q' || key == '\r' || key == '\n') break;
        taps.push_back(player.positionSec());
        std::printf("\r%zu taps  ", taps.size());
        std::fflush(stdout);
    }
    player.stop();
    device.stop();
    std::printf("\n\n");

    const TapComparison result =
        compareTaps(taps, track.grid.beats, options.tap_tolerance_sec);
    std::printf("taps                 %zu\n", result.taps);
    if (result.taps < 4) {
        std::printf("%s\n", result.verdict.c_str());
        return 1;
    }
    std::printf("your pulse           %.1f BPM\n", 60.0 / result.tap_period_sec);
    std::printf("the grid's pulse     %.1f BPM   (ratio %.2f)\n",
                60.0 / result.grid_period_sec, result.octave_ratio);
    std::printf("you tapped           %+.0f ms from the nearest beat "
                "(spread %.0f ms)\n",
                result.median_offset_sec * 1000.0,
                result.offset_spread_sec * 1000.0);
    std::printf("within %.0f ms       %zu of %zu raw, %.0f%% once your own "
                "offset is removed\n",
                options.tap_tolerance_sec * 1000.0, result.matched, result.taps,
                result.matched_after_offset * 100.0);
    std::printf("\n%s\n", result.verdict.c_str());
    return 0;
}

#else  // !TIKTAK_HAVE_DECODE

int cmdTap(const Options&) {
    std::fprintf(stderr,
                 "tiktak: this build has no decoder — rebuild with -DTIKTAK_BUILD_DECODE=ON\n");
    return 2;
}

int cmdTrack(const Options&) {
    std::fprintf(stderr,
                 "tiktak: this build has no decoder — rebuild with -DTIKTAK_BUILD_DECODE=ON\n");
    return 2;
}

#endif

}  // namespace tiktak::desktop
