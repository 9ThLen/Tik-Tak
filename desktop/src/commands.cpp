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
std::ptrdiff_t findClick(const float* window, std::size_t frames, double rate) {
    if (frames == 0) return -1;

    (void)rate;

    double peak = 0.0;
    for (std::size_t i = 0; i < frames; ++i) {
        peak = std::max(peak, std::fabs(static_cast<double>(window[i])));
    }
    if (peak < 1e-3) return -1;   // nothing arrived at all: no click, not a quiet one

    // Relative to this window's own peak, so it neither needs a calibrated
    // input level nor fires on room noise. A click reaches a third of its peak
    // within two samples of starting, which is forty microseconds — three
    // orders of magnitude under the jitter being measured.
    const double threshold = 0.3 * peak;
    for (std::size_t i = 0; i < frames; ++i) {
        if (std::fabs(static_cast<double>(window[i])) > threshold) {
            return static_cast<std::ptrdiff_t>(i);
        }
    }
    return -1;
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
        "It reports how far the bar line it picked stands above the next best\n"
        "place to put it; below a margin of 0.25 it declines to accent anything,\n"
        "because an accent on the wrong beat is harder to play to than none.\n"
        "Passing --beats overrides the meter it found — the number you type is\n"
        "an assertion about the music, the same way --hint is.\n"
        "  --no-click         the track alone, no metronome\n"
        "  --no-cache         re-analyse even when the beat grid is cached\n"
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

    std::vector<double> offsets;
    for (int k = 0;; ++k) {
        const double submitted = kStartDelaySec + beat_sec * k;
        const auto at = static_cast<std::size_t>(submitted * rate);
        if (at + window_frames >= recorded) break;

        const std::ptrdiff_t found = findClick(recording.data() + at, window_frames, rate);
        if (found < 0) continue;
        offsets.push_back(static_cast<double>(found) / rate);
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
    // The margin is what decides whether to use it at all: it says how far the
    // chosen bar line stands above the next best place to put it, and an accent
    // on the wrong beat is worse for a player to follow than no accent. Below
    // the threshold the harness says so and falls back to counting from the
    // first beat, which is at least honestly arbitrary.
    //
    // 0.25 is a placeholder and not a calibration — nothing has been measured
    // that says it is the right number. research/eval/downbeat_benchmark.py is
    // what replaces it: it sweeps the threshold, reports coverage against the
    // wrong-accent rate, and picks the most generous threshold inside a wrong
    // rate budget. That needs 30–50 annotated recordings, which do not exist
    // yet; see research/eval/README.md.
    constexpr double kMinMargin = 0.25;
    int beats_per_bar = options.beats_per_bar;
    int downbeat_offset = 0;

    if (!grid.beats.empty() && grid.beats_per_bar > 0) {
        std::printf("bar lines: %d beats to the bar (strength %.2f, margin %.2f)%s\n",
                    grid.beats_per_bar, grid.downbeat_strength, grid.downbeat_margin,
                    grid.downbeat_margin < kMinMargin ? " — too close to call" : "");
    } else if (!grid.beats.empty()) {
        std::printf("bar lines: none found — the track is too short to repeat a bar\n");
    }

    if (options.beats_per_bar_given) {
        // An explicit --beats is the user's assertion about the music and
        // outranks the analysis, exactly as an explicit --bpm does.
        if (grid.beats_per_bar > 0 && grid.beats_per_bar != options.beats_per_bar) {
            std::printf("using --beats %d over the %d the audio suggests\n",
                        options.beats_per_bar, grid.beats_per_bar);
        }
    } else if (grid.beats_per_bar > 0 && grid.downbeat_margin >= kMinMargin &&
               !grid.downbeats.empty()) {
        beats_per_bar = grid.beats_per_bar;
        const auto first = std::lower_bound(grid.beats.begin(), grid.beats.end(),
                                            grid.downbeats.front() - 1e-9);
        downbeat_offset = static_cast<int>(first - grid.beats.begin());
    }

    PlayerConfig cfg;
    cfg.sample_rate = rate;
    cfg.click.sample_rate = rate;
    cfg.beats_per_bar = beats_per_bar;
    cfg.downbeat_offset = downbeat_offset;
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

#else  // !TIKTAK_HAVE_DECODE

int cmdTrack(const Options&) {
    std::fprintf(stderr,
                 "tiktak: this build has no decoder — rebuild with -DTIKTAK_BUILD_DECODE=ON\n");
    return 2;
}

#endif

}  // namespace tiktak::desktop
