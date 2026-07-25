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
#include "wav.hpp"

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
        "\n"
        "Options:\n"
        "  --bpm N            tempo (120)\n"
        "  --beats N          beats per bar (4)\n"
        "  --sub N            subdivisions per beat, 1 = beats only (1)\n"
        "  --seconds N        how long to run (10)\n"
        "  --rate N           sample rate; 0 takes the device's own (0)\n"
        "  --latency-ms N     output latency to compensate (0) — measure it first\n"
        "  --lookahead-ms N   how far ahead beats are handed out (250)\n"
        "  --device NAME      playback device, as printed by `devices`\n"
        "  -o, --out PATH     where to write (render, measure)\n"
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

}  // namespace tiktak::desktop
