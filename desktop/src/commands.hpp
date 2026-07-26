#pragma once

#include <string>
#include <vector>

namespace tiktak::desktop {

// Everything the metronome commands share. The defaults are the ones a musician
// would expect, not zeroes, so `tiktak play` on its own does something useful.
struct Options {
    double bpm = 120.0;
    int beats_per_bar = 4;
    // Whether --beats was actually given. `track` detects the meter from the
    // audio, and needs to tell "the user asked for four" apart from "nobody
    // said anything and four is the default".
    bool beats_per_bar_given = false;
    int subdivisions = 1;
    double seconds = 10.0;
    double sample_rate = 0.0;        // 0 = whatever the device prefers
    double output_latency_sec = 0.0; // what to compensate; measure it first
    double lookahead_sec = 0.25;
    std::string device_name;
    std::string output_path;

    // The track command (Phase 4): play an analysed file with the click on its
    // beat grid.
    std::string track_path;          // the positional argument
    int count_in = 4;                // count-in beats before the music
    long long from_bar = 0;          // where to start
    long long loop_from = -1;        // --loop A:B, bars; -1 = no loop
    long long loop_to = -1;
    double hint_bpm = 0.0;           // manual-mode tempo hint; 0 = estimate

    // `listen` (Phase 6): the user's own tempo. The tracker then only looks for
    // the phase, and plays nothing until it has found one. 0 = track the tempo.
    double manual_bpm = 0.0;
    bool no_click = false;           // the track alone, cache still exercised
    bool no_cache = false;           // force a fresh analysis
};

// Returns false and fills `error` on a bad argument, rather than guessing.
bool parseOptions(const std::vector<std::string>& args, Options& options,
                  std::string& error);

int cmdDevices();
int cmdRender(const Options& options);
int cmdPlay(const Options& options);
int cmdMeasure(const Options& options);
int cmdTrack(const Options& options);
int cmdListen(const Options& options);

void printUsage();

}  // namespace tiktak::desktop
