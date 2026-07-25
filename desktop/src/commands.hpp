#pragma once

#include <string>
#include <vector>

namespace tiktak::desktop {

// Everything the metronome commands share. The defaults are the ones a musician
// would expect, not zeroes, so `tiktak play` on its own does something useful.
struct Options {
    double bpm = 120.0;
    int beats_per_bar = 4;
    int subdivisions = 1;
    double seconds = 10.0;
    double sample_rate = 0.0;        // 0 = whatever the device prefers
    double output_latency_sec = 0.0; // what to compensate; measure it first
    double lookahead_sec = 0.25;
    std::string device_name;
    std::string output_path;
};

// Returns false and fills `error` on a bad argument, rather than guessing.
bool parseOptions(const std::vector<std::string>& args, Options& options,
                  std::string& error);

int cmdDevices();
int cmdRender(const Options& options);
int cmdPlay(const Options& options);
int cmdMeasure(const Options& options);

void printUsage();

}  // namespace tiktak::desktop
