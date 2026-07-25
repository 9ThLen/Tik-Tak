// Dumps the core's ODF for a raw audio file, so the C++ can be diffed against
// the Python reference in research/tiktak/odf.py.
//
// The whole "prototype in Python, port to C++" plan rests on the two staying
// equivalent. Without a check like this, they drift apart quietly and every
// metric measured in research stops describing what ships.
//
//   dump_odf <input.f32> <sample_rate> [block_size]
//
// Input is raw 32-bit float mono, native endianness. Output is CSV on stdout:
// time_sec,full,low,high
#include "tiktak/tiktak.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s <input.f32> <sample_rate> [block_size]\n", argv[0]);
        return 2;
    }

    const char* path = argv[1];
    const double sample_rate = std::atof(argv[2]);
    // Deliberately not a multiple of the hop by default: a device hands over
    // whatever block size it likes, and the framing must not care.
    const std::size_t block = argc == 4 ? std::strtoul(argv[3], nullptr, 10) : 137;

    if (sample_rate <= 0.0 || block == 0) {
        std::fprintf(stderr, "bad sample rate or block size\n");
        return 2;
    }

    std::FILE* file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
    if (std::fread(samples.data(), sizeof(float), samples.size(), file) != samples.size()) {
        std::fprintf(stderr, "short read from %s\n", path);
        std::fclose(file);
        return 1;
    }
    std::fclose(file);

    tt_odf_config config;
    tt_odf_config_defaults(&config, sample_rate);

    tt_status status = TT_OK;
    tt_odf* odf = tt_odf_create(&config, &status);
    if (!odf) {
        std::fprintf(stderr, "tt_odf_create failed: %s\n", tt_status_string(status));
        return 1;
    }

    std::vector<tt_odf_frame> frames(block / config.hop_size + 4);
    std::printf("time_sec,full,low,high\n");

    for (std::size_t pos = 0; pos < samples.size(); pos += block) {
        const std::size_t take = std::min(block, samples.size() - pos);

        std::size_t dropped = 0;
        const std::size_t written = tt_odf_process(odf, samples.data() + pos, take,
                                                   frames.data(), frames.size(), &dropped);
        if (dropped != 0) {
            std::fprintf(stderr, "dropped %zu frames — output buffer too small\n", dropped);
            tt_odf_destroy(odf);
            return 1;
        }

        for (std::size_t i = 0; i < written; ++i) {
            std::printf("%.17g,%.9g,%.9g,%.9g\n", frames[i].time_sec, frames[i].full,
                        frames[i].low, frames[i].high);
        }
    }

    tt_odf_destroy(odf);
    return 0;
}
