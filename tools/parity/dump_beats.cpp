// Dumps the core's offline beat analysis for a raw audio file, so the C++ can
// be diffed against the Python reference in research/tiktak/.
//
// Companion to dump_odf. That one checks the front-end frame by frame; this one
// checks what the whole pipeline concludes — tempo, confidence and the beat
// grid. The two failure modes are different: the ODF can agree to seven digits
// while the tracker still picks a different beat sequence, because the dynamic
// programme makes discrete choices that a tiny difference can tip.
//
//   dump_beats <input.f32> <sample_rate> [block_size] [bpm_hint]
//
// Input is raw 32-bit float mono, native endianness. Output on stdout is a
// header block of "key=value" lines, then a blank line, then one beat time per
// line.
#include "tiktak/tiktak.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s <input.f32> <sample_rate> [block_size] [bpm_hint]\n", argv[0]);
        return 2;
    }

    const char* path = argv[1];
    const double sample_rate = std::atof(argv[2]);
    // Deliberately not a multiple of the hop by default: a decoder hands over
    // whatever block size it likes, and the framing must not care.
    const std::size_t block = argc >= 4 ? std::strtoul(argv[3], nullptr, 10) : 137;
    const double bpm_hint = argc >= 5 ? std::atof(argv[4]) : 0.0;

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

    tt_offline_config config;
    tt_offline_config_defaults(&config, sample_rate);
    config.bpm_hint = bpm_hint;

    tt_status status = TT_OK;
    tt_offline* offline = tt_offline_create(&config, &status);
    if (!offline) {
        std::fprintf(stderr, "tt_offline_create failed: %s\n", tt_status_string(status));
        return 1;
    }

    for (std::size_t pos = 0; pos < samples.size(); pos += block) {
        const std::size_t take = std::min(block, samples.size() - pos);
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

    std::printf("bpm=%.17g\n", tt_offline_bpm(offline));
    std::printf("estimated_bpm=%.17g\n", tt_offline_estimated_bpm(offline));
    std::printf("confidence=%.17g\n", tt_offline_confidence(offline));
    std::printf("frames=%zu\n", tt_offline_frame_count(offline));

    tt_tempo_candidate candidates[3];
    const std::size_t candidate_count = tt_offline_tempo_candidates(offline, candidates, 3);
    for (std::size_t i = 0; i < candidate_count; ++i) {
        std::printf("candidate%zu=%.17g,%.17g\n", i, candidates[i].bpm, candidates[i].strength);
    }

    std::vector<double> beats(tt_offline_beat_count(offline));
    if (!beats.empty()) tt_offline_beats(offline, beats.data(), beats.size());

    std::printf("\n");
    for (double beat : beats) std::printf("%.17g\n", beat);

    tt_offline_destroy(offline);
    return 0;
}
