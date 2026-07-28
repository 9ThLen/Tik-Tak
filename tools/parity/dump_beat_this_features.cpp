// Dumps the core's Beat This! log-mel spectrogram, so it can be diffed against
// research/eval/beat_this_onnx.py.
//
// The exported ONNX begins after the front end, so this spectrogram is ours to
// compute and ours to get wrong. A window that is symmetric instead of
// periodic, or the 1127*ln mel scale instead of Slaney's, still produces
// something that looks like a spectrogram and quietly feeds the network bands
// it was never fitted to.
//
//   dump_beat_this_features <input.f32> [frames]
//
// Input is raw 32-bit float mono at 22050 Hz. Output is CSV: one row per
// frame, 128 values.
#include "ml/beat_this.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input.f32> [frames]\n", argv[0]);
        return 2;
    }

    std::FILE* file = std::fopen(argv[1], "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
    if (std::fread(samples.data(), sizeof(float), samples.size(), file) != samples.size()) {
        std::fprintf(stderr, "short read\n");
        std::fclose(file);
        return 1;
    }
    std::fclose(file);

    tiktak::ml::BeatThisFeatures features;
    const std::vector<float> mel = features.compute(samples.data(), samples.size());
    const std::size_t frames = mel.size() / tiktak::ml::BeatThisFeatures::kMels;

    const std::size_t limit = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : frames;
    for (std::size_t f = 0; f < frames && f < limit; ++f) {
        const float* row = mel.data() + f * tiktak::ml::BeatThisFeatures::kMels;
        for (std::size_t m = 0; m < tiktak::ml::BeatThisFeatures::kMels; ++m) {
            std::printf(m == 0 ? "%.9g" : ",%.9g", row[m]);
        }
        std::printf("\n");
    }
    return 0;
}
