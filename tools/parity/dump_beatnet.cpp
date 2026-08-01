// Dumps the core's BeatNet features and activations, so the C++ can be diffed
// against the Python reference in research/eval/beatnet_onnx.py.
//
// The reference is not ours: it is a transcription of madmom's feature pipeline
// and the published network, and its constants are load-bearing in a way that
// nothing else in this project's DSP is — a filterbank one band too wide, or a
// window one sample off, does not look like a bug, it looks like a model that
// was never very good. So the two implementations are diffed the same way the
// ODF and the tracker are.
//
//   dump_beatnet <input.f32> <sample_rate> <weights.ttw> [block_size] [--features]
//
// Input is raw 32-bit float mono, native endianness. Output is CSV on stdout:
// time_sec,beat,downbeat — or, with --features, the 272 feature values per row.
//
// Private headers on purpose: the model is not in the C API yet, because
// nothing outside the core has asked for it in isolation. It is reached through
// the live tracker, which is where the shells want it.
#include "ml/beatnet.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool readAll(const char* path, std::vector<unsigned char>* out) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    out->resize(static_cast<std::size_t>(bytes));
    const bool ok = std::fread(out->data(), 1, out->size(), file) == out->size();
    std::fclose(file);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <input.f32> <sample_rate> <weights.ttw> "
                     "[block_size] [--features]\n",
                     argv[0]);
        return 2;
    }

    const double sample_rate = std::atof(argv[2]);
    std::size_t block = 137;
    bool dump_features = false;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--features") == 0) {
            dump_features = true;
        } else {
            block = std::strtoul(argv[i], nullptr, 10);
        }
    }
    if (sample_rate <= 0.0 || block == 0) {
        std::fprintf(stderr, "bad sample rate or block size\n");
        return 2;
    }

    std::vector<unsigned char> raw;
    if (!readAll(argv[1], &raw)) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    std::vector<float> samples(raw.size() / sizeof(float));
    std::memcpy(samples.data(), raw.data(), samples.size() * sizeof(float));

    std::vector<unsigned char> blob;
    if (!readAll(argv[3], &blob)) {
        std::fprintf(stderr, "cannot read %s\n", argv[3]);
        return 1;
    }
    tiktak::ml::BeatNetWeights weights;
    if (!weights.load(blob.data(), blob.size())) {
        std::fprintf(stderr, "%s is not a weight file this build can run\n", argv[3]);
        return 1;
    }

    if (dump_features) {
        tiktak::ml::BeatNetFeatures features(sample_rate);
        for (std::size_t pos = 0; pos < samples.size(); pos += block) {
            const std::size_t take = std::min(block, samples.size() - pos);
            features.process(samples.data() + pos, take,
                             [](const float* row, std::size_t count, double time_sec) {
                                 std::printf("%.17g", time_sec);
                                 for (std::size_t i = 0; i < count; ++i) {
                                     std::printf(",%.9g", row[i]);
                                 }
                                 std::printf("\n");
                             });
        }
        return 0;
    }

    tiktak::ml::BeatNetActivation activation(sample_rate, weights);
    std::printf("time_sec,beat,downbeat\n");
    // Fed in blocks the size a device would hand over, because the framing must
    // not depend on how the audio was cut up on the way in.
    for (std::size_t pos = 0; pos < samples.size(); pos += block) {
        const std::size_t take = std::min(block, samples.size() - pos);
        activation.process(samples.data() + pos, take,
                           [](double time_sec, double beat, double downbeat) {
                               std::printf("%.17g,%.9g,%.9g\n", time_sec, beat, downbeat);
                           });
    }
    return 0;
}
