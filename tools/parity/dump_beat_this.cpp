// Runs Beat This! end to end through the core's front end and ONNX Runtime, so
// the C++ can be diffed against research/eval/beat_this_onnx.py.
//
//   dump_beat_this <input.f32> <model.onnx>
//
// Input is raw 32-bit float mono at 22050 Hz. Output is CSV: time,beat,downbeat
// — logits, not probabilities, because that is what the peak picker reads.
#include "ml/beat_this.hpp"
#include "ml/beat_this_session.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <input.f32> <model.onnx>\n", argv[0]);
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
    const std::size_t mels = tiktak::ml::BeatThisFeatures::kMels;
    const std::size_t frames = mel.size() / mels;

    tiktak::ml::BeatThisSession session;
    if (!session.open(argv[2])) {
        std::fprintf(stderr, "%s\n", session.reason().c_str());
        return 1;
    }

    const auto activations = session.run(mel.data(), frames, mels);

    if (argc > 3 && std::string(argv[3]) == "--beats") {
        const auto grid = tiktak::ml::pickBeats(activations.beat.data(),
                                                activations.downbeat.data(),
                                                activations.beat.size());
        std::printf("beats\n");
        for (double t : grid.beats) std::printf("%.17g\n", t);
        std::printf("\ndownbeats\n");
        for (double t : grid.downbeats) std::printf("%.17g\n", t);
        return 0;
    }

    std::printf("time_sec,beat,downbeat\n");
    for (std::size_t f = 0; f < activations.beat.size(); ++f) {
        std::printf("%.17g,%.9g,%.9g\n",
                    static_cast<double>(f) / tiktak::ml::BeatThisFeatures::kFrameRate,
                    activations.beat[f], activations.downbeat[f]);
    }
    return 0;
}
