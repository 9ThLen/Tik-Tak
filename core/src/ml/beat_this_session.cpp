#include "ml/beat_this_session.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>

namespace tiktak::ml {

struct BeatThisSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "tiktak"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
};

BeatThisSession::BeatThisSession() : impl_(std::make_unique<Impl>()) {
    impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

BeatThisSession::~BeatThisSession() = default;

bool BeatThisSession::isOpen() const { return impl_ && impl_->session != nullptr; }

bool BeatThisSession::open(const std::string& model_path) {
    reason_.clear();
    impl_->session.reset();
    try {
        // ONNX Runtime takes the platform's native path type — wchar_t on
        // Windows, char everywhere else — and std::filesystem::path is the one
        // conversion that is both. It also widens through the same narrow
        // encoding the path arrived in, which hand-rolled UTF-8 widening would
        // get wrong for a non-ASCII path off the command line.
        const std::filesystem::path native_path(model_path);
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, native_path.c_str(), impl_->options);
    } catch (const Ort::Exception& error) {
        // Ordinary, not exceptional: the model is fetched separately and is
        // deliberately absent from a fresh checkout.
        reason_ = std::string("could not open ") + model_path + ": " + error.what();
        return false;
    }
    return true;
}

BeatThisSession::Activations BeatThisSession::run(const float* spectrogram,
                                                  std::size_t frames,
                                                  std::size_t mels) {
    Activations out;
    if (!isOpen() || spectrogram == nullptr || frames == 0 || mels == 0) return out;

    // -1000 is the reference's "no chunk has spoken for this frame yet". It
    // survives into the result only if the aggregation below has a hole, which
    // the choice of starts is arranged to prevent — so a -1000 in the output is
    // a bug rather than a value, and is worth being able to see.
    out.beat.assign(frames, -1000.0f);
    out.downbeat.assign(frames, -1000.0f);

    // Starts run from -border, so the first real frame is never at a chunk
    // edge, and the last start is pulled back to cover the tail exactly once.
    // That is what makes keep-first aggregation total rather than merely
    // usually total.
    const std::size_t step = kChunkFrames - 2 * kBorderFrames;
    std::vector<long long> starts;
    for (long long s = -static_cast<long long>(kBorderFrames);
         s < static_cast<long long>(frames) - static_cast<long long>(kBorderFrames);
         s += static_cast<long long>(step)) {
        starts.push_back(s);
    }
    if (starts.empty()) starts.push_back(-static_cast<long long>(kBorderFrames));
    if (frames > step) {
        starts.back() = static_cast<long long>(frames) -
                        static_cast<long long>(kChunkFrames - kBorderFrames);
    }

    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<const char*, 1> inputs{"input_spectrogram"};
    const std::array<const char*, 2> outputs{"beat", "downbeat"};

    std::vector<float> chunk(kChunkFrames * mels);
    std::vector<std::vector<float>> beat_chunks(starts.size());
    std::vector<std::vector<float>> downbeat_chunks(starts.size());

    for (std::size_t c = 0; c < starts.size(); ++c) {
        std::fill(chunk.begin(), chunk.end(), 0.0f);
        for (std::size_t j = 0; j < kChunkFrames; ++j) {
            const long long source = starts[c] + static_cast<long long>(j);
            if (source < 0 || source >= static_cast<long long>(frames)) continue;
            std::memcpy(chunk.data() + j * mels,
                        spectrogram + static_cast<std::size_t>(source) * mels,
                        mels * sizeof(float));
        }

        const std::array<std::int64_t, 3> shape{
            1, static_cast<std::int64_t>(kChunkFrames), static_cast<std::int64_t>(mels)};
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory, chunk.data(), chunk.size(), shape.data(), shape.size());

        std::vector<Ort::Value> result =
            impl_->session->Run(Ort::RunOptions{nullptr}, inputs.data(), &tensor, 1,
                                outputs.data(), outputs.size());

        const float* beat = result[0].GetTensorData<float>();
        const float* downbeat = result[1].GetTensorData<float>();
        const std::size_t produced = static_cast<std::size_t>(
            result[0].GetTensorTypeAndShapeInfo().GetElementCount());
        beat_chunks[c].assign(beat, beat + produced);
        downbeat_chunks[c].assign(downbeat, downbeat + produced);
    }

    // Backwards, so an earlier chunk overwrites a later one where they overlap.
    // The reference calls this keep_first, and the direction is the whole of
    // it: written forwards, every overlap would keep the *worse* answer, the
    // one computed nearer a chunk edge.
    for (std::size_t i = starts.size(); i-- > 0;) {
        const std::vector<float>& beat = beat_chunks[i];
        const std::vector<float>& downbeat = downbeat_chunks[i];
        std::size_t lo = kBorderFrames;
        std::size_t hi = beat.size() > 2 * kBorderFrames ? beat.size() - kBorderFrames : 0;
        if (beat.size() < 2 * kBorderFrames) {
            lo = 0;
            hi = beat.size();
        }
        for (std::size_t j = lo; j < hi; ++j) {
            const long long target = starts[i] + static_cast<long long>(j);
            if (target < 0 || target >= static_cast<long long>(frames)) continue;
            out.beat[static_cast<std::size_t>(target)] = beat[j];
            out.downbeat[static_cast<std::size_t>(target)] = downbeat[j];
        }
    }
    return out;
}

}  // namespace tiktak::ml
