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
    // The arrangement of chunks and the stitching are the core's, shared with
    // every other way of running the network; only one chunk is ours.
    Activations out;
    if (!isOpen()) return out;
    runChunked(spectrogram, frames, mels, &BeatThisSession::runChunk, this, out);
    return out;
}

bool BeatThisSession::runChunk(void* context, const float* spectrogram, std::size_t frames,
                               std::size_t mels, float* beat_logits, float* downbeat_logits) {
    auto* self = static_cast<BeatThisSession*>(context);
    if (self == nullptr || !self->isOpen() || spectrogram == nullptr || frames == 0 ||
        mels == 0 || beat_logits == nullptr || downbeat_logits == nullptr) {
        return false;
    }
    try {
        Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<const char*, 1> inputs{"input_spectrogram"};
        const std::array<const char*, 2> outputs{"beat", "downbeat"};
        const std::array<std::int64_t, 3> shape{1, static_cast<std::int64_t>(frames),
                                                static_cast<std::int64_t>(mels)};
        // The runtime takes a mutable pointer for inputs it never writes.
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(spectrogram), frames * mels, shape.data(), shape.size());

        std::vector<Ort::Value> result =
            self->impl_->session->Run(Ort::RunOptions{nullptr}, inputs.data(), &tensor, 1,
                                      outputs.data(), outputs.size());

        const std::size_t produced = static_cast<std::size_t>(
            result[0].GetTensorTypeAndShapeInfo().GetElementCount());
        if (produced != frames ||
            static_cast<std::size_t>(result[1].GetTensorTypeAndShapeInfo().GetElementCount()) !=
                frames) {
            self->reason_ = "the model returned a different number of frames than it was given";
            return false;
        }
        std::memcpy(beat_logits, result[0].GetTensorData<float>(), frames * sizeof(float));
        std::memcpy(downbeat_logits, result[1].GetTensorData<float>(), frames * sizeof(float));
    } catch (const Ort::Exception& error) {
        self->reason_ = std::string("inference failed: ") + error.what();
        return false;
    }
    return true;
}

}  // namespace tiktak::ml
