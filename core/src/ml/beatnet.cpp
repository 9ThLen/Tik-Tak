#include "ml/beatnet.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace tiktak::ml {
namespace {

std::uint32_t readU32(const std::uint8_t* p) {
    // Assembled arithmetically rather than memcpy'd so a big-endian target
    // reads the little-endian file correctly instead of quietly loading
    // byte-swapped weights, which would look like a model that never converged.
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

float readF32(const std::uint8_t* p) {
    const std::uint32_t bits = readU32(p);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

// ----------------------------------------------------------------- weights --

bool BeatNetWeights::load(const void* data, std::size_t bytes) {
    storage_.clear();
    if (data == nullptr || bytes != kFileBytes) return false;

    const auto* raw = static_cast<const std::uint8_t*>(data);
    if (std::memcmp(raw, "TTBN", 4) != 0) return false;

    const std::uint32_t expected[7] = {
        1, kFeatures, kConvChannels, kKernel, kHidden, kLayers, kClasses,
    };
    for (std::size_t i = 0; i < 7; ++i) {
        if (readU32(raw + 4 + i * sizeof(std::uint32_t)) != expected[i]) return false;
    }

    storage_.resize(kParameters);
    const std::uint8_t* payload = raw + kHeaderBytes;
    for (std::size_t i = 0; i < kParameters; ++i) {
        storage_[i] = readF32(payload + i * sizeof(float));
    }

    // The order is models/export_beatnet.py's, and neither side looks anything
    // up by name: a file that disagrees is a wrong file, caught by the size and
    // the shape header above rather than negotiated with here.
    const float* p = storage_.data();
    auto take = [&p](std::size_t count) {
        const float* here = p;
        p += count;
        return here;
    };

    conv_weight = take(kConvChannels * kKernel);
    conv_bias = take(kConvChannels);
    linear0_weight = take(kHidden * kFlat);
    linear0_bias = take(kHidden);
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        lstm_weight_ih[layer] = take(4 * kHidden * kHidden);
        lstm_weight_hh[layer] = take(4 * kHidden * kHidden);
        lstm_bias_ih[layer] = take(4 * kHidden);
        lstm_bias_hh[layer] = take(4 * kHidden);
    }
    out_weight = take(kClasses * kHidden);
    out_bias = take(kClasses);

    assert(p == storage_.data() + kParameters);
    return true;
}

// ------------------------------------------------------------------- model --

BeatNetModel::BeatNetModel(const BeatNetWeights& weights)
    : weights_(weights),
      pooled_(BeatNetWeights::kFlat, 0.0f),
      layer_in_(BeatNetWeights::kHidden, 0.0f),
      gates_(4 * BeatNetWeights::kHidden, 0.0f),
      hidden_(BeatNetWeights::kLayers * BeatNetWeights::kHidden, 0.0f),
      cell_(BeatNetWeights::kLayers * BeatNetWeights::kHidden, 0.0f) {
    assert(weights.valid());
}

void BeatNetModel::reset() {
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
    std::fill(cell_.begin(), cell_.end(), 0.0f);
}

void BeatNetModel::forward(const float* features, float* probabilities) {
    assert(features != nullptr && probabilities != nullptr);
    constexpr std::size_t kHidden = BeatNetWeights::kHidden;
    constexpr std::size_t kKernel = BeatNetWeights::kKernel;
    constexpr std::size_t kConvOut = BeatNetWeights::kConvOut;
    constexpr std::size_t kPooled = BeatNetWeights::kPooled;

    // Convolution, ReLU and max-pooling in one sweep. Fusing them is not an
    // optimisation for its own sake: the intermediate is 263 values per channel
    // that nothing else reads, and materialising it would be the largest buffer
    // in the class.
    //
    // The flattened layout is channel-major, because that is what a PyTorch
    // view of (1, channels, positions) produces, and the linear layer's columns
    // were fitted in that order.
    for (std::size_t c = 0; c < BeatNetWeights::kConvChannels; ++c) {
        const float* kernel = weights_.conv_weight + c * kKernel;
        const float bias = weights_.conv_bias[c];
        for (std::size_t m = 0; m < kPooled; ++m) {
            float best = 0.0f;  // ReLU's floor, so an all-negative pair pools to zero
            for (std::size_t half = 0; half < 2; ++half) {
                const std::size_t j = 2 * m + half;
                float sum = bias;
                for (std::size_t t = 0; t < kKernel; ++t) sum += kernel[t] * features[j + t];
                if (sum > best) best = sum;
            }
            pooled_[c * kPooled + m] = best;
        }
    }
    // 263 positions pool into 131 pairs and the last position is dropped, as
    // PyTorch drops it: max_pool1d without ceil_mode ignores a ragged tail.
    static_assert(kConvOut == 2 * kPooled + 1, "pooling assumes one dropped tail position");

    for (std::size_t o = 0; o < kHidden; ++o) {
        const float* row = weights_.linear0_weight + o * BeatNetWeights::kFlat;
        float sum = weights_.linear0_bias[o];
        for (std::size_t k = 0; k < BeatNetWeights::kFlat; ++k) sum += row[k] * pooled_[k];
        layer_in_[o] = sum;
    }

    for (std::size_t layer = 0; layer < BeatNetWeights::kLayers; ++layer) {
        float* h = hidden_.data() + layer * kHidden;
        float* c = cell_.data() + layer * kHidden;
        const float* w_ih = weights_.lstm_weight_ih[layer];
        const float* w_hh = weights_.lstm_weight_hh[layer];

        for (std::size_t g = 0; g < 4 * kHidden; ++g) {
            const float* row_ih = w_ih + g * kHidden;
            const float* row_hh = w_hh + g * kHidden;
            float sum = weights_.lstm_bias_ih[layer][g] + weights_.lstm_bias_hh[layer][g];
            for (std::size_t k = 0; k < kHidden; ++k) {
                sum += row_ih[k] * layer_in_[k] + row_hh[k] * h[k];
            }
            gates_[g] = sum;
        }

        // PyTorch packs the gates input, forget, cell, output in that order.
        for (std::size_t k = 0; k < kHidden; ++k) {
            const float in = sigmoid(gates_[k]);
            const float forget = sigmoid(gates_[kHidden + k]);
            const float candidate = std::tanh(gates_[2 * kHidden + k]);
            const float out = sigmoid(gates_[3 * kHidden + k]);
            c[k] = forget * c[k] + in * candidate;
            h[k] = out * std::tanh(c[k]);
        }
        std::copy(h, h + kHidden, layer_in_.begin());
    }

    float logits[BeatNetWeights::kClasses];
    float largest = 0.0f;
    for (std::size_t o = 0; o < BeatNetWeights::kClasses; ++o) {
        const float* row = weights_.out_weight + o * kHidden;
        float sum = weights_.out_bias[o];
        for (std::size_t k = 0; k < kHidden; ++k) sum += row[k] * layer_in_[k];
        logits[o] = sum;
        if (o == 0 || sum > largest) largest = sum;
    }

    // Softmax across the three classes: a beat is a beat *instead of* nothing
    // happening, not in addition to it. Shifted by the largest logit, which
    // changes nothing mathematically and keeps exp() away from overflow.
    float total = 0.0f;
    for (std::size_t o = 0; o < BeatNetWeights::kClasses; ++o) {
        probabilities[o] = std::exp(logits[o] - largest);
        total += probabilities[o];
    }
    for (std::size_t o = 0; o < BeatNetWeights::kClasses; ++o) probabilities[o] /= total;
}

// ---------------------------------------------------------------- features --

BeatNetFeatures::BeatNetFeatures(double sampleRate)
    : ratio_(sampleRate / kModelRate),
      dft_(kFrameSize),
      bank_(kFrameSize, kModelRate, 24, 30.0, 17000.0, 440.0),
      buffer_(kFrameSize, 0.0f),
      windowed_(kFrameSize, 0.0f),
      spectrum_(dft_.spectrumSize(), 0.0f),
      previous_(kFilters, 0.0f),
      features_(kFeatures, 0.0f) {
    assert(sampleRate > 0.0);
    assert(bank_.bands() == kFilters);
    assert(kModelRate / sampleRate < static_cast<double>(kMaxPerSample));

    // Symmetric Hann, not the periodic one dsp::hannWindow builds. The
    // difference is one sample in 1411 and it would be invisible in any
    // spectrogram, but the network was trained through numpy's symmetric
    // window and the point of transcribing a front end is that it is the same
    // front end.
    window_.resize(kFrameSize);
    for (std::size_t n = 0; n < kFrameSize; ++n) {
        window_[n] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 *
                                 static_cast<double>(n) /
                                 static_cast<double>(kFrameSize - 1)));
    }

    reset();
}

void BeatNetFeatures::reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    std::fill(previous_.begin(), previous_.end(), 0.0f);
    std::fill(features_.begin(), features_.end(), 0.0f);
    // Frame zero is centred on sample zero, so it starts life half full of the
    // silence that precedes the stream.
    fill_ = kFrameSize / 2;
    frame_index_ = 0;
    seen_frame_ = false;
    input_index_ = 0;
    output_index_ = 0;
    previous_sample_ = 0.0f;
}

std::size_t BeatNetFeatures::resample(float sample, float* out) {
    const std::size_t index = input_index_++;
    if (index == 0) {
        previous_sample_ = sample;
        return 0;
    }

    // Every output whose position falls in [index-1, index] can now be
    // interpolated, and none of them could be before this sample arrived.
    std::size_t produced = 0;
    while (produced < kMaxPerSample) {
        const double position = static_cast<double>(output_index_) * ratio_;
        if (position > static_cast<double>(index)) break;
        const double fraction = position - static_cast<double>(index - 1);
        if (fraction < 0.0) break;
        out[produced++] = previous_sample_ +
                          static_cast<float>(fraction) * (sample - previous_sample_);
        ++output_index_;
    }
    previous_sample_ = sample;
    return produced;
}

bool BeatNetFeatures::accept(float sample) {
    buffer_[fill_++] = sample;
    if (fill_ < kFrameSize) return false;

    for (std::size_t n = 0; n < kFrameSize; ++n) windowed_[n] = buffer_[n] * window_[n];
    dft_.magnitude(windowed_.data(), spectrum_.data());
    bank_.apply(spectrum_.data(), features_.data());

    for (std::size_t b = 0; b < kFilters; ++b) {
        features_[b] = std::log10(features_[b] + 1.0f);
        // The difference is to one frame back, which is not a free choice: the
        // published pipeline takes it half a window's half-power width away,
        // and for a Hann window of 1411 with a 441 hop that rounds to exactly
        // one frame. A different frame size or hop would not round there.
        const float delta = features_[b] - previous_[b];
        features_[kFilters + b] = seen_frame_ && delta > 0.0f ? delta : 0.0f;
    }
    return true;
}

void BeatNetFeatures::advance() {
    std::copy(features_.begin(), features_.begin() + kFilters, previous_.begin());
    seen_frame_ = true;
    ++frame_index_;

    const std::size_t keep = kFrameSize - kHopSize;
    std::memmove(buffer_.data(), buffer_.data() + kHopSize, keep * sizeof(float));
    fill_ = keep;
}

// ------------------------------------------------------------- activation --

BeatNetActivation::BeatNetActivation(double sampleRate, const BeatNetWeights& weights)
    : features_(sampleRate), model_(weights) {}

void BeatNetActivation::reset() {
    features_.reset();
    model_.reset();
}

}  // namespace tiktak::ml
