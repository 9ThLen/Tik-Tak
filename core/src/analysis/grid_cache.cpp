#include "analysis/grid_cache.hpp"

#include <algorithm>
#include <cstring>

namespace tiktak::analysis {
namespace {

// ------------------------------------------------------------------ SHA-256 --
//
// Written out here rather than pulled in as a dependency because tiktak_core
// having no third-party dependencies is the property that lets it build for
// any platform without a vendoring discussion — see docs/PLAN.md §3. FIPS
// 180-4, pinned by a known-answer test in the suite.

struct Sha256 {
    std::uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::uint8_t block[64] = {};
    std::size_t block_len = 0;
    std::uint64_t total_len = 0;

    static std::uint32_t rotr(std::uint32_t x, unsigned n) {
        return (x >> n) | (x << (32 - n));
    }

    void compress(const std::uint8_t* p) {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
            0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
            0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
            0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
            0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
            0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
            0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
            0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t{p[4 * i]} << 24) | (std::uint32_t{p[4 * i + 1]} << 16) |
                   (std::uint32_t{p[4 * i + 2]} << 8) | std::uint32_t{p[4 * i + 3]};
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 =
                rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 =
                rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    void update(const std::uint8_t* data, std::size_t n) {
        total_len += n;
        while (n > 0) {
            const std::size_t take = std::min(n, sizeof(block) - block_len);
            std::memcpy(block + block_len, data, take);
            block_len += take;
            data += take;
            n -= take;
            if (block_len == sizeof(block)) {
                compress(block);
                block_len = 0;
            }
        }
    }

    void finish(std::uint8_t out[32]) {
        const std::uint64_t bit_len = total_len * 8;
        const std::uint8_t pad = 0x80;
        update(&pad, 1);
        const std::uint8_t zero = 0x00;
        while (block_len != 56) update(&zero, 1);
        std::uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) {
            len_be[i] = static_cast<std::uint8_t>(bit_len >> (56 - 8 * i));
        }
        // update() counts these into total_len, but bit_len is already frozen.
        update(len_be, 8);
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = static_cast<std::uint8_t>(state[i] >> 24);
            out[4 * i + 1] = static_cast<std::uint8_t>(state[i] >> 16);
            out[4 * i + 2] = static_cast<std::uint8_t>(state[i] >> 8);
            out[4 * i + 3] = static_cast<std::uint8_t>(state[i]);
        }
    }
};

// ------------------------------------------------------------------- format --

// Bumped whenever the analysis itself changes in a way that makes old grids
// wrong, not just whenever the byte layout changes: a cache of results from a
// better-tuned tracker is stale even if it still parses.
constexpr std::uint32_t kVersion = 3;

constexpr std::uint8_t kMagic[4] = {'T', 'T', 'G', 'R'};

// Everything is written little-endian byte by byte, so the blob a phone writes
// is the blob a desktop reads, whatever either of them is running on.
void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

void put64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

void putF64(std::vector<std::uint8_t>& out, double v) {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "IEEE 754 double expected");
    std::memcpy(&bits, &v, sizeof(bits));
    put64(out, bits);
}

std::uint32_t get32(const std::uint8_t* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= std::uint32_t{p[i]} << (8 * i);
    return v;
}

std::uint64_t get64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= std::uint64_t{p[i]} << (8 * i);
    return v;
}

double getF64(const std::uint8_t* p) {
    const std::uint64_t bits = get64(p);
    double v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// FNV-1a: guards the payload against truncation and bit rot, nothing more. The
// content key needs collision resistance; a checksum only needs to notice that
// the bytes read back are not the bytes written.
std::uint64_t fnv1a(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// The fingerprint folds in every config field that shapes the result. A field
// added to any of these structs must be added here, which is why the whole
// config is spelled out rather than hashed as raw struct memory — padding
// bytes aside, hashing memory would make the fingerprint depend on layout.
std::uint64_t fingerprint(const OfflineConfig& c) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(24 * 8);
    putF64(bytes, c.odf.sampleRate);
    put64(bytes, c.odf.frameSize);
    put64(bytes, c.odf.hopSize);
    put64(bytes, c.odf.melBands);
    putF64(bytes, c.odf.melMinHz);
    putF64(bytes, c.odf.melMaxHz);
    putF64(bytes, c.odf.lowBandHz);
    putF64(bytes, c.odf.highBandHz);
    put64(bytes, c.odf.whitening ? 1 : 0);
    putF64(bytes, c.odf.whiteningTau);
    putF64(bytes, c.odf.whiteningStrength);
    putF64(bytes, c.odf.whiteningFloorRel);
    putF64(bytes, c.tempo.min_bpm);
    putF64(bytes, c.tempo.max_bpm);
    putF64(bytes, c.tempo.prior_centre_bpm);
    putF64(bytes, c.tempo.prior_width_octaves);
    put64(bytes, static_cast<std::uint64_t>(c.tempo.grid_size));
    put64(bytes, static_cast<std::uint64_t>(c.tempo.comb_harmonics));
    putF64(bytes, c.tempo.comb_weight_decay);
    putF64(bytes, c.odf.chroma ? 1.0 : 0.0);
    putF64(bytes, c.odf.chromaMinHz);
    putF64(bytes, c.odf.chromaMaxHz);
    putF64(bytes, c.tracker.tightness);
    put64(bytes, c.tracker.trim ? 1 : 0);
    putF64(bytes, c.bpm_hint);
    put64(bytes, c.find_downbeats ? 1 : 0);
    putF64(bytes, c.downbeat.low_weight);
    putF64(bytes, c.downbeat.accent_weight);
    putF64(bytes, c.downbeat.harmony_weight);
    putF64(bytes, c.downbeat.window_before);
    putF64(bytes, c.downbeat.window_after);
    put64(bytes, static_cast<std::uint64_t>(c.downbeat.min_bars));
    putF64(bytes, c.downbeat.min_salience_range);
    putF64(bytes, c.downbeat.min_phase_margin);
    putF64(bytes, c.downbeat.min_meter_margin);
    for (const MeterCandidate& m : c.downbeat.meters) {
        put64(bytes, static_cast<std::uint64_t>(m.beats_per_bar));
        putF64(bytes, m.prior);
    }
    return fnv1a(bytes.data(), bytes.size());
}

// magic, version, fingerprint, bpm, confidence, estimated bpm, frame count,
// beats per bar, downbeat strength, downbeat margin, beat count, downbeat
// count — everything before the two arrays of times.
constexpr std::size_t kHeaderSize = 4 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8;
constexpr std::size_t kChecksumSize = 8;
constexpr std::size_t kBeatCountOffset = 88;
constexpr std::size_t kDownbeatCountOffset = 96;

}  // namespace

std::string gridCacheKey(const void* bytes, std::size_t n) {
    Sha256 sha;
    if (n > 0) sha.update(static_cast<const std::uint8_t*>(bytes), n);
    std::uint8_t digest[32];
    sha.finish(digest);

    static constexpr char hex[] = "0123456789abcdef";
    std::string key(64, '0');
    for (int i = 0; i < 32; ++i) {
        key[2 * i] = hex[digest[i] >> 4];
        key[2 * i + 1] = hex[digest[i] & 0xf];
    }
    return key;
}

std::vector<std::uint8_t> serializeGrid(const OfflineResult& result,
                                        const OfflineConfig& config) {
    std::vector<std::uint8_t> out;
    out.reserve(kHeaderSize + (result.beats.size() + result.downbeats.size()) * 8 +
                kChecksumSize);

    out.insert(out.end(), kMagic, kMagic + sizeof(kMagic));
    put32(out, kVersion);
    put64(out, fingerprint(config));
    putF64(out, result.bpm);
    putF64(out, result.tempo_confidence);
    putF64(out, result.estimated_bpm);
    put64(out, result.frame_count);
    put64(out, static_cast<std::uint64_t>(result.beats_per_bar));
    putF64(out, result.downbeat_strength);
    putF64(out, result.downbeat_phase_margin);
    putF64(out, result.downbeat_meter_margin);
    put64(out, result.downbeat_confident ? 1u : 0u);
    put64(out, result.beats.size());
    put64(out, result.downbeats.size());
    for (double beat : result.beats) putF64(out, beat);
    for (double beat : result.downbeats) putF64(out, beat);

    put64(out, fnv1a(out.data(), out.size()));
    return out;
}

bool deserializeGrid(const std::uint8_t* bytes, std::size_t n,
                     const OfflineConfig& config, OfflineResult* out) {
    if (!bytes || !out) return false;
    if (n < kHeaderSize + kChecksumSize) return false;
    if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) return false;
    if (get32(bytes + 4) != kVersion) return false;
    if (get64(bytes + 8) != fingerprint(config)) return false;

    // Checked against the actual size before either count is used to index
    // anything, so a corrupted count reads as "not a grid" rather than as a
    // huge read. Compared against the payload rather than added together first,
    // because two attacker-chosen 64-bit counts can be made to sum to anything.
    const std::uint64_t payload = (n - kHeaderSize - kChecksumSize) / 8;
    const std::uint64_t beat_count = get64(bytes + kBeatCountOffset);
    const std::uint64_t downbeat_count = get64(bytes + kDownbeatCountOffset);
    if (beat_count > payload || downbeat_count > payload - beat_count) return false;
    if (n != kHeaderSize + (beat_count + downbeat_count) * 8 + kChecksumSize) return false;
    if (get64(bytes + n - kChecksumSize) != fnv1a(bytes, n - kChecksumSize)) return false;

    OfflineResult result;
    result.bpm = getF64(bytes + 16);
    result.tempo_confidence = getF64(bytes + 24);
    result.estimated_bpm = getF64(bytes + 32);
    result.frame_count = static_cast<std::size_t>(get64(bytes + 40));
    result.beats_per_bar = static_cast<int>(get64(bytes + 48));
    result.downbeat_strength = getF64(bytes + 56);
    result.downbeat_phase_margin = getF64(bytes + 64);
    result.downbeat_meter_margin = getF64(bytes + 72);
    result.downbeat_confident = get64(bytes + 80) != 0;
    result.beats.resize(static_cast<std::size_t>(beat_count));
    for (std::size_t i = 0; i < result.beats.size(); ++i) {
        result.beats[i] = getF64(bytes + kHeaderSize + i * 8);
    }
    const std::uint8_t* after_beats = bytes + kHeaderSize + result.beats.size() * 8;
    result.downbeats.resize(static_cast<std::size_t>(downbeat_count));
    for (std::size_t i = 0; i < result.downbeats.size(); ++i) {
        result.downbeats[i] = getF64(after_beats + i * 8);
    }

    *out = std::move(result);
    return true;
}

}  // namespace tiktak::analysis
