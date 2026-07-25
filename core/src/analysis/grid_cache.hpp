#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "analysis/offline.hpp"

namespace tiktak::analysis {

// The beat grid cache: an analysed track serialised for reuse, so importing the
// same file twice costs seconds once and nothing after.
//
// The core serialises to bytes and leaves storage to the shell, for the same
// reason decoding accepts bytes rather than a path: every shell can produce and
// persist bytes, while a portable notion of "the cache directory" does not
// exist — iOS has Application Support, Android has Context.getCacheDir(), the
// desktop harness has whatever it likes.
//
// The key is a hash of the *encoded file bytes*, not of the decoded samples or
// the file's name. Content-addressing means a renamed or moved file still hits,
// a re-encoded one correctly misses, and the shell never has to decode before
// deciding whether it needs to.

// 64 lowercase hex characters: the SHA-256 of the bytes. SHA-256 rather than
// something cheaper because a colliding key silently serves one track another
// track's grid — the failure would look like the analyser being wrong, and no
// amount of debugging the analyser would find it.
std::string gridCacheKey(const void* bytes, std::size_t n);

// Serialises a finished analysis. The config is part of the blob's identity:
// the same audio under a different configuration — a manual-mode bpm_hint, a
// different tightness — is a different grid, and serving one for the other
// would be a stale-cache bug the user experiences as beats in the wrong place.
std::vector<std::uint8_t> serializeGrid(const OfflineResult& result,
                                        const OfflineConfig& config);

// Restores a grid serialised earlier. Returns false — and leaves `out` alone —
// when the bytes are not a grid, were written by an incompatible version of
// this format, were produced under a different config, or fail their checksum.
// Every one of those answers means the same thing to the caller: re-analyse.
bool deserializeGrid(const std::uint8_t* bytes, std::size_t n,
                     const OfflineConfig& config, OfflineResult* out);

}  // namespace tiktak::analysis
