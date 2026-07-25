// The single translation unit that instantiates dr_libs.
//
// Kept apart from our own code so the third-party implementations compile with
// their own warning settings: they are C libraries built as C++ here, and
// holding them to this project's -Wall -Wextra -Wpedantic -Werror would mean
// either patching vendored code on every update or turning the flags off for
// files we do want them on. See core/CMakeLists.txt.

#define DR_WAV_IMPLEMENTATION
#define DR_FLAC_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION

// Ogg-in-FLAC is a container we do not accept, so leaving it out cuts both the
// binary and the amount of untrusted-file parsing that has to be trusted.
#define DR_FLAC_NO_OGG

#include "dr_flac.h"
#include "dr_mp3.h"
#include "dr_wav.h"
