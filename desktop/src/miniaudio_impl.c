/* The single translation unit that instantiates miniaudio. Kept apart from our
 * own code so the vendored implementation compiles with its own warning
 * settings, the same arrangement dr_libs has in the core. */
#define MINIAUDIO_IMPLEMENTATION

/* The harness needs devices and nothing else: no decoding (the core has its
 * own), no resampling, no node graph. Leaving them out cuts the build and makes
 * it obvious that audio format handling belongs on our side of the line. */
#define MA_NO_ENCODING
#define MA_NO_DECODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH

#include "miniaudio.h"
