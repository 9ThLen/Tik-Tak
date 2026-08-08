cmake_minimum_required(VERSION 3.20)
project(tiktak_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_IS_TOP_LEVEL)
    set(TIKTAK_IS_TOP_LEVEL ON)
else()
    set(TIKTAK_IS_TOP_LEVEL OFF)
endif()

option(TIKTAK_BUILD_TESTS "Build tiktak core tests" ${TIKTAK_IS_TOP_LEVEL})
option(TIKTAK_WERROR "Treat warnings as errors" OFF)

add_library(tiktak_core STATIC
    src/api.cpp
    src/dsp/fft.cpp
    src/dsp/window.cpp
    src/dsp/mel.cpp
    src/dsp/chroma.cpp
    src/dsp/stft.cpp
    src/dsp/dft.cpp
    src/dsp/logfilt.cpp
    src/dsp/resample.cpp
    src/ml/beatnet.cpp
    src/ml/beat_this.cpp
    src/schedule/scheduler.cpp
    src/dsp/matched.cpp
    src/dsp/odf.cpp
    src/analysis/tempo.cpp
    src/analysis/tracker.cpp
    src/analysis/downbeat.cpp
    src/analysis/offline.cpp
    src/analysis/grid_cache.cpp
    src/tracking/particle.cpp
    src/tracking/activation_tempo.cpp
    src/tracking/live.cpp
    src/tracking/bar.cpp
    src/tracking/sync.cpp
    src/render/click.cpp
    src/render/metronome.cpp
    src/render/player.cpp
    src/render/live_metronome.cpp
)

target_include_directories(tiktak_core
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_options(tiktak_core PRIVATE /W4)
    if(TIKTAK_WERROR)
        target_compile_options(tiktak_core PRIVATE /WX)
    endif()
else()
    target_compile_options(tiktak_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if(TIKTAK_WERROR)
        target_compile_options(tiktak_core PRIVATE -Werror)
    endif()
endif()

# The core stays free of platform SDK dependencies so the same source
# cross-compiles to iOS and Android unchanged.
set_target_properties(tiktak_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ------------------------------------------------------------------ decoding --
#
# A separate target on purpose. tiktak_core has no third-party dependencies and
# that property is worth keeping: it is what lets the analysis cross-compile
# anywhere without a vendoring story. Decoding cannot have that property — it
# needs codec implementations — so it lives beside the core rather than inside
# it, and a platform that would rather use AVAssetReader or MediaCodec can link
# the core alone.

option(TIKTAK_BUILD_DECODE "Build the audio file decoder" ${TIKTAK_IS_TOP_LEVEL})

if(TIKTAK_BUILD_DECODE)
    include(FetchContent)

    # Pinned to a commit because dr_libs publishes no tags. Updating it is a
    # deliberate act, not something a fresh checkout does silently.
    FetchContent_Declare(
        dr_libs
        GIT_REPOSITORY https://github.com/mackron/dr_libs.git
        GIT_TAG        34a89ffe6bfc4d78db6888fef76cd408dba18185
        GIT_SHALLOW    FALSE
        # dr_libs is consumed as headers only. Point MakeAvailable at a
        # deliberately absent subdirectory so it populates the source without
        # adding dr_libs' own test targets to this build.
        SOURCE_SUBDIR  _tiktak_headers_only
    )
    FetchContent_MakeAvailable(dr_libs)

    add_library(tiktak_decode STATIC
        src/decode/dr_libs.cpp
        src/decode/decoder.cpp
        src/decode_api.cpp
    )

    target_link_libraries(tiktak_decode PUBLIC tiktak_core)
    target_include_directories(tiktak_decode
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE ${dr_libs_SOURCE_DIR}
    )

    if(MSVC)
        target_compile_options(tiktak_decode PRIVATE /W4)
        if(TIKTAK_WERROR)
            target_compile_options(tiktak_decode PRIVATE /WX)
        endif()
    else()
        target_compile_options(tiktak_decode PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
        if(TIKTAK_WERROR)
            target_compile_options(tiktak_decode PRIVATE -Werror)
        endif()
    endif()

    # Vendored C code compiled as C++ does not meet this project's warning bar,
    # and patching it on every update would be worse than exempting the one
    # translation unit that instantiates it.
    set_source_files_properties(src/decode/dr_libs.cpp PROPERTIES COMPILE_OPTIONS "-w")
    if(MSVC)
        set_source_files_properties(src/decode/dr_libs.cpp PROPERTIES COMPILE_OPTIONS "/w")
    endif()

    set_target_properties(tiktak_decode PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# ----------------------------------------------------------------- inference --
#
# Another separate target, for the same reason decoding is one. tiktak_core has
# no third-party dependencies and that is what lets the analysis cross-compile
# anywhere; an inference runtime cannot have that property, so Beat This! runs
# beside the core rather than inside it.
#
# Note the asymmetry with src/ml/beatnet, which *is* in the core. That model is
# 0.40 M parameters at 20 MMAC/s — smaller than the runtime that would host it,
# so the core carries the forward pass itself. This one is a transformer run
# once over a whole file off the audio thread, where a runtime is a fair trade.
# The rule is the size of the model against the size of the runtime, not a
# preference for one style.

option(TIKTAK_BUILD_ML "Build the ONNX inference library" OFF)

if(TIKTAK_BUILD_ML)
    include(FetchContent)

    # Pinned by URL and hash, all three verified by downloading the archives
    # and hashing them rather than by copying numbers from a page.
    set(TIKTAK_ORT_VERSION "1.22.0")
    if(WIN32)
        set(TIKTAK_ORT_ARCHIVE "onnxruntime-win-x64-${TIKTAK_ORT_VERSION}.zip")
        set(TIKTAK_ORT_HASH "SHA256=174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da")
    elseif(APPLE)
        set(TIKTAK_ORT_ARCHIVE "onnxruntime-osx-universal2-${TIKTAK_ORT_VERSION}.tgz")
        set(TIKTAK_ORT_HASH "SHA256=cfa6f6584d87555ed9f6e7e8a000d3947554d589efe3723b8bfa358cd263d03c")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(TIKTAK_ORT_ARCHIVE "onnxruntime-linux-x64-${TIKTAK_ORT_VERSION}.tgz")
        set(TIKTAK_ORT_HASH "SHA256=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3")
    else()
        message(FATAL_ERROR "TIKTAK_BUILD_ML: no ONNX Runtime archive is pinned for ${CMAKE_SYSTEM_NAME}")
    endif()
    set(TIKTAK_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${TIKTAK_ORT_VERSION}/${TIKTAK_ORT_ARCHIVE}")

    FetchContent_Declare(onnxruntime URL "${TIKTAK_ORT_URL}" URL_HASH "${TIKTAK_ORT_HASH}")
    FetchContent_MakeAvailable(onnxruntime)

    add_library(tiktak_ml STATIC src/ml/beat_this_session.cpp)
    target_link_libraries(tiktak_ml PUBLIC tiktak_core)
    target_include_directories(tiktak_ml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # SYSTEM, so the runtime's own headers are not held to this project's
    # warning bar. They shadow locals in several places; patching a vendored
    # header on every update would be worse than exempting it, which is the
    # same call already made for dr_libs.
    target_include_directories(tiktak_ml SYSTEM PUBLIC ${onnxruntime_SOURCE_DIR}/include)
    find_library(TIKTAK_ORT_LIB onnxruntime
                 PATHS ${onnxruntime_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(tiktak_ml PUBLIC ${TIKTAK_ORT_LIB})

    # Windows links against an import library and loads the DLL at run time, so
    # the DLL has to sit beside whatever executable ends up linking this. Told
    # to consumers rather than assumed: TIKTAK_ORT_RUNTIME_FILES is the list to
    # copy.
    if(WIN32)
        file(GLOB TIKTAK_ORT_RUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dll")
    else()
        set(TIKTAK_ORT_RUNTIME_FILES "")
    endif()
    set(TIKTAK_ORT_RUNTIME_FILES "${TIKTAK_ORT_RUNTIME_FILES}" CACHE INTERNAL
        "ONNX Runtime shared libraries that must sit beside the executable")

    if(MSVC)
        target_compile_options(tiktak_ml PRIVATE /W4)
        if(TIKTAK_WERROR)
            target_compile_options(tiktak_ml PRIVATE /WX)
        endif()
    else()
        target_compile_options(tiktak_ml PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
        if(TIKTAK_WERROR)
            target_compile_options(tiktak_ml PRIVATE -Werror)
        endif()
    endif()

    set_target_properties(tiktak_ml PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

if(TIKTAK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
