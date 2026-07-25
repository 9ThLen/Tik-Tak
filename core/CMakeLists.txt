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
    src/dsp/stft.cpp
    src/schedule/scheduler.cpp
    src/dsp/odf.cpp
    src/analysis/tempo.cpp
    src/analysis/tracker.cpp
    src/analysis/offline.cpp
    src/analysis/grid_cache.cpp
    src/tracking/particle.cpp
    src/tracking/live.cpp
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

# The core must stay free of platform SDK dependencies so it cross-compiles to
# both iOS and Android unchanged — see docs/adr/0001-portable-cpp-core.md.
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

if(TIKTAK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
