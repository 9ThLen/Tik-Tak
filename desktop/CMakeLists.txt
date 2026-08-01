cmake_minimum_required(VERSION 3.20)
project(tiktak_desktop VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Built against the core in this repository. Nothing here is allowed to hold
# logic the phone will also need — see docs/adr/0003-windows-host-no-mac.md. The
# harness connects real devices to the core and measures; everything it measures
# lives in core/.
if(NOT TARGET tiktak_core)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_CURRENT_BINARY_DIR}/core)
endif()

include(FetchContent)

# Single-header, public domain, and it speaks WASAPI on Windows, CoreAudio on
# macOS and ALSA/PulseAudio on Linux without any of them being a build
# dependency — the backends are loaded at runtime. That last property is why the
# harness builds in CI on a machine with no sound card at all.
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        0.11.21
    GIT_SHALLOW    TRUE
    # The harness includes miniaudio.h directly; none of the upstream examples
    # or tests should become part of this build.
    SOURCE_SUBDIR  _tiktak_headers_only
)
FetchContent_MakeAvailable(miniaudio)

add_executable(tiktak
    src/main.cpp
    src/commands.cpp
    src/device.cpp
    src/keys.cpp
    src/tap.cpp
    src/wav.cpp
    src/miniaudio_impl.c
)

target_include_directories(tiktak PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${miniaudio_SOURCE_DIR}
    # The harness uses the C++ classes rather than the C API: it is not a
    # platform shell and gains nothing from the ABI boundary, and going through
    # the C API would hide type errors the compiler could catch.
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/src
)

target_link_libraries(tiktak PRIVATE tiktak_core)

if(TARGET tiktak_decode)
    target_link_libraries(tiktak PRIVATE tiktak_decode)
    target_compile_definitions(tiktak PRIVATE TIKTAK_HAVE_DECODE=1)
endif()

if(MSVC)
    target_compile_options(tiktak PRIVATE /W4)
else()
    target_compile_options(tiktak PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
endif()

# Vendored C, compiled with its own warning settings for the same reason
# dr_libs is — see core/CMakeLists.txt.
if(MSVC)
    set_source_files_properties(src/miniaudio_impl.c PROPERTIES COMPILE_FLAGS "/w")
else()
    set_source_files_properties(src/miniaudio_impl.c PROPERTIES COMPILE_FLAGS "-w")
endif()

# The harness's own tests, when the core is building its own — gtest is already
# fetched by then, so this costs nothing and adds no dependency of its own.
if(TARGET GTest::gtest_main)
    enable_testing()
    add_subdirectory(tests)
endif()

find_package(Threads REQUIRED)
target_link_libraries(tiktak PRIVATE Threads::Threads)
if(UNIX AND NOT APPLE)
    target_link_libraries(tiktak PRIVATE ${CMAKE_DL_LIBS} m)
endif()
