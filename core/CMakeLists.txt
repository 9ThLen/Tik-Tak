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
    src/dsp/odf.cpp
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

if(TIKTAK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
