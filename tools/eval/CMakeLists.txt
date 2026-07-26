cmake_minimum_required(VERSION 3.20)
project(tiktak_eval LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The decoder is on, unlike in tools/parity: the point of this tool is to score
# the analysis on real recordings, and a real recording is an MP3.
set(TIKTAK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TIKTAK_BUILD_DECODE ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../core ${CMAKE_BINARY_DIR}/core)

add_executable(dump_analysis dump_analysis.cpp)
target_link_libraries(dump_analysis PRIVATE tiktak_core)

if(TARGET tiktak_decode)
    target_link_libraries(dump_analysis PRIVATE tiktak_decode)
    target_include_directories(dump_analysis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src)
    target_compile_definitions(dump_analysis PRIVATE TIKTAK_HAVE_DECODE=1)
endif()

if(MSVC)
    target_compile_options(dump_analysis PRIVATE /W4)
else()
    target_compile_options(dump_analysis PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
endif()
