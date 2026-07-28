cmake_minimum_required(VERSION 3.20)
project(tiktak_parity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The core is consumed exactly as an app would consume it — through its public
# header only. Tests are off; this builds the tool, not the suite.
set(TIKTAK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../core ${CMAKE_BINARY_DIR}/core)

add_executable(dump_odf dump_odf.cpp)
target_link_libraries(dump_odf PRIVATE tiktak_core)

add_executable(dump_beats dump_beats.cpp)
target_link_libraries(dump_beats PRIVATE tiktak_core)

# The exception to the paragraph above: BeatNet is not in the C API, because
# nothing outside the core wants the model on its own — the shells reach it
# through the live tracker. Diffing it against the reference still has to be
# possible, so this one tool sees the private headers.
add_executable(dump_beatnet dump_beatnet.cpp)
target_link_libraries(dump_beatnet PRIVATE tiktak_core)
target_include_directories(dump_beatnet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src)
