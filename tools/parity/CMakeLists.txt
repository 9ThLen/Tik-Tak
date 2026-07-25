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
