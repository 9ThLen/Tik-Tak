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

add_executable(dump_beat_this_features dump_beat_this_features.cpp)
target_link_libraries(dump_beat_this_features PRIVATE tiktak_core)
target_include_directories(dump_beat_this_features
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src)

# End to end: the core's front end plus the runtime. Off unless asked for —
# configure with -DTIKTAK_BUILD_ML=ON. It downloads ONNX Runtime and needs the
# ONNX model, neither of which is in git, so CI gets neither the dependency nor
# a check it could not run anyway.
if(TARGET tiktak_ml)
    add_executable(dump_beat_this dump_beat_this.cpp)
    target_link_libraries(dump_beat_this PRIVATE tiktak_ml)
    target_include_directories(dump_beat_this PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src)
    foreach(runtime IN LISTS TIKTAK_ORT_RUNTIME_FILES)
        add_custom_command(TARGET dump_beat_this POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${runtime}" "$<TARGET_FILE_DIR:dump_beat_this>")
    endforeach()
endif()
