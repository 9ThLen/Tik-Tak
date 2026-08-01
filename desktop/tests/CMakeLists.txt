# Two pieces of logic worth testing.
#
# `compareTaps` turns a person's uneven tapping into a verdict about the
# tracker, and a verdict that is wrong in a plausible direction is worse than no
# bench at all, because it would be believed.
#
# The device layer was for a while described here as thin and verified by
# running it. It was not: naming a microphone could not work, and the rate the
# tracker was built for was not the rate the device ran at. Neither showed in a
# run on this machine. See test_device.cpp. gtest comes from the core's own
# fetch, so this adds no dependency; miniaudio loads its backends at runtime, so
# linking it does not require a sound card to build or to run.
add_executable(tiktak_desktop_tests
    test_tap.cpp
    test_device.cpp
    ../src/tap.cpp
    ../src/device.cpp
    ../src/miniaudio_impl.c
)
target_link_libraries(tiktak_desktop_tests PRIVATE GTest::gtest_main)
target_include_directories(tiktak_desktop_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${miniaudio_SOURCE_DIR}
)
find_package(Threads REQUIRED)
target_link_libraries(tiktak_desktop_tests PRIVATE Threads::Threads)
if(UNIX AND NOT APPLE)
    target_link_libraries(tiktak_desktop_tests PRIVATE ${CMAKE_DL_LIBS} m)
endif()
if(MSVC)
    set_source_files_properties(../src/miniaudio_impl.c PROPERTIES COMPILE_FLAGS "/w")
else()
    set_source_files_properties(../src/miniaudio_impl.c PROPERTIES COMPILE_FLAGS "-w")
endif()

include(GoogleTest)
gtest_discover_tests(tiktak_desktop_tests)
