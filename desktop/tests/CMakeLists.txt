# The harness has one piece of logic worth testing, and this is it.
#
# Everything else here is device access and argument parsing — thin, and
# verified by running it. `compareTaps` is different: it turns a person's
# uneven tapping into a verdict about the tracker, and a verdict that is wrong
# in a plausible direction is worse than no bench at all, because it would be
# believed. gtest comes from the core's own fetch, so this adds no dependency.
add_executable(tiktak_desktop_tests test_tap.cpp ../src/tap.cpp)
target_link_libraries(tiktak_desktop_tests PRIVATE GTest::gtest_main)
target_include_directories(tiktak_desktop_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

include(GoogleTest)
gtest_discover_tests(tiktak_desktop_tests)
