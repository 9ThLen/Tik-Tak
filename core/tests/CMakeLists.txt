include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tiktak_core_tests
    test_fft.cpp
    test_window.cpp
    test_mel.cpp
    test_stft.cpp
    test_odf.cpp
    test_scheduler.cpp
    test_tempo.cpp
    test_tracker.cpp
    test_offline.cpp
    test_api.cpp
)

target_link_libraries(tiktak_core_tests PRIVATE tiktak_core GTest::gtest_main)

# Tests reach into the private headers on purpose: the DSP building blocks are
# implementation detail of the C API but still need direct coverage.
target_include_directories(tiktak_core_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

include(GoogleTest)
gtest_discover_tests(tiktak_core_tests)
