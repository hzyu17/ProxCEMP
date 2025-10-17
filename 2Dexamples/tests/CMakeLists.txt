# Collect all test sources
set(TEST_SOURCES
    test_determinism.cpp
)

add_executable(test_pcem ${TEST_SOURCES})

target_link_libraries(test_pcem
    PRIVATE
    gtest_main           # GoogleTest entry point
    Eigen3::Eigen
    yaml-cpp
    ${GTEST_LIBRARIES}
)

# Stubs include path (private to tests)
target_include_directories(test_pcem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs
    GTest::gtest_main
)

# Discover and run tests
include(GoogleTest)
gtest_discover_tests(test_pcem)