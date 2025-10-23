cmake_minimum_required(VERSION 3.10)
project(2DExample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific configuration
if(APPLE)
    message(STATUS "Configuring for macOS")
    set(YAML_CPP_LIB /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib)
    set(GTEST_INCLUDE_DIRS /opt/homebrew/include)
    set(GTEST_LIBRARIES
        /opt/homebrew/lib/libgtest.dylib
        /opt/homebrew/lib/libgtest_main.dylib
    )
    link_directories(/opt/homebrew/lib)
    set(PLATFORM_LIBS)  # No stdc++fs needed on macOS
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    set(YAML_CPP_LIB yaml-cpp)
    set(GTEST_INCLUDE_DIRS)  # Use system paths
    set(GTEST_LIBRARIES)     # Will be found by find_package
    set(PLATFORM_LIBS stdc++fs)  # Required for GCC 9.x filesystem support
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Find SFML package
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)

# Find the YAML-CPP library
find_package(yaml-cpp REQUIRED)

# Find Eigen3 library
find_package(Eigen3 REQUIRED)

# Find GoogleTest for unit testing
find_package(GTest QUIET)
if(NOT GTest_FOUND AND APPLE)
    message(WARNING "GTest not found via find_package. Using brew paths.")
    set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES})
elseif(NOT GTest_FOUND)
    message(WARNING "GTest not found.")
endif()

# Include for all targets
if(GTEST_INCLUDE_DIRS)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
endif()

# Add executables
add_executable(main src/main.cpp)
add_executable(visualize_collision_checking src/visualize_collision_checking.cpp)
add_executable(visualize_noisy_trj src/visualize_noisy_trj.cpp)

# Include directories
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_collision_checking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_noisy_trj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Link libraries for 'main'
target_link_libraries(main PRIVATE
    SFML::System
    SFML::Window
    SFML::Graphics
    ${YAML_CPP_LIB}
    Eigen3::Eigen
    ${PLATFORM_LIBS}
)

# Link libraries for 'visualize_collision_checking'
target_link_libraries(visualize_collision_checking PRIVATE
    SFML::System
    SFML::Window
    SFML::Graphics
    ${YAML_CPP_LIB}
    Eigen3::Eigen
    ${PLATFORM_LIBS}
)

# Link libraries for 'visualize_noisy_trj'
target_link_libraries(visualize_noisy_trj PRIVATE
    SFML::System
    SFML::Window
    SFML::Graphics
    ${YAML_CPP_LIB}
    Eigen3::Eigen
    ${PLATFORM_LIBS}
)

# Set standard C++ features for modern development
set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
set_target_properties(visualize_collision_checking PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
set_target_properties(visualize_noisy_trj PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Configuration check
if(SFML_FOUND)
    message(STATUS "SFML found and configured successfully!")
else()
    message(FATAL_ERROR "SFML not found. Please ensure SFML is installed.")
endif()