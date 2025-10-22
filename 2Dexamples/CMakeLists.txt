cmake_minimum_required(VERSION 3.10)
project(ObstacleMapGenerator CXX)

# Find SFML package and specify the required components:
# system, window, and graphics are needed for the window and drawing.
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)

# Find the YAML-CPP library
find_package(yaml-cpp REQUIRED)

# Find Eigen3 library
find_package(Eigen3 REQUIRED)

# Find GoogleTest for unit testing
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(WARNING "GTest not found via find_package. Falling back to brew paths.")
    # Add brew include/lib paths manually
    set(GTEST_INCLUDE_DIRS 
        /opt/homebrew/include  # Apple Silicon; change to /usr/local/include for Intel
    )
    set(GTEST_LIBRARIES 
        /opt/homebrew/lib/libgtest.dylib
        /opt/homebrew/lib/libgtest_main.dylib
    )
    set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES})
else()
    message(STATUS "Found GTest via find_package")
endif()

# Include for all targets
include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})

# Add the executable target: 'main'
# This tells CMake to build an executable from the source file.
add_executable(main src/main.cpp)
add_executable(visualize_collision_checking src/visualize_collision_checking.cpp)
add_executable(visualize_noisy_trj src/visualize_noisy_trj.cpp)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_collision_checking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_noisy_trj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_directories(/opt/homebrew/lib)

# Link the SFML libraries to the executable
# This is how the compiler finds the SFML functions used in the C++ file.
target_link_libraries(main PRIVATE 
        SFML::System 
        SFML::Window 
        SFML::Graphics 
        /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib
        Eigen3::Eigen)
target_link_libraries(visualize_collision_checking 
        PRIVATE SFML::System 
        SFML::Window 
        SFML::Graphics 
        /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib
        Eigen3::Eigen)
target_link_libraries(visualize_noisy_trj 
        PRIVATE SFML::System 
        SFML::Window 
        SFML::Graphics 
        /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib
        Eigen3::Eigen)

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

# Configuration check for user feedback
if(SFML_FOUND)
message(STATUS "SFML found and configured successfully!")
else()
message(FATAL_ERROR "SFML not found. Please ensure SFML is installed.")
endif()
