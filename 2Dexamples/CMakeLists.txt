cmake_minimum_required(VERSION 3.10)
project(ObstacleMapGenerator CXX)

# Find SFML package and specify the required components:
# system, window, and graphics are needed for the window and drawing.
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)

# Find the YAML-CPP library
find_package(yaml-cpp REQUIRED)

# Add the executable target: 'main'
# This tells CMake to build an executable from the source file.
add_executable(main src/main.cpp src/MotionPlanner.cpp)
add_executable(visualize_collision_checking src/visualize_collision_checking.cpp src/MotionPlanner.cpp)
add_executable(visualize_noisy_trj src/visualize_noisy_trj.cpp src/MotionPlanner.cpp)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(visualize_collision_checking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(visualize_noisy_trj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(/opt/homebrew/lib)

# Link the SFML libraries to the executable
# This is how the compiler finds the SFML functions used in the C++ file.
target_link_libraries(main PRIVATE SFML::System SFML::Window SFML::Graphics /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib)
target_link_libraries(visualize_collision_checking PRIVATE SFML::System SFML::Window SFML::Graphics /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib)
target_link_libraries(visualize_noisy_trj PRIVATE SFML::System SFML::Window SFML::Graphics /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib)

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