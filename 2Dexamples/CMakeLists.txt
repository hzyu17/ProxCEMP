cmake_minimum_required(VERSION 3.10)
project(ObstacleMapGenerator CXX)

# Find SFML package and specify the required components:
# system, window, and graphics are needed for the window and drawing.
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)

# Find the YAML-CPP library
find_package(yaml-cpp REQUIRED)

# Add the executable target: 'map_generator'
# This tells CMake to build an executable from the source file.
add_executable(map_generator src/main.cpp src/ObstacleMap.cpp src/Trajectory.cpp src/MotionPlanner.cpp)

target_include_directories(map_generator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(/opt/homebrew/lib)

# Link the SFML libraries to the executable
# This is how the compiler finds the SFML functions used in the C++ file.
target_link_libraries(map_generator PRIVATE SFML::System SFML::Window SFML::Graphics /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib)

# Set standard C++ features for modern development
set_target_properties(map_generator PROPERTIES
CXX_STANDARD 17
CXX_STANDARD_REQUIRED ON
)

# Configuration check for user feedback
if(SFML_FOUND)
message(STATUS "SFML found and configured successfully!")
else()
message(FATAL_ERROR "SFML not found. Please ensure SFML is installed.")
endif()