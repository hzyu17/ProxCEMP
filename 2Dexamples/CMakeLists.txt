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
    set(PLATFORM_LIBS)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    set(YAML_CPP_LIB yaml-cpp)
    set(GTEST_INCLUDE_DIRS)
    set(GTEST_LIBRARIES)
    set(PLATFORM_LIBS stdc++fs)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Find SFML package
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)

# Find the YAML-CPP library
find_package(yaml-cpp REQUIRED)

# Find Eigen3 library
find_package(Eigen3 REQUIRED)

# ============================================================================
# CasADi Detection - Check pip installation
# ============================================================================
# Common pip installation paths
set(CASADI_SEARCH_PATHS
    /usr/local/lib/python3.12/dist-packages/casadi
    /usr/local/lib/python3.11/dist-packages/casadi
    /usr/local/lib/python3.10/dist-packages/casadi
    /usr/lib/python3/dist-packages/casadi
    $ENV{HOME}/.local/lib/python3.12/site-packages/casadi
    $ENV{HOME}/.local/lib/python3.11/site-packages/casadi
    $ENV{HOME}/.local/lib/python3.10/site-packages/casadi
)

# Find CasADi header (pip installs headers in include/ subdirectory)
find_path(CASADI_INCLUDE_DIR 
    NAMES casadi/casadi.hpp
    PATHS ${CASADI_SEARCH_PATHS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

# Also check system paths
if(NOT CASADI_INCLUDE_DIR)
    find_path(CASADI_INCLUDE_DIR 
        NAMES casadi/casadi.hpp
        PATHS /usr/local/include /usr/include /opt/casadi/include
    )
endif()

# Find CasADi library
find_library(CASADI_LIBRARY 
    NAMES casadi
    PATHS ${CASADI_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

# Also check system paths
if(NOT CASADI_LIBRARY)
    find_library(CASADI_LIBRARY 
        NAMES casadi
        PATHS /usr/local/lib /usr/lib /opt/casadi/lib
    )
endif()

if(CASADI_INCLUDE_DIR AND CASADI_LIBRARY)
    set(CASADI_FOUND TRUE)
    message(STATUS "CasADi found:")
    message(STATUS "  Include: ${CASADI_INCLUDE_DIR}")
    message(STATUS "  Library: ${CASADI_LIBRARY}")
    
    # Get the directory containing the library for RPATH
    get_filename_component(CASADI_LIBRARY_DIR ${CASADI_LIBRARY} DIRECTORY)
    message(STATUS "  Lib dir: ${CASADI_LIBRARY_DIR}")
else()
    message(FATAL_ERROR 
        "CasADi not found!\n"
        "Include dir: ${CASADI_INCLUDE_DIR}\n"
        "Library: ${CASADI_LIBRARY}\n"
        "Please install CasADi: pip install casadi")
endif()

# Include CasADi headers
include_directories(SYSTEM ${CASADI_INCLUDE_DIR})

# Add library directory to linker search path
link_directories(${CASADI_LIBRARY_DIR})

# ============================================================================
# GoogleTest (optional)
# ============================================================================
find_package(GTest QUIET)
if(NOT GTest_FOUND AND APPLE)
    message(WARNING "GTest not found via find_package. Using brew paths.")
    set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES})
elseif(NOT GTest_FOUND)
    message(WARNING "GTest not found.")
endif()

if(GTEST_INCLUDE_DIRS)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
endif()

# ============================================================================
# Executables
# ============================================================================
add_executable(main src/main.cpp)
add_executable(visualize_collision_checking src/visualize_collision_checking.cpp)
add_executable(visualize_noisy_trj src/visualize_noisy_trj.cpp)

# Include directories
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_collision_checking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_noisy_trj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Link libraries for 'main' (with CasADi)
target_link_libraries(main PRIVATE
    SFML::System
    SFML::Window
    SFML::Graphics
    ${YAML_CPP_LIB}
    Eigen3::Eigen
    ${CASADI_LIBRARY}
    ${PLATFORM_LIBS}
)

# Set RPATH so the executable can find CasADi at runtime
set_target_properties(main PROPERTIES
    BUILD_RPATH "${CASADI_LIBRARY_DIR}"
    INSTALL_RPATH "${CASADI_LIBRARY_DIR}"
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

# C++ standard properties
set_target_properties(main visualize_collision_checking visualize_noisy_trj
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Status messages
message(STATUS "=== Configuration Summary ===")
message(STATUS "SFML: Found")
message(STATUS "CasADi library: ${CASADI_LIBRARY}")