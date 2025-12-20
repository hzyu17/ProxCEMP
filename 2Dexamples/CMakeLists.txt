cmake_minimum_required(VERSION 3.10)
project(2DExample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

# --- Platform-specific configuration ---
if(APPLE)
    set(YAML_CPP_LIB /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib/libyaml-cpp.dylib)
    set(GTEST_LIBRARIES /opt/homebrew/lib/libgtest.dylib /opt/homebrew/lib/libgtest_main.dylib)
    include_directories(SYSTEM /opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
elseif(UNIX)
    set(YAML_CPP_LIB yaml-cpp)
    set(PLATFORM_LIBS stdc++fs)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# --- Dependencies ---
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(stomp REQUIRED)

# --- CasADi Detection ---
set(CASADI_SEARCH_PATHS
    /usr/local/lib/python3.12/dist-packages/casadi /usr/local/lib/python3.11/dist-packages/casadi
    /usr/local/lib/python3.10/dist-packages/casadi /usr/lib/python3/dist-packages/casadi
    $ENV{HOME}/.local/lib/python3.12/site-packages/casadi $ENV{HOME}/.local/lib/python3.11/site-packages/casadi
    $ENV{HOME}/.local/lib/python3.10/site-packages/casadi
    /usr/local /usr /opt/casadi
)

find_path(CASADI_INCLUDE_DIR NAMES casadi/casadi.hpp PATHS ${CASADI_SEARCH_PATHS} PATH_SUFFIXES include)
find_library(CASADI_LIBRARY NAMES casadi PATHS ${CASADI_SEARCH_PATHS} PATH_SUFFIXES lib)

if(NOT (CASADI_INCLUDE_DIR AND CASADI_LIBRARY))
    message(FATAL_ERROR "CasADi not found! Please install: pip install casadi")
endif()

get_filename_component(CASADI_LIBRARY_DIR ${CASADI_LIBRARY} DIRECTORY)
include_directories(SYSTEM ${CASADI_INCLUDE_DIR})
link_directories(${CASADI_LIBRARY_DIR})

# --- Executables and Linking ---
set(ALL_TARGETS main main_mute test_solvers test_R cart_pole visualize_collision_checking visualize_noisy_trj)

foreach(exe ${ALL_TARGETS})
    add_executable(${exe} src/${exe}.cpp)
    target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
    
    # Base libraries
    target_link_libraries(${exe} PRIVATE SFML::System SFML::Window SFML::Graphics ${YAML_CPP_LIB} Eigen3::Eigen ${PLATFORM_LIBS} ${CASADI_LIBRARY} OpenMP::OpenMP_CXX stomp)

    target_compile_options(${exe} PRIVATE 
        -O3 -march=native -ffast-math
    )
    
    # CasADi specific (main targets only)
    if(exe MATCHES "main")
        target_link_libraries(${exe} PRIVATE ${CASADI_LIBRARY})
        set_target_properties(${exe} PROPERTIES BUILD_RPATH "${CASADI_LIBRARY_DIR}" INSTALL_RPATH "${CASADI_LIBRARY_DIR}")
    endif()
endforeach()

# --- Summary ---
message(STATUS "=== Configuration Summary ===\nSFML: Found\nCasADi library: ${CASADI_LIBRARY}")
