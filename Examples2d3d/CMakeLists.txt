cmake_minimum_required(VERSION 3.10)
project(2DExample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# =============================================================================
# Platform-specific configuration
# =============================================================================
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

# =============================================================================
# Dependencies
# =============================================================================
find_package(SFML 3 COMPONENTS System Window Graphics Audio Network REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# =============================================================================
# CasADi Detection
# =============================================================================
set(CASADI_SEARCH_PATHS
    /usr/local/lib/python3.12/dist-packages/casadi
    /usr/local/lib/python3.11/dist-packages/casadi
    /usr/local/lib/python3.10/dist-packages/casadi
    /usr/lib/python3/dist-packages/casadi
    $ENV{HOME}/.local/lib/python3.12/site-packages/casadi
    $ENV{HOME}/.local/lib/python3.11/site-packages/casadi
    $ENV{HOME}/.local/lib/python3.10/site-packages/casadi
    /usr/local
    /usr
    /opt/casadi
)

find_path(CASADI_INCLUDE_DIR NAMES casadi/casadi.hpp PATHS ${CASADI_SEARCH_PATHS} PATH_SUFFIXES include)
find_library(CASADI_LIBRARY NAMES casadi PATHS ${CASADI_SEARCH_PATHS} PATH_SUFFIXES lib)

if(NOT (CASADI_INCLUDE_DIR AND CASADI_LIBRARY))
    message(FATAL_ERROR "CasADi not found! Please install: pip install casadi")
endif()

get_filename_component(CASADI_LIBRARY_DIR ${CASADI_LIBRARY} DIRECTORY)
include_directories(SYSTEM ${CASADI_INCLUDE_DIR})
link_directories(${CASADI_LIBRARY_DIR})

# =============================================================================
# Define Target Groups
# =============================================================================

# CasADi targets - CANNOT use ASan (incompatible with IPOPT plugin)
# These are SKIPPED when ENABLE_STOMP_ASAN=ON
set(CASADI_TARGETS main_casadi main main3d)

# STOMP/PCE/NGD targets - use ASan when ENABLE_STOMP_ASAN=ON
set(STOMP_TARGETS main_stomp main_mute)

# Visualization targets - use ASan when ENABLE_STOMP_ASAN=ON
set(VIZ_TARGETS visualize_collision visualize_collision3d visualize_noisy_trj visualize_noisy_trj3d)

# =============================================================================
# Common include directories and libraries
# =============================================================================
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples
)

set(COMMON_LIBS
    SFML::System
    SFML::Window
    SFML::Graphics
    ${YAML_CPP_LIB}
    Eigen3::Eigen
    ${PLATFORM_LIBS}
    OpenMP::OpenMP_CXX
    stomp
)

# =============================================================================
# Build CasADi Targets (ONLY when ASan is OFF)
# =============================================================================
if(ENABLE_STOMP_ASAN)
    message(STATUS "")
    message(STATUS "*** CasADi targets SKIPPED (incompatible with ASan) ***")
    message(STATUS "*** To build main_casadi, use: cmake -DENABLE_STOMP_ASAN=OFF .. ***")
    message(STATUS "")
else()
    foreach(exe ${CASADI_TARGETS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${exe}.cpp")
            add_executable(${exe} src/${exe}.cpp)
            
            target_include_directories(${exe} PRIVATE ${COMMON_INCLUDE_DIRS})
            
            target_link_libraries(${exe} PRIVATE 
                ${COMMON_LIBS}
                ${CASADI_LIBRARY}
            )
            
            # Optimization flags (NO ASan)
            target_compile_options(${exe} PRIVATE -O3 -march=native -ffast-math)
            
            # CasADi RPATH
            set_target_properties(${exe} PROPERTIES 
                BUILD_RPATH "${CASADI_LIBRARY_DIR}" 
                INSTALL_RPATH "${CASADI_LIBRARY_DIR}"
            )
            
            message(STATUS "  ${exe}: CasADi target (NO ASan)")
        else()
            message(WARNING "Source file not found: src/${exe}.cpp")
        endif()
    endforeach()
endif()

# =============================================================================
# Build STOMP/PCE/NGD Targets
# =============================================================================
foreach(exe ${STOMP_TARGETS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${exe}.cpp")
        add_executable(${exe} src/${exe}.cpp)
        
        target_include_directories(${exe} PRIVATE ${COMMON_INCLUDE_DIRS})
        
        target_link_libraries(${exe} PRIVATE 
            ${COMMON_LIBS}
            ${CASADI_LIBRARY}
        )
        
        # CasADi RPATH
        set_target_properties(${exe} PROPERTIES 
            BUILD_RPATH "${CASADI_LIBRARY_DIR}" 
            INSTALL_RPATH "${CASADI_LIBRARY_DIR}"
        )
        
        if(ENABLE_STOMP_ASAN)
            # ASan flags already set globally, just add debug info
            target_compile_options(${exe} PRIVATE -g -O1)
            message(STATUS "  ${exe}: STOMP target (WITH ASan)")
        else()
            target_compile_options(${exe} PRIVATE -O3 -march=native -ffast-math)
            message(STATUS "  ${exe}: STOMP target (NO ASan)")
        endif()
    else()
        message(WARNING "Source file not found: src/${exe}.cpp")
    endif()
endforeach()

# =============================================================================
# Build Visualization Targets
# =============================================================================
foreach(exe ${VIZ_TARGETS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${exe}.cpp")
        add_executable(${exe} src/${exe}.cpp)
        
        target_include_directories(${exe} PRIVATE ${COMMON_INCLUDE_DIRS})
        
        target_link_libraries(${exe} PRIVATE 
            ${COMMON_LIBS}
            ${CASADI_LIBRARY}
        )
        
        set_target_properties(${exe} PROPERTIES 
            BUILD_RPATH "${CASADI_LIBRARY_DIR}" 
            INSTALL_RPATH "${CASADI_LIBRARY_DIR}"
        )
        
        if(ENABLE_STOMP_ASAN)
            target_compile_options(${exe} PRIVATE -g -O1)
            message(STATUS "  ${exe}: Viz target (WITH ASan)")
        else()
            target_compile_options(${exe} PRIVATE -O3 -march=native -ffast-math)
            message(STATUS "  ${exe}: Viz target (NO ASan)")
        endif()
    else()
        message(WARNING "Source file not found: src/${exe}.cpp")
    endif()
endforeach()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== Examples2d3d Configuration Summary ===")
message(STATUS "SFML: Found")
message(STATUS "CasADi library: ${CASADI_LIBRARY}")
message(STATUS "ENABLE_STOMP_ASAN: ${ENABLE_STOMP_ASAN}")
message(STATUS "")
if(ENABLE_STOMP_ASAN)
    message(STATUS "Mode: STOMP DEBUGGING (ASan enabled)")
    message(STATUS "  - stomp library: WITH ASan")
    message(STATUS "  - main_casadi:   SKIPPED (incompatible)")
    message(STATUS "  - main, main3d:  WITH ASan")
else()
    message(STATUS "Mode: PRODUCTION (ASan disabled)")
    message(STATUS "  - stomp library: NO ASan")
    message(STATUS "  - main_casadi:   Built (works)")
    message(STATUS "  - main, main3d:  Built (STOMP may crash)")
endif()
message(STATUS "")