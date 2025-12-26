cmake_minimum_required(VERSION 3.10)
project(PCEMotionPlanner VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# Build Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

# Build type (Release for performance, Debug for debugging)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# Options
# =============================================================================
# ASan for STOMP/PCE/NGD (NOT for CasADi - incompatible with IPOPT)
option(ENABLE_STOMP_ASAN "Enable Address Sanitizer for STOMP library and executables" OFF)

message(STATUS "")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "STOMP ASan enabled: ${ENABLE_STOMP_ASAN}")

# =============================================================================
# Compiler Flags - Set BEFORE add_subdirectory(external/stomp)
# =============================================================================
# This ensures the stomp library itself gets compiled with ASan when enabled
if(ENABLE_STOMP_ASAN)
    # ASan flags for stomp library and STOMP executables
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
    message(STATUS "ASan enabled globally for stomp library")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# =============================================================================
# Subdirectories - stomp MUST come first to get ASan flags
# =============================================================================
add_subdirectory(external/stomp)

# Examples comes after - it will handle CasADi targets specially
add_subdirectory(Examples2d3d)

# =============================================================================
# Find Dependencies
# =============================================================================
find_package(Eigen3 REQUIRED)
find_package(CASADI REQUIRED)
find_package(OpenMP REQUIRED)

# Include CasADi headers
include_directories(${CASADI_INCLUDE_DIRS})

# =============================================================================
# Header-Only Library Interface
# =============================================================================
add_library(pce_motion_planner INTERFACE)

target_include_directories(pce_motion_planner INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/examples>
    $<INSTALL_INTERFACE:include>
    ${CASADI_INCLUDE_DIRS}
)

target_link_libraries(pce_motion_planner INTERFACE 
    Eigen3::Eigen 
    ${CASADI_LIBRARIES} 
    OpenMP::OpenMP_CXX 
    stomp
)

# =============================================================================
# Installation Rules
# =============================================================================
include(GNUInstallDirs)

# Install header files
install(FILES
    include/PCEMotionPlanner.h
    include/NGDMotionPlanner.h
    include/task.h
    include/Trajectory.h
    include/MotionPlanner.h
    include/ForwardKinematics.h
    include/CasadiMotionPlanner.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pce
)

# Install library target
install(TARGETS pce_motion_planner
    EXPORT PCEMotionPlannerTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install CMake config files (for find_package support)
install(EXPORT PCEMotionPlannerTargets
    FILE PCEMotionPlannerTargets.cmake
    NAMESPACE PCE::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PCEMotionPlanner
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PCEMotionPlannerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PCEMotionPlannerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PCEMotionPlannerConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PCEMotionPlanner
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PCEMotionPlannerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/PCEMotionPlannerConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PCEMotionPlanner
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== PCEMotionPlanner Configuration ===")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  STOMP ASan:        ${ENABLE_STOMP_ASAN}")
message(STATUS "  CasADi ASan:       OFF (incompatible with IPOPT plugin)")
message(STATUS "")
if(ENABLE_STOMP_ASAN)
    message(STATUS "NOTE: stomp library compiled WITH ASan")
    message(STATUS "NOTE: main_casadi compiled WITHOUT ASan (special handling)")
    message(STATUS "NOTE: All other executables compiled WITH ASan")
else()
    message(STATUS "NOTE: All targets compiled WITHOUT ASan")
endif()
message(STATUS "")