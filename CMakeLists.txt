cmake_minimum_required(VERSION 3.10)
project(PCEMotionPlanner VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

# Build type (Release for performance, Debug for debugging)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Optional: Enable ASan for debugging (disabled by default)
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

add_subdirectory(external/stomp)
add_subdirectory(Examples2d3d)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(CASADI REQUIRED)
find_package(OpenMP REQUIRED)

# Include Casadi headers
include_directories(${CASADI_INCLUDE_DIRS})

# Define header-only library interface
add_library(pce_motion_planner INTERFACE)

# Specify include directories for the library
target_include_directories(pce_motion_planner INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CASADI_INCLUDE_DIRS}
)

# Link Eigen3
target_link_libraries(pce_motion_planner INTERFACE 
    Eigen3::Eigen 
    ${CASADI_LIBRARIES} 
    OpenMP::OpenMP_CXX 
    stomp
)

# Installation rules
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