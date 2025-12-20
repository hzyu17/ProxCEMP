cmake_minimum_required(VERSION 3.10)
project(stomp_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(console_bridge REQUIRED)

# Find STOMP - adjust path as needed
find_package(stomp QUIET)

if(NOT stomp_FOUND)
    # Manual find if package config not available
    find_library(STOMP_LIBRARY NAMES stomp)
    find_path(STOMP_INCLUDE_DIR stomp/stomp.h)
    
    if(NOT STOMP_LIBRARY OR NOT STOMP_INCLUDE_DIR)
        message(FATAL_ERROR "STOMP library not found")
    endif()
    
    message(STATUS "STOMP library: ${STOMP_LIBRARY}")
    message(STATUS "STOMP include: ${STOMP_INCLUDE_DIR}")
endif()

# Print compiler info for debugging
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

add_executable(stomp_test stomp_test.cpp)

if(stomp_FOUND)
    target_link_libraries(stomp_test 
        stomp::stomp
        Eigen3::Eigen
        ${console_bridge_LIBRARIES}
    )
else()
    target_include_directories(stomp_test PRIVATE ${STOMP_INCLUDE_DIR})
    target_link_libraries(stomp_test 
        ${STOMP_LIBRARY}
        Eigen3::Eigen
        ${console_bridge_LIBRARIES}
    )
endif()

# Optional: Add address sanitizer for debugging
option(USE_ASAN "Use Address Sanitizer" OFF)
if(USE_ASAN)
    target_compile_options(stomp_test PRIVATE -fsanitize=address -g)
    target_link_options(stomp_test PRIVATE -fsanitize=address)
endif()