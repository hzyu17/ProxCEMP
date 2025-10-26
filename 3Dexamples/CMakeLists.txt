cmake_minimum_required(VERSION 3.10)
project(MotionPlanning3DVisualization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# GLM (header-only, may need manual path if not in system)
set(GLM_INCLUDE_DIR "/usr/include" CACHE PATH "Path to GLM headers")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
)

# 3D Visualization executable
add_executable(visualization_3d
    src/visualize3D.cpp
)

add_executable(visualize_noisy_trj 
    src/visualize_noisy_trj.cpp
)

add_executable(main 
    src/main.cpp
)

target_include_directories(visualization_3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(visualize_noisy_trj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(main
    OpenGL::GL
    GLEW::GLEW
    glfw
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
)

target_link_libraries(visualization_3d
    OpenGL::GL
    GLEW::GLEW
    glfw
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
)

target_link_libraries(visualize_noisy_trj
    OpenGL::GL
    GLEW::GLEW
    glfw
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
)

# Optional: 2D Visualization executable (if you want to keep both)
# add_executable(visualization_2d
#     main_visualization_2d.cpp
# )
# 
# target_link_libraries(visualization_2d
#     motion_planning
#     sfml-graphics
#     sfml-window
#     sfml-system
#     Eigen3::Eigen
#     ${YAML_CPP_LIBRARIES}
# )

# Installation
install(TARGETS visualization_3d
    RUNTIME DESTINATION bin
)

# Copy config file to build directory
configure_file(
    ${PROJECT_SOURCE_DIR}/configs/config.yaml
    ${CMAKE_BINARY_DIR}/configs/config.yaml
    COPYONLY
)
