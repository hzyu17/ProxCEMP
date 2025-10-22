cmake_minimum_required(VERSION 3.22)
project(moveit_planners_pcem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# PCEM planner library
add_library(moveit_pcem_planner SHARED
  src/pcem_planner_manager.cpp
  src/pcem_planning_context.cpp
  src/pcem_optimizer.cpp
)

target_include_directories(moveit_pcem_planner
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/moveit_planners_pcem>
)

ament_target_dependencies(moveit_pcem_planner
  moveit_core
  moveit_ros_planning
  pluginlib
  rclcpp
  Eigen3
)

# Plugin description
pluginlib_export_plugin_description_file(moveit_core pcem_planner_plugin_description.xml)

# Install
install(
  TARGETS moveit_pcem_planner
  EXPORT moveit_pcem_plannerTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include/moveit_planners_pcem
)

ament_export_targets(moveit_pcem_plannerTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  moveit_core
  moveit_ros_planning
  pluginlib
  rclcpp
  Eigen3
)

ament_package()
