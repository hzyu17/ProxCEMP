cmake_minimum_required(VERSION 2.8.12)
project(prox_ce_planner)

# C++11 required
add_compile_options(-std=c++11)

# Find catkin and dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  moveit_core
  moveit_ros_planning
  pluginlib
  class_loader
)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
    roscpp
    moveit_core
    moveit_ros_planning
    pluginlib
  DEPENDS EIGEN3
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Build the planner library
add_library(${PROJECT_NAME}
  src/prox_ce_planner.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# Mark executables and libraries for installation
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install header files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install plugin description
install(FILES prox_ce_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)