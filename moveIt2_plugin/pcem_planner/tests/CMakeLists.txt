cmake_minimum_required(VERSION 3.10)
project(pce_motion_planner_tests)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(Eigen3 REQUIRED)

# Find your PCE motion planner package
find_package(pce_motion_planner REQUIRED)

# Find GTest
find_package(ament_cmake_gtest REQUIRED)

# Add test executable
ament_add_gtest(test_pce_motion_planner
    test_pce_motion_planner.cpp
)

# Include directories
target_include_directories(test_pce_motion_planner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

# Link libraries
ament_target_dependencies(test_pce_motion_planner
    rclcpp
    geometry_msgs
    nav_msgs
    visualization_msgs
    moveit_msgs
    moveit_core
    moveit_ros_planning
    pce_motion_planner
)

target_link_libraries(test_pce_motion_planner
    Eigen3::Eigen
)

# Install test executable
install(TARGETS test_pce_motion_planner
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)

ament_package()
