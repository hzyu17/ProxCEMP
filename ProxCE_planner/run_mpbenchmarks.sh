#!/bin/bash

# --- ROS Motion Planning Benchmarking Script ---

# 1. Define Workspace and Setup
CATKIN_WS=~/catkin_ws
SETUP_FILE="$CATKIN_WS/devel/setup.bash"

# Check if the workspace exists and source the setup file
if [ ! -f "$SETUP_FILE" ]; then
    echo "FATAL ERROR: ROS setup file not found at $SETUP_FILE"
    echo "Please ensure you have built your catkin_ws and the path is correct."
    exit 1
fi

echo "Sourcing ROS environment from $SETUP_FILE..."
source "$SETUP_FILE"

# --- 2. Benchmark Configuration ---

# This script launches the 'benchmark.launch' file from the 'motion_bench_maker' 
# package, specifying the dataset location and the list of OMPL planners to test.

# Variables for the benchmark (optional, but good practice)
BENCHMARK_PACKAGE="motion_bench_maker"
LAUNCH_FILE="benchmark.launch"
DATASET_PATH="package://motion_bench_maker/problems/box_fetch/"
PLANNERS_LIST="RRTConnect,ProxCE"
# RESULT_PATH="$HOME/git/ProxCEMP/benchmark_logs"
RESULT_PATH="$CATKIN_WS/src/ProxCEMP/motion_bench_maker/ProxCE_planner/benchmark_logs"

echo "Starting motion planning benchmark..."
echo "  Dataset: ${DATASET_PATH}"
echo "  Planners: ${PLANNERS_LIST}"

# --- 3. Execute Benchmark ---

# The core roslaunch command
roslaunch "${BENCHMARK_PACKAGE}" "${LAUNCH_FILE}" \
    dataset:="${DATASET_PATH}" \
    planners:="${PLANNERS_LIST}" \
    results:="${RESULT_PATH}"

# Check the exit status of the roslaunch command
if [ $? -eq 0 ]; then
    echo "Benchmark command executed successfully. Results should be available in the output directory defined in the benchmark configuration."
else
    echo "ERROR: Benchmark execution failed. Check the ROS logs for details."
fi
