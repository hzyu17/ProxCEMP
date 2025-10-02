#!/usr/bin/env bash

### --- Config (you can edit these defaults) ---
# Dataset to visualize:
DATASET="${DATASET:-package://motion_bench_maker/problems/box_fetch/}"

# Whether to load sensed and/or geometric scenes:
SENSED="${SENSED:-true}"
GEOMETRIC="${GEOMETRIC:-true}"

# Task index (1-based). You can override by passing an arg, e.g. ./visualize_mbm_task.sh 7
TASK_INDEX="${1:-${TASK_INDEX:-1}}"
### -------------------------------------------

# Basic validation
if ! [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] || [ "$TASK_INDEX" -lt 1 ]; then
  echo "Error: TASK_INDEX must be a positive integer. Got: $TASK_INDEX" >&2
  exit 1
fi

# Source ROS (adjust if you use a different workspace/distro)
if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
  source "$HOME/catkin_ws/devel/setup.bash"
fi

echo "Launching MotionBenchMaker visualizer"
echo "  dataset   : $DATASET"
echo "  sensed    : $SENSED"
echo "  geometric : $GEOMETRIC"
echo "  task index: $TASK_INDEX"

# Pass the same index as both start and end to show a single task
exec roslaunch motion_bench_maker visualize.launch \
  dataset:="$DATASET" \
  sensed:="$SENSED" \
  geometric:="$GEOMETRIC" \
  start:="$TASK_INDEX" \
  end:="$TASK_INDEX"
