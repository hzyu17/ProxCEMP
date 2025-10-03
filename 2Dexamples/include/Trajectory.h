#pragma once

#include <vector>

/**
 * @brief Represents a single point along the trajectory with a collision checking radius.
 */
struct PathNode {
    float x;
    float y;
    float radius; // Collision checking radius for this specific point
};

/**
 * @brief Represents a sequence of connected PathNodes forming a time-parametrized trajectory.
 */
struct Trajectory {
    std::vector<PathNode> nodes;
    float total_time;       // Total time duration of the trajectory (seconds)
    size_t start_index = 0; // Index of the start node (usually 0)
    size_t goal_index;      // Index of the goal node (usually nodes.size() - 1)
};

/**
 * @brief Generates a straight line trajectory between start and goal using linear interpolation.
 * @param start Start point of the trajectory.
 * @param goal Goal point of the trajectory.
 * @param num_steps The total number of nodes to generate (including start/goal).
 * @param total_time The total time duration for the trajectory.
 * @return The generated Trajectory structure.
 */
Trajectory generateInterpolatedTrajectoryLinear(const PathNode& start, const PathNode& goal, size_t num_steps, float total_time);

/**
 * @brief Generates a smooth trajectory between start and goal using cubic Bezier interpolation.
 * @param start Start point of the trajectory.
 * @param goal Goal point of the trajectory.
 * @param num_steps The total number of nodes to generate (including start/goal).
 * @param total_time The total time duration for the trajectory.
 * @return The generated Trajectory structure.
 */
Trajectory generateInterpolatedTrajectoryBezier(const PathNode& start, const PathNode& goal, size_t num_steps, float total_time);
