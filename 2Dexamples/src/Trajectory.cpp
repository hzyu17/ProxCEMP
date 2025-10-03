#include "../include/Trajectory.h"
#include <iostream>
#include <algorithm>
#include <cmath>

/**
 * @brief Linear interpolation function for PathNodes (x, y, radius).
 * @param p1 Start point.
 * @param p2 End point.
 * @param alpha Interpolation parameter (0.0 to 1.0).
 * @return Interpolated point (PathNode).
 */
static PathNode lerp(const PathNode& p1, const PathNode& p2, float alpha) {
    float x = p1.x + alpha * (p2.x - p1.x);
    float y = p1.y + alpha * (p2.y - p1.y);
    float radius = p1.radius + alpha * (p2.radius - p1.radius);
    
    return {x, y, radius};
}

// Cubic Bezier helper function implementation (uses P0, P1, P2, P3 control points)
static PathNode cubic_bezier(const PathNode& P0, const PathNode& P1, const PathNode& P2, const PathNode& P3, float t) {
    float one_minus_t = 1.0f - t;
    float one_minus_t2 = one_minus_t * one_minus_t;
    float one_minus_t3 = one_minus_t2 * one_minus_t;
    float t2 = t * t;
    float t3 = t2 * t;

    // Blending functions
    float b0 = one_minus_t3;
    float b1 = 3 * one_minus_t2 * t;
    float b2 = 3 * one_minus_t * t2;
    float b3 = t3;

    float x = b0 * P0.x + b1 * P1.x + b2 * P2.x + b3 * P3.x;
    float y = b0 * P0.y + b1 * P1.y + b2 * P2.y + b3 * P3.y;
    float radius = b0 * P0.radius + b1 * P1.radius + b2 * P2.radius + b3 * P3.radius;

    return {x, y, radius};
}


Trajectory generateInterpolatedTrajectoryLinear(const PathNode& start, const PathNode& goal, size_t num_steps, float total_time) {
    if (num_steps < 2) num_steps = 2;

    Trajectory traj;
    traj.total_time = total_time; // Set total time
    traj.goal_index = num_steps - 1;
    
    for (size_t i = 0; i < num_steps; ++i) {
        float alpha = (float)i / (num_steps - 1); // 0.0 to 1.0
        PathNode node = lerp(start, goal, alpha);
        traj.nodes.push_back(node);
    }

    return traj;
}

Trajectory generateInterpolatedTrajectoryBezier(const PathNode& start, const PathNode& goal, size_t num_steps, float total_time) {
    if (num_steps < 2) num_steps = 2;

    Trajectory traj;
    traj.total_time = total_time; // Set total time
    traj.goal_index = num_steps - 1;

    // --- Control Point Generation (P1 and P2) ---
    float dx = goal.x - start.x;
    float dy = goal.y - start.y;
    float length = std::sqrt(dx * dx + dy * dy);
    
    // Normalized vector from start to goal
    float nx = dx / length;
    float ny = dy / length;

    // Perpendicular vector
    float px = -ny;
    float py = nx;
    
    // Curve magnitude (30% of total length)
    float curve_mag = 0.3f * length; 
    
    // P1: Offset forward and sideways
    PathNode P1 = {
        start.x + nx * curve_mag + px * curve_mag,
        start.y + ny * curve_mag + py * curve_mag,
        start.radius + 0.33f * (goal.radius - start.radius)
    };

    // P2: Offset backward from goal and sideways
    PathNode P2 = {
        goal.x - nx * curve_mag + px * curve_mag,
        goal.y - ny * curve_mag + py * curve_mag,
        start.radius + 0.66f * (goal.radius - start.radius)
    };

    // P0 and P3 are the given start and goal points
    const PathNode P0 = start;
    const PathNode P3 = goal;

    for (size_t i = 0; i < num_steps; ++i) {
        float t = (float)i / (num_steps - 1); // 0.0 to 1.0
        PathNode node = cubic_bezier(P0, P1, P2, P3, t);
        traj.nodes.push_back(node);
    }
    
    return traj;
}
