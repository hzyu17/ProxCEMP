#pragma once

#include <vector>
#include <Eigen/Dense>

/**
 * @brief Represents a single point along the trajectory with a collision checking radius.
 * Pure configuration space representation - no workspace coordinates.
 */
struct TrajectoryNode {
    Eigen::VectorXf position;  // N-dimensional configuration space position
    float radius;              // Collision checking radius
    
    // Default constructor
    TrajectoryNode() : position(Eigen::VectorXf::Zero(2)), radius(0.0f) {}
    
    // N-dimensional constructor
    TrajectoryNode(const Eigen::VectorXf& pos, float r) : position(pos), radius(r) {}
    
    // 2D convenience constructor
    TrajectoryNode(float q1, float q2, float r) : radius(r) {
        position.resize(2);
        position << q1, q2;
    }
    
    // 3D convenience constructor
    TrajectoryNode(float q1, float q2, float q3, float r) : radius(r) {
        position.resize(3);
        position << q1, q2, q3;
    }
    
    // Get dimensionality
    size_t dimensions() const { 
        return static_cast<size_t>(position.size()); 
    }
};

/**
 * @brief Represents a sequence of connected PathNodes forming a time-parametrized trajectory.
 * Pure configuration space representation.
 */
struct Trajectory {
    std::vector<TrajectoryNode> nodes;
    float total_time;       // Total time duration of the trajectory (seconds)
    size_t start_index = 0; // Index of the start node (usually 0)
    size_t goal_index;      // Index of the goal node (usually nodes.size() - 1)
    
    // Get dimensionality of the trajectory
    size_t dimensions() const {
        return nodes.empty() ? 0 : nodes[0].dimensions();
    }
};

/**
 * @brief Linear interpolation function for PathNodes (position, radius).
 * @param p1 Start point.
 * @param p2 End point.
 * @param alpha Interpolation parameter (0.0 to 1.0).
 * @return Interpolated point (TrajectoryNode).
 */
inline TrajectoryNode lerp(const TrajectoryNode& p1, const TrajectoryNode& p2, float alpha) {
    if (p1.dimensions() != p2.dimensions()) {
        throw std::runtime_error("Cannot interpolate PathNodes with different dimensions");
    }
    
    Eigen::VectorXf position = p1.position + alpha * (p2.position - p1.position);
    float radius = p1.radius + alpha * (p2.radius - p1.radius);
    
    TrajectoryNode result(position, radius);
    return result;
}

/**
 * @brief Cubic Bezier helper function for N-dimensional PathNodes
 * @param P0 Start control point
 * @param P1 First intermediate control point
 * @param P2 Second intermediate control point
 * @param P3 End control point
 * @param t Parameter (0.0 to 1.0)
 * @return Interpolated point
 */
inline TrajectoryNode cubic_bezier(const TrajectoryNode& P0, const TrajectoryNode& P1, 
                             const TrajectoryNode& P2, const TrajectoryNode& P3, float t) {
    if (P0.dimensions() != P1.dimensions() || 
        P1.dimensions() != P2.dimensions() || 
        P2.dimensions() != P3.dimensions()) {
        throw std::runtime_error("All control points must have the same dimensions");
    }
    
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

    // Blend positions using Eigen's vectorized operations
    Eigen::VectorXf position = b0 * P0.position + b1 * P1.position + 
                               b2 * P2.position + b3 * P3.position;
    
    // Blend radius
    float radius = b0 * P0.radius + b1 * P1.radius + 
                   b2 * P2.radius + b3 * P3.radius;

    return TrajectoryNode(position, radius);
}

/**
 * @brief Generates a straight line trajectory between start and goal using linear interpolation.
 * Works for any dimensionality.
 * @param start Start point of the trajectory.
 * @param goal Goal point of the trajectory.
 * @param num_steps The total number of nodes to generate (including start/goal).
 * @param total_time The total time duration for the trajectory.
 * @return The generated Trajectory structure.
 */
inline Trajectory generateInterpolatedTrajectoryLinear(const TrajectoryNode& start, 
                                                       const TrajectoryNode& goal, 
                                                       size_t num_steps, 
                                                       float total_time) {
    if (num_steps < 2) num_steps = 2;
    
    if (start.dimensions() != goal.dimensions()) {
        throw std::runtime_error("Start and goal must have the same dimensions");
    }

    Trajectory traj;
    traj.total_time = total_time;
    traj.goal_index = num_steps - 1;
    traj.nodes.reserve(num_steps);
    
    for (size_t i = 0; i < num_steps; ++i) {
        float alpha = static_cast<float>(i) / (num_steps - 1);
        TrajectoryNode node = lerp(start, goal, alpha);
        traj.nodes.push_back(node);
    }

    return traj;
}

/**
 * @brief Generates a smooth trajectory between start and goal using cubic Bezier interpolation.
 * Works for any dimensionality N >= 2.
 * @param start Start point of the trajectory.
 * @param goal Goal point of the trajectory.
 * @param num_steps The total number of nodes to generate (including start/goal).
 * @param total_time The total time duration for the trajectory.
 * @return The generated Trajectory structure.
 */
inline Trajectory generateInterpolatedTrajectoryBezier(const TrajectoryNode& start, 
                                                       const TrajectoryNode& goal, 
                                                       size_t num_steps, 
                                                       float total_time) {
    if (num_steps < 2) num_steps = 2;
    
    if (start.dimensions() != goal.dimensions()) {
        throw std::runtime_error("Start and goal must have the same dimensions");
    }
    
    const size_t dim = start.dimensions();
    if (dim < 2) {
        throw std::runtime_error("Bezier trajectory requires at least 2 dimensions");
    }

    Trajectory traj;
    traj.total_time = total_time;
    traj.goal_index = num_steps - 1;
    traj.nodes.reserve(num_steps);

    // Control Point Generation
    Eigen::VectorXf direction = goal.position - start.position;
    float length = direction.norm();
    Eigen::VectorXf n = direction / length;
    
    // Choose a perpendicular vector for offset
    Eigen::VectorXf p = Eigen::VectorXf::Zero(dim);
    
    if (dim == 2) {
        p << -n(1), n(0);
    } else if (dim >= 3) {
        Eigen::VectorXf candidate = Eigen::VectorXf::Zero(dim);
        candidate(1) = 1.0f;
        
        if (std::abs(n.dot(candidate)) > 0.9f) {
            candidate = Eigen::VectorXf::Zero(dim);
            candidate(0) = 1.0f;
        }
        
        p = candidate - n.dot(candidate) * n;
        p.normalize();
    }
    
    float curve_mag = 0.3f * length;
    
    TrajectoryNode P1(start.position + n * curve_mag + p * curve_mag,
                start.radius + 0.33f * (goal.radius - start.radius));
    TrajectoryNode P2(goal.position - n * curve_mag + p * curve_mag,
                start.radius + 0.66f * (goal.radius - start.radius));

    const TrajectoryNode& P0 = start;
    const TrajectoryNode& P3 = goal;

    for (size_t i = 0; i < num_steps; ++i) {
        float t = static_cast<float>(i) / (num_steps - 1);
        TrajectoryNode node = cubic_bezier(P0, P1, P2, P3, t);
        traj.nodes.push_back(node);
    }
    
    return traj;
}

