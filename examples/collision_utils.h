#pragma once

#include "Trajectory.h"
#include "ObstacleMap.h"
#include <vector>

/**
 * @brief Checks if a node is in collision with any obstacle
 */
inline bool isNodeInCollision(const TrajectoryNode& node, const std::vector<ObstacleND>& obstacles) {
    for (const auto& obs : obstacles) {
        if (node.dimensions() != obs.dimensions()) {
            continue;
        }
        
        // Calculate signed distance
        float dist_to_center = (node.position - obs.center).norm();
        float signed_distance = dist_to_center - obs.radius - node.radius;
        
        // In collision if signed distance is negative
        if (signed_distance < 0.0f) {
            return true;
        }
    }
    return false;
}
