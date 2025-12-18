#pragma once
#include "../include/PCEMotionPlanner.h"
#include "../include/Trajectory.h"

inline float computeTotalCost(const ProximalCrossEntropyMotionPlanner& planner, const Trajectory& traj) {
    return planner.computeStateCost(traj, planner.getObstacles()) +
           planner.computeSmoothnessCost(traj);
}