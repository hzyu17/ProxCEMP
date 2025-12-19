/**
 * @file TestTask.h
 * @brief Simple convex test task for solver verification
 * 
 * QuadraticBowlTask: state cost = ||x - x*||^2 where x* = straight line
 * Both J_state and J_smooth have the same optimal → solvers should reach cost ~0
 */
#pragma once

#include "task.h"
#include "Trajectory.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace pce {

/**
 * @brief Quadratic Bowl Task (CONVEX)
 * 
 * State cost = sum ||p_i - target_i||^2
 * Target = straight line interpolation
 * 
 * Since smoothness cost also has straight line as optimal,
 * the combined cost has a unique global minimum at the straight line.
 * All solvers should converge to total cost ≈ 0.
 */
class QuadraticBowlTask : public Task {
public:
    QuadraticBowlTask() = default;
    
    void initialize(size_t num_dimensions,
                   const TrajectoryNode& start,
                   const TrajectoryNode& goal,
                   size_t num_nodes,
                   float total_time) override {
        num_dims_ = num_dimensions;
        num_nodes_ = num_nodes;
        start_ = start;
        goal_ = goal;
        
        // Target = straight line (the known global optimum)
        targets_.resize(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            float t = static_cast<float>(i) / (num_nodes - 1);
            targets_[i] = (1.0f - t) * start.position + t * goal.position;
        }
        
        std::cout << "QuadraticBowlTask: optimal = straight line, expected cost = 0\n";
    }
    
    float computeStateCost(const Trajectory& traj) const override {
        return computeStateCostSimple(traj);
    }
    
    float computeStateCostSimple(const Trajectory& traj) const override {
        float cost = 0.0f;
        for (size_t i = 1; i < traj.nodes.size() - 1; ++i) {
            if (i < targets_.size()) {
                cost += (traj.nodes[i].position - targets_[i]).squaredNorm();
            }
        }
        return cost;
    }
    
    bool filterTrajectory(Trajectory& traj, int max_iter) override {
        return false;
    }
    
    void done(bool success, int iterations, float final_cost,
              const Trajectory& final_traj) override {
        float state_cost = computeStateCostSimple(final_traj);
        std::cout << "QuadraticBowlTask: state_cost=" << state_cost 
                  << (state_cost < 0.5f ? " PASS" : " FAIL") << "\n";
    }

private:
    size_t num_dims_ = 2;
    size_t num_nodes_ = 10;
    TrajectoryNode start_, goal_;
    std::vector<Eigen::VectorXf> targets_;
};

}  // namespace pce