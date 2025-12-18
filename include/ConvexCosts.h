/**
 * @file TestTask.h
 * @brief Test tasks with known global minima for verifying CasADi solvers
 * 
 * These tasks provide simple, convex cost functions where:
 * 1. The global minimum is analytically known
 * 2. Any local minimum is also a global minimum (convexity)
 * 3. Solvers should reliably converge to the optimal solution
 */
#pragma once

#include "task.h"
#include "Trajectory.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

namespace pce {

/**
 * @brief Test Task 1: Quadratic Attractor (CONVEX)
 * 
 * Cost = sum over waypoints of ||p_i - target_i||^2
 * 
 * The optimal trajectory is the straight line from start to goal.
 * This is strictly convex, so solvers MUST find the global minimum.
 */
class QuadraticAttractorTask : public Task {
public:
    QuadraticAttractorTask() = default;
    
    void initialize(size_t num_dimensions,
                   const TrajectoryNode& start,
                   const TrajectoryNode& goal,
                   size_t num_nodes,
                   float total_time) override {
        num_dims_ = num_dimensions;
        num_nodes_ = num_nodes;
        start_ = start;
        goal_ = goal;
        
        // Set target trajectory as straight line (known optimal)
        target_positions_.clear();
        target_positions_.resize(num_nodes);
        
        for (size_t i = 0; i < num_nodes; ++i) {
            float t = static_cast<float>(i) / (num_nodes - 1);
            target_positions_[i] = (1.0f - t) * start.position + t * goal.position;
        }
        
        std::cout << "QuadraticAttractorTask initialized:\n";
        std::cout << "  Dimensions: " << num_dims_ << "\n";
        std::cout << "  Nodes: " << num_nodes_ << "\n";
        std::cout << "  Optimal collision cost: 0.0 (straight line)\n";
    }
    
    float computeStateCost(const Trajectory& traj) const override {
        return computeStateCostSimple(traj);
    }
    
    float computeStateCostSimple(const Trajectory& traj) const override {
        float cost = 0.0f;
        
        size_t N = std::min(traj.nodes.size(), target_positions_.size());
        
        // Only penalize interior waypoints (start/goal are fixed)
        for (size_t i = 1; i < N - 1; ++i) {
            Eigen::VectorXf diff = traj.nodes[i].position - target_positions_[i];
            cost += diff.squaredNorm();
        }
        
        return cost;
    }
    
    bool filterTrajectory(Trajectory& traj, int max_iter) override {
        (void)traj;
        (void)max_iter;
        return false;
    }
    
    void done(bool success, int iterations, float final_cost,
              const Trajectory& final_traj) override {
        std::cout << "\n=== QuadraticAttractorTask Results ===\n";
        std::cout << "Success: " << (success ? "yes" : "no") << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Final total cost: " << final_cost << "\n";
        
        float collision_cost = computeStateCostSimple(final_traj);
        std::cout << "Final collision cost: " << collision_cost << "\n";
        std::cout << "Expected optimal: 0.0\n";
        
        if (collision_cost < 1e-3) {
            std::cout << "✓ PASSED: Solver found global minimum!\n";
        } else if (collision_cost < 0.1) {
            std::cout << "~ CLOSE: Solver nearly converged (cost=" << collision_cost << ")\n";
        } else {
            std::cout << "✗ FAILED: Solver did not converge to global minimum\n";
        }
    }
    
    std::vector<Eigen::VectorXf> getOptimalPositions() const {
        return target_positions_;
    }

private:
    size_t num_dims_ = 2;
    size_t num_nodes_ = 10;
    TrajectoryNode start_, goal_;
    std::vector<Eigen::VectorXf> target_positions_;
};


/**
 * @brief Test Task 2: Quadratic Bowl (Simplest Possible)
 * 
 * Cost = ||Y - Y*||^2 where Y* is the straight line configuration
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
        
        optimal_positions_.resize(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            float t = static_cast<float>(i) / (num_nodes - 1);
            optimal_positions_[i] = (1.0f - t) * start.position + t * goal.position;
        }
        
        std::cout << "QuadraticBowlTask initialized (simplest convex test)\n";
        std::cout << "  Global minimum collision cost: 0.0\n";
    }
    
    float computeStateCost(const Trajectory& traj) const override {
        return computeStateCostSimple(traj);
    }
    
    float computeStateCostSimple(const Trajectory& traj) const override {
        float cost = 0.0f;
        
        for (size_t i = 1; i < traj.nodes.size() - 1; ++i) {
            if (i < optimal_positions_.size()) {
                Eigen::VectorXf diff = traj.nodes[i].position - optimal_positions_[i];
                cost += diff.squaredNorm();
            }
        }
        
        return cost;
    }
    
    bool filterTrajectory(Trajectory& traj, int max_iter) override {
        (void)traj;
        (void)max_iter;
        return false;
    }
    
    void done(bool success, int iterations, float final_cost,
              const Trajectory& final_traj) override {
        float coll_cost = computeStateCostSimple(final_traj);
        
        std::cout << "\n=== QuadraticBowlTask Results ===\n";
        std::cout << "Final collision cost: " << coll_cost << "\n";
        std::cout << "Expected: 0.0\n";
        
        bool passed = (coll_cost < 1e-3);
        std::cout << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    }

private:
    size_t num_dims_ = 2;
    size_t num_nodes_ = 10;
    TrajectoryNode start_, goal_;
    std::vector<Eigen::VectorXf> optimal_positions_;
};


/**
 * @brief Test Task 3: Zero Collision Cost (smoothness-only test)
 */
class ZeroCollisionTask : public Task {
public:
    void initialize(size_t num_dimensions,
                   const TrajectoryNode& start,
                   const TrajectoryNode& goal,
                   size_t num_nodes,
                   float total_time) override {
        std::cout << "ZeroCollisionTask: Testing smoothness-only optimization\n";
        std::cout << "  Optimal: straight line trajectory\n";
        start_ = start;
        goal_ = goal;
        num_nodes_ = num_nodes;
        num_dims_ = num_dimensions;
    }
    
    float computeStateCost(const Trajectory& traj) const override {
        return 0.0f;
    }
    
    float computeStateCostSimple(const Trajectory& traj) const override {
        return 0.0f;
    }
    
    bool filterTrajectory(Trajectory& traj, int max_iter) override {
        return false;
    }
    
    void done(bool success, int iterations, float final_cost,
              const Trajectory& final_traj) override {
        std::cout << "\n=== ZeroCollisionTask Results ===\n";
        
        float max_deviation = 0.0f;
        for (size_t i = 0; i < final_traj.nodes.size(); ++i) {
            float t = static_cast<float>(i) / (final_traj.nodes.size() - 1);
            Eigen::VectorXf expected = (1.0f - t) * start_.position + t * goal_.position;
            float dev = (final_traj.nodes[i].position - expected).norm();
            max_deviation = std::max(max_deviation, dev);
        }
        
        std::cout << "Max deviation from straight line: " << max_deviation << "\n";
        std::cout << "Final smoothness cost: " << final_cost << "\n";
        std::cout << (max_deviation < 0.1f ? "✓ PASSED" : "✗ FAILED") << "\n";
    }

private:
    TrajectoryNode start_, goal_;
    size_t num_nodes_;
    size_t num_dims_;
};


/**
 * @brief Test Task 4: Offset Attractor (curved optimal path)
 * 
 * Optimal trajectory passes through a via-point offset from the straight line.
 * Still convex, but tests if solver can move away from straight-line initialization.
 */
class OffsetAttractorTask : public Task {
public:
    OffsetAttractorTask(float offset = 2.0f) : offset_(offset) {}
    
    void initialize(size_t num_dimensions,
                   const TrajectoryNode& start,
                   const TrajectoryNode& goal,
                   size_t num_nodes,
                   float total_time) override {
        num_dims_ = num_dimensions;
        num_nodes_ = num_nodes;
        start_ = start;
        goal_ = goal;
        
        // Create curved optimal path through via-point
        Eigen::VectorXf midpoint = 0.5f * (start.position + goal.position);
        Eigen::VectorXf offset_vec = Eigen::VectorXf::Zero(num_dimensions);
        if (num_dimensions >= 2) {
            offset_vec(1) = offset_;
        }
        via_point_ = midpoint + offset_vec;
        
        // Quadratic Bezier curve
        target_positions_.resize(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            float t = static_cast<float>(i) / (num_nodes - 1);
            target_positions_[i] = (1-t)*(1-t)*start.position + 
                                   2*(1-t)*t*via_point_ + 
                                   t*t*goal.position;
        }
        
        std::cout << "OffsetAttractorTask initialized:\n";
        std::cout << "  Offset: " << offset_ << "\n";
        std::cout << "  Optimal path curves through via-point\n";
    }
    
    float computeStateCost(const Trajectory& traj) const override {
        return computeStateCostSimple(traj);
    }
    
    float computeStateCostSimple(const Trajectory& traj) const override {
        float cost = 0.0f;
        
        for (size_t i = 1; i < traj.nodes.size() - 1; ++i) {
            if (i < target_positions_.size()) {
                Eigen::VectorXf diff = traj.nodes[i].position - target_positions_[i];
                cost += diff.squaredNorm();
            }
        }
        
        return cost;
    }
    
    bool filterTrajectory(Trajectory& traj, int max_iter) override {
        return false;
    }
    
    void done(bool success, int iterations, float final_cost,
              const Trajectory& final_traj) override {
        float coll_cost = computeStateCostSimple(final_traj);
        
        std::cout << "\n=== OffsetAttractorTask Results ===\n";
        std::cout << "Final collision cost: " << coll_cost << "\n";
        
        size_t mid_idx = final_traj.nodes.size() / 2;
        float via_dist = (final_traj.nodes[mid_idx].position - via_point_).norm();
        std::cout << "Distance from via-point at midpoint: " << via_dist << "\n";
        
        bool passed = (coll_cost < 0.1f);
        std::cout << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    }

private:
    size_t num_dims_ = 2;
    size_t num_nodes_ = 10;
    float offset_ = 2.0f;
    TrajectoryNode start_, goal_;
    Eigen::VectorXf via_point_;
    std::vector<Eigen::VectorXf> target_positions_;
};

}  // namespace pce