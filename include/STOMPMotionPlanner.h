/**
 * @file StompMotionPlanner.h
 * @brief STOMP-based motion planner wrapper
 * 
 * Provides a unified interface for STOMP optimization compatible
 * with the PCE/NGD planner comparison framework.
 */
#pragma once

#include <stomp/stomp.h>
#include "StompTask.h"
#include "CollisionAvoidanceTask.h"
#include "Trajectory.h"
#include <memory>
#include <vector>
#include <iostream>

namespace pce {

/**
 * @brief STOMP Motion Planner
 * 
 * Wrapper class that integrates STOMP optimization with the
 * collision avoidance motion planning framework.
 */
class StompMotionPlanner {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXf = Eigen::VectorXf;
    
    explicit StompMotionPlanner(std::shared_ptr<CollisionAvoidanceTask> task)
        : collision_task_(task)
        , initialized_(false)
    {
    }
    
    /**
     * @brief Initialize the planner with configuration
     */
    bool initialize(const StompPlannerConfig& config) {
        config_ = config;
        
        // Validate configuration
        if (config_.start_position.size() != config_.num_dimensions ||
            config_.goal_position.size() != config_.num_dimensions) {
            std::cerr << "Error: Start/goal position dimensions mismatch\n";
            return false;
        }
        
        // Create STOMP task adapter
        stomp_task_ = std::make_shared<StompCollisionTask>(collision_task_, config_);
        
        // Initialize trajectory
        initializeTrajectory();
        
        // Create STOMP optimizer
        stomp::StompConfiguration stomp_config = config_.toStompConfig();
        stomp_ = std::make_unique<stomp::Stomp>(stomp_config, stomp_task_);
        
        initialized_ = true;
        
        std::cout << "STOMP Planner initialized:\n"
                  << "  Dimensions: " << config_.num_dimensions << "\n"
                  << "  Timesteps: " << config_.num_timesteps << "\n"
                  << "  Iterations: " << config_.num_iterations << "\n"
                  << "  Rollouts: " << config_.num_rollouts << "\n"
                  << "  Noise stddev: " << config_.noise_stddev << "\n";
        
        return true;
    }
    
    /**
     * @brief Run STOMP optimization
     */
    bool solve() {
        if (!initialized_) {
            std::cerr << "Error: Planner not initialized\n";
            return false;
        }
        
        std::cout << "\n=== Running STOMP Optimization ===\n";
        
        // Clear previous history
        stomp_task_->clearHistory();
        trajectory_history_.clear();
        
        // Store initial trajectory
        trajectory_history_.push_back(current_trajectory_);
        
        // Convert start/goal to Eigen vectors
        VectorXd start(config_.num_dimensions);
        VectorXd goal(config_.num_dimensions);
        for (size_t d = 0; d < config_.num_dimensions; ++d) {
            start(d) = config_.start_position[d];
            goal(d) = config_.goal_position[d];
        }
        
        // Run STOMP optimization
        MatrixXd optimized_params;
        bool success = stomp_->solve(start, goal, optimized_params);
        
        // Update current trajectory
        if (success || optimized_params.size() > 0) {
            current_trajectory_ = stomp_task_->matrixToTrajectory(optimized_params);
        }
        
        // Collect trajectory history from task
        const auto& task_history = stomp_task_->getTrajectoryHistory();
        for (const auto& traj : task_history) {
            trajectory_history_.push_back(traj);
        }
        
        return success;
    }
    
    /**
     * @brief Cancel optimization (thread-safe)
     */
    bool cancel() {
        if (stomp_) {
            return stomp_->cancel();
        }
        return false;
    }
    
    /**
     * @brief Reset the planner
     */
    void reset() {
        if (stomp_) {
            stomp_->clear();
        }
        initializeTrajectory();
        trajectory_history_.clear();
        if (stomp_task_) {
            stomp_task_->clearHistory();
        }
    }
    
    // Accessors
    const Trajectory& getCurrentTrajectory() const { return current_trajectory_; }
    const std::vector<Trajectory>& getTrajectoryHistory() const { return trajectory_history_; }
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Compute smoothness cost for a trajectory
     */
    float computeSmoothnessCost(const Trajectory& trajectory) const {
        if (trajectory.nodes.size() < 3) return 0.0f;
        
        float cost = 0.0f;
        for (size_t i = 1; i < trajectory.nodes.size() - 1; ++i) {
            // Compute acceleration (second derivative)
            VectorXf acc = trajectory.nodes[i+1].position 
                         - 2.0f * trajectory.nodes[i].position 
                         + trajectory.nodes[i-1].position;
            cost += acc.squaredNorm();
        }
        
        return cost * config_.control_cost_weight / (config_.delta_t * config_.delta_t);
    }

private:
    void initializeTrajectory() {
        // Create linear interpolation from start to goal
        current_trajectory_.nodes.resize(config_.num_timesteps);
        
        VectorXf start(config_.num_dimensions);
        VectorXf goal(config_.num_dimensions);
        for (size_t d = 0; d < config_.num_dimensions; ++d) {
            start(d) = config_.start_position[d];
            goal(d) = config_.goal_position[d];
        }
        
        for (size_t t = 0; t < config_.num_timesteps; ++t) {
            float alpha = static_cast<float>(t) / (config_.num_timesteps - 1);
            current_trajectory_.nodes[t].position = (1.0f - alpha) * start + alpha * goal;
        }
    }
    
    std::shared_ptr<CollisionAvoidanceTask> collision_task_;
    std::shared_ptr<StompCollisionTask> stomp_task_;
    std::unique_ptr<stomp::Stomp> stomp_;
    
    StompPlannerConfig config_;
    Trajectory current_trajectory_;
    std::vector<Trajectory> trajectory_history_;
    
    bool initialized_;
};

} // namespace pce
