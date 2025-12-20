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
        , stomp_(nullptr)
    {
    }
    
    ~StompMotionPlanner() {
        // Clean up STOMP first (it holds reference to task)
        if (stomp_) {
            delete stomp_;
            stomp_ = nullptr;
        }
        // shared_ptr handles task cleanup automatically
    }
    
    /**
     * @brief Initialize the planner with configuration
     */
    bool initialize(const StompPlannerConfig& config) {
        config_ = config;
        
        // Validate configuration thoroughly
        if (config_.num_dimensions == 0) {
            std::cerr << "Error: num_dimensions cannot be 0\n";
            return false;
        }
        
        if (config_.num_timesteps < 10) {
            std::cerr << "Error: num_timesteps must be at least 10 (STOMP uses finite differences)\n";
            return false;
        }
        
        if (config_.start_position.size() != config_.num_dimensions) {
            std::cerr << "Error: start_position size (" << config_.start_position.size() 
                      << ") != num_dimensions (" << config_.num_dimensions << ")\n";
            return false;
        }
        
        if (config_.goal_position.size() != config_.num_dimensions) {
            std::cerr << "Error: goal_position size (" << config_.goal_position.size() 
                      << ") != num_dimensions (" << config_.num_dimensions << ")\n";
            return false;
        }
        
        if (config_.num_rollouts == 0) {
            std::cerr << "Error: num_rollouts cannot be 0\n";
            return false;
        }
        
        if (config_.max_rollouts <= config_.num_rollouts) {
            config_.max_rollouts = config_.num_rollouts + 1;
            std::cout << "Warning: Adjusted max_rollouts to " << config_.max_rollouts << "\n";
        }
        
        // Initialize trajectory first
        initializeTrajectory();
        
        // Create STOMP configuration
        stomp::StompConfiguration stomp_config = config_.toStompConfig();
        
        // Debug output
        std::cout << "STOMP Config Debug:\n"
                  << "  num_dimensions: " << stomp_config.num_dimensions << "\n"
                  << "  num_timesteps: " << stomp_config.num_timesteps << "\n"
                  << "  num_iterations: " << stomp_config.num_iterations << "\n"
                  << "  num_rollouts: " << stomp_config.num_rollouts << "\n"
                  << "  max_rollouts: " << stomp_config.max_rollouts << "\n"
                  << "  delta_t: " << stomp_config.delta_t << "\n";
        
        // Create STOMP task adapter using simple constructor (like minimal test)
        std::cout << "Creating STOMP task adapter...\n";
        auto stomp_task = std::make_shared<StompCollisionTask>();
        
        if (!stomp_task) {
            std::cerr << "Error: Failed to create STOMP task adapter\n";
            return false;
        }
        std::cout << "STOMP task adapter created (size=" << sizeof(StompCollisionTask) << ")\n";
        
        // Store for later use
        stomp_task_ptr_ = stomp_task;
        
        // Explicitly cast to stomp::TaskPtr
        stomp::TaskPtr task_for_stomp = std::static_pointer_cast<stomp::Task>(stomp_task_ptr_);
        
        std::cout << "Task pointer created, use_count: " << task_for_stomp.use_count() 
                  << ", creating STOMP optimizer...\n";
        
        // Create STOMP optimizer (like minimal test)
        try {
            stomp_ = new stomp::Stomp(stomp_config, task_for_stomp);
        } catch (const std::exception& e) {
            std::cerr << "Error creating STOMP optimizer: " << e.what() << "\n";
            return false;
        } catch (...) {
            std::cerr << "Error: Unknown exception while creating STOMP optimizer\n";
            return false;
        }
        
        if (!stomp_) {
            std::cerr << "Error: STOMP optimizer is null after creation\n";
            return false;
        }
        
        std::cout << "STOMP created successfully!\n";
        
        // NOW setup the task with collision data (after STOMP is created)
        stomp_task_ptr_->setup(collision_task_, config_.noise_stddev, config_.noise_decay);
        
        initialized_ = true;
        
        std::cout << "STOMP Planner initialized successfully:\n"
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
        if (stomp_task_ptr_) {
            stomp_task_ptr_->clearHistory();
        }
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
        bool success = false;
        
        try {
            success = stomp_->solve(start, goal, optimized_params);
        } catch (const std::exception& e) {
            std::cerr << "Exception during STOMP solve: " << e.what() << "\n";
            return false;
        }
        
        // Update current trajectory
        if (success || optimized_params.size() > 0) {
            current_trajectory_ = matrixToTrajectory(optimized_params);
        }
        
        // Collect trajectory history from task
        if (stomp_task_ptr_) {
            const auto& task_history = stomp_task_ptr_->getTrajectoryHistory();
            for (const auto& traj : task_history) {
                trajectory_history_.push_back(traj);
            }
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
        if (stomp_task_ptr_) {
            stomp_task_ptr_->clearHistory();
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
    
    Trajectory matrixToTrajectory(const MatrixXd& params) const {
        Trajectory traj;
        if (params.size() == 0) return traj;
        
        size_t num_dims = params.rows();
        size_t num_timesteps = params.cols();
        
        traj.nodes.resize(num_timesteps);
        for (size_t t = 0; t < num_timesteps; ++t) {
            traj.nodes[t].position.resize(num_dims);
            for (size_t d = 0; d < num_dims; ++d) {
                traj.nodes[t].position(d) = static_cast<float>(params(d, t));
            }
        }
        return traj;
    }
    
    std::shared_ptr<CollisionAvoidanceTask> collision_task_;
    std::shared_ptr<StompCollisionTask> stomp_task_ptr_;  // Shared ptr for STOMP
    stomp::Stomp* stomp_;                                  // Raw pointer to avoid ABI issues
    
    StompPlannerConfig config_;
    Trajectory current_trajectory_;
    std::vector<Trajectory> trajectory_history_;
    
    bool initialized_;
};

} // namespace pce