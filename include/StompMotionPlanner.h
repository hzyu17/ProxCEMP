/**
 * @file StompMotionPlanner.h
 * @brief STOMP-based motion planner wrapper (Refactored for visualization consistency)
 * 
 * Modified to match PCEMotionPlanner's data storage patterns:
 * - Stores samples per iteration for visualization
 * - Uses trajectory_history_ naming convention
 * - Provides OptimizationHistory-compatible interface
 */
#pragma once

#include <stomp/stomp.h>
#include "StompTask.h"
#include "CollisionAvoidanceTask.h"
#include "Trajectory.h"
#include <memory>
#include <vector>
#include <random>
#include <iostream>

namespace pce {

/**
 * @brief Data for a single optimization iteration (matches PCE visualization format)
 */
struct StompIterationData {
    std::vector<Trajectory> samples;    // Noisy rollouts for this iteration
    Trajectory mean_trajectory;          // Mean/updated trajectory
    float cost = 0.0f;
    int iteration_number = 0;
};

/**
 * @brief Complete optimization history (compatible with visualization)
 */
struct StompOptimizationHistory {
    std::vector<StompIterationData> iterations;
    
    // Convenience accessors matching OptimizationHistory interface
    size_t size() const { return iterations.size(); }
    bool empty() const { return iterations.empty(); }
};

/**
 * @brief STOMP Motion Planner - holds all optimization state
 * 
 * Refactored to be consistent with PCEMotionPlanner:
 * - Stores sample trajectories per iteration
 * - Provides trajectory_history_ style interface
 * - Compatible with showTrajectoryEvolution visualization
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
        , noise_stddev_(20.0f)
        , noise_decay_(0.95f)
        , current_iteration_(0)
        , final_cost_(0.0f)
        , converged_(false)
    {
        rng_.seed(std::random_device{}());
    }
    
    ~StompMotionPlanner() {
        if (stomp_) {
            delete stomp_;
            stomp_ = nullptr;
        }
    }
    
    bool initialize(const StompPlannerConfig& config) {
        config_ = config;
        noise_stddev_ = config.noise_stddev;
        noise_decay_ = config.noise_decay;
        
        // Validate configuration
        if (config_.num_dimensions == 0) {
            std::cerr << "Error: num_dimensions cannot be 0\n";
            return false;
        }
        
        if (config_.num_timesteps < 10) {
            std::cerr << "Error: num_timesteps must be at least 10\n";
            return false;
        }
        
        if (config_.start_position.size() != config_.num_dimensions ||
            config_.goal_position.size() != config_.num_dimensions) {
            std::cerr << "Error: Start/goal position dimensions mismatch\n";
            return false;
        }
        
        if (config_.max_rollouts <= config_.num_rollouts) {
            config_.max_rollouts = config_.num_rollouts + 1;
        }
        
        initializeTrajectory();
        
        stomp::StompConfiguration stomp_config = config_.toStompConfig();
        
        std::cout << "STOMP Config Debug:\n"
                  << "  num_dimensions: " << stomp_config.num_dimensions << "\n"
                  << "  num_timesteps: " << stomp_config.num_timesteps << "\n"
                  << "  num_iterations: " << stomp_config.num_iterations << "\n"
                  << "  num_rollouts: " << stomp_config.num_rollouts << "\n"
                  << "  max_rollouts: " << stomp_config.max_rollouts << "\n";
        
        // Create minimal stateless task
        std::cout << "Creating STOMP task adapter...\n";
        stomp_task_ = std::make_shared<StompCollisionTask>();
        std::cout << "Task size: " << sizeof(StompCollisionTask) << " bytes\n";
        
        // Set back-pointer BEFORE creating STOMP
        stomp_task_->setPlanner(this);
        
        std::cout << "Creating STOMP optimizer...\n";
        try {
            stomp_ = new stomp::Stomp(stomp_config, stomp_task_);
        } catch (const std::exception& e) {
            std::cerr << "Error creating STOMP: " << e.what() << "\n";
            return false;
        }
        
        if (!stomp_) {
            std::cerr << "Error: STOMP is null\n";
            return false;
        }
        
        initialized_ = true;
        std::cout << "STOMP Planner initialized successfully\n";
        return true;
    }
    
    bool solve() {
        if (!initialized_) {
            std::cerr << "Error: Planner not initialized\n";
            return false;
        }
        
        std::cout << "\n=== Running STOMP Optimization ===\n";
        
        // Clear history and prepare for new optimization
        optimization_history_.iterations.clear();
        trajectory_history_.clear();
        current_iteration_samples_.clear();
        
        // Store initial trajectory as iteration 0
        StompIterationData initial_data;
        initial_data.mean_trajectory = current_trajectory_;
        initial_data.iteration_number = 0;
        initial_data.cost = 0.0f;
        optimization_history_.iterations.push_back(initial_data);
        trajectory_history_.push_back(current_trajectory_);
        
        VectorXd start(config_.num_dimensions);
        VectorXd goal(config_.num_dimensions);
        for (size_t d = 0; d < config_.num_dimensions; ++d) {
            start(d) = config_.start_position[d];
            goal(d) = config_.goal_position[d];
        }
        
        MatrixXd optimized_params;
        bool success = false;
        
        try {
            success = stomp_->solve(start, goal, optimized_params);
        } catch (const std::exception& e) {
            std::cerr << "Exception during solve: " << e.what() << "\n";
            return false;
        }
        
        if (success || optimized_params.size() > 0) {
            current_trajectory_ = matrixToTrajectory(optimized_params);
        }
        
        return success;
    }
    
    // ==================== Task callback implementations ====================
    
    bool taskGenerateNoisyParameters(const MatrixXd& parameters,
                                     std::size_t num_timesteps,
                                     int iteration_number,
                                     int rollout_number,
                                     MatrixXd& parameters_noise,
                                     MatrixXd& noise) 
    {
        float decayed_stddev = noise_stddev_ * std::pow(noise_decay_, iteration_number);
        std::normal_distribution<double> dist(0.0, decayed_stddev);
        
        noise.resize(parameters.rows(), parameters.cols());
        parameters_noise.resize(parameters.rows(), parameters.cols());
        
        for (int d = 0; d < parameters.rows(); ++d) {
            for (int t = 0; t < parameters.cols(); ++t) {
                if (t == 0 || t == parameters.cols() - 1) {
                    noise(d, t) = 0.0;
                } else {
                    noise(d, t) = dist(rng_);
                }
            }
        }
        
        parameters_noise = parameters + noise;
        
        // Store this sample for visualization (like PCE stores sample_trajectories)
        if (rollout_number >= 0) {
            Trajectory sample_traj = matrixToTrajectory(parameters_noise);
            current_iteration_samples_.push_back(sample_traj);
        }
        
        return true;
    }
    
    bool taskComputeCosts(const MatrixXd& parameters,
                          std::size_t num_timesteps,
                          int rollout_number,
                          VectorXd& costs,
                          bool& validity) 
    {
        costs.resize(num_timesteps);
        costs.setZero();
        
        if (collision_task_) {
            const auto& obs = collision_task_->getObstacleSoA();
            if (!obs.empty()) {
                const float epsilon = collision_task_->getConfig().epsilon_sdf;
                const float sigma = collision_task_->getConfig().sigma_obs;
                
                for (size_t t = 1; t < num_timesteps - 1; ++t) {
                    VectorXf pos(parameters.rows());
                    for (int d = 0; d < parameters.rows(); ++d) {
                        pos(d) = static_cast<float>(parameters(d, t));
                    }
                    
                    Eigen::MatrixXf diff = obs.centers.colwise() - pos;
                    Eigen::VectorXf distances = diff.colwise().norm();
                    Eigen::VectorXf sdfs = distances - obs.combined_radii;
                    Eigen::VectorXf hinges = (epsilon - sdfs.array()).max(0.0f);
                    costs(t) = sigma * hinges.squaredNorm();
                }
            }
        }
        
        validity = (costs.sum() < 1e-3);
        return true;
    }
    
    bool taskFilterUpdates(MatrixXd& updates) {
        updates.col(0).setZero();
        updates.col(updates.cols() - 1).setZero();
        return true;
    }
    
    void taskPostIteration(int iteration_number, double cost, const MatrixXd& parameters) {
        current_iteration_ = iteration_number;
        
        Trajectory mean_traj = matrixToTrajectory(parameters);
        
        // Store iteration data with samples (like PCE's trajectory_history_)
        StompIterationData iter_data;
        iter_data.samples = std::move(current_iteration_samples_);
        iter_data.mean_trajectory = mean_traj;
        iter_data.cost = static_cast<float>(cost);
        iter_data.iteration_number = iteration_number;
        
        optimization_history_.iterations.push_back(iter_data);
        trajectory_history_.push_back(mean_traj);
        
        // Clear for next iteration
        current_iteration_samples_.clear();
        
        if (iteration_number % 10 == 0) {
            std::cout << "  STOMP Iteration " << iteration_number 
                      << ": cost = " << cost 
                      << ", samples = " << iter_data.samples.size() << "\n";
        }
    }
    
    void taskDone(bool success, int total_iterations, double final_cost) {
        std::cout << "STOMP " << (success ? "succeeded" : "failed")
                  << " after " << total_iterations << " iterations"
                  << " with cost " << final_cost << "\n";
        converged_ = success;
        final_cost_ = static_cast<float>(final_cost);
    }
    
    // ==================== Accessors (consistent with PCE interface) ====================
    
    const Trajectory& getCurrentTrajectory() const { return current_trajectory_; }
    
    /** @brief Get trajectory history (mean trajectories, like PCE's trajectory_history_) */
    const std::vector<Trajectory>& getTrajectoryHistory() const { 
        return trajectory_history_; 
    }
    
    /** @brief Get full optimization history with samples (for visualization) */
    const StompOptimizationHistory& getOptimizationHistory() const { 
        return optimization_history_; 
    }
    
    /** @brief Get number of iterations completed */
    size_t getNumIterations() const { 
        return optimization_history_.iterations.size(); 
    }
    
    bool isInitialized() const { return initialized_; }
    bool hasConverged() const { return converged_; }
    float getFinalCost() const { return final_cost_; }
    
    float computeSmoothnessCost(const Trajectory& trajectory) const {
        if (trajectory.nodes.size() < 3) return 0.0f;
        
        float cost = 0.0f;
        for (size_t i = 1; i < trajectory.nodes.size() - 1; ++i) {
            VectorXf acc = trajectory.nodes[i+1].position 
                         - 2.0f * trajectory.nodes[i].position 
                         + trajectory.nodes[i-1].position;
            cost += acc.squaredNorm();
        }
        
        return cost * config_.control_cost_weight / (config_.delta_t * config_.delta_t);
    }

private:
    void initializeTrajectory() {
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
    
    // Core references
    std::shared_ptr<CollisionAvoidanceTask> collision_task_;
    std::shared_ptr<StompCollisionTask> stomp_task_;
    stomp::Stomp* stomp_;
    
    // Configuration
    StompPlannerConfig config_;
    
    // Optimization state
    float noise_stddev_;
    float noise_decay_;
    std::mt19937 rng_;
    int current_iteration_;
    
    // Results - consistent with PCE naming
    Trajectory current_trajectory_;
    std::vector<Trajectory> trajectory_history_;              // Mean trajectories (like PCE)
    StompOptimizationHistory optimization_history_;           // Full history with samples
    std::vector<Trajectory> current_iteration_samples_;       // Temp storage during iteration
    
    float final_cost_;
    bool converged_;
    bool initialized_;
};

// ==================== StompCollisionTask method implementations ====================

inline bool StompCollisionTask::generateNoisyParameters(
    const MatrixXd& parameters,
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    int rollout_number,
    MatrixXd& parameters_noise,
    MatrixXd& noise) 
{
    if (!planner_) return false;
    return planner_->taskGenerateNoisyParameters(parameters, num_timesteps, iteration_number, 
                                                  rollout_number, parameters_noise, noise);
}

inline bool StompCollisionTask::computeNoisyCosts(
    const MatrixXd& parameters,
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    int rollout_number,
    VectorXd& costs,
    bool& validity) 
{
    if (!planner_) {
        costs = VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    return planner_->taskComputeCosts(parameters, num_timesteps, rollout_number, costs, validity);
}

inline bool StompCollisionTask::computeCosts(
    const MatrixXd& parameters,
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    VectorXd& costs,
    bool& validity) 
{
    return computeNoisyCosts(parameters, start_timestep, num_timesteps, 
                             iteration_number, -1, costs, validity);
}

inline bool StompCollisionTask::filterNoisyParameters(
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    int rollout_number,
    MatrixXd& parameters,
    bool& filtered) 
{
    filtered = false;
    return true;
}

inline bool StompCollisionTask::filterParameterUpdates(
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    const MatrixXd& parameters,
    MatrixXd& updates) 
{
    if (!planner_) return true;
    return planner_->taskFilterUpdates(updates);
}

inline void StompCollisionTask::postIteration(
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    double cost,
    const MatrixXd& parameters) 
{
    if (planner_) {
        planner_->taskPostIteration(iteration_number, cost, parameters);
    }
}

inline void StompCollisionTask::done(
    bool success, 
    int total_iterations, 
    double final_cost,
    const MatrixXd& parameters) 
{
    if (planner_) {
        planner_->taskDone(success, total_iterations, final_cost);
    }
}

} // namespace pce