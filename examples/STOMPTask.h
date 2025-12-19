/**
 * @file StompTask.h
 * @brief Adapter from CollisionAvoidanceTask to STOMP's Task interface
 * 
 * Bridges the pce::CollisionAvoidanceTask with STOMP's optimization framework.
 */
#pragma once

#include <stomp/task.h>
#include <stomp/utils.h>
#include "CollisionAvoidanceTask.h"
#include "Trajectory.h"
#include <Eigen/Dense>
#include <random>
#include <memory>
#include <iostream>

namespace pce {

/**
 * @brief Configuration for STOMP-based motion planning
 */
struct StompPlannerConfig {
    // STOMP algorithm parameters
    size_t num_timesteps = 50;
    size_t num_iterations = 100;
    size_t num_rollouts = 10;
    size_t max_rollouts = 20;
    size_t num_iterations_after_valid = 5;
    float delta_t = 0.1f;
    float control_cost_weight = 0.0f;
    float exponentiated_cost_sensitivity = 10.0f;
    
    // Noise generation parameters
    float noise_stddev = 20.0f;
    float noise_decay = 0.95f;
    
    // Motion planning parameters
    size_t num_dimensions = 2;
    std::vector<float> start_position;
    std::vector<float> goal_position;
    
    // Convergence criteria
    float cost_tolerance = 1e-4f;
    
    bool loadFromFile(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            
            if (const auto& mp = config["motion_planning"]) {
                num_timesteps = mp["num_nodes"].as<size_t>(num_timesteps);
                num_dimensions = mp["num_dimensions"].as<size_t>(num_dimensions);
                
                if (mp["start_position"]) {
                    start_position = mp["start_position"].as<std::vector<float>>();
                }
                if (mp["goal_position"]) {
                    goal_position = mp["goal_position"].as<std::vector<float>>();
                }
            }
            
            if (const auto& stomp = config["stomp"]) {
                num_iterations = stomp["num_iterations"].as<size_t>(num_iterations);
                num_rollouts = stomp["num_rollouts"].as<size_t>(num_rollouts);
                max_rollouts = stomp["max_rollouts"].as<size_t>(max_rollouts);
                num_iterations_after_valid = stomp["num_iterations_after_valid"].as<size_t>(num_iterations_after_valid);
                delta_t = stomp["delta_t"].as<float>(delta_t);
                control_cost_weight = stomp["control_cost_weight"].as<float>(control_cost_weight);
                exponentiated_cost_sensitivity = stomp["exponentiated_cost_sensitivity"].as<float>(exponentiated_cost_sensitivity);
                noise_stddev = stomp["noise_stddev"].as<float>(noise_stddev);
                noise_decay = stomp["noise_decay"].as<float>(noise_decay);
                cost_tolerance = stomp["cost_tolerance"].as<float>(cost_tolerance);
            }
            
            // Ensure max_rollouts > num_rollouts
            if (max_rollouts <= num_rollouts) {
                max_rollouts = num_rollouts + 1;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading STOMP config: " << e.what() << std::endl;
            return false;
        }
    }
    
    stomp::StompConfiguration toStompConfig() const {
        stomp::StompConfiguration cfg;
        cfg.num_timesteps = static_cast<int>(num_timesteps);
        cfg.num_iterations = static_cast<int>(num_iterations);
        cfg.num_rollouts = static_cast<int>(num_rollouts);
        cfg.max_rollouts = static_cast<int>(max_rollouts);
        cfg.num_iterations_after_valid = static_cast<int>(num_iterations_after_valid);
        cfg.num_dimensions = static_cast<int>(num_dimensions);
        cfg.delta_t = delta_t;
        cfg.control_cost_weight = control_cost_weight;
        cfg.exponentiated_cost_sensitivity = exponentiated_cost_sensitivity;
        cfg.initialization_method = stomp::TrajectoryInitializations::LINEAR_INTERPOLATION;
        return cfg;
    }
};


/**
 * @brief STOMP Task adapter for CollisionAvoidanceTask
 * 
 * Implements the stomp::Task interface to enable STOMP optimization
 * on collision avoidance problems.
 */
class StompCollisionTask : public stomp::Task {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXf = Eigen::VectorXf;
    
    StompCollisionTask(std::shared_ptr<CollisionAvoidanceTask> collision_task,
                       const StompPlannerConfig& config)
        : collision_task_(collision_task)
        , config_(config)
        , current_iteration_(0)
        , rng_(std::random_device{}())
    {
        // Initialize noise covariance matrix (diagonal)
        noise_stddev_ = config.noise_stddev;
    }
    
    /**
     * @brief Generate noisy parameters for exploration
     */
    bool generateNoisyParameters(const MatrixXd& parameters,
                                 std::size_t start_timestep,
                                 std::size_t num_timesteps,
                                 int iteration_number,
                                 int rollout_number,
                                 MatrixXd& parameters_noise,
                                 MatrixXd& noise) override
    {
        // Compute decayed noise standard deviation
        float decayed_stddev = noise_stddev_ * std::pow(config_.noise_decay, iteration_number);
        std::normal_distribution<double> dist(0.0, decayed_stddev);
        
        // Generate noise for each dimension and timestep
        noise.resize(parameters.rows(), parameters.cols());
        parameters_noise.resize(parameters.rows(), parameters.cols());
        
        for (int d = 0; d < parameters.rows(); ++d) {
            for (int t = 0; t < parameters.cols(); ++t) {
                // Don't add noise to start and end points
                if (t == 0 || t == parameters.cols() - 1) {
                    noise(d, t) = 0.0;
                } else {
                    noise(d, t) = dist(rng_);
                }
            }
        }
        
        parameters_noise = parameters + noise;
        return true;
    }
    
    /**
     * @brief Filter noisy parameters (optional smoothing)
     */
    bool filterNoisyParameters(std::size_t start_timestep,
                               std::size_t num_timesteps,
                               int iteration_number,
                               int rollout_number,
                               MatrixXd& parameters,
                               bool& filtered) override
    {
        // Apply simple moving average smoothing
        filtered = false;
        
        if (parameters.cols() < 5) return true;
        
        MatrixXd smoothed = parameters;
        const int window = 3;
        
        for (int d = 0; d < parameters.rows(); ++d) {
            for (int t = window/2; t < parameters.cols() - window/2 - 1; ++t) {
                double sum = 0.0;
                for (int w = -window/2; w <= window/2; ++w) {
                    sum += parameters(d, t + w);
                }
                smoothed(d, t) = sum / window;
            }
        }
        
        // Preserve start and end
        smoothed.col(0) = parameters.col(0);
        smoothed.col(parameters.cols() - 1) = parameters.col(parameters.cols() - 1);
        
        parameters = smoothed;
        filtered = true;
        
        return true;
    }
    
    /**
     * @brief Compute costs for noisy rollouts
     */
    bool computeNoisyCosts(const MatrixXd& parameters,
                           std::size_t start_timestep,
                           std::size_t num_timesteps,
                           int iteration_number,
                           int rollout_number,
                           VectorXd& costs,
                           bool& validity) override
    {
        // Convert to trajectory and compute collision cost
        Trajectory traj = matrixToTrajectory(parameters);
        
        costs.resize(num_timesteps);
        costs.setZero();
        
        // Compute per-timestep collision costs
        const auto& obstacles = collision_task_->getObstacleSoA();
        if (!obstacles.empty()) {
            for (size_t t = 1; t < num_timesteps - 1; ++t) {
                VectorXf pos = traj.nodes[t].position;
                costs(t) = computePositionCost(pos, obstacles);
            }
        }
        
        // Check validity (no collisions)
        validity = (costs.sum() < 1e-3);
        
        return true;
    }
    
    /**
     * @brief Compute costs for optimized trajectory
     */
    bool computeCosts(const MatrixXd& parameters,
                      std::size_t start_timestep,
                      std::size_t num_timesteps,
                      int iteration_number,
                      VectorXd& costs,
                      bool& validity) override
    {
        return computeNoisyCosts(parameters, start_timestep, num_timesteps,
                                 iteration_number, -1, costs, validity);
    }
    
    /**
     * @brief Filter parameter updates (optional)
     */
    bool filterParameterUpdates(std::size_t start_timestep,
                                std::size_t num_timesteps,
                                int iteration_number,
                                const MatrixXd& parameters,
                                MatrixXd& updates) override
    {
        // Don't update start and end points
        updates.col(0).setZero();
        updates.col(updates.cols() - 1).setZero();
        return true;
    }
    
    /**
     * @brief Post-iteration callback
     */
    void postIteration(std::size_t start_timestep,
                       std::size_t num_timesteps,
                       int iteration_number,
                       double cost,
                       const MatrixXd& parameters) override
    {
        current_iteration_ = iteration_number;
        
        // Store trajectory for history
        Trajectory traj = matrixToTrajectory(parameters);
        trajectory_history_.push_back(traj);
        cost_history_.push_back(static_cast<float>(cost));
        
        if (iteration_number % 10 == 0) {
            std::cout << "  STOMP Iteration " << iteration_number 
                      << ": cost = " << cost << std::endl;
        }
    }
    
    /**
     * @brief Optimization complete callback
     */
    void done(bool success, int total_iterations, double final_cost,
              const MatrixXd& parameters) override
    {
        std::cout << "STOMP optimization " << (success ? "succeeded" : "failed")
                  << " after " << total_iterations << " iterations"
                  << " with cost " << final_cost << std::endl;
                  
        final_trajectory_ = matrixToTrajectory(parameters);
        final_cost_ = static_cast<float>(final_cost);
        converged_ = success;
    }
    
    // Accessors
    const std::vector<Trajectory>& getTrajectoryHistory() const { return trajectory_history_; }
    const std::vector<float>& getCostHistory() const { return cost_history_; }
    const Trajectory& getFinalTrajectory() const { return final_trajectory_; }
    float getFinalCost() const { return final_cost_; }
    bool hasConverged() const { return converged_; }
    
    void clearHistory() {
        trajectory_history_.clear();
        cost_history_.clear();
    }
    
    /**
     * @brief Convert trajectory to STOMP matrix format
     */
    static MatrixXd trajectoryToMatrix(const Trajectory& traj) {
        if (traj.nodes.empty()) return MatrixXd();
        
        size_t num_dims = traj.nodes[0].position.size();
        size_t num_timesteps = traj.nodes.size();
        
        MatrixXd params(num_dims, num_timesteps);
        for (size_t t = 0; t < num_timesteps; ++t) {
            for (size_t d = 0; d < num_dims; ++d) {
                params(d, t) = traj.nodes[t].position(d);
            }
        }
        return params;
    }
    
    /**
     * @brief Convert STOMP matrix to trajectory
     */
    Trajectory matrixToTrajectory(const MatrixXd& params) const {
        Trajectory traj;
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

private:
    float computePositionCost(const VectorXf& pos, const ObstacleDataSoA& obs) const {
        if (obs.empty()) return 0.0f;
        
        const float epsilon = collision_task_->getConfig().epsilon_sdf;
        const float sigma = collision_task_->getConfig().sigma_obs;
        
        // Vectorized distance computation
        Eigen::MatrixXf diff = obs.centers.colwise() - pos;
        Eigen::VectorXf distances = diff.colwise().norm();
        Eigen::VectorXf sdfs = distances - obs.combined_radii;
        Eigen::VectorXf hinges = (epsilon - sdfs.array()).max(0.0f);
        
        return sigma * hinges.squaredNorm();
    }
    
    std::shared_ptr<CollisionAvoidanceTask> collision_task_;
    StompPlannerConfig config_;
    
    float noise_stddev_;
    int current_iteration_;
    std::mt19937 rng_;
    
    std::vector<Trajectory> trajectory_history_;
    std::vector<float> cost_history_;
    Trajectory final_trajectory_;
    float final_cost_ = 0.0f;
    bool converged_ = false;
};

} // namespace pce