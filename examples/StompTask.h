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
            
            // Load motion planning parameters first (these are required)
            if (const auto& mp = config["motion_planning"]) {
                num_timesteps = mp["num_nodes"].as<size_t>(num_timesteps);
                num_dimensions = mp["num_dimensions"].as<size_t>(num_dimensions);
                
                if (mp["start_position"]) {
                    start_position = mp["start_position"].as<std::vector<float>>();
                }
                if (mp["goal_position"]) {
                    goal_position = mp["goal_position"].as<std::vector<float>>();
                }
            } else {
                std::cerr << "Warning: No motion_planning section found in config\n";
            }
            
            // Load STOMP-specific parameters (optional, has defaults)
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
            } else {
                std::cout << "Note: No stomp section found, using default STOMP parameters\n";
            }
            
            // Ensure max_rollouts > num_rollouts
            if (max_rollouts <= num_rollouts) {
                max_rollouts = num_rollouts + 1;
            }
            
            // Validate required fields
            if (start_position.empty() || goal_position.empty()) {
                std::cerr << "Error: start_position and goal_position are required\n";
                return false;
            }
            
            if (start_position.size() != num_dimensions) {
                std::cerr << "Error: start_position size doesn't match num_dimensions\n";
                return false;
            }
            
            if (goal_position.size() != num_dimensions) {
                std::cerr << "Error: goal_position size doesn't match num_dimensions\n";
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading STOMP config: " << e.what() << std::endl;
            return false;
        }
    }
    
    stomp::StompConfiguration toStompConfig() const {
        stomp::StompConfiguration cfg;
        
        // Ensure all fields are explicitly set
        cfg.num_timesteps = static_cast<int>(num_timesteps);
        cfg.num_iterations = static_cast<int>(num_iterations);
        cfg.num_rollouts = static_cast<int>(num_rollouts);
        cfg.num_dimensions = static_cast<int>(num_dimensions);
        cfg.delta_t = delta_t;
        cfg.control_cost_weight = control_cost_weight;
        cfg.exponentiated_cost_sensitivity = exponentiated_cost_sensitivity;
        cfg.initialization_method = stomp::TrajectoryInitializations::LINEAR_INTERPOLATION;
        
        // Ensure max_rollouts > num_rollouts (STOMP requirement)
        cfg.max_rollouts = static_cast<int>(max_rollouts);
        if (cfg.max_rollouts <= cfg.num_rollouts) {
            cfg.max_rollouts = cfg.num_rollouts + 1;
        }
        
        // num_iterations_after_valid
        cfg.num_iterations_after_valid = static_cast<int>(num_iterations_after_valid);
        
        return cfg;
    }
};


/**
 * @brief STOMP Task adapter for CollisionAvoidanceTask
 * 
 * Based on working StompTaskMinimal pattern.
 */
class StompCollisionTask : public stomp::Task {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXf = Eigen::VectorXf;
    
    // Default constructor - matches working pattern
    StompCollisionTask() {
        std::cout << "StompCollisionTask constructor (size=" << sizeof(*this) << ")\n";
    }
    
    ~StompCollisionTask() {
        std::cout << "StompCollisionTask destructor\n";
    }
    
    // Deferred initialization - call after STOMP is created
    void setup(std::shared_ptr<CollisionAvoidanceTask> collision_task,
               float noise_stddev, float noise_decay) {
        collision_task_ = collision_task;
        noise_stddev_ = noise_stddev;
        noise_decay_ = noise_decay;
        rng_.seed(std::random_device{}());
        std::cout << "StompCollisionTask setup complete\n";
    }
    
    bool generateNoisyParameters(const MatrixXd& parameters,
                                 std::size_t start_timestep,
                                 std::size_t num_timesteps,
                                 int iteration_number,
                                 int rollout_number,
                                 MatrixXd& parameters_noise,
                                 MatrixXd& noise) override 
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
        return true;
    }
    
    bool computeNoisyCosts(const MatrixXd& parameters,
                           std::size_t start_timestep,
                           std::size_t num_timesteps,
                           int iteration_number,
                           int rollout_number,
                           VectorXd& costs,
                           bool& validity) override 
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
                    
                    // Vectorized collision cost
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
    
    bool filterNoisyParameters(std::size_t start_timestep,
                               std::size_t num_timesteps,
                               int iteration_number,
                               int rollout_number,
                               MatrixXd& parameters,
                               bool& filtered) override 
    {
        filtered = false;
        return true;
    }
    
    bool filterParameterUpdates(std::size_t start_timestep,
                                std::size_t num_timesteps,
                                int iteration_number,
                                const MatrixXd& parameters,
                                MatrixXd& updates) override 
    {
        updates.col(0).setZero();
        updates.col(updates.cols() - 1).setZero();
        return true;
    }
    
    void postIteration(std::size_t start_timestep,
                       std::size_t num_timesteps,
                       int iteration_number,
                       double cost,
                       const MatrixXd& parameters) override 
    {
        current_iteration_ = iteration_number;
        
        // Store trajectory
        Trajectory traj;
        traj.nodes.resize(num_timesteps);
        for (size_t t = 0; t < num_timesteps; ++t) {
            traj.nodes[t].position.resize(parameters.rows());
            for (int d = 0; d < parameters.rows(); ++d) {
                traj.nodes[t].position(d) = static_cast<float>(parameters(d, t));
            }
        }
        trajectory_history_.push_back(traj);
        
        if (iteration_number % 10 == 0) {
            std::cout << "  STOMP Iteration " << iteration_number << ": cost = " << cost << "\n";
        }
    }
    
    void done(bool success, int total_iterations, double final_cost,
              const MatrixXd& parameters) override 
    {
        std::cout << "STOMP " << (success ? "succeeded" : "failed")
                  << " after " << total_iterations << " iterations"
                  << " with cost " << final_cost << "\n";
        converged_ = success;
        final_cost_ = static_cast<float>(final_cost);
    }
    
    // Accessors
    const std::vector<Trajectory>& getTrajectoryHistory() const { return trajectory_history_; }
    void clearHistory() { trajectory_history_.clear(); }
    bool hasConverged() const { return converged_; }
    float getFinalCost() const { return final_cost_; }

private:
    // Member variables - initialized in setup() after STOMP construction
    std::shared_ptr<CollisionAvoidanceTask> collision_task_;
    float noise_stddev_ = 20.0f;
    float noise_decay_ = 0.95f;
    std::mt19937 rng_;
    int current_iteration_ = 0;
    
    std::vector<Trajectory> trajectory_history_;
    float final_cost_ = 0.0f;
    bool converged_ = false;
};

} // namespace pce