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
 * @brief Forward declaration
 */
class StompMotionPlanner;

/**
 * @brief Stateless STOMP Task adapter - all state lives in StompMotionPlanner
 * 
 * This class is intentionally minimal (just a vtable pointer + one raw pointer = 16 bytes)
 * to avoid memory issues with the STOMP library.
 */
class StompCollisionTask : public stomp::Task {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    
    StompCollisionTask() : planner_(nullptr) {
        std::cout << "StompCollisionTask constructor (size=" << sizeof(*this) << ")\n";
    }
    
    ~StompCollisionTask() {
        std::cout << "StompCollisionTask destructor\n";
    }
    
    void setPlanner(StompMotionPlanner* planner) {
        planner_ = planner;
    }
    
    // All virtual methods delegate to planner
    bool generateNoisyParameters(const MatrixXd& parameters,
                                 std::size_t start_timestep,
                                 std::size_t num_timesteps,
                                 int iteration_number,
                                 int rollout_number,
                                 MatrixXd& parameters_noise,
                                 MatrixXd& noise) override;
    
    bool computeNoisyCosts(const MatrixXd& parameters,
                           std::size_t start_timestep,
                           std::size_t num_timesteps,
                           int iteration_number,
                           int rollout_number,
                           VectorXd& costs,
                           bool& validity) override;
    
    bool computeCosts(const MatrixXd& parameters,
                      std::size_t start_timestep,
                      std::size_t num_timesteps,
                      int iteration_number,
                      VectorXd& costs,
                      bool& validity) override;
    
    bool filterNoisyParameters(std::size_t start_timestep,
                               std::size_t num_timesteps,
                               int iteration_number,
                               int rollout_number,
                               MatrixXd& parameters,
                               bool& filtered) override;
    
    bool filterParameterUpdates(std::size_t start_timestep,
                                std::size_t num_timesteps,
                                int iteration_number,
                                const MatrixXd& parameters,
                                MatrixXd& updates) override;
    
    void postIteration(std::size_t start_timestep,
                       std::size_t num_timesteps,
                       int iteration_number,
                       double cost,
                       const MatrixXd& parameters) override;
    
    void done(bool success, int total_iterations, double final_cost,
              const MatrixXd& parameters) override;

private:
    StompMotionPlanner* planner_;  // Raw pointer to avoid shared_ptr overhead
};

} // namespace pce