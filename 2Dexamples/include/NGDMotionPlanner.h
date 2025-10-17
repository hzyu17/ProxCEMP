#pragma once

#include "MotionPlanner.h"
#include <random>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <algorithm> 
#include <Eigen/Dense>

/**
 * @brief Natural Gradient Descent (NGD) for Trajectory Optimization.
 */
class NGDMotionPlanner : public MotionPlanner {
public:
    /**
     * @brief Constructor that initializes the planner with obstacles and config.
     * @param obs A reference to the list of obstacles in the environment.
     * @param config The YAML node containing the planner's hyperparameters.
     */
    NGDMotionPlanner() {
        std::random_device rd;
        random_engine_.seed(rd());
    }

    bool solve(const std::string& config_file_path) override {
        if (!loadPlannerConfig(config_file_path)) {
            return false;
        }
        return MotionPlanner::solve(config_file_path);
    }

    std::string getPlannerName() const override {
        return "NGD";
    }

    const std::vector<ObstacleND>& getObstacles() const override {
        return obstacles_;
    }


protected:
    /**
     * @brief Runs the NGD optimization loop.
     */
    bool optimize() override {
        log("Update Rule: Y_{k+1} = (1-η)Y_k - η E[S(Ỹ)ε], where Ỹ ~ N(Y_k, R^{-1})");
        log("");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t M = num_samples_;
        
        storeTrajectory();
        
        float collision_cost = computeCollisionCost(current_trajectory_, obstacles_);
        float smoothness_cost = computeSmoothnessCost(current_trajectory_);
        float cost = collision_cost + smoothness_cost;
        
        const float alpha = 0.99f;
        float best_cost = cost;
        Trajectory best_trajectory = current_trajectory_;

        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            learning_rate_ = learning_rate_ * std::pow(alpha, iteration - 1);
            
            // ✓ Use log instead of std::cout
            logf("------- Iteration %zu : Cost = %.2f ------ ", iteration, cost);
            logf("  Collision: %.4f, Smoothness: %.4f, LR: %.6f", 
                collision_cost, smoothness_cost, learning_rate_);
                    
            Eigen::MatrixXf Y_k = trajectoryToMatrix();
            std::vector<Eigen::MatrixXf> epsilon_samples = sampleNoiseMatrices(M, N, D);
            
            Eigen::MatrixXf natural_gradient = Eigen::MatrixXf::Zero(D, N);
            
            for (size_t m = 0; m < M; ++m) {
                Trajectory perturbed_traj = createPerturbedTrajectory(Y_k, epsilon_samples[m]);
                float perturbed_collision = computeCollisionCost(perturbed_traj, obstacles_);
                natural_gradient += perturbed_collision * epsilon_samples[m];
            }
            natural_gradient /= M;
            
            Eigen::MatrixXf Y_new = (1.0f - learning_rate_) * Y_k - learning_rate_ * natural_gradient;
            updateTrajectoryFromMatrix(Y_new);

            storeTrajectory();
            collision_cost = computeCollisionCost(current_trajectory_, obstacles_);
            smoothness_cost = computeSmoothnessCost(current_trajectory_);
            float new_cost = collision_cost + smoothness_cost;

            // ✓ Use log instead of std::cout
            logf("  New Cost = %.2f (Δ = %.2f)", new_cost, cost - new_cost);
            
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_trajectory = current_trajectory_;
            }
            
            if (std::isnan(new_cost) || std::isinf(new_cost) || new_cost > 1e10) {
                log("WARNING: Optimization diverged! Restoring best trajectory.");
                current_trajectory_ = best_trajectory;
                break;
            }
            
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_) {
                log("Converged!");
                break;
            }
            cost = new_cost;
        }
        
        current_trajectory_ = best_trajectory;
        collision_cost = computeCollisionCost(current_trajectory_, obstacles_);
        smoothness_cost = computeSmoothnessCost(current_trajectory_);
        
        // ✓ Use log instead of std::cout
        logf("NGD finished. Best Cost: %.2f (Collision: %.4f, Smoothness: %.4f)", 
            best_cost, collision_cost, smoothness_cost);
        
        return true;
    }

    bool loadPlannerConfig(const std::string& config_file_path) {
        // Similar to PCEM
        try {
            YAML::Node config = YAML::LoadFile(config_file_path);
            const YAML::Node& planner_config = config["ngd_planner"];
            
            num_samples_ = planner_config["num_samples"].as<size_t>();
            num_iterations_ = planner_config["num_iterations"].as<size_t>();
            learning_rate_ = planner_config["learning_rate"].as<float>();
            convergence_threshold_ = planner_config["convergence_threshold"].as<float>();
            
            if (planner_config["cost"]) {
                const YAML::Node& cost_config = planner_config["cost"];
                epsilon_sdf_ = cost_config["epsilon_sdf"].as<float>();
                sigma_obs_ = cost_config["sigma_obs"].as<float>();
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading NGD config: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * @brief Logs NGD-specific configuration
     */
    void logPlannerSpecificConfig() override {
        log("--- NGD Planner Parameters ---");
        log("  Algorithm:            Natural Gradient Descent");
        logf("  Number of samples:    %zu", num_samples_);
        logf("  Number of iterations: %zu", num_iterations_);
        logf("  Initial learning rate: %.6f", learning_rate_);
        logf("  Learning rate decay:  %.4f (alpha)", 0.99f);
        logf("  Convergence threshold: %.6f", convergence_threshold_);
        log("");
        
        log("--- Cost Function Parameters ---");
        logf("  Epsilon SDF:          %.2f", epsilon_sdf_);
        logf("  Sigma obs:            %.4f", sigma_obs_);
        log("");
        
        // Log YAML config if available
        if (config_) {
            logYAMLConfig(config_);
        }
    }


private:
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float learning_rate_ = 0.01f;
    float convergence_threshold_ = 0.01f;

};
