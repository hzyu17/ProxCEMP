#pragma once

#include "MotionPlanner.h"
#include <random>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <algorithm> // Needed for std::max

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
    NGDMotionPlanner(const std::vector<Obstacle>& obs, const YAML::Node& config)
        : obstacles_(obs)
    {
        std::random_device rd;
        random_engine_.seed(rd());

        if (config["ngd_planner"]) {
            const YAML::Node& planner_config = config["ngd_planner"];

            if (planner_config["num_samples"]) {
                num_samples_ = planner_config["num_samples"].as<size_t>();
            } else {
                throw std::runtime_error("Required parameter 'num_samples' not found in ngd_planner section.");
            }

            if (planner_config["num_iterations"]) {
                num_iterations_ = planner_config["num_iterations"].as<size_t>();
            } else {
                throw std::runtime_error("Required parameter 'num_iterations' not found in ngd_planner section.");
            }
            
            if (planner_config["learning_rate"]) {
                learning_rate_ = planner_config["learning_rate"].as<float>();
            } else {
                throw std::runtime_error("Required parameter 'learning_rate' not found in ngd_planner section.");
            }

            if (planner_config["convergence_threshold"]) {
                convergence_threshold_ = planner_config["convergence_threshold"].as<float>();
            } else {
                throw std::runtime_error("Required parameter 'convergence_threshold' not found in ngd_planner section.");
            }
            
            if (planner_config["cost"]) {
                const YAML::Node& cost_config = planner_config["cost"];
                if (cost_config["epsilon_sdf"]) {
                    epsilon_sdf_ = cost_config["epsilon_sdf"].as<float>();
                } else {
                    throw std::runtime_error("Required parameter 'epsilon_sdf' not found in cost section.");
                }

                if (cost_config["sigma_obs"]) {
                    sigma_obs_ = cost_config["sigma_obs"].as<float>();
                } else {
                    throw std::runtime_error("Required parameter 'sigma_obs' not found in cost section.");
                }
            } else {
                throw std::runtime_error("Required section 'cost' not found in ngd_planner section.");
            }
        } else {
            std::cerr << "Warning: 'ngd_planner' section not found in config.yaml. Using default hyperparameters.\n";
        }

        std::cout << "NGD planner initialized with hyperparameters:\n"
                  << "  num_samples: " << num_samples_ << "\n"
                  << "  num_iterations: " << num_iterations_ << "\n"
                  << "  learning_rate: " << learning_rate_ << "\n"
                  << "  convergence_threshold: " << convergence_threshold_ << "\n"
                  << "  epsilon_sdf: " << epsilon_sdf_ << "\n"
                  << "  sigma_obs: " << sigma_obs_ << "\n";
    }

    /**
     * @brief Runs the NGD optimization loop.
     */
    bool optimize() override {
        std::cout << "\n--- Starting NGD Optimization ---\n";
        // NOTE: The update rule is corrected here to stabilize the algorithm.
        // It uses Y_{k+1} = Y_k - η * E[S(Y_k + ε)ε], omitting the destructive Y_k proximal term.
        std::cout << "Update Rule: Y_{k+1} = Y_k - η * E[S(Y_k + ε)ε] \n\n";
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        
        storeTrajectory();
        float cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);
        
        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            
            std::cout << "------- Iteration " << iteration << " : Cost = " << cost << "------ \n";

            float collision_cost = computeCollisionCost(current_trajectory_, obstacles_);
            float smoothness_cost = computeSmoothnessCost(current_trajectory_);
            std::cout << "Collision cost: " << collision_cost 
                    << ", Smoothness cost: " << smoothness_cost << "\n";
                    
            // Step 1: Extract Y_k
            std::vector<float> Y_k_x(N);
            std::vector<float> Y_k_y(N);
            for (size_t i = 0; i < N; ++i) {
                Y_k_x[i] = current_trajectory_.nodes[i].x;
                Y_k_y[i] = current_trajectory_.nodes[i].y;
            }

            // Step 2: Sample M noise vectors (epsilon) from N(0, R^{-1})
            std::cout << "Iteration " << iteration << " - Sampling " << M << " noise vectors...\n";
            std::vector<std::vector<float>> epsilon_x_samples(M);
            std::vector<std::vector<float>> epsilon_y_samples(M);
            
            for (size_t m = 0; m < M; ++m) {
                epsilon_x_samples[m] = sampleSmoothnessNoise(N, random_engine_);
                epsilon_y_samples[m] = sampleSmoothnessNoise(N, random_engine_);
            }
            
            // Step 3: Compute the stochastic gradient term E[S(Y_k + ε)ε]
            std::vector<float> stochastic_grad_x(N, 0.0f);
            std::vector<float> stochastic_grad_y(N, 0.0f);
            
            for (size_t m = 0; m < M; ++m) {
                // Create a perturbed trajectory Y_k + epsilon_m
                Trajectory perturbed_traj = current_trajectory_;
                for (size_t i = 0; i < N; ++i) {
                    perturbed_traj.nodes[i].x = Y_k_x[i] + epsilon_x_samples[m][i];
                    perturbed_traj.nodes[i].y = Y_k_y[i] + epsilon_y_samples[m][i];
                }
                
                // Compute the collision cost of the perturbed trajectory
                float perturbed_cost = computeCollisionCost(perturbed_traj, obstacles_);
                
                // Accumulate the weighted noise vectors
                for (size_t i = 0; i < N; ++i) {
                    stochastic_grad_x[i] += perturbed_cost * epsilon_x_samples[m][i];
                    stochastic_grad_y[i] += perturbed_cost * epsilon_y_samples[m][i];
                }
            }
            
            // Normalize the stochastic gradient by the number of samples
            for (size_t i = 0; i < N; ++i) {
                stochastic_grad_x[i] /= M;
                stochastic_grad_y[i] /= M;
            }
            
            // Step 4: Update the trajectory Y_{k+1} = Y_k - η * E[S(Y_k + ε)ε]
            // We use only the stochastic collision gradient term to drive the update
            for (size_t i = 0; i < N; ++i) {
                // Y_{k+1} = Y_k - η * (stochastic_grad)
                current_trajectory_.nodes[i].x = Y_k_x[i] - learning_rate_ * stochastic_grad_x[i];
                current_trajectory_.nodes[i].y = Y_k_y[i] - learning_rate_ * stochastic_grad_y[i];
            }

            // Fix the start and goal nodes (index 0 and N-1)
            current_trajectory_.nodes[0].x = start_node_.x;
            current_trajectory_.nodes[0].y = start_node_.y;
            current_trajectory_.nodes[N-1].x = goal_node_.x;
            current_trajectory_.nodes[N-1].y = goal_node_.y;

            // Step 5: Store new trajectory and check convergence
            storeTrajectory();
            float new_cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);

            std::cout << "Iteration " << iteration << ": New Cost = " << new_cost << "\n";
            
            // Check if cost improvement is too small
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_) {
                std::cout << "Cost improvement negligible. Stopping NGD optimization.\n";
                break;
            }
            cost = new_cost;
        }
        
        std::cout << "NGD finished. Final Cost: " << computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_) << "\n";
        return true;
    }

    const std::vector<Obstacle>& getObstacles() const override {
        return obstacles_;
    }

private:
    const std::vector<Obstacle>& obstacles_;

    size_t num_samples_ = 1000; // Increased default for stability
    size_t num_iterations_ = 20; // Increased default
    float learning_rate_ = 0.5f; // Adjusted default
    float convergence_threshold_ = 0.01f;

    // Cost function parameters
    float epsilon_sdf_ = 20.0f;
    float sigma_obs_ = 1.0f;
    
    std::mt19937 random_engine_;

    /**
     * @brief Specialized collision cost calculation using Hinge Loss, designed for NGD/PCEM sampling.
     * @param traj The specific trajectory to evaluate (usually a sampled trajectory).
     * @param obstacles The list of static obstacles.
     * @return The weighted squared hinge loss collision cost.
     */
    float computeCollisionCost(const Trajectory& traj, const std::vector<Obstacle>& obstacles) const {
        float total_cost = 0.0f;
        for (const auto& node : traj.nodes) {
            float x = node.x;
            float y = node.y;
            
            for (const auto& obs : obstacles) {
                float dx = x - obs.x;
                float dy = y - obs.y;
                float dist_to_center = std::sqrt(dx*dx + dy*dy);
                // Signed distance to obstacle surface
                float signed_distance = dist_to_center - obs.radius - node.radius; 
                
                // Hinge loss: max(0, epsilon_sdf - signed_distance)
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - signed_distance);
                
                // Weighted squared hinge loss
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
            }
        }
        return total_cost;
    }
};
