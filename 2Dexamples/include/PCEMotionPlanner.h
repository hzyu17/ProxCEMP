#pragma once

#include "MotionPlanner.h"
#include <yaml-cpp/yaml.h> 

/**
 * @brief Proximal Cross-Entropy Method (PCEM) for Trajectory Optimization.
 * This class is header-only, with all methods implemented inline.
 */
class ProximalCrossEntropyMotionPlanner : public MotionPlanner {
public:
    ProximalCrossEntropyMotionPlanner(const std::vector<Obstacle>& obs, const YAML::Node& config)
        : obstacles_(obs)
    {
        // Initialize random engine (using a time-based seed)
        std::random_device rd;
        random_engine_.seed(rd());

        // Read hyperparameters from the YAML config
        if (config["pce_planner"]) {
            const YAML::Node& planner_config = config["pce_planner"];
            
            if (planner_config["num_samples"]) {
                num_samples_ = planner_config["num_samples"].as<size_t>();
            } else {
                throw std::runtime_error("Required parameter 'num_samples' not found in pce_planner section.");
            }

            if (planner_config["num_iterations"]) {
                num_iterations_ = planner_config["num_iterations"].as<size_t>();
            } else {
                throw std::runtime_error("Required parameter 'num_iterations' not found in pce_planner section.");
            }
            
            if (planner_config["temperature"]) {
                temperature_ = planner_config["temperature"].as<float>();
            } else {
                throw std::runtime_error("Required parameter 'temperature' not found in pce_planner section.");
            }

            if (planner_config["eta"]) {
                eta_ = planner_config["eta"].as<float>();
                gamma_ = eta_ / (eta_ + 1);
            } else {
                throw std::runtime_error("Required parameter 'eta' not found in pce_planner section.");
            }
            
            if (planner_config["convergence_threshold"]) {
                convergence_threshold_ = planner_config["convergence_threshold"].as<float>();
            } else {
                throw std::runtime_error("Required parameter 'convergence_threshold' not found in pce_planner section.");
            }
            
            // Read obstacle cost parameters
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
                throw std::runtime_error("Required section 'cost' not found in pce_planner section.");
            }
        } else {
            std::cerr << "Warning: 'pce_planner' section not found in config.yaml. Using default hyperparameters.\n";
        }
        
        std::cout << "PCEM planner initialized with hyperparameters:\n"
                  << "  num_samples: " << num_samples_ << "\n"
                  << "  num_iterations: " << num_iterations_ << "\n"
                  << "  temperature: " << temperature_ << "\n"
                  << "  eta: " << eta_ << "\n"
                  << "  gamma: " << gamma_ << "\n"
                  << "  convergence_threshold: " << convergence_threshold_ << "\n"
                  << "  epsilon_sdf: " << epsilon_sdf_ << "\n"
                  << "  sigma_obs: " << sigma_obs_ << "\n";
    }

    /**
     * @brief Runs the PCEM optimization loop.
     */
    bool optimize() override {
        std::cout << "\n--- Starting PCEM Optimization ---\n";
        std::cout << "Update Rule: Y_{k+1} = Σ w_m (Y_k + ε_m), where w ∝ exp(-γ(S(Ỹ) - Ỹ^T*R*Y_k))\n\n";
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        
        // Store initial trajectory Y_0 (Iteration 0)
        storeTrajectory();
        float cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);

        float alpha = 0.99f;
        float alpha_temp = 0.99f;
        
        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {

            // Update the learning rate iteratively 
            gamma_ = gamma_ * std::pow(alpha, iteration-1);

            // Adaptive temperature
            temperature_ = temperature_ * alpha_temp;

            std::cout << "------- Iteration " << iteration << " : Cost = " << cost << "------ \n";

            float collision_cost = computeCollisionCost(current_trajectory_, obstacles_);
            float smoothness_cost = computeSmoothnessCost(current_trajectory_);
            std::cout << "Collision cost: " << collision_cost 
                    << ", Smoothness cost: " << smoothness_cost << "\n";
                    
            std::cout << "Iteration " << iteration << " - Sampling " << M << " trajectories...\n";
            
            // --- Step 1: Sample M noise vectors ---
            std::vector<std::vector<float>> epsilon_x_samples(M);
            std::vector<std::vector<float>> epsilon_y_samples(M);
            
            for (size_t m = 0; m < M; ++m) {
                epsilon_x_samples[m] = sampleSmoothnessNoise(N, random_engine_);
                epsilon_y_samples[m] = sampleSmoothnessNoise(N, random_engine_);
            }
            
            // --- Step 2: Extract Y_k (current trajectory) as vectors ---
            std::vector<float> Y_k_x(N);
            std::vector<float> Y_k_y(N);
            for (size_t i = 0; i < N; ++i) {
                Y_k_x[i] = current_trajectory_.nodes[i].x;
                Y_k_y[i] = current_trajectory_.nodes[i].y;
            }
            
            // --- Step 3: Compute R * Y_k for bias correction term ---
            std::vector<float> R_Y_k_x = applyRMatrix(Y_k_x);
            std::vector<float> R_Y_k_y = applyRMatrix(Y_k_y);
            
            // --- Step 4: Evaluate costs and compute weights ---
            std::vector<float> weights(M);
            std::vector<Trajectory> sampled_trajectories(M);
            float max_exponent = -std::numeric_limits<float>::infinity();
            
            // Diagnostics
            std::vector<float> collision_costs(M);
            std::vector<float> bias_terms(M);
            
            for (size_t m = 0; m < M; ++m) {
                // Create sampled trajectory: Ỹ = Y_k + ε_m
                sampled_trajectories[m] = current_trajectory_;
                for (size_t i = 0; i < N; ++i) {
                    sampled_trajectories[m].nodes[i].x = Y_k_x[i] + epsilon_x_samples[m][i];
                    sampled_trajectories[m].nodes[i].y = Y_k_y[i] + epsilon_y_samples[m][i];
                }
                
                // Compute collision cost S(Ỹ)
                float collision_cost = computeCollisionCost(sampled_trajectories[m], obstacles_);
                
                // Compute bias term: Ỹ^T * R * Y_k = (Y_k + ε)^T * R * Y_k
                // Since Y_k^T * R * Y_k is constant across samples, we only need ε^T * R * Y_k
                float bias_x = 0.0f;
                float bias_y = 0.0f;
                for (size_t i = 0; i < N; ++i) {
                    bias_x += epsilon_x_samples[m][i] * R_Y_k_x[i];
                    bias_y += epsilon_y_samples[m][i] * R_Y_k_y[i];
                }
                float bias_term = bias_x + bias_y;
                
                // Store for diagnostics
                collision_costs[m] = collision_cost;
                bias_terms[m] = bias_term;
                
                // Compute exponent: -γ * (S(Ỹ) - Ỹ^T * R * Y_k)
                // Equivalently: -γ * (S(Ỹ) - ε^T * R * Y_k) since constant cancels in normalization
                float exponent = -gamma_ * (collision_cost + bias_term) / temperature_;
                
                // Track maximum for numerical stability
                if (exponent > max_exponent) {
                    max_exponent = exponent;
                }                
                weights[m] = exponent;  // Store exponent temporarily
            }
            
            // Print diagnostic statistics
            auto minmax_collision = std::minmax_element(collision_costs.begin(), collision_costs.end());
            auto minmax_bias = std::minmax_element(bias_terms.begin(), bias_terms.end());
            std::cout << "Collision cost range: [" << *minmax_collision.first << ", " 
                      << *minmax_collision.second << "]\n";
            std::cout << "Bias term range: [" << *minmax_bias.first << ", " 
                      << *minmax_bias.second << "]\n";

            // Print statistics BEFORE exp
            float min_log_weight = *std::min_element(weights.begin(), weights.end());
            float max_log_weight = *std::max_element(weights.begin(), weights.end());
            std::cout << "Log-weight range: [" << min_log_weight << ", " << max_log_weight 
                    << "], span = " << (max_log_weight - min_log_weight) << "\n";
            
            // --- Step 5: Normalize weights (subtract max for numerical stability) ---
            float weight_sum = 0.0f;
            for (size_t m = 0; m < M; ++m) {
                weights[m] = std::exp(weights[m] - max_exponent);
                weight_sum += weights[m];
            }
            
            // Normalize to sum to 1
            for (size_t m = 0; m < M; ++m) {
                weights[m] /= weight_sum;
            }

            // After normalization, check weight distribution
            std::vector<float> sorted_weights = weights;
            std::sort(sorted_weights.begin(), sorted_weights.end(), std::greater<float>());
            std::cout << "Top 5 weights: ";
            for (size_t i = 0; i < std::min(5ul, M); ++i) {
                std::cout << sorted_weights[i] << " ";
            }
            std::cout << "\nBottom 5 weights: ";
            for (size_t i = M-5; i < M; ++i) {
                std::cout << sorted_weights[i] << " ";
            }
            std::cout << "\nMax weight: " << sorted_weights[0] << "\n";
            
            // --- Step 6: Compute Y_{k+1} via weighted mean update ---
            // Y_{k+1} = Σ w_m (Y_k + ε_m)
            std::vector<float> Y_kplus1_x(N, 0.0f);
            std::vector<float> Y_kplus1_y(N, 0.0f);
            
            for (size_t m = 0; m < M; ++m) {
                for (size_t i = 0; i < N; ++i) {
                    Y_kplus1_x[i] += weights[m] * sampled_trajectories[m].nodes[i].x;
                    Y_kplus1_y[i] += weights[m] * sampled_trajectories[m].nodes[i].y;
                }
            }
            
            // --- Step 7: Update current trajectory: Y_k ← Y_{k+1} ---
            for (size_t i = 0; i < N; ++i) {
                current_trajectory_.nodes[i].x = Y_kplus1_x[i];
                current_trajectory_.nodes[i].y = Y_kplus1_y[i];
            }
            
            // --- Step 8: Store Y_{k+1} and check convergence ---
            storeTrajectory();
            float new_cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);
            
            // Compute effective sample size (ESS) for diagnostics
            float ess = 0.0f;
            for (size_t m = 0; m < M; ++m) {
                ess += weights[m] * weights[m];
            }
            ess = 1.0f / ess;
            
            std::cout << "Iteration " << iteration << ": Cost = " << new_cost 
                    << ", ESS = " << ess << "/" << M << "\n";
            
            // Check if cost improvement is too small
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_ && cost - new_cost > 0) {
                std::cout << "Cost improvement negligible. Stopping.\n";
                break;
            }
            
            cost = new_cost;
        }
        
        std::cout << "PCEM finished. Final Cost: " << computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_) << "\n";
        return true;
    }

    /**
     * @brief Provides access to the obstacle map (required by MotionPlanner base class).
     */
    const std::vector<Obstacle>& getObstacles() const override {
        return obstacles_;
    }

private:
    const std::vector<Obstacle>& obstacles_;

    size_t num_samples_ = 100;
    size_t num_iterations_ = 10;
    float temperature_ = 1.0f;
    float eta_ = 1.0;
    float gamma_ = 1.0;
    float convergence_threshold_ = 0.01f;

    // Cost function parameters
    float epsilon_sdf_ = 20.0f;
    float sigma_obs_ = 1.0f;
    
    // Internal state for optimization
    std::mt19937 random_engine_;

    /**
     * @brief Applies the R matrix to a vector: computes R * epsilon
     * R is a pentadiagonal smoothness matrix
     */
    std::vector<float> applyRMatrix(const std::vector<float>& epsilon) const {
        const size_t N = epsilon.size();
        if (N < 3) return epsilon;
        
        // Get R matrix diagonals
        RMatrixDiagonals R_bands = getSmoothnessMatrixRDiagonals(N);
        
        // R * epsilon for pentadiagonal matrix
        std::vector<float> result(N, 0.0f);
        
        for (size_t i = 0; i < N; ++i) {
            // Main diagonal
            result[i] += R_bands.main_diag[i] * epsilon[i];
            
            // First off-diagonal (upper and lower due to symmetry)
            if (i > 0) {
                result[i] += R_bands.diag1[i-1] * epsilon[i-1];
            }
            if (i < N - 1) {
                result[i] += R_bands.diag1[i] * epsilon[i+1];
            }
            
            // Second off-diagonal (upper and lower due to symmetry)
            if (i > 1) {
                result[i] += R_bands.diag2[i-2] * epsilon[i-2];
            }
            if (i < N - 2) {
                result[i] += R_bands.diag2[i] * epsilon[i+2];
            }
        }
        
        return result;
    }


public:
    float computeCollisionCost(const Trajectory& traj, const std::vector<Obstacle>& obstacles) const {
        float total_cost = 0.0f;
        
        // For each node in trajectory
        for (const auto& node : traj.nodes) {
            float x = node.x;
            float y = node.y;
            
            // Compute hinge loss for each obstacle
            for (const auto& obs : obstacles) {
                // Signed distance to obstacle
                float dx = x - obs.x;
                float dy = y - obs.y;
                float dist_to_center = std::sqrt(dx*dx + dy*dy);
                float signed_distance = dist_to_center - obs.radius - node.radius;
                
                // Hinge loss function with cut-off epsilon_sdf
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - signed_distance);
                
                // Weighted squared hinge loss
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
            }
        }
        
        return total_cost;
    }
};