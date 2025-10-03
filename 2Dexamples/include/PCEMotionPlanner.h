#pragma once

#include "MotionPlanner.h"
#include <random>
#include <iostream>
#include <algorithm>
#include <vector>
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
            
            // --- Key Change: Check for key existence before casting ---
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
        std::cout << "Update Rule: Y_{k+1} = Σ w_m S(Y_k + ε_m), where ε_m ~ N(0, R^{-1})\n\n";
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;  // Number of Monte Carlo samples (e.g., 100-500)
        
        // Store initial trajectory Y_0 (Iteration 0)
        storeTrajectory();
        float cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);
        
        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {

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
            // Y_k is the current trajectory at iteration k
            std::vector<float> Y_k_x(N);
            std::vector<float> Y_k_y(N);
            for (size_t i = 0; i < N; ++i) {
                Y_k_x[i] = current_trajectory_.nodes[i].x;
                Y_k_y[i] = current_trajectory_.nodes[i].y;
            }
            
            // --- Step 3: Compute R^{-1} * Y_k for bias correction ---
            std::vector<float> R_inv_Y_k_x = solveRInverse(Y_k_x);
            std::vector<float> R_inv_Y_k_y = solveRInverse(Y_k_y);
            
            // --- Step 4: Evaluate costs and compute weights ---
            std::vector<float> weights(M);
            std::vector<Trajectory> sampled_trajectories(M);
            float max_exponent = -std::numeric_limits<float>::infinity();
            
            for (size_t m = 0; m < M; ++m) {
                // Create sampled trajectory: (Y_k + ε_m)
                // This represents a candidate trajectory perturbed from current Y_k
                sampled_trajectories[m] = current_trajectory_;
                for (size_t i = 0; i < N; ++i) {
                    sampled_trajectories[m].nodes[i].x = Y_k_x[i] + epsilon_x_samples[m][i];
                    sampled_trajectories[m].nodes[i].y = Y_k_y[i] + epsilon_y_samples[m][i];
                }
                
                // Compute collision cost S(Y_k + epsilon_m)
                float collision_cost = computeCollisionCost(sampled_trajectories[m], obstacles_);
                
                // Compute bias correction: epsilon_m^T * R^{-1} * Y_k
                float bias_x = 0.0f;
                float bias_y = 0.0f;
                for (size_t i = 0; i < N; ++i) {
                    bias_x += epsilon_x_samples[m][i] * R_inv_Y_k_x[i];
                    bias_y += epsilon_y_samples[m][i] * R_inv_Y_k_y[i];
                }
                float bias_correction = bias_x + bias_y;
                
                // Compute exponent: -gamma * (S(Y_k + epsilon_m) - epsilon_m^T R^{-1} Y_k)
                float exponent = -gamma_ * (collision_cost - bias_correction);
                
                // Track maximum for numerical stability
                if (exponent > max_exponent) {
                    max_exponent = exponent;
                }                
                weights[m] = exponent;  // Store exponent temporarily
            }

            // ============ DEBUG ============
            std::vector<float> log_weights_raw(M);
            for (size_t m = 0; m < M; ++m) {
                log_weights_raw[m] = weights[m]; // Save the raw log-weights before exp
            }

            // Print statistics BEFORE exp
            float min_log_weight = *std::min_element(weights.begin(), weights.end());
            float max_log_weight = *std::max_element(weights.begin(), weights.end());
            std::cout << "Log-weight range: [" << min_log_weight << ", " << max_log_weight 
                    << "], span = " << (max_log_weight - min_log_weight) << "\n";

            // ============ DEBUG ============
            
            // --- Step 5: Normalize weights (subtract max for numerical stability) ---
            float weight_sum = 0.0f;
            for (size_t m = 0; m < M; ++m) {
                weights[m] = std::exp((weights[m] - max_exponent) / temperature_);
                weight_sum += weights[m];
            }
            
            // Normalize to sum to 1
            for (size_t m = 0; m < M; ++m) {
                weights[m] /= weight_sum;
            }

            // After normalization, check weight distribution
            std::sort(weights.begin(), weights.end(), std::greater<float>());
            std::cout << "Top 5 weights: ";
            for (size_t i = 0; i < std::min(5ul, M); ++i) {
                std::cout << weights[i] << " ";
            }
            std::cout << "\nBottom 5 weights: ";
            for (size_t i = M-5; i < M; ++i) {
                std::cout << weights[i] << " ";
            }
            std::cout << "\nMax weight: " << weights[0] << "\n";
            
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
            // current_trajectory_ now contains Y_{k+1}, which becomes Y_k in the next iteration
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
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_) {
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

    // Cost function parameters (now initialized from config)
    float epsilon_sdf_ = 20.0f;
    float sigma_obs_ = 1.0f;
    
    // Internal state for optimization
    std::mt19937 random_engine_; // Random number generator engine


    /**
     * @brief Solves R^{-1} * y for a given vector y using Cholesky factorization.
     * Since R = L * L^T, we solve: L * L^T * x = y
     * First solve L * z = y (forward substitution), then solve L^T * x = z (back substitution)
     */
    std::vector<float> solveRInverse(const std::vector<float>& y) const {
        const size_t N = y.size();
        if (N < 3) return y;
        
        // Extract free nodes (indices 1 to N-2)
        size_t N_free = N - 2;
        std::vector<float> y_free(N_free);
        for (size_t i = 0; i < N_free; ++i) {
            y_free[i] = y[i + 1];
        }
        
        // Get R matrix diagonals and perform Cholesky factorization
        RMatrixDiagonals R_bands = getSmoothnessMatrixRDiagonals(N);
        
        // Extract submatrix for free nodes
        std::vector<float> R_main(N_free);
        std::vector<float> R_diag1(N_free - 1);
        std::vector<float> R_diag2(N_free - 2);
        
        for (size_t i = 0; i < N_free; ++i) {
            R_main[i] = R_bands.main_diag[i + 1];
        }
        for (size_t i = 0; i < N_free - 1; ++i) {
            R_diag1[i] = R_bands.diag1[i + 1];
        }
        for (size_t i = 0; i < N_free - 2; ++i) {
            R_diag2[i] = R_bands.diag2[i + 1];
        }
        
        // Cholesky factorization: R = L * L^T
        std::vector<float> L0(N_free);
        std::vector<float> L1(N_free - 1);
        std::vector<float> L2(N_free - 2);
        
        // Compute L (same as in sampleSmoothnessNoise)
        L0[0] = std::sqrt(R_main[0]);
        if (N_free > 1) {
            L1[0] = R_diag1[0] / L0[0];
        }
        if (N_free > 2) {
            L2[0] = R_diag2[0] / L0[0];
            L0[1] = std::sqrt(R_main[1] - L1[0] * L1[0]);
            L1[1] = (R_diag1[1] - L2[0] * L1[0]) / L0[1];
        }
        if (N_free > 3) {
            L2[1] = R_diag2[1] / L0[1];
        }
        
        for (size_t i = 2; i < N_free - 2; ++i) {
            L0[i] = std::sqrt(R_main[i] - L1[i-1] * L1[i-1] - L2[i-2] * L2[i-2]);
            L1[i] = (R_diag1[i] - L2[i-1] * L1[i-1]) / L0[i];
            L2[i] = R_diag2[i] / L0[i];
        }
        
        if (N_free > 2) {
            L0[N_free-2] = std::sqrt(R_main[N_free-2] - L1[N_free-3] * L1[N_free-3] - L2[N_free-4] * L2[N_free-4]);
            if (N_free > 3) {
                L1[N_free-2] = (R_diag1[N_free-2] - L2[N_free-3] * L1[N_free-3]) / L0[N_free-2];
            }
            L0[N_free-1] = std::sqrt(R_main[N_free-1] - L1[N_free-2] * L1[N_free-2] - L2[N_free-3] * L2[N_free-3]);
        }
        
        // Step 1: Solve L * z = y_free (forward substitution)
        std::vector<float> z(N_free);
        for (size_t i = 0; i < N_free; ++i) {
            float sum = y_free[i];
            if (i >= 1) sum -= L1[i-1] * z[i-1];
            if (i >= 2) sum -= L2[i-2] * z[i-2];
            z[i] = sum / L0[i];
        }
        
        // Step 2: Solve L^T * x_free = z (backward substitution)
        std::vector<float> x_free(N_free);
        for (int i = N_free - 1; i >= 0; --i) {
            float sum = z[i];
            if (i + 1 < static_cast<int>(N_free)) sum -= L1[i] * x_free[i+1];
            if (i + 2 < static_cast<int>(N_free)) sum -= L2[i] * x_free[i+2];
            x_free[i] = sum / L0[i];
        }
        
        // Construct full vector with zeros at boundaries
        std::vector<float> x(N, 0.0f);
        for (size_t i = 0; i < N_free; ++i) {
            x[i + 1] = x_free[i];
        }
        
        return x;
    }

public:
    float computeCollisionCost(const Trajectory& traj, const std::vector<Obstacle>& obstacles) const {
        float total_cost = 0.0f;
        
        // For each node in trajectory
        for (const auto& node : traj.nodes) {
            // F(X) = X (forward kinematics is identity in simplest case)
            float x = node.x;
            float y = node.y;
            
            // Compute hinge loss for each obstacle
            for (const auto& obs : obstacles) {
                // S(F(X)): Signed distance to obstacle
                // Positive = outside obstacle, Negative = inside obstacle
                float dx = x - obs.x;
                float dy = y - obs.y;
                float dist_to_center = std::sqrt(dx*dx + dy*dy);
                float signed_distance = dist_to_center - obs.radius - node.radius;
                
                // h_tilde(d): Hinge loss function with cut-off epsilon_sdf
                // h(d) = max(0, epsilon_sdf - d)
                // This penalizes when d < epsilon_sdf (too close or in collision)
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - signed_distance);
                
                // ||h_tilde||²_Σ: Weighted squared hinge loss
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
            }
        }
        
        return total_cost;
    }

};
