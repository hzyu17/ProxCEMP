#pragma once

#include "MotionPlanner.h"
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * @brief Obstacle structure extended for N-dimensional spaces
 */
struct ObstacleND {
    Eigen::VectorXf center;  // N-dimensional center position
    float radius;            // Obstacle radius
    
    // Constructors
    ObstacleND() : center(Eigen::VectorXf::Zero(2)), radius(0.0f) {}
    
    ObstacleND(const Eigen::VectorXf& c, float r) : center(c), radius(r) {}
    
    // 2D convenience constructor
    ObstacleND(float x, float y, float r) : center(2), radius(r) {
        center << x, y;
    }
    
    // 3D convenience constructor
    ObstacleND(float x, float y, float z, float r) : center(3), radius(r) {
        center << x, y, z;
    }
    
    size_t dimensions() const { return center.size(); }
};

/**
 * @brief Proximal Cross-Entropy Method (PCEM) for Trajectory Optimization.
 * Fully Eigen-based implementation supporting arbitrary N-dimensional configuration spaces.
 */
class ProximalCrossEntropyMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    ProximalCrossEntropyMotionPlanner(const std::vector<ObstacleND>& obs, const YAML::Node& config)
        : obstacles_(obs)
    {
        // Initialize random engine (using a time-based seed)
        std::random_device rd;
        random_engine_.seed(rd());

        // Read hyperparameters from the YAML config
        if (config["pce_planner"]) {
            const YAML::Node& planner_config = config["pce_planner"];
            
            // Read dimensionality (default to 2 for backward compatibility)
            if (planner_config["num_dimensions"]) {
                num_dimensions_ = planner_config["num_dimensions"].as<size_t>();
            } else {
                num_dimensions_ = 2;
                std::cout << "Warning: 'num_dimensions' not specified, defaulting to 2D.\n";
            }
            
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
            num_dimensions_ = 2;
        }
        
        std::cout << "STOMP planner initialized with hyperparameters:\n"
                  << "  num_dimensions: " << num_dimensions_ << "\n"
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
        const size_t D = num_dimensions_;
        
        // Validate trajectory dimensionality
        if (current_trajectory_.dimensions() != D) {
            throw std::runtime_error("Trajectory dimensionality doesn't match configured num_dimensions");
        }
        
        // Build the R matrix once (sparse pentadiagonal matrix)
        R_matrix_ = buildSparseRMatrix(N);
        
        // Store initial trajectory Y_0 (Iteration 0)
        storeTrajectory();
        float cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);

        float alpha = 0.99f;
        float alpha_temp = 0.99f;

        float best_cost = cost;
        size_t best_iteration = 0;
        
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
            
            // --- Step 1: Sample M noise matrices (each is N x D) ---
            std::vector<MatrixXf> epsilon_samples(M);
            for (size_t m = 0; m < M; ++m) {
                epsilon_samples[m] = sampleSmoothnessNoiseMatrix(N, D);
            }
            
            // --- Step 2: Extract Y_k (current trajectory) as N x D matrix ---
            MatrixXf Y_k = trajectoryToMatrix(current_trajectory_);
            
            // --- Step 3: Compute R * Y_k for bias correction term ---
            // R_Y_k is N x D matrix where each column is R * Y_k[:, d]
            MatrixXf R_Y_k = R_matrix_ * Y_k;
            
            // --- Step 4: Evaluate costs and compute weights ---
            VectorXf weights(M);
            std::vector<Trajectory> sampled_trajectories(M);
            float max_exponent = -std::numeric_limits<float>::infinity();
            
            // Diagnostics
            VectorXf collision_costs(M);
            VectorXf bias_terms(M);
            
            for (size_t m = 0; m < M; ++m) {
                // Create sampled trajectory: Ỹ = Y_k + ε_m
                MatrixXf Y_tilde = Y_k + epsilon_samples[m];
                sampled_trajectories[m] = matrixToTrajectory(Y_tilde, current_trajectory_);
                
                // Compute collision cost S(Ỹ)
                float collision_cost = computeCollisionCost(sampled_trajectories[m], obstacles_);
                collision_costs(m) = collision_cost;
                
                // Compute bias term: ε_m^T * R * Y_k
                // This is the sum of element-wise products: sum(epsilon_m .* R_Y_k)
                float bias_term = (epsilon_samples[m].array() * R_Y_k.array()).sum();
                bias_terms(m) = bias_term;
                
                // Compute exponent: -γ * (S(Ỹ) - ε^T * R * Y_k) / T
                float exponent = -gamma_ * (collision_cost + bias_term) / temperature_;
                
                // Track maximum for numerical stability
                if (exponent > max_exponent) {
                    max_exponent = exponent;
                }                
                weights(m) = exponent;  // Store exponent temporarily
            }
            
            // Print diagnostic statistics
            std::cout << "Collision cost range: [" << collision_costs.minCoeff() << ", " 
                      << collision_costs.maxCoeff() << "]\n";
            std::cout << "Bias term range: [" << bias_terms.minCoeff() << ", " 
                      << bias_terms.maxCoeff() << "]\n";
            std::cout << "Log-weight range: [" << weights.minCoeff() << ", " << weights.maxCoeff() 
                    << "], span = " << (weights.maxCoeff() - weights.minCoeff()) << "\n";
            
            // --- Step 5: Normalize weights (subtract max for numerical stability) ---
            weights = (weights.array() - max_exponent).exp();
            float weight_sum = weights.sum();
            weights /= weight_sum;
            
            // After normalization, check weight distribution
            std::vector<float> sorted_weights(weights.data(), weights.data() + M);
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
            // Y_{k+1} = Σ w_m * (Y_k + ε_m)
            MatrixXf Y_kplus1 = MatrixXf::Zero(N, D);
            for (size_t m = 0; m < M; ++m) {
                Y_kplus1 += weights(m) * (Y_k + epsilon_samples[m]);
            }
            
            // --- Step 7: Update current trajectory: Y_k ← Y_{k+1} ---
            current_trajectory_ = matrixToTrajectory(Y_kplus1, current_trajectory_);
            
            // --- Step 8: Store Y_{k+1} and check convergence ---
            storeTrajectory();
            float new_cost = computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_);

            // Track best trajectory
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_iteration = iteration;
            }
            
            // Compute effective sample size (ESS) for diagnostics
            float ess = 1.0f / weights.array().square().sum();
            
            std::cout << "Iteration " << iteration << ": Cost = " << new_cost 
                    << ", ESS = " << ess << "/" << M << "\n";
            
            // Check if cost improvement is too small
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_ && cost - new_cost > 0) {
                std::cout << "Cost improvement negligible. Stopping.\n";
                break;
            }
            
            cost = new_cost;
        }

        // Restore the best trajectory found during optimization
        if (best_iteration < trajectory_history_.size()) {
            current_trajectory_ = trajectory_history_[best_iteration];

            std::cout << "\n*** Restoring best trajectory from iteration " << best_iteration 
                      << " with cost " << best_cost << " ***\n";

            // Remove all trajectories after the best iteration
            trajectory_history_.erase(
                trajectory_history_.begin() + best_iteration + 1,
                trajectory_history_.end()
            );
        }
        
        std::cout << "PCEM finished. Final Cost: " << computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_) << "\n";
        return true;
    }

    /**
     * @brief Provides access to the obstacle map (required by MotionPlanner base class).
     */
    const std::vector<ObstacleND>& getObstacles() const {
        return obstacles_;
    }

private:
    const std::vector<ObstacleND>& obstacles_;

    size_t num_dimensions_ = 2;
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
    SparseMatrixXf R_matrix_;  // Cached smoothness matrix

    /**
     * @brief Builds sparse pentadiagonal R matrix for smoothness regularization
     */
    SparseMatrixXf buildSparseRMatrix(size_t N) const {
        if (N < 3) {
            SparseMatrixXf R(N, N);
            R.setIdentity();
            return R;
        }
        
        // Get R matrix diagonals
        RMatrixDiagonals R_bands = getSmoothnessMatrixRDiagonals(N);
        
        // Build sparse matrix with pentadiagonal structure
        std::vector<Eigen::Triplet<float>> triplets;
        triplets.reserve(N * 5);  // At most 5 non-zeros per row
        
        for (size_t i = 0; i < N; ++i) {
            // Main diagonal
            triplets.emplace_back(i, i, R_bands.main_diag[i]);
            
            // First off-diagonal (upper and lower)
            if (i > 0) {
                triplets.emplace_back(i, i-1, R_bands.diag1[i-1]);
            }
            if (i < N - 1) {
                triplets.emplace_back(i, i+1, R_bands.diag1[i]);
            }
            
            // Second off-diagonal (upper and lower)
            if (i > 1) {
                triplets.emplace_back(i, i-2, R_bands.diag2[i-2]);
            }
            if (i < N - 2) {
                triplets.emplace_back(i, i+2, R_bands.diag2[i]);
            }
        }
        
        SparseMatrixXf R(N, N);
        R.setFromTriplets(triplets.begin(), triplets.end());
        
        return R;
    }

    /**
     * @brief Samples a smoothness noise matrix (N x D)
     */
    MatrixXf sampleSmoothnessNoiseMatrix(size_t N, size_t D) const {
        MatrixXf noise(N, D);
        
        // Sample each dimension independently
        for (size_t d = 0; d < D; ++d) {
            std::vector<float> noise_vec = sampleSmoothnessNoise(N, random_engine_);
            for (size_t i = 0; i < N; ++i) {
                noise(i, d) = noise_vec[i];
            }
        }
        
        return noise;
    }


public:
    /**
     * @brief Computes collision cost for N-dimensional trajectories using Eigen
     */
    float computeCollisionCost(const Trajectory& traj, const std::vector<ObstacleND>& obstacles) const {
        float total_cost = 0.0f;
        
        // For each node in trajectory
        for (const auto& node : traj.nodes) {
            // Compute hinge loss for each obstacle
            for (const auto& obs : obstacles) {
                // Ensure dimensionality matches
                if (node.dimensions() != obs.dimensions()) {
                    throw std::runtime_error("Node and obstacle dimensionality mismatch");
                }
                
                // Compute N-dimensional Euclidean distance using Eigen
                VectorXf diff = node.position - obs.center;
                float dist_to_center = diff.norm();
                float signed_distance = dist_to_center - obs.radius - node.radius;
                
                // Hinge loss function with cut-off epsilon_sdf
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - signed_distance);
                
                // Weighted squared hinge loss
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
            }
        }
        
        return total_cost;
    }
    
    /**
     * @brief Converts legacy 2D obstacles to ObstacleND format
     */
    static std::vector<ObstacleND> convertObstacles(const std::vector<Obstacle>& legacy_obstacles) {
        std::vector<ObstacleND> obstacles_nd;
        obstacles_nd.reserve(legacy_obstacles.size());
        
        for (const auto& obs : legacy_obstacles) {
            obstacles_nd.emplace_back(obs.x, obs.y, obs.radius);
        }
        
        return obstacles_nd;
    }
};