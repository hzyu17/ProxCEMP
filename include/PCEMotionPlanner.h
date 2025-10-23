#pragma once

#include "MotionPlanner.h" 
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits> // Added for infinity check

/**
 * @brief Proximal Cross-Entropy Method (PCEM) for Trajectory Optimization.
 * Fully Eigen-based implementation supporting arbitrary N-dimensional configuration spaces.
 */
class ProximalCrossEntropyMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    

    /**
     * @brief Solve the motion planning problem using PCEM
     */
    bool solve(const std::string& config_file_path) override {
        // Load planner-specific config
        if (!loadPlannerConfig(config_file_path)) {
            return false;
        }
        
        // Call base class solve (handles everything)
        return MotionPlanner::solve(config_file_path);
    }

    // Constructor
    ProximalCrossEntropyMotionPlanner()
    {
        // Initialize random engine
        std::random_device rd;
        seedRandomEngine(rd());
    }

    std::string getPlannerName() const override {
        return "PCEM";
    }

    /**
     * @brief Initializes the trajectory and precomputes the Cholesky factorization L0.
     */
    virtual void initialize(const size_t num_dims,
                            const PathNode& start, 
                            const PathNode& goal, 
                            size_t num_nodes, 
                            float total_time, 
                            InterpolationMethod method, 
                            ObstacleMap& obstacle_map,
                            float clearance_dist) override
    {
        // 1. Call base class initialize to set up current_trajectory_
        MotionPlanner::initialize(num_dims, start, goal, num_nodes, total_time, method, obstacle_map, clearance_dist);

        const size_t N = current_trajectory_.nodes.size();

        if (N < 2) {
            throw std::invalid_argument("num_nodes must be >= 2");
        }
        
    }

    const std::vector<ObstacleND>& getObstacles() const override {
        return obstacles_;
    }

    size_t getNumSamples() const { return num_samples_; }
    size_t getNumIterations() const { return num_iterations_; }
    float getTemperature() const { return temperature_; }
    float getEta() const { return eta_; }
    float getGamma() const { return gamma_; }
    float getConvergenceThreshold() const { return convergence_threshold_; }
    float getEpsilonSdf() const { return epsilon_sdf_; }
    float getSigmaObs() const { return sigma_obs_; }

    void seedRandomEngine(unsigned int seed) {
        random_engine_.seed(seed);
    }

    const SparseMatrixXf& getRMatrix() const {
        return R_matrix_;
    }

protected:
    
    // --- TEMPORARY FUNCTION FOR TESTING CORE LOGIC (OVERRIDE IS ASSUMED MOCKED IN BASE) ---
    // Note: The original signature was: float computeCollisionCost(const Trajectory& trajectory, const std::vector<ObstacleND>& obstacles) const override 
    // We modify the implementation to ignore arguments and return a constant.
    // float computeCollisionCost(const Trajectory& /*trajectory*/, const std::vector<ObstacleND>& /*obstacles*/) const override 
    // {
    //     return 5.0f; // Arbitrary constant value
    // }
    // -----------------------------------------------------------------------------------


    /**
     * @brief Runs the PCEM optimization loop. (Eigen Implementation, full history preserved)
     */
    bool optimize() override {

        log("\n--- Starting PCEM Optimization ---\n");
        log("Update Rule: Y_{k+1} = Σ w_m (Y_k + ε_m), where w ∝ exp(-γ(S(Ỹ) + ε^T*R*Y_k))\n\n");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        const size_t D = num_dimensions_;
        
        if (N < 3 || current_trajectory_.dimensions() != D) {
            std::cerr << "Error: Trajectory size, dimensionality mismatch, or not initialized correctly.\n";
            return false;
        }

        // --- MODIFIED CALLER SITE: Arguments ignored due to temporary implementation ---
        float collision_cost = computeCollisionCost(current_trajectory_, obstacles_); 
        float smoothness_cost = computeSmoothnessCost(current_trajectory_);
        
        float cost = collision_cost + smoothness_cost;

        float alpha = 0.99f;
        float alpha_temp = 1.01f;

        float best_cost = cost;
        size_t best_iteration = 0;

        storeTrajectory();
        
        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {

            // Update parameters
            gamma_ = gamma_ * std::pow(alpha, iteration-1);
            temperature_ = temperature_ * alpha_temp;

            logf("------- Iteration %zu : Cost = %.2f ------ ", iteration, cost);
            logf("Collision Cost = %.4f, Smoothness Cost = %.4f", collision_cost, smoothness_cost);

            // Extract trajectory as matrix
            Eigen::MatrixXf Y_k = trajectoryToMatrix();

            // Sample noise matrices
            std::vector<Eigen::MatrixXf> epsilon_samples = sampleNoiseMatrices(M, N, D);
            
            // Compute weights
            Eigen::VectorXf weights(M);
            float max_exponent = -std::numeric_limits<float>::infinity();

            for (size_t m = 0; m < M; ++m) {
                // ✓ Create perturbed trajectory
                Trajectory sample_traj = createPerturbedTrajectory(Y_k, epsilon_samples[m]);
            
                // --- MODIFIED CALLER SITE: Arguments ignored due to temporary implementation ---
                float sample_collision = computeCollisionCost(sample_traj, obstacles_);
                
                // Regularization term per dimension
                float reg_term = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    // Extract full vectors for dimension d
                    Eigen::VectorXf epsilon_d = epsilon_samples[m].row(d).transpose();  // N x 1
                    Eigen::VectorXf Y_k_d = Y_k.row(d).transpose();  // N x 1
                    
                    // Compute ε^T * R * Y for this dimension
                    // Note: R_matrix_ is sparse, so use sparse matrix-vector multiply
                    Eigen::VectorXf R_Y_d = R_matrix_ * Y_k_d;
                    reg_term += epsilon_d.dot(R_Y_d);
                }

                float exponent = -gamma_ * (sample_collision + reg_term) / temperature_;

                if (exponent > max_exponent) {
                    max_exponent = exponent;
                }                
                weights(m) = exponent; 
            }
            
            // --- Step 5: Normalize weights ---
            weights = (weights.array() - max_exponent).exp();
            float weight_sum = weights.sum();
            weights /= weight_sum;
            
            // --- Step 6: Compute Y_{k+1} via weighted mean update ---
            MatrixXf Y_kplus1 = MatrixXf::Zero(D, N);
            for (size_t m = 0; m < M; ++m) {
                Eigen::MatrixXf temp = Y_k + epsilon_samples[m];
                Y_kplus1 += weights(m) * temp;
                // Y_kplus1 += weights(m) * (Y_k + epsilon_samples[m]);
            }
            updateTrajectoryFromMatrix(Y_kplus1);

            // Fix start and goal
            current_trajectory_.nodes[0].position = start_node_.position;
            current_trajectory_.nodes[N - 1].position = goal_node_.position;

            log("  Updated trajectory computed.\n");
            
            // --- Step 8: Store Y_{k+1} and check convergence ---
            storeTrajectory();

            // ✓ RECALCULATE costs for the updated trajectory
            // --- MODIFIED CALLER SITE: Arguments ignored due to temporary implementation ---
            collision_cost = computeCollisionCost(current_trajectory_, obstacles_);
            smoothness_cost = computeSmoothnessCost(current_trajectory_);
            float new_cost = collision_cost + smoothness_cost;
            
            // Track best trajectory
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_iteration = iteration;
            }
            
            // Check if cost improvement is too small
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_ && cost - new_cost > 0) {
                log("Cost improvement negligible. Stopping.\n");
                break;
            }
            
            cost = new_cost;
        }

        // Restore the best trajectory found during optimization
        if (best_iteration < trajectory_history_.size()) {
            current_trajectory_ = trajectory_history_[best_iteration];
            logf("\n*** Restoring best trajectory from iteration %zu with cost %.2f ***", best_iteration, best_cost);
            
            // History truncation removed: all trajectories are kept for visualization.
        }
        
        // --- MODIFIED CALLER SITE: Arguments ignored due to temporary implementation ---
        logf("PCEM finished. Final Cost: %.2f (Collision: %.4f, Smoothness: %.4f)", 
         computeCollisionCost(current_trajectory_, obstacles_) + computeSmoothnessCost(current_trajectory_), 
         computeCollisionCost(current_trajectory_, obstacles_), 
         computeSmoothnessCost(current_trajectory_));

        log("\nLog saved to: " + getLogFilename());

        return true;
    }

    /**
     * @brief Load PCEM-specific configuration
     */
    bool loadPlannerConfig(const std::string& config_file_path) {
        try {
            YAML::Node config = YAML::LoadFile(config_file_path);
            
            if (!config["pce_planner"]) {
                std::cerr << "Error: 'pce_planner' section not found in config\n";
                return false;
            }
            
            const YAML::Node& planner_config = config["pce_planner"];
            
            // Read hyperparameters
            num_samples_ = planner_config["num_samples"].as<size_t>();
            num_iterations_ = planner_config["num_iterations"].as<size_t>();
            temperature_ = planner_config["temperature"].as<float>();
            eta_ = planner_config["eta"].as<float>();
            convergence_threshold_ = planner_config["convergence_threshold"].as<float>();
            
            // Compute gamma from eta
            gamma_ = eta_ / temperature_;
            
            // Read cost parameters
            if (planner_config["cost"]) {
                const YAML::Node& cost_config = planner_config["cost"];
                epsilon_sdf_ = cost_config["epsilon_sdf"].as<float>();
                sigma_obs_ = cost_config["sigma_obs"].as<float>();
            }
            
            // std::cout << "PCEM hyperparameters loaded:\n"
            //           << "  num_samples: " << num_samples_ << "\n"
            //           << "  num_iterations: " << num_iterations_ << "\n"
            //           << "  temperature: " << temperature_ << "\n"
            //           << "  eta: " << eta_ << "\n"
            //           << "  gamma: " << gamma_ << "\n";
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading PCEM config: " << e.what() << "\n";
            return false;
        }
    }


    void logPlannerSpecificConfig() override {
        log("--- PCEM Planner Parameters ---");
        log("  Algorithm:            Proximal Cross-Entropy Method");
        logf("  Number of samples:    %zu", num_samples_);
        logf("  Number of iterations: %zu", num_iterations_);
        logf("  Initial temperature:  %.4f", temperature_);
        logf("  Temperature scaling:  %.4f (alpha_temp)", 1.01f);
        logf("  Initial eta:          %.4f", eta_);
        logf("  Initial gamma:        %.4f", gamma_);
        logf("  Gamma decay:          %.4f (alpha)", 0.99f);
        logf("  Convergence threshold: %.6f", convergence_threshold_);
        log("");
        
        log("--- Cost Function Parameters ---");
        logf("  Epsilon SDF:          %.2f", epsilon_sdf_);
        logf("  Sigma obs:            %.4f", sigma_obs_);
        log("");
    }

    
private:
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float temperature_ = 1.5f;
    float eta_ = 1.0f;
    float gamma_ = 0.5f;
    float convergence_threshold_ = 0.01f;
    
    std::mt19937 random_engine_;

    
};
