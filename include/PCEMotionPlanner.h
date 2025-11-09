/**
 * @file PCEMotionPlanner.h (Refactored Version - Config in Initialization)
 * @brief Proximal Cross-Entropy Method planner using Task interface
 * 
 */
#pragma once

#include "MotionPlanner.h"
#include "task.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>


/**
 * @brief Configuration for Proximal Cross-Entropy Method planner
 * 
 * Extends MotionPlannerConfig with PCE-specific algorithm parameters
 */
struct PCEConfig : public MotionPlannerConfig {
    // PCE algorithm parameters
    size_t num_samples = 3000;
    size_t num_iterations = 10;
    float eta = 1.0f;
    float temperature = 1.5f;
    float convergence_threshold = 0.01f;
    float collision_clearance = 0.1f;
    float collision_threshold = 0.1f;
    
    // Derived parameter (computed from eta and temperature)
    float gamma = 0.5f;
    
    /**
     * @brief Load PCE-specific configuration from YAML node
     * @param config YAML node containing configuration
     * @return true if loading successful
     */
    bool loadFromYAML(const YAML::Node& config) override {
        // First load base configuration
        if (!MotionPlannerConfig::loadFromYAML(config)) {
            return false;
        }
        
        try {
            // Load PCE-specific parameters
            if (config["pce_planner"]) {
                const YAML::Node& pce = config["pce_planner"];
                
                if (pce["num_samples"]) {
                    num_samples = pce["num_samples"].as<size_t>();
                }
                if (pce["num_iterations"]) {
                    num_iterations = pce["num_iterations"].as<size_t>();
                }
                if (pce["eta"]) {
                    eta = pce["eta"].as<float>();
                }
                if (pce["temperature"]) {
                    temperature = pce["temperature"].as<float>();
                }
                if (pce["convergence_threshold"]) {
                    convergence_threshold = pce["convergence_threshold"].as<float>();
                }
                if (pce["collision_clearance"]) {
                    collision_clearance = pce["collision_clearance"].as<float>();
                }
                if (pce["collision_threshold"]) {
                    collision_threshold = pce["collision_threshold"].as<float>();
                }
                
                // Compute gamma from eta and temperature
                if (temperature > 0.0f) {
                    gamma = eta / temperature;
                }

            }

            print();
            
            return validate();
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading PCE config from YAML: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * @brief Validate PCE configuration parameters
     * @return true if configuration is valid
     */
    bool validate() const override {
        // Validate base configuration first
        if (!MotionPlannerConfig::validate()) {
            return false;
        }
        
        // Validate PCE-specific parameters
        if (num_samples == 0) {
            std::cerr << "Error: num_samples must be > 0\n";
            return false;
        }
        
        if (num_iterations == 0) {
            std::cerr << "Error: num_iterations must be > 0\n";
            return false;
        }
        
        if (temperature <= 0.0f) {
            std::cerr << "Error: temperature must be positive\n";
            return false;
        }
        
        if (eta < 0.0f) {
            std::cerr << "Error: eta must be non-negative\n";
            return false;
        }
        
        if (convergence_threshold < 0.0f) {
            std::cerr << "Error: convergence_threshold must be non-negative\n";
            return false;
        }

        if (collision_clearance < 0.0f) {
            std::cerr << "Error: collision_clearance must be non-negative\n";
            return false;
        }

        if (collision_threshold < 0.0f) {
            std::cerr << "Error: collision_threshold must be non-negative\n";
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Print PCE configuration to console
     */
    void print() const override {
        // Print base configuration
        MotionPlannerConfig::print();
        
        // Print PCE-specific parameters
        std::cout << "=== PCE Planner Configuration ===\n";
        std::cout << "Algorithm:              Proximal Cross-Entropy Method\n";
        std::cout << "Number of samples:      " << num_samples << "\n";
        std::cout << "Number of iterations:   " << num_iterations << "\n";
        std::cout << "Initial temperature:    " << temperature << "\n";
        std::cout << "Temperature scaling:    1.01 (alpha_temp)\n";
        std::cout << "Initial eta:            " << eta << "\n";
        std::cout << "Initial gamma:          " << gamma << "\n";
        std::cout << "Gamma decay:            0.99 (alpha)\n";
        std::cout << "Convergence threshold:  " << convergence_threshold << "\n";
        std::cout << "Collision clearance:    " << collision_clearance << "\n";
        std::cout << "Collision threshold:    " << collision_threshold << "\n";
        std::cout << "\n";
    }
};


/**
 * @brief Proximal Cross-Entropy Method (PCEM) for Trajectory Optimization.
 * 
 * This planner is fully task-agnostic:
 * - NO knowledge of obstacles
 * - NO knowledge of collision detection
 * - ALL problem-specific logic in Task
 * 
 * Configuration is loaded via PCEConfig object, solve() runs optimization.
 */
class ProximalCrossEntropyMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    /**
     * @brief Constructor
     * @param task Shared pointer to task defining the optimization problem
     */
    ProximalCrossEntropyMotionPlanner(pce::TaskPtr task = nullptr)
        : task_(task)
    {
        std::random_device rd;
        seedRandomEngine(rd());
    }

    /**
     * @brief Set the task for this planner
     */
    void setTask(pce::TaskPtr task) {
        task_ = task;
    }

    /**
     * @brief Get the current task
     */
    pce::TaskPtr getTask() const {
        return task_;
    }

    /**
     * @brief Initialize planner with PCE configuration
     * @param config PCE configuration object
     * @return true if initialization successful
     */
    bool initialize(const PCEConfig& config) 
    {
        
        // Store PCE config (also stores base config)
        try {
            pce_config_ = std::make_shared<PCEConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating pce_config_: " << e.what() << std::endl;
            return false;
        }
        
        // Extract PCE-specific parameters
        num_samples_ = config.num_samples;
        num_iterations_ = config.num_iterations;
        temperature_ = config.temperature;
        
        eta_ = config.eta;
        gamma_ = config.gamma;
        convergence_threshold_ = config.convergence_threshold;
                
        // Call base class initialize with base config portion
        bool result = false;
        try {
            result = MotionPlanner::initialize(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in MotionPlanner::initialize: " << e.what() << std::endl;
            return false;
        }
        
        if (result) {
            pce_config_->print();
        }
        
        return result;
    }

    /**
     * @brief Get planner name
     */
    std::string getPlannerName() const override {
        return "PCEM";
    }

    // Getters for algorithm parameters
    size_t getNumSamples() const { return num_samples_; }
    size_t getNumIterations() const { return num_iterations_; }
    float getTemperature() const { return temperature_; }
    float getEta() const { return eta_; }
    float getGamma() const { return gamma_; }
    float getConvergenceThreshold() const { return convergence_threshold_; }

    /**
     * @brief Get PCE configuration
     */
    std::shared_ptr<const PCEConfig> getPCEConfig() const {
        return pce_config_;
    }

    void seedRandomEngine(unsigned int seed) {
        random_engine_.seed(seed);
    }

    /**
     * @brief Override collision cost computation to use Task
     */
    float computeCollisionCost(const Trajectory& trajectory) const override {
        if (!task_) {
            std::cerr << "Error: No task set for collision cost computation\n";
            return std::numeric_limits<float>::infinity();
        }
        return task_->computeCollisionCost(trajectory);
    }

    /**
     * @brief Runs the PCEM optimization loop
     */
    bool optimize() override {
        if (!task_) {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }

        log("\n--- Starting PCEM Optimization ---\n");
        log("Update Rule: Y_{k+1} = Σ w_m (Y_k + ε_m), where w ∝ exp(-γ(S(ỹ) + ε^T*R*Y_k))\n\n");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        const size_t D = num_dimensions_;

        // Validate trajectory
        if (N == 0) {
            std::cerr << "Error: Current trajectory has no nodes!\n";
            return false;
        }
        
        if (current_trajectory_.nodes[0].position.size() != D) {
            std::cerr << "Error: Dimension mismatch in trajectory!\n";
            return false;
        }

        // Compute initial costs
        float collision_cost = task_->computeCollisionCost(current_trajectory_);
        float smoothness_cost = computeSmoothnessCost(current_trajectory_);
        float cost = collision_cost + smoothness_cost;

        float alpha = 0.99f;
        float alpha_temp = 1.01f;

        float best_cost = cost;
        size_t best_iteration = 0;

        storeTrajectory();
        
        // Main optimization loop
        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            // Update parameters
            gamma_ = gamma_ * std::pow(alpha, iteration-1);
            temperature_ = temperature_ * alpha_temp;

            logf("------- Iteration %zu : Cost = %.2f ------ ", iteration, cost);
            logf("Collision Cost = %.4f, Smoothness Cost = %.4f", collision_cost, smoothness_cost);

            // Extract trajectory as matrix
            Eigen::MatrixXf Y_k = trajectoryToMatrix();
            
            if (Y_k.rows() != static_cast<long>(D) || Y_k.cols() != static_cast<long>(N)) {
                std::cerr << "Error: Matrix size mismatch!\n";
                return false;
            }

            // Sample noise matrices
            std::vector<Eigen::MatrixXf> epsilon_samples = sampleNoiseMatrices(M, N, D);
            
            if (epsilon_samples.size() != M) {
                std::cerr << "Error: Wrong number of noise matrices!\n";
                return false;
            }
            
            // Create perturbed trajectories
            std::vector<Trajectory> sample_trajectories;
            sample_trajectories.reserve(M);
            
            for (size_t m = 0; m < M; ++m) {
                Trajectory sample_traj = createPerturbedTrajectory(Y_k, epsilon_samples[m]);
                
                if (sample_traj.nodes.size() != N) {
                    std::cerr << "Error: Sample trajectory has wrong size!\n";
                    return false;
                }
                
                sample_trajectories.push_back(sample_traj);
            }
            
            // Batch evaluate collision costs
            std::vector<float> sample_collisions = task_->computeCollisionCost(sample_trajectories);
            
            if (sample_collisions.size() != M) {
                std::cerr << "Error: Wrong number of collision costs!\n";
                return false;
            }
            
            // Check for invalid costs and fix
            for (size_t m = 0; m < sample_collisions.size(); ++m) {
                if (!std::isfinite(sample_collisions[m])) {
                    sample_collisions[m] = 1e6f;  // Large penalty
                }
            }
            
            // Compute weights
            Eigen::VectorXf weights(M);
            float max_exponent = -std::numeric_limits<float>::infinity();

            for (size_t m = 0; m < M; ++m) {
                float sample_collision = sample_collisions[m];
                
                // Regularization term per dimension
                float reg_term = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    Eigen::VectorXf epsilon_d = epsilon_samples[m].row(d).transpose();
                    Eigen::VectorXf Y_k_d = Y_k.row(d).transpose();
                    
                    Eigen::VectorXf R_Y_d = R_matrix_ * Y_k_d;
                    reg_term += epsilon_d.dot(R_Y_d);
                }

                float exponent = -gamma_ * (sample_collision + reg_term) / temperature_;
                
                if (!std::isfinite(exponent)) {
                    exponent = -1e6f;  // Very small weight
                }

                if (exponent > max_exponent) {
                    max_exponent = exponent;
                }
                
                weights(m) = exponent;
            }
            
            // Normalize weights
            weights = (weights.array() - max_exponent).exp();
            float weight_sum = weights.sum();
            
            if (!std::isfinite(weight_sum) || weight_sum < 1e-10f) {
                weights.setConstant(1.0f / M);  // Uniform weights as fallback
            } else {
                weights /= weight_sum;
            }
            
            // Compute Y_{k+1} via weighted mean update
            MatrixXf Y_new = MatrixXf::Zero(D, N);
            
            for (size_t m = 0; m < M; ++m) {
                Eigen::MatrixXf temp = Y_k + epsilon_samples[m];
                Y_new += weights(m) * temp;
            }
            
            updateTrajectoryFromMatrix(Y_new);

            // Fix start and goal
            current_trajectory_.nodes[0].position = start_node_.position;
            current_trajectory_.nodes[N - 1].position = goal_node_.position;

            // Apply task filtering if available
            bool filtered = task_->filterTrajectory(current_trajectory_, iteration);
            if (filtered) {
                log("  Trajectory filtered by task\n");
            }

            log("  Updated trajectory computed.\n");
            
            // Store trajectory
            storeTrajectory();

            // Recalculate costs
            collision_cost = task_->computeCollisionCost(current_trajectory_);
            smoothness_cost = computeSmoothnessCost(current_trajectory_);
            float new_cost = collision_cost + smoothness_cost;
            
            // Track best trajectory
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_iteration = iteration;
            }
            
            // Check convergence
            if (iteration > 1 && std::abs(cost - new_cost) < convergence_threshold_ && cost - new_cost > 0) {
                log("Cost improvement negligible. Stopping.\n");
                break;
            }
            
            // Notify task of iteration completion
            task_->postIteration(iteration, new_cost, current_trajectory_);
            
            cost = new_cost;
        }

        // Restore best trajectory
        if (best_iteration < trajectory_history_.size()) {
            current_trajectory_ = trajectory_history_[best_iteration];
            logf("\n*** Restoring best trajectory from iteration %zu with cost %.2f ***", 
                best_iteration, best_cost);
        }
        
        // Notify task of completion
        bool success = (best_cost < std::numeric_limits<float>::infinity());
        task_->done(success, num_iterations_, best_cost, current_trajectory_);
        
        logf("PCEM finished. Final Cost: %.2f (Collision: %.4f, Smoothness: %.4f)", 
            task_->computeCollisionCost(current_trajectory_) + computeSmoothnessCost(current_trajectory_),
            task_->computeCollisionCost(current_trajectory_), 
            computeSmoothnessCost(current_trajectory_));

        log("\nLog saved to: " + getLogFilename());

        return success;
    }


protected:
    
    /**
     * @brief Initialize task with trajectory parameters
     */
    void initializeTask() override {
        if (!task_) {
            std::cerr << "Warning: No task set for initialization\n";
            return;
        }
        
        if (!pce_config_) {
            std::cerr << "Error: No PCE configuration loaded\n";
            return;
        }
        
        // Task handles obstacle clearing and setup
        task_->initialize(num_dimensions_, start_node_, goal_node_, 
                        num_nodes_, total_time_);
        
        std::cout << "Task initialized\n";
    }

    /**
     * @brief Log PCEM-specific configuration
     */
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
    }

private:
    // Task defining the optimization problem
    pce::TaskPtr task_;
    
    // PCE-specific configuration
    std::shared_ptr<PCEConfig> pce_config_;
    
    // Algorithm hyperparameters (extracted from config)
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float temperature_ = 1.5f;
    float eta_ = 1.0f;
    float gamma_ = 0.5f;
    float convergence_threshold_ = 0.01f;
    
    std::mt19937 random_engine_;
};
