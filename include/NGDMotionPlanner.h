/**
 * @file NGDMotionPlanner.h (Refactored Version - Using Task Interface)
 * @brief Natural Gradient Descent planner using Task interface
 * 
 * This planner is fully task-agnostic:
 * - NO knowledge of obstacles
 * - NO knowledge of collision detection
 * - ALL problem-specific logic in Task
 */
#pragma once

#include "MotionPlanner.h"
#include "task.h"
#include <random>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <algorithm> 
#include <Eigen/Dense>

/**
 * @brief Configuration for Natural Gradient Descent planner
 * 
 * Extends MotionPlannerConfig with NGD-specific algorithm parameters
 */
struct NGDConfig : public MotionPlannerConfig {
    // NGD algorithm parameters
    size_t num_samples = 3000;
    size_t num_iterations = 10;
    float learning_rate = 0.01f;
    float temperature = 1.0f;  // Temperature for gradient scaling
    float temperature_final = 0.1f; // Added: T_final for scheduling
    float convergence_threshold = 0.01f;
    
    // Cost function parameters
    float epsilon_sdf = 2.0f;
    float sigma_obs = 0.5f;
    
    /**
     * @brief Load NGD-specific configuration from YAML node
     * @param config YAML node containing configuration
     * @return true if loading successful
     */
    bool loadFromYAML(const YAML::Node& config) override {
        // First load base configuration
        if (!MotionPlannerConfig::loadFromYAML(config)) {
            return false;
        }
        
        try {
            // Load NGD-specific parameters
            if (config["ngd_planner"]) {
                const YAML::Node& ngd = config["ngd_planner"];
                
                if (ngd["num_samples"]) {
                    num_samples = ngd["num_samples"].as<size_t>();
                }
                if (ngd["num_iterations"]) {
                    num_iterations = ngd["num_iterations"].as<size_t>();
                }
                if (ngd["learning_rate"]) {
                    learning_rate = ngd["learning_rate"].as<float>();
                }
                if (ngd["temperature"]) {
                    temperature = ngd["temperature"].as<float>();
                }
                if (ngd["convergence_threshold"]) {
                    convergence_threshold = ngd["convergence_threshold"].as<float>();
                }
                
                // Load cost function parameters
                if (ngd["cost"]) {
                    const YAML::Node& cost_config = ngd["cost"];
                    if (cost_config["epsilon_sdf"]) {
                        epsilon_sdf = cost_config["epsilon_sdf"].as<float>();
                    }
                    if (cost_config["sigma_obs"]) {
                        sigma_obs = cost_config["sigma_obs"].as<float>();
                    }
                }
            }
            
            print();
            
            return validate();
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading NGD config from YAML: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * @brief Validate NGD configuration parameters
     * @return true if configuration is valid
     */
    bool validate() const override {
        // Validate base configuration first
        if (!MotionPlannerConfig::validate()) {
            return false;
        }
        
        // Validate NGD-specific parameters
        if (num_samples == 0) {
            std::cerr << "Error: num_samples must be > 0\n";
            return false;
        }
        
        if (num_iterations == 0) {
            std::cerr << "Error: num_iterations must be > 0\n";
            return false;
        }
        
        if (learning_rate <= 0.0f) {
            std::cerr << "Error: learning_rate must be positive\n";
            return false;
        }
        
        if (temperature <= 0.0f) {
            std::cerr << "Error: temperature must be positive\n";
            return false;
        }
        
        if (convergence_threshold < 0.0f) {
            std::cerr << "Error: convergence_threshold must be non-negative\n";
            return false;
        }
        
        if (epsilon_sdf < 0.0f) {
            std::cerr << "Error: epsilon_sdf must be non-negative\n";
            return false;
        }
        
        if (sigma_obs <= 0.0f) {
            std::cerr << "Error: sigma_obs must be positive\n";
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Print NGD configuration to console
     */
    void print() const override {
        // Print base configuration
        // MotionPlannerConfig::print();
        
        // Print NGD-specific parameters
        printf("=== NGD Planner Configuration ===\n");
        printf("Algorithm:              Natural Gradient Descent\n");
        printf("Number of samples:      %zu\n", num_samples);
        printf("Number of iterations:   %zu\n", num_iterations);
        printf("Initial learning rate:  %.6f\n", learning_rate);
        printf("Learning rate decay:    0.99 (alpha)\n");
        printf("Initial temperature:    %.4f\n", temperature);
        printf("Temperature scaling:    1.01 (alpha_temp)\n");
        printf("Convergence threshold:  %.6f\n", convergence_threshold);
        printf("Epsilon SDF:            %.2f\n", epsilon_sdf);
        printf("Sigma obs:              %.4f\n", sigma_obs);
        printf("\n");
    }

    std::string getScheduleName() const {
        switch (cov_schedule) {
            case CovarianceSchedule::CONSTANT:    return "constant";
            case CovarianceSchedule::LINEAR:      return "linear";
            case CovarianceSchedule::EXPONENTIAL: return "exponential";
            case CovarianceSchedule::COSINE:      return "cosine";
            case CovarianceSchedule::STEP:        return "step";
            case CovarianceSchedule::ADAPTIVE:    return "adaptive";
            default:                              return "unknown";
        }
    }
};

/**
 * @brief Natural Gradient Descent (NGD) for Trajectory Optimization.
 * 
 * This planner is fully task-agnostic:
 * - NO knowledge of obstacles
 * - NO knowledge of collision detection
 * - ALL problem-specific logic in Task
 * 
 * Configuration is loaded via NGDConfig object, solve() runs optimization.
 */
class NGDMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    /**
     * @brief Constructor
     * @param task Shared pointer to task defining the optimization problem
     */
    NGDMotionPlanner(pce::TaskPtr task = nullptr)
        : task_(task)
    {
        std::random_device rd;
        random_engine_.seed(rd());
    }

    /**
     * @brief Set the task for this planner
     */
    void setTask(pce::TaskPtr task) { task_ = task; }

    /**
     * @brief Get the current task
     */
    pce::TaskPtr getTask() const { return task_; }

    /**
     * @brief Initialize planner with NGD configuration
     * @param config NGD configuration object
     * @return true if initialization successful
     */
    bool initialize(const NGDConfig& config) 
    {
        // Store NGD config (also stores base config)
        try {
            ngd_config_ = std::make_shared<NGDConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating ngd_config_: " << e.what() << std::endl;
            return false;
        }
        
        // Extract NGD-specific parameters
        num_samples_ = config.num_samples;
        num_iterations_ = config.num_iterations;
        learning_rate_ = config.learning_rate;
        temperature_ = config.temperature;
        convergence_threshold_ = config.convergence_threshold;
        epsilon_sdf_ = config.epsilon_sdf;
        sigma_obs_ = config.sigma_obs;

        // Extract covariance scheduling parameters
        cov_schedule_ = config.cov_schedule;
        cov_scale_initial_ = config.cov_scale_initial;
        cov_scale_final_ = config.cov_scale_final;
        cov_decay_rate_ = config.cov_decay_rate;
        cov_step_interval_ = config.cov_step_interval;
        cov_step_factor_ = config.cov_step_factor;
        cov_adaptive_threshold_ = config.cov_adaptive_threshold;
        
        // Initialize current covariance scale
        cov_scale_current_ = cov_scale_initial_;
        
        // Call base class initialize with base config portion
        bool result = false;
        try {
            result = MotionPlanner::initialize(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in MotionPlanner::initialize(): " << e.what() << std::endl;
            return false;
        }
        
        if (!result) {
            std::cerr << "ERROR: Base initialization failed\n";
            return false;
        }
        
        // Initialize task after trajectory is set up
        try {
            initializeTask();
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in initializeTask(): " << e.what() << std::endl;
            return false;
        }
        
        return true;
    }

    /**
     * @brief Get planner name
     */
    std::string getPlannerName() const override {
        return "NGD";
    }

    const Eigen::SparseMatrix<float>& getRMatrix() const {
        return R_matrix_;
    }

    // Covariance scheduling getters
    float getCovarianceScale() const { return cov_scale_current_; }
    CovarianceSchedule getCovarianceSchedule() const { return cov_schedule_; }


    /**
     * @brief Runs the NGD optimization loop using Task interface
     */
    bool optimize() override {
        if (!task_) {
            std::cerr << "Error: No task set for optimization\n";
            return false;
        }

        log("Update Rule: Y_{k+1} = (1-η)Y_k + η Y_k - η E[S(Ỹ)ε], where Ỹ ~ N(Y_k, R^{-1})");
        log("");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t M = num_samples_;
        
        // Store initial parameters (FIX for compounding decay)
        const float initial_learning_rate = learning_rate_;
        const float initial_temperature = temperature_;
        const float alpha = 0.99f;
        const float alpha_temp = 1.01f;

        /**
         * @brief Modifications =======================================
         * 
         */
        float current_lr = initial_learning_rate;
        float best_cost = std::numeric_limits<float>::infinity();
        size_t best_iteration = 0;
        float prev_cost = std::numeric_limits<float>::infinity();

        // Reset history
        trajectory_history_.clear();
        covariance_scale_history_.clear();
        // ============================================================

        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            Eigen::MatrixXf Y_k = trajectoryToMatrix();

            // --- Temperature Schedule Calculation ---
            float progress = static_cast<float>(iteration - 1) / std::max(1.0f, static_cast<float>(num_iterations_ - 1));
            float current_temp = ngd_config_->temperature * std::pow((ngd_config_->temperature_final / ngd_config_->temperature), progress);
            Eigen::MatrixXf natural_gradient = Eigen::MatrixXf::Zero(D, N);

            // 2. Sampling and Evaluation
            // std::vector<Eigen::MatrixXf> epsilon_samples = sampleNoiseMatrices(M, N, D);
            std::vector<Eigen::MatrixXf> epsilon_samples = sampleScaledNoiseMatrices(M, cov_scale_current_);
            std::vector<Trajectory> sample_trajectories;
            sample_trajectories.reserve(M);
            
            for (size_t m = 0; m < M; ++m) {
                sample_trajectories.push_back(createPerturbedTrajectory(Y_k, epsilon_samples[m]));
            }
            
            // Batch evaluate
            std::vector<float> sample_collisions = task_->computeStateCostSimple(sample_trajectories);
                        
            // 3. Update current trajectory

            // Compute natural gradient with temperature scaling
            for (size_t m = 0; m < M; ++m) {
                natural_gradient += (sample_collisions[m] / current_temp) * epsilon_samples[m];
            }
            natural_gradient /= static_cast<float>(M);
            
            Eigen::MatrixXf Y_new = Y_k - current_lr * natural_gradient;
            
            updateTrajectoryFromMatrix(Y_new);
            
            // 4. Cleanup & Constraints
            current_trajectory_.nodes[0].position = start_node_.position;
            current_trajectory_.nodes[N - 1].position = goal_node_.position;
            task_->filterTrajectory(current_trajectory_, iteration);

            // 5. Logging and Convergence
            float current_collision = task_->computeStateCostSimple(current_trajectory_);
            float current_smoothness = computeSmoothnessCost(current_trajectory_);
            float current_total_cost = current_collision + current_smoothness;

            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);
            cov_scale_current_ = computeCovarianceScale(iteration + 1, prev_cost, current_total_cost);
            
            if (current_total_cost < best_cost) {
                best_cost = current_total_cost;
                best_iteration = iteration;
            }

            // --- Every 10 steps ---
            if (iteration == 1 || iteration % 10 == 0 || iteration == num_iterations_) {
                logf("Iteration %zu - Cost: %.2f (Collision: %.4f, Smoothness: %.4f) LR: %.6f, Temp: %.4f",
                    iteration, current_total_cost, current_collision, current_smoothness, current_lr, current_temp);
            }
            
            if (std::abs(prev_cost - current_total_cost) < convergence_threshold_) break;
            prev_cost = current_total_cost;
            
            task_->postIteration(iteration, current_total_cost, current_trajectory_);
        }
        
        current_trajectory_ = trajectory_history_[best_iteration];
        task_->done(true, num_iterations_, best_cost, current_trajectory_);
        return true;

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
        
        if (!ngd_config_) {
            std::cerr << "Error: No NGD configuration loaded\n";
            return;
        }
        
        // Task handles obstacle clearing and setup
        task_->initialize(num_dimensions_, start_node_, goal_node_, 
                        num_nodes_, total_time_);
        
        log("Task initialized");
    }

    /**
     * @brief Log NGD-specific configuration
     */
    void logPlannerSpecificConfig() override {
        log("--- NGD Planner Parameters ---");
        log("  Algorithm:            Natural Gradient Descent");
        logf("  Number of samples:    %zu", num_samples_);
        logf("  Number of iterations: %zu", num_iterations_);
        logf("  Initial learning rate: %.6f", learning_rate_);
        logf("  Learning rate decay:  %.4f (alpha)", 0.99f);
        logf("  Initial temperature:  %.4f", temperature_);
        logf("  Temperature scaling:  %.4f (alpha_temp)", 1.01f);
        logf("  Convergence threshold: %.6f", convergence_threshold_);
        log("");
        
        log("--- Cost Function Parameters ---");
        logf("  Epsilon SDF:          %.2f", epsilon_sdf_);
        logf("  Sigma obs:            %.4f", sigma_obs_);
        log("");
    }

private:
    // Task defining the optimization problem
    pce::TaskPtr task_;
    
    // NGD-specific configuration
    std::shared_ptr<NGDConfig> ngd_config_;
    
    // Algorithm hyperparameters (extracted from config)
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float learning_rate_ = 0.01f;
    float temperature_ = 1.0f;
    float convergence_threshold_ = 0.01f;
    
    // Cost function parameters
    float epsilon_sdf_ = 2.0f;
    float sigma_obs_ = 0.5f;

    // Covariance scheduling parameters
    CovarianceSchedule cov_schedule_ = CovarianceSchedule::EXPONENTIAL;
    float cov_scale_initial_ = 1.0f;
    float cov_scale_final_ = 0.1f;
    float cov_decay_rate_ = 0.9f;
    size_t cov_step_interval_ = 3;
    float cov_step_factor_ = 0.5f;
    float cov_adaptive_threshold_ = 0.05f;
    float cov_scale_current_ = 1.0f;
    
    std::vector<float> covariance_scale_history_;

};