/**
 * @file PCEMotionPlanner.h (Refactored Version - Config in Initialization)
 * @brief Proximal Cross-Entropy Method planner using Task interface
 * 
 * Features covariance scheduling for exploration-exploitation tradeoff:
 * - High covariance initially for broad exploration
 * - Gradually decreasing covariance for fine-tuning
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
 * @brief Covariance scheduling strategies
 */
enum class CovarianceSchedule {
    CONSTANT,       // No scheduling - fixed covariance
    LINEAR,         // Linear decay: σ(t) = σ_init * (1 - t/T) + σ_final * (t/T)
    EXPONENTIAL,    // Exponential decay: σ(t) = σ_init * α^t
    COSINE,         // Cosine annealing: σ(t) = σ_final + 0.5*(σ_init - σ_final)*(1 + cos(πt/T))
    STEP,           // Step decay: reduce by factor every k iterations
    ADAPTIVE        // Adaptive based on cost improvement
};


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
    
    // === Covariance Scheduling Parameters ===
    CovarianceSchedule cov_schedule = CovarianceSchedule::COSINE;
    float cov_scale_initial = 1.0f;      // Initial covariance scale (σ_init)
    float cov_scale_final = 0.1f;        // Final covariance scale (σ_final)
    float cov_decay_rate = 0.9f;         // Decay rate for exponential schedule (α)
    size_t cov_step_interval = 3;        // Interval for step schedule
    float cov_step_factor = 0.5f;        // Reduction factor for step schedule
    float cov_adaptive_threshold = 0.05f; // Cost improvement threshold for adaptive
    
    // === EMA ===
    float ema_alpha = 0.5f; // 1.0 = no EMA (standard PCE), 0.1 = heavy smoothing


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
                
                // === Load covariance scheduling parameters ===
                if (pce["covariance_schedule"]) {
                    std::string schedule_str = pce["covariance_schedule"].as<std::string>();
                    if (schedule_str == "constant") {
                        cov_schedule = CovarianceSchedule::CONSTANT;
                    } else if (schedule_str == "linear") {
                        cov_schedule = CovarianceSchedule::LINEAR;
                    } else if (schedule_str == "exponential") {
                        cov_schedule = CovarianceSchedule::EXPONENTIAL;
                    } else if (schedule_str == "cosine") {
                        cov_schedule = CovarianceSchedule::COSINE;
                    } else if (schedule_str == "step") {
                        cov_schedule = CovarianceSchedule::STEP;
                    } else if (schedule_str == "adaptive") {
                        cov_schedule = CovarianceSchedule::ADAPTIVE;
                    } else {
                        std::cerr << "Warning: Unknown covariance schedule '" << schedule_str 
                                  << "', using exponential\n";
                        cov_schedule = CovarianceSchedule::EXPONENTIAL;
                    }
                }
                
                if (pce["cov_scale_initial"]) {
                    cov_scale_initial = pce["cov_scale_initial"].as<float>();
                }
                if (pce["cov_scale_final"]) {
                    cov_scale_final = pce["cov_scale_final"].as<float>();
                }
                if (pce["cov_decay_rate"]) {
                    cov_decay_rate = pce["cov_decay_rate"].as<float>();
                }
                if (pce["cov_step_interval"]) {
                    cov_step_interval = pce["cov_step_interval"].as<size_t>();
                }
                if (pce["cov_step_factor"]) {
                    cov_step_factor = pce["cov_step_factor"].as<float>();
                }
                if (pce["cov_adaptive_threshold"]) {
                    cov_adaptive_threshold = pce["cov_adaptive_threshold"].as<float>();
                }

                if (pce["ema_alpha"]) {
                    ema_alpha = pce["ema_alpha"].as<float>();
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

        if (ema_alpha <= 0.0f || ema_alpha > 1.0f) {
            std::cerr << "Error: ema_alpha must be in (0, 1]\n";
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
        
        // Validate covariance scheduling parameters
        if (cov_scale_initial <= 0.0f) {
            std::cerr << "Error: cov_scale_initial must be positive\n";
            return false;
        }
        
        if (cov_scale_final < 0.0f) {
            std::cerr << "Error: cov_scale_final must be non-negative\n";
            return false;
        }
        
        if (cov_scale_final > cov_scale_initial) {
            std::cerr << "Warning: cov_scale_final > cov_scale_initial (covariance will increase)\n";
        }
        
        if (cov_decay_rate <= 0.0f || cov_decay_rate > 1.0f) {
            std::cerr << "Error: cov_decay_rate must be in (0, 1]\n";
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Get schedule name as string
     */
    std::string getScheduleName() const {
        switch (cov_schedule) {
            case CovarianceSchedule::CONSTANT: return "constant";
            case CovarianceSchedule::LINEAR: return "linear";
            case CovarianceSchedule::EXPONENTIAL: return "exponential";
            case CovarianceSchedule::COSINE: return "cosine";
            case CovarianceSchedule::STEP: return "step";
            case CovarianceSchedule::ADAPTIVE: return "adaptive";
            default: return "unknown";
        }
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
        
        // Print covariance scheduling parameters
        std::cout << "=== Covariance Scheduling ===\n";
        std::cout << "Schedule type:          " << getScheduleName() << "\n";
        std::cout << "Initial scale (σ_init): " << cov_scale_initial << "\n";
        std::cout << "Final scale (σ_final):  " << cov_scale_final << "\n";
        
        switch (cov_schedule) {
            case CovarianceSchedule::EXPONENTIAL:
                std::cout << "Decay rate (α):         " << cov_decay_rate << "\n";
                std::cout << "Formula: σ(t) = σ_init * α^t\n";
                break;
            case CovarianceSchedule::LINEAR:
                std::cout << "Formula: σ(t) = σ_init + (σ_final - σ_init) * t/T\n";
                break;
            case CovarianceSchedule::COSINE:
                std::cout << "Formula: σ(t) = σ_final + 0.5*(σ_init - σ_final)*(1 + cos(πt/T))\n";
                break;
            case CovarianceSchedule::STEP:
                std::cout << "Step interval:          " << cov_step_interval << "\n";
                std::cout << "Step factor:            " << cov_step_factor << "\n";
                break;
            case CovarianceSchedule::ADAPTIVE:
                std::cout << "Improvement threshold:  " << cov_adaptive_threshold << "\n";
                break;
            default:
                break;
        }
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
 * Features covariance scheduling for exploration-exploitation tradeoff.
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

        // EMA schedule parameter
        ema_alpha_ = config.ema_alpha;
                
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
    
    // Covariance scheduling getters
    float getCovarianceScale() const { return cov_scale_current_; }
    CovarianceSchedule getCovarianceSchedule() const { return cov_schedule_; }

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
     * @brief Compute covariance scale for given iteration
     * @param iteration Current iteration (1-indexed)
     * @param prev_cost Previous iteration cost (for adaptive schedule)
     * @param curr_cost Current iteration cost (for adaptive schedule)
     * @return Covariance scale factor
     */
    float computeCovarianceScale(size_t iteration, float prev_cost = 0.0f, float curr_cost = 0.0f) {
        float t = static_cast<float>(iteration - 1);  // 0-indexed for formulas
        float T = static_cast<float>(num_iterations_);
        
        switch (cov_schedule_) {
            case CovarianceSchedule::CONSTANT:
                return cov_scale_initial_;
                
            case CovarianceSchedule::LINEAR:
                // Linear interpolation from initial to final
                return cov_scale_initial_ + (cov_scale_final_ - cov_scale_initial_) * (t / T);
                
            case CovarianceSchedule::EXPONENTIAL:
                // Exponential decay: σ(t) = σ_init * α^t
                // Optionally clamp to final value
                return std::max(cov_scale_final_, 
                               cov_scale_initial_ * std::pow(cov_decay_rate_, t));
                
            case CovarianceSchedule::COSINE:
                // Cosine annealing (smooth transition)
                return cov_scale_final_ + 0.5f * (cov_scale_initial_ - cov_scale_final_) * 
                       (1.0f + std::cos(M_PI * t / T));
                
            case CovarianceSchedule::STEP:
                // Step decay: reduce by factor every k iterations
                {
                    size_t num_steps = (iteration - 1) / cov_step_interval_;
                    float scale = cov_scale_initial_ * std::pow(cov_step_factor_, 
                                                                 static_cast<float>(num_steps));
                    return std::max(cov_scale_final_, scale);
                }
                
            case CovarianceSchedule::ADAPTIVE:
                // Adaptive: reduce if cost improved, else keep or increase slightly
                {
                    if (iteration <= 1) {
                        return cov_scale_current_;
                    }
                    
                    float improvement = (prev_cost - curr_cost) / (std::abs(prev_cost) + 1e-6f);
                    
                    if (improvement > cov_adaptive_threshold_) {
                        // Good improvement - reduce covariance to focus search
                        cov_scale_current_ *= cov_decay_rate_;
                    } else if (improvement < 0) {
                        // Cost increased - slightly increase covariance to explore more
                        cov_scale_current_ *= (1.0f + 0.1f * (1.0f - cov_decay_rate_));
                    }
                    // else: small improvement - keep current scale
                    
                    // Clamp to bounds
                    cov_scale_current_ = std::max(cov_scale_final_, 
                                                  std::min(cov_scale_initial_, cov_scale_current_));
                    return cov_scale_current_;
                }
                
            default:
                return cov_scale_initial_;
        }
    }

    /**
     * @brief Sample noise matrices with covariance scaling
     * 
     * Samples ε ~ N(0, σ² * Σ) where:
     * - Σ = R⁻¹ is the structured covariance from the smoothness prior
     * - σ is the covariance scale factor
     * 
     * Implementation:
     * 1. Sample z ~ N(0, I) 
     * 2. Transform via Cholesky: ε_base ~ N(0, Σ) by solving L·ε = z
     * 3. Scale: ε = σ · ε_base ~ N(0, σ²·Σ)
     * 
     * @param num_samples Number of samples to generate
     * @param scale Covariance scale factor σ
     * @return Vector of scaled noise matrices
     */
    std::vector<Eigen::MatrixXf> sampleScaledNoiseMatrices(size_t num_samples, float scale) {
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        
        // Sample from structured prior (ε ~ N(0, Σ))
        std::vector<Eigen::MatrixXf> epsilon_samples = sampleNoiseMatrices(num_samples, N, D);
        
        // Scale samples: σ·ε ~ N(0, σ²·Σ)
        // This preserves the correlation structure while adjusting variance
        if (std::abs(scale - 1.0f) > 1e-6f) {
            for (size_t m = 0; m < num_samples; ++m) {
                epsilon_samples[m] *= scale;
            }
        }
        
        return epsilon_samples;
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
        log("Update Rule: Y_{k+1} = Σ w_m (Y_k + σ_k * ε_m)\n");
        log("  where ε_m ~ N(0, R⁻¹) samples from smoothness prior\n");
        log("  and σ_k is the covariance scale at iteration k\n");
        logf("Covariance Schedule: %s (σ_init=%.3f → σ_final=%.3f)\n\n",
             pce_config_->getScheduleName().c_str(), cov_scale_initial_, cov_scale_final_);
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        const size_t D = num_dimensions_;

        // Validate trajectory
        if (N == 0 || current_trajectory_.nodes[0].position.size() != D) {
            std::cerr << "Error: Invalid trajectory!\n";
            return false;
        }

        float alpha = 0.99f;
        float alpha_temp = 1.01f;

        float best_cost = std::numeric_limits<float>::infinity();
        size_t best_iteration = 0;
        
        // Reset covariance scale
        cov_scale_current_ = cov_scale_initial_;

        // Clear history
        trajectory_history_.clear();
        covariance_scale_history_.clear();
        
        // Iteration 0: Evaluate and store initial trajectory
        float prev_cost = 0.0f;
        {
            float collision_cost = task_->computeCollisionCostSimple(current_trajectory_);
            float smoothness_cost = computeSmoothnessCost(current_trajectory_);
            float cost = collision_cost + smoothness_cost;
            
            logf("Iter 0 (Init) - Cost: %.2f (Coll: %.4f, Smooth: %.4f), σ=%.4f",
                cost, collision_cost, smoothness_cost, cov_scale_current_);
            
            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);
            
            best_cost = cost;
            best_iteration = 0;
            prev_cost = cost;
        }
        
        // Main optimization loop
        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            // Update parameters
            gamma_ = gamma_ * std::pow(alpha, iteration - 1);
            temperature_ = temperature_ * alpha_temp;

            // Extract trajectory as matrix
            Eigen::MatrixXf Y_k = trajectoryToMatrix();
            
            if (Y_k.rows() != static_cast<long>(D) || Y_k.cols() != static_cast<long>(N)) {
                std::cerr << "Error: Matrix size mismatch!\n";
                return false;
            }

            // Sample noise matrices with current covariance scale
            // ε ~ N(0, σ² * R⁻¹) - scaled structured noise
            std::vector<Eigen::MatrixXf> epsilon_samples = sampleScaledNoiseMatrices(M, cov_scale_current_);
            
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
            std::vector<float> sample_collisions = task_->computeCollisionCostSimple(sample_trajectories);
            
            if (sample_collisions.size() != M) {
                std::cerr << "Error: Wrong number of collision costs!\n";
                return false;
            }
            
            // Check for invalid costs
            for (size_t m = 0; m < sample_collisions.size(); ++m) {
                if (!std::isfinite(sample_collisions[m])) {
                    sample_collisions[m] = 1e6f;
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
                    exponent = -1e6f;
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
                weights.setConstant(1.0f / M);
            } else {
                weights /= weight_sum;
            }
            
            // 1. Compute Y_{k+1} via weighted mean update
            MatrixXf Y_weighted = MatrixXf::Zero(D, N);

            for (size_t m = 0; m < M; ++m) {
                Eigen::MatrixXf temp = Y_k + epsilon_samples[m];
                Y_weighted += weights(m) * temp; // EMA updates
            }

            // 2. Apply EMA Update: Y_{k+1} = (1 - α) * Y_k + α * Y_weighted
            // This acts as an adaptive step size.
            MatrixXf Y_new = (1.0f - ema_alpha_) * Y_k + ema_alpha_ * Y_weighted;
            
            updateTrajectoryFromMatrix(Y_new);

            // Fix start and goal
            current_trajectory_.nodes[0].position = start_node_.position;
            current_trajectory_.nodes[N - 1].position = goal_node_.position;

            // Apply task filtering
            bool filtered = task_->filterTrajectory(current_trajectory_, iteration);
            if (filtered) {
                log("  Trajectory filtered by task\n");
            }

            // Store trajectory
            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);

            // Compute costs
            float collision_cost = task_->computeCollisionCostSimple(current_trajectory_);
            float smoothness_cost = computeSmoothnessCost(current_trajectory_);
            float cost = collision_cost + smoothness_cost;
            
            logf("Iter %2zu - Cost: %.2f (Coll: %.4f, Smooth: %.4f), σ=%.4f",
                iteration, cost, collision_cost, smoothness_cost, cov_scale_current_);
            
            // Track best trajectory
            if (cost < best_cost) {
                logf("  New best! (previous: %.2f at iteration %zu)", best_cost, best_iteration);
                best_cost = cost;
                best_iteration = iteration;
            }
            
            // Update covariance scale for next iteration
            cov_scale_current_ = computeCovarianceScale(iteration + 1, prev_cost, cost);
            
            // Check convergence
            if (iteration > 1) {
                if (std::abs(prev_cost - cost) < convergence_threshold_ && prev_cost - cost > 0) {
                    log("Cost improvement negligible. Stopping.\n");
                    break;
                }
            }
            
            prev_cost = cost;
            
            // Notify task
            task_->postIteration(iteration, cost, current_trajectory_);
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
        
        // Final summary
        float final_collision = task_->computeCollisionCostSimple(current_trajectory_);
        float final_smoothness = computeSmoothnessCost(current_trajectory_);
        float final_cost = final_collision + final_smoothness;
        
        logf("PCEM finished. Final Cost: %.2f (Collision: %.4f, Smoothness: %.4f)", 
            final_cost, final_collision, final_smoothness);
        
        // Log covariance schedule summary
        log("\nCovariance Schedule History:");
        for (size_t i = 0; i < covariance_scale_history_.size(); ++i) {
            logf("  Iter %zu: σ = %.4f", i, covariance_scale_history_[i]);
        }

        log("\nLog saved to: " + getLogFilename());

        return success;
    }
    
    /**
     * @brief Get covariance scale history
     */
    const std::vector<float>& getCovarianceScaleHistory() const {
        return covariance_scale_history_;
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
        log("--- Covariance Scheduling ---");
        logf("  Schedule:             %s", pce_config_->getScheduleName().c_str());
        logf("  Initial scale:        %.4f", cov_scale_initial_);
        logf("  Final scale:          %.4f", cov_scale_final_);
        logf("  Decay rate:           %.4f", cov_decay_rate_);
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

    float ema_alpha_ = 0.5f;
    
    // Covariance scheduling parameters
    CovarianceSchedule cov_schedule_ = CovarianceSchedule::EXPONENTIAL;
    float cov_scale_initial_ = 1.0f;
    float cov_scale_final_ = 0.1f;
    float cov_decay_rate_ = 0.9f;
    size_t cov_step_interval_ = 3;
    float cov_step_factor_ = 0.5f;
    float cov_adaptive_threshold_ = 0.05f;
    float cov_scale_current_ = 1.0f;
    
    // History tracking
    std::vector<float> covariance_scale_history_;
    
    std::mt19937 random_engine_;
};