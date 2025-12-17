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
 * @brief Configuration for Proximal Cross-Entropy Method planner
 * 
 * Extends MotionPlannerConfig with PCE-specific algorithm parameters
 */
struct PCEConfig : public MotionPlannerConfig {
    // PCE algorithm parameters
    size_t num_samples = 3000;
    size_t num_iterations = 10;
    float eta = 1.0f;
    float temperature = 1.5f;       // T_initial
    float temperature_final = 0.1f; // Added: T_final for scheduling
    float convergence_threshold = 0.01f;
    float collision_clearance = 0.1f;
    float collision_threshold = 0.1f;
    
    // Updated fixed gamma calculation
    float gamma = 0.5f; 

    // Elite portion
    float elite_ratio = 0.1f;
        
    // === EMA ===
    float ema_alpha = 0.5f;

    bool loadFromYAML(const YAML::Node& config) override {
        if (!MotionPlannerConfig::loadFromYAML(config)) return false;
        
        try {
            if (config["pce_planner"]) {
                const YAML::Node& pce = config["pce_planner"];
                
                if (pce["num_samples"]) num_samples = pce["num_samples"].as<size_t>();
                if (pce["num_iterations"]) num_iterations = pce["num_iterations"].as<size_t>();
                if (pce["eta"]) {
                    eta = pce["eta"].as<float>();
                    // Apply your fixed formulation: gamma = eta / (eta + 1)
                    gamma = eta / (eta + 1.0f);
                }
                
                if (pce["temperature"]) temperature = pce["temperature"].as<float>();
                if (pce["temperature_final"]) {
                    temperature_final = pce["temperature_final"].as<float>();
                } else {
                    temperature_final = temperature; // Default to constant temp
                }

                if (pce["convergence_threshold"]) convergence_threshold = pce["convergence_threshold"].as<float>();
                if (pce["elite_ratio"]) elite_ratio = pce["elite_ratio"].as<float>();
                if (pce["ema_alpha"]) ema_alpha = pce["ema_alpha"].as<float>();
                if (pce["collision_clearance"]) collision_clearance = pce["collision_clearance"].as<float>();
                if (pce["collision_threshold"]) collision_threshold = pce["collision_threshold"].as<float>();
                
                // // Covariance Schedule
                // if (pce["covariance_schedule"]) {
                //     std::string s = pce["covariance_schedule"].as<std::string>();
                //     if (s == "constant") cov_schedule = CovarianceSchedule::CONSTANT;
                //     else if (s == "linear") cov_schedule = CovarianceSchedule::LINEAR;
                //     else if (s == "exponential") cov_schedule = CovarianceSchedule::EXPONENTIAL;
                //     else if (s == "cosine") cov_schedule = CovarianceSchedule::COSINE;
                //     else if (s == "step") cov_schedule = CovarianceSchedule::STEP;
                //     else if (s == "adaptive") cov_schedule = CovarianceSchedule::ADAPTIVE;
                // }
                
                // if (pce["cov_scale_initial"]) cov_scale_initial = pce["cov_scale_initial"].as<float>();
                // if (pce["cov_scale_final"])   cov_scale_final = pce["cov_scale_final"].as<float>();
                // if (pce["cov_decay_rate"])    cov_decay_rate = pce["cov_decay_rate"].as<float>();
                // if (pce["cov_step_interval"]) cov_step_interval = pce["cov_step_interval"].as<size_t>();
                // if (pce["cov_step_factor"])   cov_step_factor = pce["cov_step_factor"].as<float>();
            }
            return validate();
        } catch (const std::exception& e) {
            std::cerr << "Error loading PCE config: " << e.what() << "\n";
            return false;
        }
    }

    void print() const override {
        MotionPlannerConfig::print();
        std::cout << "--- PCE Hyperparameters (Updated) ---\n";
        std::cout << "  Eta:         " << eta << " (Fixed Gamma: " << gamma << ")\n";
        std::cout << "  Temperature: " << temperature << " -> " << temperature_final << "\n";
        std::cout << "  Elite Ratio: " << elite_ratio << "\n";
        std::cout << "  EMA Alpha:   " << ema_alpha << "\n";
        std::cout << "  Cov Scale:   " << cov_scale_initial << " -> " << cov_scale_final << "\n";
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

        // Elite ratio
        elite_ratio_ = config.elite_ratio;

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
        
        // if (result) {
        //     pce_config_->print();
        // }
        
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
    
    // /**
    //  * @brief Compute covariance scale for given iteration
    //  * @param iteration Current iteration (1-indexed)
    //  * @param prev_cost Previous iteration cost (for adaptive schedule)
    //  * @param curr_cost Current iteration cost (for adaptive schedule)
    //  * @return Covariance scale factor
    //  */
    // float computeCovarianceScale(size_t iteration, float prev_cost = 0.0f, float curr_cost = 0.0f) {
    //     float t = static_cast<float>(iteration - 1);  // 0-indexed for formulas
    //     float T = static_cast<float>(num_iterations_);
        
    //     switch (cov_schedule_) {
    //         case CovarianceSchedule::CONSTANT:
    //             return cov_scale_initial_;
                
    //         case CovarianceSchedule::LINEAR:
    //             // Linear interpolation from initial to final
    //             return cov_scale_initial_ + (cov_scale_final_ - cov_scale_initial_) * (t / T);
                
    //         case CovarianceSchedule::EXPONENTIAL:
    //             // Exponential decay: σ(t) = σ_init * α^t
    //             // Optionally clamp to final value
    //             return std::max(cov_scale_final_, 
    //                            cov_scale_initial_ * std::pow(cov_decay_rate_, t));
                
    //         case CovarianceSchedule::COSINE:
    //             // Cosine annealing (smooth transition)
    //             return cov_scale_final_ + 0.5f * (cov_scale_initial_ - cov_scale_final_) * 
    //                    (1.0f + std::cos(M_PI * t / T));
                
    //         case CovarianceSchedule::STEP:
    //             // Step decay: reduce by factor every k iterations
    //             {
    //                 size_t num_steps = (iteration - 1) / cov_step_interval_;
    //                 float scale = cov_scale_initial_ * std::pow(cov_step_factor_, 
    //                                                              static_cast<float>(num_steps));
    //                 return std::max(cov_scale_final_, scale);
    //             }
                
    //         case CovarianceSchedule::ADAPTIVE:
    //             // Adaptive: reduce if cost improved, else keep or increase slightly
    //             {
    //                 if (iteration <= 1) {
    //                     return cov_scale_current_;
    //                 }
                    
    //                 float improvement = (prev_cost - curr_cost) / (std::abs(prev_cost) + 1e-6f);
                    
    //                 if (improvement > cov_adaptive_threshold_) {
    //                     // Good improvement - reduce covariance to focus search
    //                     cov_scale_current_ *= cov_decay_rate_;
    //                 } else if (improvement < 0) {
    //                     // Cost increased - slightly increase covariance to explore more
    //                     cov_scale_current_ *= (1.0f + 0.1f * (1.0f - cov_decay_rate_));
    //                 }
    //                 // else: small improvement - keep current scale
                    
    //                 // Clamp to bounds
    //                 cov_scale_current_ = std::max(cov_scale_final_, 
    //                                               std::min(cov_scale_initial_, cov_scale_current_));
    //                 return cov_scale_current_;
    //             }
                
    //         default:
    //             return cov_scale_initial_;
    //     }
    // }

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

        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        const size_t D = num_dimensions_;

        float best_cost = std::numeric_limits<float>::infinity();
        size_t best_iteration = 0;
        float prev_cost = std::numeric_limits<float>::infinity();

        // Reset history
        trajectory_history_.clear();
        covariance_scale_history_.clear();
        mean_cost_history_.clear(); // Added history reset

        gamma_ = pce_config_->gamma;

        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            Eigen::MatrixXf Y_k = trajectoryToMatrix();

            // --- Temperature Schedule Calculation ---
            float progress = static_cast<float>(iteration - 1) / std::max(1.0f, static_cast<float>(num_iterations_ - 1));
            float current_temp = pce_config_->temperature * std::pow((pce_config_->temperature_final / pce_config_->temperature), progress);

            // 2. Sampling and Evaluation
            std::vector<Eigen::MatrixXf> epsilon_samples = sampleScaledNoiseMatrices(M, cov_scale_current_);
            std::vector<Trajectory> sample_trajectories;
            sample_trajectories.reserve(M);
            
            for (size_t m = 0; m < M; ++m) {
                sample_trajectories.push_back(createPerturbedTrajectory(Y_k, epsilon_samples[m]));
            }
            
            std::vector<float> sample_collisions = task_->computeCollisionCostSimple(sample_trajectories);

            // 3. Elite Selection
            std::vector<size_t> indices(M);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return sample_collisions[a] < sample_collisions[b];
            });

            const size_t num_elites = std::max(static_cast<size_t>(1), static_cast<size_t>(M * elite_ratio_));
            
            // 4. Weight Calculation (using scheduled temperature)
            Eigen::VectorXf weights = Eigen::VectorXf::Zero(M);
            float max_exponent = -std::numeric_limits<float>::infinity();
            
            for (size_t i = 0; i < num_elites; ++i) {
                size_t m = indices[i];
                float reg_term = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    reg_term += epsilon_samples[m].row(d).dot(R_matrix_ * Y_k.row(d).transpose());
                }

                float exponent = (-gamma_ * (sample_collisions[m] + reg_term)) / current_temp;
                weights(m) = exponent;
                if (exponent > max_exponent) max_exponent = exponent;
            }

            // 5. Normalized Softmax
            float weight_sum = 0.0f;
            for (size_t i = 0; i < num_elites; ++i) {
                size_t m = indices[i];
                weights(m) = std::exp(weights(m) - max_exponent);
                weight_sum += weights(m);
            }
            weights /= (weight_sum + 1e-10f);

            // 6. EMA Update Rule
            MatrixXf Y_weighted = MatrixXf::Zero(D, N);
            for (size_t i = 0; i < num_elites; ++i) {
                size_t m = indices[i];
                Y_weighted += weights(m) * (Y_k + epsilon_samples[m]);
            }

            MatrixXf Y_new = (1.0f - ema_alpha_) * Y_k + ema_alpha_ * Y_weighted;
            updateTrajectoryFromMatrix(Y_new);

            // 7. Cleanup & Constraints
            current_trajectory_.nodes[0].position = start_node_.position;
            current_trajectory_.nodes[N - 1].position = goal_node_.position;
            task_->filterTrajectory(current_trajectory_, iteration);

            // 8. Logging and Convergence
            float current_collision = task_->computeCollisionCostSimple(current_trajectory_);
            float current_smoothness = computeSmoothnessCost(current_trajectory_);
            float current_total_cost = current_collision + current_smoothness;

            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);
            mean_cost_history_.push_back(current_total_cost); // Record history

            if (current_total_cost < best_cost) {
                best_cost = current_total_cost;
                best_iteration = iteration;
            }

            // --- Every 10 steps ---
            if (iteration == 1 || iteration % 10 == 0 || iteration == num_iterations_) {
                logf("Iter %zu: Cost=%.3f, Temp=%.4f, σ=%.4f", 
                    iteration, current_total_cost, current_temp, cov_scale_current_);
            }

            cov_scale_current_ = computeCovarianceScale(iteration + 1, prev_cost, current_total_cost);
            if (std::abs(prev_cost - current_total_cost) < convergence_threshold_) break;
            prev_cost = current_total_cost;
            
            task_->postIteration(iteration, current_total_cost, current_trajectory_);
        }

        current_trajectory_ = trajectory_history_[best_iteration];
        task_->done(true, num_iterations_, best_cost, current_trajectory_);
        return true;
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
        if (!pce_config_) return;

        log("--- PCEM Planner Parameters ---");
        log("  Algorithm:            Proximal Cross-Entropy Method");
        logf("  Number of samples:    %zu", num_samples_);
        logf("  Number of iterations: %zu", num_iterations_);
        
        // Updated to reflect the new scheduling logic
        logf("  Initial Temp:         %.4f", pce_config_->temperature);
        logf("  Final Temp:           %.4f", pce_config_->temperature_final);
        
        logf("  Initial eta:          %.4f", eta_);
        logf("  Fixed gamma:          %.4f", gamma_);
        logf("  Convergence threshold: %.6f", convergence_threshold_);
        log("");
        log("--- Covariance Scheduling ---");
        
        // Accessing getScheduleName() through the shared_ptr
        logf("  Schedule:             %s", pce_config_->getScheduleName().c_str());
        logf("  Initial scale:        %.4f", cov_scale_initial_);
        logf("  Final scale:          %.4f", cov_scale_final_);
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
    float elite_ratio_ = 0.1f;
    
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
    std::vector<float> mean_cost_history_;
    
    std::mt19937 random_engine_;
};