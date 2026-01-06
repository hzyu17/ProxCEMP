/**
 * @file PCEMotionPlanner_STOMP_Inspired.h
 * @brief Optimized PCEM planner with STOMP-inspired efficiency patterns
 * 
 * Key patterns borrowed from STOMP:
 * 1. Rollout struct grouping all per-sample data
 * 2. Rollout reuse from previous iterations (elite preservation)
 * 3. Optimized trajectory stored as a rollout
 * 4. Pre-allocation of all rollouts at initialization
 * 5. Importance weighting for samples
 * 6. Modular cost computation functions
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
#include <numeric>

// Forward declaration
struct PCEConfig;

/**
 * @brief Rollout structure (STOMP-inspired)
 * 
 * Groups all per-sample data together for:
 * - Cache locality during iteration
 * - Easy reuse across iterations
 * - Clean memory management
 */
struct PCERollout {
    // Noise added to mean trajectory
    Eigen::MatrixXf noise;
    
    // Full noisy parameters (mean + noise)
    Eigen::MatrixXf parameters_noise;
    
    // Trajectory representation for task interface
    Trajectory trajectory;
    
    // Cost components
    float state_cost = 0.0f;      // Collision/task cost
    float control_cost = 0.0f;    // Smoothness cost
    float total_cost = 0.0f;      // Combined cost
    
    // Probability/weight for this rollout
    float probability = 0.0f;
    
    // Importance weight (for reused rollouts)
    float importance_weight = 1.0f;
    
    // Is this rollout valid (collision-free)?
    bool is_valid = false;
    
    /**
     * @brief Initialize rollout dimensions
     */
    void resize(size_t num_dimensions, size_t num_timesteps) {
        noise.resize(num_dimensions, num_timesteps);
        noise.setZero();
        parameters_noise.resize(num_dimensions, num_timesteps);
        parameters_noise.setZero();
    }
    
    /**
     * @brief Reset costs and weights for new iteration
     */
    void resetCosts() {
        state_cost = 0.0f;
        control_cost = 0.0f;
        total_cost = 0.0f;
        probability = 0.0f;
    }
};


/**
 * @brief Configuration for PCE planner with STOMP-inspired rollout reuse
 */
struct PCEConfig : public MotionPlannerConfig {
    // === Core Algorithm Parameters ===
    size_t num_samples = 3000;
    size_t num_iterations = 10;
    float eta = 1.0f;
    float temperature = 1.5f;
    float temperature_final = 0.1f;
    float convergence_threshold = 0.01f;
    float collision_clearance = 0.1f;
    float collision_threshold = 0.1f;
    float gamma = 0.5f;
    float elite_ratio = 0.1f;
    float ema_alpha = 0.5f;
    
    // === Rollout Reuse Parameters (STOMP-inspired) ===
    bool enable_rollout_reuse = true;           // Enable/disable rollout reuse feature
    size_t max_rollouts = 0;                    // Max rollouts to store (0 = auto-compute)
    size_t num_rollouts_reuse = 0;              // How many to reuse (0 = auto-compute)
    float reuse_ratio = 0.5f;                   // Fraction of elites to reuse (0.0-1.0)
    float exponentiated_cost_sensitivity = 10.0f;  // 'h' parameter for selection weighting
    float importance_weight_decay = 0.9f;       // Decay factor for reused rollout weights
    float importance_weight_min = 0.1f;         // Minimum importance weight before discard

    [[nodiscard]] constexpr float computeGamma() const noexcept {
        return eta / (eta + 1.0f);
    }
    
    /**
     * @brief Compute derived rollout reuse parameters
     */
    void computeDerivedParameters() {
        gamma = computeGamma();
        
        if (enable_rollout_reuse) {
            size_t num_elites = static_cast<size_t>(num_samples * elite_ratio);
            
            // Auto-compute num_rollouts_reuse if not set
            if (num_rollouts_reuse == 0) {
                num_rollouts_reuse = static_cast<size_t>(num_elites * reuse_ratio);
            }
            
            // Auto-compute max_rollouts if not set
            if (max_rollouts == 0) {
                max_rollouts = num_samples + num_rollouts_reuse + 1;  // +1 for optimized
            }
        } else {
            num_rollouts_reuse = 0;
            max_rollouts = num_samples + 1;  // Just samples + optimized
        }
    }

    bool loadFromYAML(const YAML::Node& config) override {
        if (!MotionPlannerConfig::loadFromYAML(config)) return false;
        
        try {
            if (auto pce = config["pce_planner"]) {
                // Helper lambda for clean YAML loading
                auto load = [&pce]<typename T>(const char* key, T& out) {
                    if (auto node = pce[key]) out = node.template as<T>();
                };
                
                // Core parameters
                load("num_samples", num_samples);
                load("num_iterations", num_iterations);
                load("eta", eta);
                load("temperature", temperature);
                load("temperature_final", temperature_final);
                load("convergence_threshold", convergence_threshold);
                load("elite_ratio", elite_ratio);
                load("ema_alpha", ema_alpha);
                load("collision_clearance", collision_clearance);
                load("collision_threshold", collision_threshold);
                
                // Rollout reuse parameters
                load("enable_rollout_reuse", enable_rollout_reuse);
                load("max_rollouts", max_rollouts);
                load("num_rollouts_reuse", num_rollouts_reuse);
                load("reuse_ratio", reuse_ratio);
                load("exponentiated_cost_sensitivity", exponentiated_cost_sensitivity);
                load("importance_weight_decay", importance_weight_decay);
                load("importance_weight_min", importance_weight_min);
                
                // Compute derived values
                computeDerivedParameters();
            }
            return validate();
        } catch (const std::exception& e) {
            std::cerr << "Error loading PCE config: " << e.what() << "\n";
            return false;
        }
    }
    
    bool validate() const override {
        if (!MotionPlannerConfig::validate()) return false;
        
        // Validate reuse parameters
        if (reuse_ratio < 0.0f || reuse_ratio > 1.0f) {
            std::cerr << "Error: reuse_ratio must be in [0, 1]\n";
            return false;
        }
        if (importance_weight_decay <= 0.0f || importance_weight_decay > 1.0f) {
            std::cerr << "Error: importance_weight_decay must be in (0, 1]\n";
            return false;
        }
        if (exponentiated_cost_sensitivity <= 0.0f) {
            std::cerr << "Error: exponentiated_cost_sensitivity must be > 0\n";
            return false;
        }
        return true;
    }

    void print() const override {
        MotionPlannerConfig::print();
        std::cout << "--- PCE Hyperparameters (STOMP-Inspired) ---\n"
                  << "  Num Samples:     " << num_samples << "\n"
                  << "  Num Iterations:  " << num_iterations << "\n"
                  << "  Temperature:     " << temperature << " -> " << temperature_final << "\n"
                  << "  Eta/Gamma:       " << eta << " / " << gamma << "\n"
                  << "  Elite Ratio:     " << elite_ratio << "\n"
                  << "  EMA Alpha:       " << ema_alpha << "\n"
                  << "--- Rollout Reuse ---\n"
                  << "  Enabled:         " << (enable_rollout_reuse ? "yes" : "no") << "\n";
        if (enable_rollout_reuse) {
            std::cout << "  Max Rollouts:    " << max_rollouts << "\n"
                      << "  Reuse Count:     " << num_rollouts_reuse << " (" << (reuse_ratio * 100) << "% of elites)\n"
                      << "  Cost Sensitivity: " << exponentiated_cost_sensitivity << "\n"
                      << "  Weight Decay:    " << importance_weight_decay << "\n"
                      << "  Weight Min:      " << importance_weight_min << "\n";
        }
    }
};


/**
 * @brief STOMP-inspired Proximal Cross-Entropy Motion Planner
 */
class ProximalCrossEntropyMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    
    explicit ProximalCrossEntropyMotionPlanner(pce::TaskPtr task = nullptr)
        : task_(std::move(task))
    {
        std::random_device rd;
        seedRandomEngine(rd());
    }

    void setTask(pce::TaskPtr task) { task_ = std::move(task); }
    [[nodiscard]] pce::TaskPtr getTask() const { return task_; }

    bool initialize(const PCEConfig& config) {
        try {
            pce_config_ = std::make_shared<PCEConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to create config: " << e.what() << '\n';
            return false;
        }
        
        // Store core parameters
        num_samples_ = config.num_samples;
        num_iterations_ = config.num_iterations;
        temperature_ = config.temperature;
        eta_ = config.eta;
        gamma_ = config.gamma;
        convergence_threshold_ = config.convergence_threshold;
        elite_ratio_ = config.elite_ratio;
        ema_alpha_ = config.ema_alpha;
        
        // Store rollout reuse parameters (STOMP-inspired)
        enable_rollout_reuse_ = config.enable_rollout_reuse;
        max_rollouts_ = config.max_rollouts;
        num_rollouts_reuse_ = config.num_rollouts_reuse;
        reuse_ratio_ = config.reuse_ratio;
        exponentiated_cost_sensitivity_ = config.exponentiated_cost_sensitivity;
        importance_weight_decay_ = config.importance_weight_decay;
        importance_weight_min_ = config.importance_weight_min;
        
        // Covariance scheduling
        cov_schedule_ = config.cov_schedule;
        cov_scale_initial_ = config.cov_scale_initial;
        cov_scale_final_ = config.cov_scale_final;
        cov_decay_rate_ = config.cov_decay_rate;
        cov_scale_current_ = cov_scale_initial_;
        
        // Initialize base class first (sets dimensions)
        if (!MotionPlanner::initialize(config)) {
            return false;
        }
        
        // Now allocate rollouts (STOMP pattern)
        if (!resetVariables()) {
            return false;
        }
        
        return true;
    }

    [[nodiscard]] std::string getPlannerName() const override { return "PCEM-STOMP"; }

    void seedRandomEngine(unsigned int seed) { random_engine_.seed(seed); }

    [[nodiscard]] float computeStateCost(const Trajectory& trajectory) const override {
        if (!task_) {
            std::cerr << "Error: No task set\n";
            return std::numeric_limits<float>::infinity();
        }
        return task_->computeStateCost(trajectory);
    }

    /**
     * @brief Main optimization loop (STOMP-inspired structure)
     */
    bool optimize() override {
        if (!task_) {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }

        current_iteration_ = 1;
        current_lowest_cost_ = std::numeric_limits<float>::infinity();
        float prev_cost = std::numeric_limits<float>::infinity();
        
        // Compute initial trajectory cost
        if (!computeOptimizedCost()) {
            std::cerr << "Failed to compute initial cost\n";
            return false;
        }

        trajectory_history_.clear();
        trajectory_history_.reserve(num_iterations_);
        covariance_scale_history_.clear();
        
        size_t best_iteration = 0;

        while (current_iteration_ <= num_iterations_ && runSingleIteration()) {
            float current_cost = parameters_total_cost_;
            
            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);
            
            if (current_cost < current_lowest_cost_) {
                current_lowest_cost_ = current_cost;
                best_iteration = current_iteration_;
            }
            
            // Logging
            if (current_iteration_ == 1 || current_iteration_ % 10 == 0 || 
                current_iteration_ == num_iterations_) {
                logf("Iter %zu - Cost: %.2f (Active rollouts: %zu)", 
                     current_iteration_, current_cost, num_active_rollouts_);
            }
            
            // Convergence check
            if (std::abs(prev_cost - current_cost) < convergence_threshold_) {
                log("Converged!");
                break;
            }
            
            prev_cost = current_cost;
            cov_scale_current_ = computeCovarianceScale(current_iteration_ + 1, prev_cost, current_cost);
            current_iteration_++;
        }

        // Use best trajectory
        if (best_iteration > 0 && best_iteration <= trajectory_history_.size()) {
            current_trajectory_ = trajectory_history_[best_iteration - 1];
        }
        
        task_->done(true, num_iterations_, current_lowest_cost_, current_trajectory_);
        return true;
    }

    [[nodiscard]] const std::vector<float>& getCovarianceScaleHistory() const {
        return covariance_scale_history_;
    }

protected:
    void initializeTask() override {
        if (!task_ || !pce_config_) return;
        task_->initialize(num_dimensions_, start_node_, goal_node_, num_nodes_, total_time_);
    }

    void logPlannerSpecificConfig() override {
        if (!pce_config_) return;
        log("--- PCEM-STOMP Parameters ---");
        logf("  Samples: %zu, Iterations: %zu", num_samples_, num_iterations_);
        logf("  Temperature: %.2f -> %.2f", pce_config_->temperature, pce_config_->temperature_final);
        logf("  Elite ratio: %.2f, EMA alpha: %.2f", elite_ratio_, ema_alpha_);
        log("--- Rollout Reuse ---");
        logf("  Enabled: %s", enable_rollout_reuse_ ? "yes" : "no");
        if (enable_rollout_reuse_) {
            logf("  Max rollouts: %zu", max_rollouts_);
            logf("  Reuse count: %zu (%.0f%% of elites)", num_rollouts_reuse_, reuse_ratio_ * 100);
            logf("  Cost sensitivity (h): %.1f", exponentiated_cost_sensitivity_);
            logf("  Weight decay: %.2f, min: %.2f", importance_weight_decay_, importance_weight_min_);
        }
    }

private:
    /**
     * @brief Reset and pre-allocate all variables (STOMP pattern)
     * 
     * This is the key efficiency pattern from STOMP - allocate once at init
     */
    bool resetVariables() {
        proceed_ = true;
        parameters_total_cost_ = 0;
        num_active_rollouts_ = 0;
        
        const size_t D = num_dimensions_;
        const size_t N = num_nodes_;
        
        // Pre-allocate all rollouts (STOMP pattern)
        rollouts_.resize(max_rollouts_);
        reused_rollouts_.resize(max_rollouts_);
        
        for (size_t r = 0; r < max_rollouts_; ++r) {
            rollouts_[r].resize(D, N);
            rollouts_[r].trajectory = current_trajectory_;  // Copy structure
            reused_rollouts_[r].resize(D, N);
            reused_rollouts_[r].trajectory = current_trajectory_;
        }
        
        // Pre-allocate update matrix
        parameters_updates_.resize(D, N);
        parameters_updates_.setZero();
        
        // Pre-allocate optimized parameters matrix
        parameters_optimized_.resize(D, N);
        for (size_t i = 0; i < N; ++i) {
            parameters_optimized_.col(i) = current_trajectory_.nodes[i].position;
        }
        
        // Pre-allocate cost sorting structure (STOMP pattern)
        rollout_cost_sorter_.reserve(max_rollouts_);
        
        // Pre-allocate index vector for elite selection
        indices_.resize(max_rollouts_);
        
        return true;
    }

    /**
     * @brief Run single optimization iteration (STOMP structure)
     */
    bool runSingleIteration() {
        if (!proceed_) return false;
        
        bool success = generateNoisyRollouts() &&
                       computeRolloutsCosts() &&
                       computeProbabilities() &&
                       updateParameters() &&
                       computeOptimizedCost();
        
        // Notify task
        task_->postIteration(current_iteration_, parameters_total_cost_, current_trajectory_);
        
        return success;
    }

    /**
     * @brief Generate noisy rollouts with optional reuse (STOMP pattern)
     */
    bool generateNoisyRollouts() {
        const size_t D = num_dimensions_;
        const size_t N = num_nodes_;
        
        size_t rollouts_generate = num_samples_;
        size_t rollouts_reuse = 0;
        
        // Calculate rollouts to reuse from previous iteration (if enabled)
        if (enable_rollout_reuse_ && current_iteration_ > 1) {
            size_t rollouts_stored = (num_active_rollouts_ > 1) ? num_active_rollouts_ - 1 : 0;
            size_t rollouts_total = rollouts_generate + rollouts_stored + 1;
            rollouts_reuse = (rollouts_total <= max_rollouts_) ? 
                             rollouts_stored : 
                             max_rollouts_ - (rollouts_generate + 1);
            
            rollouts_reuse = std::min(rollouts_reuse, num_rollouts_reuse_);
            
            // Select best rollouts to reuse (STOMP pattern)
            if (rollouts_reuse > 0 && rollouts_stored > 0) {
                selectRolloutsForReuse(rollouts_stored, rollouts_reuse, rollouts_generate);
            }
        }
        
        // Add optimized trajectory as the last rollout (STOMP pattern)
        size_t opt_idx = rollouts_generate + rollouts_reuse;
        rollouts_[opt_idx].parameters_noise = parameters_optimized_;
        rollouts_[opt_idx].noise.setZero();
        rollouts_[opt_idx].state_cost = parameters_state_cost_;
        rollouts_[opt_idx].control_cost = parameters_control_cost_;
        rollouts_[opt_idx].total_cost = parameters_total_cost_;
        rollouts_[opt_idx].importance_weight = 1.0f;
        
        // Update trajectory for optimized rollout
        for (size_t i = 0; i < N; ++i) {
            rollouts_[opt_idx].trajectory.nodes[i].position = parameters_optimized_.col(i);
        }
        
        // Generate new noisy rollouts
        for (size_t r = 0; r < rollouts_generate; ++r) {
            if (!proceed_) return false;
            
            // Sample noise using structured covariance
            sampleNoiseInPlace(rollouts_[r].noise, cov_scale_current_);
            
            // Compute noisy parameters
            rollouts_[r].parameters_noise.noalias() = parameters_optimized_ + rollouts_[r].noise;
            
            // Update trajectory
            for (size_t i = 0; i < N; ++i) {
                rollouts_[r].trajectory.nodes[i].position = rollouts_[r].parameters_noise.col(i);
            }
            
            rollouts_[r].importance_weight = 1.0f;
            rollouts_[r].resetCosts();
        }
        
        num_active_rollouts_ = rollouts_generate + rollouts_reuse + 1;
        return true;
    }

    /**
     * @brief Select best rollouts from previous iteration for reuse (STOMP pattern)
     * 
     * Uses exponential cost weighting and importance weight decay.
     * Rollouts below importance_weight_min_ are discarded.
     */
    void selectRolloutsForReuse(size_t rollouts_stored, size_t rollouts_reuse, 
                                 size_t rollouts_generate) {
        // Find min/max cost for normalization
        float min_cost = std::numeric_limits<float>::max();
        float max_cost = std::numeric_limits<float>::lowest();
        
        for (size_t r = 0; r < rollouts_stored; ++r) {
            // Skip rollouts with weight below minimum
            if (rollouts_[r].importance_weight < importance_weight_min_) {
                continue;
            }
            float c = rollouts_[r].total_cost;
            min_cost = std::min(min_cost, c);
            max_cost = std::max(max_cost, c);
        }
        
        float cost_denom = std::max(max_cost - min_cost, 1e-8f);
        
        // Compute weighted probabilities and sort (STOMP pattern)
        rollout_cost_sorter_.clear();
        for (size_t r = 0; r < rollouts_stored; ++r) {
            // Skip rollouts with weight below minimum
            if (rollouts_[r].importance_weight < importance_weight_min_) {
                continue;
            }
            
            // Update noise relative to new optimized trajectory
            rollouts_[r].noise = rollouts_[r].parameters_noise - parameters_optimized_;
            
            float cost_prob = std::exp(-exponentiated_cost_sensitivity_ * 
                                       (rollouts_[r].total_cost - min_cost) / cost_denom);
            float weighted_prob = cost_prob * rollouts_[r].importance_weight;
            
            // Negative for descending sort (best first)
            rollout_cost_sorter_.emplace_back(-weighted_prob, r);
        }
        
        // Adjust rollouts_reuse if we filtered some out
        size_t available_rollouts = rollout_cost_sorter_.size();
        rollouts_reuse = std::min(rollouts_reuse, available_rollouts);
        
        if (rollouts_reuse == 0) return;
        
        // Partial sort - only need top rollouts_reuse (efficiency!)
        if (rollouts_reuse < available_rollouts) {
            std::nth_element(rollout_cost_sorter_.begin(),
                           rollout_cost_sorter_.begin() + rollouts_reuse,
                           rollout_cost_sorter_.end());
        }
        std::sort(rollout_cost_sorter_.begin(),
                 rollout_cost_sorter_.begin() + rollouts_reuse);
        
        // Copy best rollouts to reused buffer with weight decay
        for (size_t r = 0; r < rollouts_reuse; ++r) {
            size_t reuse_idx = rollout_cost_sorter_[r].second;
            reused_rollouts_[r] = rollouts_[reuse_idx];
            
            // Apply importance weight decay (STOMP-inspired)
            reused_rollouts_[r].importance_weight *= importance_weight_decay_;
        }
        
        // Copy back to main rollouts array (after newly generated ones)
        for (size_t r = 0; r < rollouts_reuse; ++r) {
            rollouts_[rollouts_generate + r] = reused_rollouts_[r];
        }
    }

    /**
     * @brief Sample noise in-place with structured covariance
     */
    void sampleNoiseInPlace(MatrixXf& noise, float scale) {
        const size_t D = noise.rows();
        const size_t N = noise.cols();
        
        for (size_t d = 0; d < D; ++d) {
            auto sampled = sampleSmoothnessNoise(N, random_engine_);
            for (size_t i = 0; i < N; ++i) {
                noise(d, i) = sampled[i] * scale;
            }
        }
    }

    /**
     * @brief Compute costs for all rollouts (STOMP pattern - batched)
     */
    bool computeRolloutsCosts() {
        // Batch compute state costs
        std::vector<Trajectory> trajectories;
        trajectories.reserve(num_active_rollouts_);
        
        for (size_t r = 0; r < num_active_rollouts_; ++r) {
            trajectories.push_back(rollouts_[r].trajectory);
        }
        
        // Batch cost computation
        std::vector<float> state_costs = task_->computeStateCostSimple(trajectories);
        
        // Temperature scaling
        float progress = static_cast<float>(current_iteration_ - 1) / 
                        std::max(1.0f, static_cast<float>(num_iterations_ - 1));
        float current_temp = pce_config_->temperature * 
                            std::pow(pce_config_->temperature_final / pce_config_->temperature, progress);
        float inv_temp = 1.0f / current_temp;
        
        // Assign costs to rollouts
        for (size_t r = 0; r < num_active_rollouts_; ++r) {
            rollouts_[r].state_cost = state_costs[r] * inv_temp;
            rollouts_[r].control_cost = computeRolloutControlCost(rollouts_[r]);
            rollouts_[r].total_cost = rollouts_[r].state_cost + rollouts_[r].control_cost;
        }
        
        return true;
    }

    /**
     * @brief Compute control cost for a single rollout
     */
    float computeRolloutControlCost(const PCERollout& rollout) const {
        return computeSmoothnessCost(rollout.trajectory);
    }

    /**
     * @brief Compute probabilities/weights for all rollouts (STOMP-inspired)
     */
    bool computeProbabilities() {
        const size_t num_elites = std::max(size_t{1}, 
                                           static_cast<size_t>(num_active_rollouts_ * elite_ratio_));
        
        // Sort by total cost to find elites
        std::iota(indices_.begin(), indices_.begin() + num_active_rollouts_, size_t{0});
        
        // Use nth_element for O(n) elite selection
        std::nth_element(indices_.begin(), 
                        indices_.begin() + num_elites,
                        indices_.begin() + num_active_rollouts_,
                        [this](size_t a, size_t b) {
                            return rollouts_[a].total_cost < rollouts_[b].total_cost;
                        });
        
        // Only sort elites
        std::sort(indices_.begin(), indices_.begin() + num_elites,
                 [this](size_t a, size_t b) {
                     return rollouts_[a].total_cost < rollouts_[b].total_cost;
                 });
        
        // Compute elite statistics for normalization
        float mean_cost = 0.0f;
        for (size_t i = 0; i < num_elites; ++i) {
            mean_cost += rollouts_[indices_[i]].total_cost;
        }
        mean_cost /= static_cast<float>(num_elites);
        
        float var_cost = 0.0f;
        for (size_t i = 0; i < num_elites; ++i) {
            float diff = rollouts_[indices_[i]].total_cost - mean_cost;
            var_cost += diff * diff;
        }
        float std_cost = std::sqrt(var_cost / static_cast<float>(num_elites) + 1e-8f);
        float inv_std = 1.0f / std_cost;
        
        // Compute weights with log-sum-exp stability
        float max_exponent = -std::numeric_limits<float>::infinity();
        
        for (size_t i = 0; i < num_elites; ++i) {
            size_t r = indices_[i];
            float normalized_cost = (rollouts_[r].total_cost - mean_cost) * inv_std;
            float exponent = -gamma_ * normalized_cost;
            rollouts_[r].probability = exponent;  // Store exponent temporarily
            max_exponent = std::max(max_exponent, exponent);
        }
        
        // Zero out non-elite probabilities
        for (size_t i = num_elites; i < num_active_rollouts_; ++i) {
            rollouts_[indices_[i]].probability = 0.0f;
        }
        
        // Apply softmax with stability
        float prob_sum = 0.0f;
        for (size_t i = 0; i < num_elites; ++i) {
            size_t r = indices_[i];
            rollouts_[r].probability = rollouts_[r].importance_weight * 
                                       std::exp(rollouts_[r].probability - max_exponent);
            prob_sum += rollouts_[r].probability;
        }
        
        // Normalize
        float inv_sum = 1.0f / (prob_sum + 1e-10f);
        for (size_t i = 0; i < num_elites; ++i) {
            rollouts_[indices_[i]].probability *= inv_sum;
        }
        
        return true;
    }

    /**
     * @brief Update parameters using weighted combination (STOMP pattern)
     */
    bool updateParameters() {
        const size_t D = num_dimensions_;
        const size_t N = num_nodes_;
        const size_t num_elites = std::max(size_t{1}, 
                                           static_cast<size_t>(num_active_rollouts_ * elite_ratio_));
        
        // Compute weighted update (STOMP pattern: convex combination)
        MatrixXf weighted_params = MatrixXf::Zero(D, N);
        
        for (size_t i = 0; i < num_elites; ++i) {
            size_t r = indices_[i];
            weighted_params.noalias() += rollouts_[r].probability * rollouts_[r].parameters_noise;
        }
        
        // EMA blend
        parameters_updates_.noalias() = weighted_params - parameters_optimized_;
        parameters_optimized_.noalias() = (1.0f - ema_alpha_) * parameters_optimized_ + 
                                          ema_alpha_ * weighted_params;
        
        // Apply boundary constraints
        parameters_optimized_.col(0) = start_node_.position;
        parameters_optimized_.col(N - 1) = goal_node_.position;
        
        // Update current trajectory
        for (size_t i = 0; i < N; ++i) {
            current_trajectory_.nodes[i].position = parameters_optimized_.col(i);
        }
        
        // Apply task filter
        task_->filterTrajectory(current_trajectory_, current_iteration_);
        
        // Sync back to parameters
        for (size_t i = 0; i < N; ++i) {
            parameters_optimized_.col(i) = current_trajectory_.nodes[i].position;
        }
        
        return true;
    }

    /**
     * @brief Compute cost of optimized trajectory (STOMP pattern)
     */
    bool computeOptimizedCost() {
        parameters_state_cost_ = task_->computeStateCostSimple(current_trajectory_);
        parameters_control_cost_ = computeSmoothnessCost(current_trajectory_);
        parameters_total_cost_ = parameters_state_cost_ + parameters_control_cost_;
        
        return true;
    }

    // Task and configuration
    pce::TaskPtr task_;
    std::shared_ptr<PCEConfig> pce_config_;
    
    // Core algorithm parameters
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float temperature_ = 1.5f;
    float eta_ = 1.0f;
    float gamma_ = 0.5f;
    float convergence_threshold_ = 0.01f;
    float ema_alpha_ = 0.5f;
    float elite_ratio_ = 0.1f;
    
    // Rollout reuse parameters (STOMP-inspired)
    bool enable_rollout_reuse_ = true;
    size_t max_rollouts_ = 0;
    size_t num_rollouts_reuse_ = 0;
    float reuse_ratio_ = 0.5f;
    float exponentiated_cost_sensitivity_ = 10.0f;
    float importance_weight_decay_ = 0.9f;
    float importance_weight_min_ = 0.1f;
    
    // Covariance scheduling
    float cov_scale_current_ = 1.0f;
    
    // Rollout storage (STOMP pattern - pre-allocated)
    std::vector<PCERollout> rollouts_;
    std::vector<PCERollout> reused_rollouts_;
    size_t num_active_rollouts_ = 0;
    
    // Optimization state
    MatrixXf parameters_optimized_;
    MatrixXf parameters_updates_;
    float parameters_state_cost_ = 0.0f;
    float parameters_control_cost_ = 0.0f;
    float parameters_total_cost_ = 0.0f;
    float current_lowest_cost_ = 0.0f;
    size_t current_iteration_ = 0;
    bool proceed_ = true;
    
    // Sorting helper (STOMP pattern)
    std::vector<std::pair<float, size_t>> rollout_cost_sorter_;
    std::vector<size_t> indices_;
    
    // History
    std::vector<float> covariance_scale_history_;
    
    std::mt19937 random_engine_;
};