/**
 * @file PCEMotionPlanner_Lightweight.h
 * @brief Lightweight PCEM with optional rollout reuse and early termination
 * 
 * Key features:
 * 1. NO Trajectory storage in rollouts - just matrices (fast)
 * 2. Create trajectories on-demand for batch evaluation
 * 3. Rollout reuse is optional and lightweight
 * 4. Early termination when valid solution found (STOMP-inspired)
 * 
 * Task Interface Requirements:
 * - computeStateCostSimple(trajectory) -> float
 * - computeStateCostSimple(vector<trajectory>) -> vector<float>
 * - isTrajectoryValid(trajectory) -> bool (optional, uses cost threshold if not available)
 * - filterTrajectory(trajectory, iteration) -> void
 * - postIteration(iteration, cost, trajectory) -> void
 * - done(valid, iterations, cost, trajectory) -> void
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
#include <chrono>

/**
 * @brief Lightweight rollout data - NO Trajectory storage
 */
struct LightweightRollout {
    float total_cost = 0.0f;
    float importance_weight = 1.0f;
    size_t noise_index = 0;  // Index into noise buffer (avoids copying matrices)
};


/**
 * @brief Configuration for PCE planner
 */
struct PCEConfig : public MotionPlannerConfig {
    // Core parameters
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
    
    // Early termination parameters (STOMP-inspired)
    size_t num_iterations_after_valid = 10;  // Continue refining after valid solution found
    bool enable_early_termination = true;    // Enable/disable early termination
    float validity_cost_threshold = 1e-2f;   // Cost below which trajectory is "valid"
    
    // Rollout reuse parameters
    bool enable_rollout_reuse = true;
    size_t num_rollouts_reuse = 0;
    float reuse_ratio = 0.5f;
    float exponentiated_cost_sensitivity = 10.0f;
    float importance_weight_decay = 0.9f;
    float importance_weight_min = 0.1f;

    [[nodiscard]] float computeGamma() const noexcept {
        return eta / (eta + 1.0f);
    }

    bool loadFromYAML(const YAML::Node& config) override {
        if (!MotionPlannerConfig::loadFromYAML(config)) return false;
        
        try {
            if (auto pce = config["pce_planner"]) {
                auto load = [&pce]<typename T>(const char* key, T& out) {
                    if (auto node = pce[key]) out = node.template as<T>();
                };
                
                load("num_samples", num_samples);
                load("num_iterations", num_iterations);
                load("eta", eta);
                gamma = computeGamma();
                
                load("temperature", temperature);
                load("temperature_final", temperature_final);
                load("convergence_threshold", convergence_threshold);
                load("elite_ratio", elite_ratio);
                load("ema_alpha", ema_alpha);
                
                // Early termination
                load("num_iterations_after_valid", num_iterations_after_valid);
                load("enable_early_termination", enable_early_termination);
                load("validity_cost_threshold", validity_cost_threshold);
                
                // Rollout reuse
                load("enable_rollout_reuse", enable_rollout_reuse);
                load("num_rollouts_reuse", num_rollouts_reuse);
                load("reuse_ratio", reuse_ratio);
                load("exponentiated_cost_sensitivity", exponentiated_cost_sensitivity);
                load("importance_weight_decay", importance_weight_decay);
                load("importance_weight_min", importance_weight_min);
                
                // Auto-compute reuse count
                if (enable_rollout_reuse && num_rollouts_reuse == 0) {
                    size_t num_elites = static_cast<size_t>(num_samples * elite_ratio);
                    num_rollouts_reuse = static_cast<size_t>(num_elites * reuse_ratio);
                }
            }
            return validate();
        } catch (const std::exception& e) {
            std::cerr << "Error loading PCE config: " << e.what() << "\n";
            return false;
        }
    }

    void print() const override {
        MotionPlannerConfig::print();
        std::cout << "--- PCE Lightweight Config ---\n"
                  << "  Samples: " << num_samples << ", Iterations: " << num_iterations << "\n"
                  << "  Early termination: " << (enable_early_termination ? "enabled" : "disabled");
        if (enable_early_termination) {
            std::cout << " (+" << num_iterations_after_valid << " after valid)";
        }
        std::cout << "\n  Reuse: " << (enable_rollout_reuse ? "enabled" : "disabled");
        if (enable_rollout_reuse) {
            std::cout << " (" << num_rollouts_reuse << " rollouts)";
        }
        std::cout << "\n";
    }
};


/**
 * @brief Lightweight Proximal Cross-Entropy Motion Planner
 * 
 * Optimized for speed - similar structure to NGD but with PCE update rule
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
        pce_config_ = std::make_shared<PCEConfig>(config);
        
        // Store parameters
        num_samples_ = config.num_samples;
        num_iterations_ = config.num_iterations;
        temperature_ = config.temperature;
        eta_ = config.eta;
        gamma_ = config.gamma;
        convergence_threshold_ = config.convergence_threshold;
        elite_ratio_ = config.elite_ratio;
        ema_alpha_ = config.ema_alpha;
        
        // Early termination (STOMP-inspired)
        enable_early_termination_ = config.enable_early_termination;
        num_iterations_after_valid_ = config.num_iterations_after_valid;
        validity_cost_threshold_ = config.validity_cost_threshold;
        
        // Rollout reuse
        enable_rollout_reuse_ = config.enable_rollout_reuse;
        num_rollouts_reuse_ = config.num_rollouts_reuse;
        exponentiated_cost_sensitivity_ = config.exponentiated_cost_sensitivity;
        importance_weight_decay_ = config.importance_weight_decay;
        importance_weight_min_ = config.importance_weight_min;
        
        // Covariance
        cov_schedule_ = config.cov_schedule;
        cov_scale_initial_ = config.cov_scale_initial;
        cov_scale_final_ = config.cov_scale_final;
        cov_decay_rate_ = config.cov_decay_rate;
        cov_scale_current_ = cov_scale_initial_;
        
        if (!MotionPlanner::initialize(config)) {
            return false;
        }
        
        // Pre-allocate buffers
        preAllocateBuffers();
        
        return true;
    }

    [[nodiscard]] std::string getPlannerName() const override { return "PCEM-Light"; }
    void seedRandomEngine(unsigned int seed) { random_engine_.seed(seed); }

    [[nodiscard]] float computeStateCost(const Trajectory& trajectory) const override {
        if (!task_) return std::numeric_limits<float>::infinity();
        return task_->computeStateCost(trajectory);
    }

    /**
     * @brief Main optimization - lightweight like NGD with early termination
     */
    bool optimize() override {
        // Defensive checks
        if (!task_) {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }
        
        if (!pce_config_) {
            std::cerr << "Error: PCE config not set!\n";
            return false;
        }
        
        if (current_trajectory_.nodes.empty()) {
            std::cerr << "Error: Current trajectory not initialized!\n";
            return false;
        }

        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        const size_t D = num_dimensions_;
        
        if (N == 0 || M == 0 || D == 0) {
            std::cerr << "Error: Invalid dimensions N=" << N << " M=" << M << " D=" << D << "\n";
            return false;
        }
        
        // Ensure buffers are allocated
        if (epsilon_samples_.empty() || epsilon_samples_.size() < M) {
            std::cerr << "Error: Buffers not properly allocated. Reallocating...\n";
            preAllocateBuffers();
        }
        
        const size_t num_elites = std::max(size_t{1}, static_cast<size_t>(M * elite_ratio_));

        // Precompute temperature schedule values
        const float temp_ratio = pce_config_->temperature_final / pce_config_->temperature;
        const float inv_max_iter = 1.0f / std::max(1.0f, static_cast<float>(num_iterations_ - 1));

        float best_cost = std::numeric_limits<float>::infinity();
        size_t best_iteration = 0;
        float prev_cost = std::numeric_limits<float>::infinity();

        trajectory_history_.clear();
        trajectory_history_.reserve(num_iterations_);
        covariance_scale_history_.clear();

        gamma_ = pce_config_->gamma;
        
        // Reuse buffer: stores (noise_matrix, cost, importance_weight)
        size_t num_reused = 0;
        
        // Early termination state (STOMP-inspired)
        size_t valid_iterations = 0;
        bool parameters_valid = false;
        bool parameters_valid_prev = false;
        
        log("Starting optimization loop...");
        
        // Timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Compute initial trajectory validity (safely)
        float initial_cost = task_->computeStateCostSimple(current_trajectory_);
        float initial_smoothness = computeSmoothnessCost(current_trajectory_);
        parameters_valid = (initial_cost < validity_cost_threshold_);
        parameters_valid_prev = parameters_valid;
        
        logf("Initial: Collision=%.4f, Smoothness=%.4f, Valid=%s", 
             initial_cost, initial_smoothness, parameters_valid ? "yes" : "no");
        log("-----------------------------------------------------------------------------------------");
        log("Iter     |       Cost |  Collision | Smoothness | Valid | Reused | Time (ms)");
        log("-----------------------------------------------------------------------------------------");

        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            
            // Current trajectory as matrix
            MatrixXf Y_k = trajectoryToMatrix();

            // Temperature schedule
            const float progress = static_cast<float>(iteration - 1) * inv_max_iter;
            const float current_temp = pce_config_->temperature * std::pow(temp_ratio, progress);
            const float inv_temp = 1.0f / current_temp;

            // === SAMPLING PHASE (like NGD) ===
            // Sample fresh noise matrices
            size_t num_fresh = M;
            if (enable_rollout_reuse_ && num_reused > 0) {
                num_fresh = M - std::min(num_reused, num_rollouts_reuse_);
            }
            
            // Sample new noise
            for (size_t m = 0; m < num_fresh; ++m) {
                if (m >= epsilon_samples_.size()) {
                    std::cerr << "Error: epsilon_samples_ index " << m << " out of bounds (size=" << epsilon_samples_.size() << ")\n";
                    return false;
                }
                sampleNoiseInPlace(epsilon_samples_[m], cov_scale_current_);
            }
            
            // If reusing, shift reused samples to end of fresh samples
            size_t actual_reused = 0;
            if (enable_rollout_reuse_ && num_reused > 0) {
                actual_reused = std::min(num_reused, num_rollouts_reuse_);
                for (size_t r = 0; r < actual_reused; ++r) {
                    epsilon_samples_[num_fresh + r] = reused_epsilon_[r];
                }
            }
            
            size_t total_samples = num_fresh + actual_reused;

            // === CREATE TRAJECTORIES AND EVALUATE (like NGD) ===
            sample_trajectories_.resize(total_samples);
            for (size_t m = 0; m < total_samples; ++m) {
                sample_trajectories_[m] = createPerturbedTrajectory(Y_k, epsilon_samples_[m]);
            }
            
            // Batch evaluate costs
            std::vector<float> batch_costs = task_->computeStateCostSimple(sample_trajectories_);
            
            // Copy to pre-allocated buffer
            for (size_t m = 0; m < std::min(batch_costs.size(), total_samples); ++m) {
                sample_costs_[m] = batch_costs[m];
            }
            
            // Apply temperature and importance weighting
            for (size_t m = 0; m < num_fresh; ++m) {
                sample_costs_[m] *= inv_temp;
            }
            for (size_t m = num_fresh; m < total_samples; ++m) {
                sample_costs_[m] *= inv_temp * reused_weights_[m - num_fresh];
            }
            
            // Ensure num_elites doesn't exceed total_samples
            size_t actual_num_elites = std::min(num_elites, total_samples);
            
            // === ELITE SELECTION (O(n) with nth_element) ===
            std::iota(indices_.begin(), indices_.begin() + total_samples, size_t{0});
            
            if (actual_num_elites > 0 && actual_num_elites < total_samples) {
                std::nth_element(indices_.begin(), 
                                indices_.begin() + actual_num_elites,
                                indices_.begin() + total_samples,
                                [this](size_t a, size_t b) {
                                    return sample_costs_[a] < sample_costs_[b];
                                });
            }

            // === COMPUTE WEIGHTS (softmax over elites) ===
            // Compute mean/std for normalization
            float mean_cost = 0.0f;
            for (size_t i = 0; i < actual_num_elites; ++i) {
                mean_cost += sample_costs_[indices_[i]];
            }
            mean_cost /= static_cast<float>(actual_num_elites);
            
            float var_cost = 0.0f;
            for (size_t i = 0; i < actual_num_elites; ++i) {
                float diff = sample_costs_[indices_[i]] - mean_cost;
                var_cost += diff * diff;
            }
            float std_cost = std::sqrt(var_cost / static_cast<float>(actual_num_elites) + 1e-8f);
            float inv_std = 1.0f / std_cost;

            // Compute softmax weights with log-sum-exp stability
            float max_exp = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < actual_num_elites; ++i) {
                float norm_cost = (sample_costs_[indices_[i]] - mean_cost) * inv_std;
                weights_[i] = -gamma_ * norm_cost;
                max_exp = std::max(max_exp, weights_[i]);
            }
            
            float weight_sum = 0.0f;
            for (size_t i = 0; i < actual_num_elites; ++i) {
                weights_[i] = std::exp(weights_[i] - max_exp);
                weight_sum += weights_[i];
            }
            float inv_sum = 1.0f / (weight_sum + 1e-10f);
            for (size_t i = 0; i < actual_num_elites; ++i) {
                weights_[i] *= inv_sum;
            }

            // === UPDATE (weighted mean of elites) ===
            Y_weighted_.setZero();
            for (size_t i = 0; i < actual_num_elites; ++i) {
                size_t idx = indices_[i];
                Y_weighted_.noalias() += weights_[i] * (Y_k + epsilon_samples_[idx]);
            }
            
            // Store pre-update trajectory for potential reversion
            MatrixXf Y_prev = Y_k;
            
            // EMA blend
            MatrixXf Y_new = (1.0f - ema_alpha_) * Y_k + ema_alpha_ * Y_weighted_;
            updateTrajectoryFromMatrix(Y_new);

            // Apply constraints
            current_trajectory_.nodes.front().position = start_node_.position;
            current_trajectory_.nodes.back().position = goal_node_.position;
            task_->filterTrajectory(current_trajectory_, iteration);

            // === PREPARE REUSE FOR NEXT ITERATION ===
            if (enable_rollout_reuse_) {
                // Store elite noise for potential reuse
                num_reused = std::min(actual_num_elites, num_rollouts_reuse_);
                for (size_t i = 0; i < num_reused; ++i) {
                    size_t idx = indices_[i];
                    if (idx >= epsilon_samples_.size()) {
                        std::cerr << "Warning: idx " << idx << " >= epsilon_samples_.size() " << epsilon_samples_.size() << "\n";
                        break;
                    }
                    reused_epsilon_[i] = epsilon_samples_[idx];
                    // Decay importance weight
                    if (idx >= num_fresh) {
                        // This was already a reused sample
                        size_t reuse_idx = idx - num_fresh;
                        if (reuse_idx < reused_weights_.size()) {
                            reused_weights_[i] = reused_weights_[reuse_idx] * importance_weight_decay_;
                        } else {
                            reused_weights_[i] = importance_weight_decay_;
                        }
                    } else {
                        reused_weights_[i] = importance_weight_decay_;
                    }
                    // Filter out samples below minimum weight
                    if (reused_weights_[i] < importance_weight_min_) {
                        num_reused = i;
                        break;
                    }
                }
            }

            // === COMPUTE COSTS AND CHECK VALIDITY ===
            float current_collision = task_->computeStateCostSimple(current_trajectory_);
            float current_smoothness = computeSmoothnessCost(current_trajectory_);
            float current_total_cost = current_collision + current_smoothness;
            
            // Check trajectory validity (collision-free)
            parameters_valid = checkTrajectoryValidity(current_trajectory_);

            // === COST IMPROVEMENT CHECK (STOMP-inspired) ===
            // If we had a valid solution and this update made it worse, revert
            if (current_total_cost < best_cost) {
                best_cost = current_total_cost;
                best_iteration = iteration;
                parameters_valid_prev = parameters_valid;
            } else if (parameters_valid_prev && !parameters_valid) {
                // Lost validity - revert to previous
                updateTrajectoryFromMatrix(Y_prev);
                current_trajectory_.nodes.front().position = start_node_.position;
                current_trajectory_.nodes.back().position = goal_node_.position;
                parameters_valid = parameters_valid_prev;
                logf("Iteration %zu - Reverted (lost validity)", iteration);
            }

            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);
            cov_scale_current_ = computeCovarianceScale(iteration + 1, prev_cost, current_total_cost);

            // === LOGGING (every iteration) ===
            auto iter_end = std::chrono::high_resolution_clock::now();
            double iter_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
            
            logf("%4zu     | %10.4f | %10.4f | %10.4f | %5s | %6zu | %9.2f",
                 iteration, current_total_cost, current_collision, current_smoothness,
                 parameters_valid ? "YES" : "no", actual_reused, iter_ms);

            // === EARLY TERMINATION CHECK (STOMP-inspired) ===
            if (enable_early_termination_) {
                if (parameters_valid) {
                    valid_iterations++;
                    
                    if (valid_iterations > num_iterations_after_valid_) {
                        logf("Early termination: Valid solution refined for %zu iterations", 
                             num_iterations_after_valid_);
                        break;
                    }
                } else {
                    // Lost validity, reset counter
                    valid_iterations = 0;
                }
            }

            // === CONVERGENCE CHECK ===
            if (std::abs(prev_cost - current_total_cost) < convergence_threshold_) {
                log("Converged (cost change below threshold)");
                break;
            }
            prev_cost = current_total_cost;
            
            task_->postIteration(iteration, current_total_cost, current_trajectory_);
        }
        
        // Get actual iterations completed (loop variable persists after break)
        size_t iterations_completed = trajectory_history_.size();
        
        log("-----------------------------------------------------------------------------------------");
        
        // Total runtime
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double total_sec = total_ms / 1000.0;

        // Use best trajectory
        if (best_iteration > 0) {
            current_trajectory_ = trajectory_history_[best_iteration - 1];
        }
        
        // Final status
        parameters_valid_ = checkTrajectoryValidity(current_trajectory_);
        float final_collision = task_->computeStateCostSimple(current_trajectory_);
        float final_smoothness = computeSmoothnessCost(current_trajectory_);
        
        logf("Final: Collision=%.4f, Smoothness=%.4f, Total=%.4f", 
             final_collision, final_smoothness, final_collision + final_smoothness);
        logf("Runtime: %.2f ms (%.3f sec) | %zu iterations | Avg: %.2f ms/iter", 
             total_ms, total_sec, iterations_completed, 
             total_ms / static_cast<double>(iterations_completed > 0 ? iterations_completed : 1));
        
        if (parameters_valid_) {
            logf("PCEM found VALID solution at iteration %zu", best_iteration);
        } else {
            logf("PCEM finished WITHOUT valid solution (best iteration: %zu)", best_iteration);
        }
        
        task_->done(parameters_valid_, num_iterations_, best_cost, current_trajectory_);
        return parameters_valid_;
    }

    [[nodiscard]] const std::vector<float>& getCovarianceScaleHistory() const {
        return covariance_scale_history_;
    }
    
    /**
     * @brief Check if the final trajectory is valid (collision-free)
     */
    [[nodiscard]] bool isTrajectoryValid() const {
        return parameters_valid_;
    }

protected:
    void initializeTask() override {
        if (!task_ || !pce_config_) return;
        task_->initialize(num_dimensions_, start_node_, goal_node_, num_nodes_, total_time_);
    }

    void logPlannerSpecificConfig() override {
        if (!pce_config_) return;
        log("--- PCEM-Lightweight ---");
        logf("  Samples: %zu, Elite ratio: %.2f", num_samples_, elite_ratio_);
        logf("  Early termination: %s", enable_early_termination_ ? "enabled" : "disabled");
        if (enable_early_termination_) {
            logf("    Iterations after valid: %zu", num_iterations_after_valid_);
        }
        logf("  Rollout reuse: %s", enable_rollout_reuse_ ? "enabled" : "disabled");
        if (enable_rollout_reuse_) {
            logf("    Max reuse: %zu rollouts", num_rollouts_reuse_);
        }
    }

private:
    /**
     * @brief Check if trajectory is valid (collision-free)
     * 
     * A trajectory is considered valid if its collision cost is below threshold.
     * This is a simple heuristic - zero collision cost means no collisions.
     */
    bool checkTrajectoryValidity(const Trajectory& trajectory) {
        if (!task_) return false;
        
        // Check if collision cost is essentially zero (below threshold)
        float collision_cost = task_->computeStateCostSimple(trajectory);
        return collision_cost < validity_cost_threshold_;
    }
    void preAllocateBuffers() {
        const size_t M = num_samples_;
        const size_t N = num_nodes_;
        const size_t D = num_dimensions_;
        
        // Safety checks
        if (M == 0 || N == 0 || D == 0) {
            std::cerr << "Warning: Cannot pre-allocate with M=" << M << " N=" << N << " D=" << D << "\n";
            return;
        }
        
        const size_t max_reuse = std::max(num_rollouts_reuse_, size_t{10}) + 10;  // Buffer for reuse
        
        // Noise matrices (M + space for reused)
        epsilon_samples_.resize(M + max_reuse);
        for (auto& eps : epsilon_samples_) {
            eps.resize(D, N);
            eps.setZero();
        }
        
        // Reuse buffers
        reused_epsilon_.resize(max_reuse);
        for (auto& eps : reused_epsilon_) {
            eps.resize(D, N);
            eps.setZero();
        }
        reused_weights_.resize(max_reuse, 1.0f);
        
        // Cost and index buffers
        sample_costs_.resize(M + max_reuse, 0.0f);
        indices_.resize(M + max_reuse, 0);
        
        // Weight buffer for elites
        size_t max_elites = static_cast<size_t>(M * elite_ratio_) + 10;
        weights_.resize(max_elites, 0.0f);
        
        // Weighted mean buffer
        Y_weighted_.resize(D, N);
        Y_weighted_.setZero();
        
        // Trajectory buffer for batch evaluation
        sample_trajectories_.clear();
        sample_trajectories_.reserve(M + max_reuse);
        
        logf("Pre-allocated buffers: M=%zu N=%zu D=%zu max_reuse=%zu", M, N, D, max_reuse);
    }

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

    // Task and config
    pce::TaskPtr task_;
    std::shared_ptr<PCEConfig> pce_config_;
    
    // Algorithm parameters
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float temperature_ = 1.5f;
    float eta_ = 1.0f;
    float gamma_ = 0.5f;
    float convergence_threshold_ = 0.01f;
    float ema_alpha_ = 0.5f;
    float elite_ratio_ = 0.1f;
    
    // Early termination (STOMP-inspired)
    bool enable_early_termination_ = true;
    size_t num_iterations_after_valid_ = 10;
    bool parameters_valid_ = false;  // Is current trajectory valid (collision-free)?
    float validity_cost_threshold_ = 1e-2f;  // Cost threshold for validity
    
    // Rollout reuse
    bool enable_rollout_reuse_ = true;
    size_t num_rollouts_reuse_ = 0;
    float exponentiated_cost_sensitivity_ = 10.0f;
    float importance_weight_decay_ = 0.9f;
    float importance_weight_min_ = 0.1f;
    
    // Covariance
    float cov_scale_current_ = 1.0f;
    
    // Pre-allocated buffers (lightweight - NO Trajectory storage)
    std::vector<MatrixXf> epsilon_samples_;      // Noise matrices
    std::vector<MatrixXf> reused_epsilon_;       // Reused noise from elites
    std::vector<float> reused_weights_;          // Importance weights for reused
    std::vector<float> sample_costs_;
    std::vector<size_t> indices_;
    std::vector<float> weights_;
    MatrixXf Y_weighted_;
    std::vector<Trajectory> sample_trajectories_; // Created on-demand (like NGD)
    
    std::vector<float> covariance_scale_history_;
    std::mt19937 random_engine_;
};