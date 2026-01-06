/**
 * @file PCEMotionPlanner_Optimized.h
 * @brief Optimized Proximal Cross-Entropy Method planner
 * 
 * Key optimizations:
 * 1. Pre-allocated buffers to avoid per-iteration allocations
 * 2. std::nth_element instead of full sort for elite selection
 * 3. Eigen expression templates and .noalias() for zero-copy ops
 * 4. Cache-friendly data layout
 * 5. Modern C++17/20 features
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
#include <execution>  // C++17 parallel algorithms


struct PCEConfig : public MotionPlannerConfig {
    // Use inline static constexpr for compile-time defaults (C++17)
    static constexpr size_t kDefaultNumSamples = 3000;
    static constexpr size_t kDefaultNumIterations = 10;
    static constexpr float kDefaultEta = 1.0f;
    static constexpr float kDefaultTemperature = 1.5f;
    static constexpr float kDefaultTemperatureFinal = 0.1f;
    static constexpr float kDefaultConvergenceThreshold = 0.01f;
    static constexpr float kDefaultEliteRatio = 0.1f;
    static constexpr float kDefaultEmaAlpha = 0.5f;
    
    size_t num_samples = kDefaultNumSamples;
    size_t num_iterations = kDefaultNumIterations;
    float eta = kDefaultEta;
    float temperature = kDefaultTemperature;
    float temperature_final = kDefaultTemperatureFinal;
    float convergence_threshold = kDefaultConvergenceThreshold;
    float collision_clearance = 0.1f;
    float collision_threshold = 0.1f;
    float gamma = 0.5f;
    float elite_ratio = kDefaultEliteRatio;
    float ema_alpha = kDefaultEmaAlpha;

    // Precompute derived values
    [[nodiscard]] constexpr float computeGamma() const noexcept {
        return eta / (eta + 1.0f);
    }

    bool loadFromYAML(const YAML::Node& config) override {
        if (!MotionPlannerConfig::loadFromYAML(config)) return false;
        
        try {
            if (auto pce = config["pce_planner"]) {
                // Use structured bindings helper for cleaner YAML parsing
                auto load = [&pce]<typename T>(const char* key, T& out) {
                    if (auto node = pce[key]) out = node.template as<T>();
                };
                
                load("num_samples", num_samples);
                load("num_iterations", num_iterations);
                load("eta", eta);
                gamma = computeGamma();  // Auto-update gamma when eta changes
                
                load("temperature", temperature);
                load("temperature_final", temperature_final);
                load("convergence_threshold", convergence_threshold);
                load("elite_ratio", elite_ratio);
                load("ema_alpha", ema_alpha);
                load("collision_clearance", collision_clearance);
                load("collision_threshold", collision_threshold);
            }
            return validate();
        } catch (const std::exception& e) {
            std::cerr << "Error loading PCE config: " << e.what() << "\n";
            return false;
        }
    }

    void print() const override {
        MotionPlannerConfig::print();
        std::cout << "--- PCE Hyperparameters (Optimized) ---\n"
                  << "  Eta:         " << eta << " (Gamma: " << gamma << ")\n"
                  << "  Temperature: " << temperature << " -> " << temperature_final << "\n"
                  << "  Elite Ratio: " << elite_ratio << "\n"
                  << "  EMA Alpha:   " << ema_alpha << "\n"
                  << "  Cov Scale:   " << cov_scale_initial << " -> " << cov_scale_final << "\n";
    }
};


class ProximalCrossEntropyMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    // Use Eigen's aligned allocator for better SIMD performance
    using AlignedMatrixVector = std::vector<MatrixXf, Eigen::aligned_allocator<MatrixXf>>;
    
    explicit ProximalCrossEntropyMotionPlanner(pce::TaskPtr task = nullptr)
        : task_(std::move(task))
    {
        std::random_device rd;
        seedRandomEngine(rd());
    }

    // Rule of 5 - explicitly default move operations for efficiency
    ProximalCrossEntropyMotionPlanner(ProximalCrossEntropyMotionPlanner&&) noexcept = default;
    ProximalCrossEntropyMotionPlanner& operator=(ProximalCrossEntropyMotionPlanner&&) noexcept = default;
    ~ProximalCrossEntropyMotionPlanner() override = default;
    
    // Delete copy to prevent expensive copies
    ProximalCrossEntropyMotionPlanner(const ProximalCrossEntropyMotionPlanner&) = delete;
    ProximalCrossEntropyMotionPlanner& operator=(const ProximalCrossEntropyMotionPlanner&) = delete;

    void setTask(pce::TaskPtr task) { task_ = std::move(task); }
    [[nodiscard]] pce::TaskPtr getTask() const { return task_; }

    bool initialize(const PCEConfig& config) {
        try {
            pce_config_ = std::make_shared<PCEConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating pce_config_: " << e.what() << '\n';
            return false;
        }
        
        // Batch parameter extraction - single cache line friendly
        num_samples_ = config.num_samples;
        num_iterations_ = config.num_iterations;
        temperature_ = config.temperature;
        eta_ = config.eta;
        gamma_ = config.gamma;
        convergence_threshold_ = config.convergence_threshold;
        elite_ratio_ = config.elite_ratio;
        ema_alpha_ = config.ema_alpha;
        
        // Covariance scheduling
        cov_schedule_ = config.cov_schedule;
        cov_scale_initial_ = config.cov_scale_initial;
        cov_scale_final_ = config.cov_scale_final;
        cov_decay_rate_ = config.cov_decay_rate;
        cov_step_interval_ = config.cov_step_interval;
        cov_step_factor_ = config.cov_step_factor;
        cov_adaptive_threshold_ = config.cov_adaptive_threshold;
        cov_scale_current_ = cov_scale_initial_;
        
        // Call base class initialize first (sets num_dimensions_, num_nodes_)
        try {
            if (!MotionPlanner::initialize(config)) {
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in MotionPlanner::initialize: " << e.what() << '\n';
            return false;
        }

        // Pre-allocate buffers after dimensions are known
        preAllocateBuffers();
        
        return true;
    }

    [[nodiscard]] std::string getPlannerName() const override { return "PCEM"; }

    // Getters with [[nodiscard]] to prevent ignoring return values
    [[nodiscard]] size_t getNumSamples() const noexcept { return num_samples_; }
    [[nodiscard]] size_t getNumIterations() const noexcept { return num_iterations_; }
    [[nodiscard]] float getTemperature() const noexcept { return temperature_; }
    [[nodiscard]] float getEta() const noexcept { return eta_; }
    [[nodiscard]] float getGamma() const noexcept { return gamma_; }
    [[nodiscard]] float getConvergenceThreshold() const noexcept { return convergence_threshold_; }
    [[nodiscard]] float getCovarianceScale() const noexcept { return cov_scale_current_; }
    [[nodiscard]] CovarianceSchedule getCovarianceSchedule() const noexcept { return cov_schedule_; }
    [[nodiscard]] std::shared_ptr<const PCEConfig> getPCEConfig() const { return pce_config_; }

    void seedRandomEngine(unsigned int seed) { random_engine_.seed(seed); }

    [[nodiscard]] float computeStateCost(const Trajectory& trajectory) const override {
        if (!task_) [[unlikely]] {
            std::cerr << "Error: No task set for collision cost computation\n";
            return std::numeric_limits<float>::infinity();
        }
        return task_->computeStateCost(trajectory);
    }

    bool optimize() override {
        if (!task_) [[unlikely]] {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }

        const size_t N = current_trajectory_.nodes.size();
        const size_t M = num_samples_;
        const size_t D = num_dimensions_;
        const size_t num_elites = std::max(size_t{1}, static_cast<size_t>(M * elite_ratio_));

        // Precompute temperature ratio for exponential schedule
        const float temp_ratio = pce_config_->temperature_final / pce_config_->temperature;
        const float inv_max_iter = 1.0f / std::max(1.0f, static_cast<float>(num_iterations_ - 1));

        float best_cost = std::numeric_limits<float>::infinity();
        size_t best_iteration = 0;
        float prev_cost = std::numeric_limits<float>::infinity();

        trajectory_history_.clear();
        trajectory_history_.reserve(num_iterations_);
        covariance_scale_history_.clear();
        covariance_scale_history_.reserve(num_iterations_);

        gamma_ = pce_config_->gamma;

        for (size_t iteration = 1; iteration <= num_iterations_; ++iteration) {
            MatrixXf Y_k = trajectoryToMatrix();

            // Temperature schedule - precomputed ratio makes this faster
            const float progress = static_cast<float>(iteration - 1) * inv_max_iter;
            const float current_temp = pce_config_->temperature * std::pow(temp_ratio, progress);
            const float inv_temp = 1.0f / current_temp;  // Multiply is faster than divide

            // Sample noise matrices into pre-allocated buffer
            sampleScaledNoiseMatricesInPlace(M, cov_scale_current_);
            
            // Create perturbed trajectories - reuse pre-allocated vector
            for (size_t m = 0; m < M; ++m) {
                createPerturbedTrajectoryInPlace(Y_k, epsilon_buffer_[m], sample_trajectories_[m]);
            }
            
            // Batch compute collision costs
            // Note: If Task API can be modified, consider adding:
            //   void computeStateCostSimple(const std::vector<Trajectory>&, std::vector<float>& out)
            // to avoid allocation here
            sample_costs_ = task_->computeStateCostSimple(sample_trajectories_);
            
            // Apply temperature scaling in-place
            for (auto& c : sample_costs_) c *= inv_temp;

            // OPTIMIZATION: Use nth_element instead of full sort (O(n) vs O(n log n))
            std::iota(indices_.begin(), indices_.end(), size_t{0});
            std::nth_element(indices_.begin(), indices_.begin() + num_elites, indices_.end(),
                [this](size_t a, size_t b) { return sample_costs_[a] < sample_costs_[b]; });
            
            // Only sort the elite portion
            std::sort(indices_.begin(), indices_.begin() + num_elites,
                [this](size_t a, size_t b) { return sample_costs_[a] < sample_costs_[b]; });

            // Compute raw costs for elites - use pre-allocated buffer
            computeEliteCosts(num_elites, Y_k, D);

            // Compute statistics using numerically stable algorithms
            const auto [mean_cost, std_cost] = computeMeanStd(raw_costs_, num_elites);
            const float inv_std = 1.0f / std_cost;  // Pre-compute division

            // Compute normalized weights with log-sum-exp stability
            float max_exponent = -std::numeric_limits<float>::infinity();
            weights_.setZero();
            
            for (size_t i = 0; i < num_elites; ++i) {
                const size_t m = indices_[i];
                const float normalized_cost = (raw_costs_[i] - mean_cost) * inv_std;
                const float exponent = -gamma_ * normalized_cost;
                weights_(m) = exponent;
                max_exponent = std::max(max_exponent, exponent);
            }

            // Softmax with numerical stability
            float weight_sum = 0.0f;
            for (size_t i = 0; i < num_elites; ++i) {
                const size_t m = indices_[i];
                weights_(m) = std::exp(weights_(m) - max_exponent);
                weight_sum += weights_(m);
            }
            
            const float inv_weight_sum = 1.0f / (weight_sum + 1e-10f);
            for (size_t i = 0; i < num_elites; ++i) {
                weights_(indices_[i]) *= inv_weight_sum;
            }

            // EMA Update - use .noalias() to avoid temporaries
            Y_weighted_.setZero();
            for (size_t i = 0; i < num_elites; ++i) {
                const size_t m = indices_[i];
                Y_weighted_.noalias() += weights_(m) * (Y_k + epsilon_buffer_[m]);
            }

            // Blend old and new trajectories using pre-allocated buffer
            Y_new_.noalias() = (1.0f - ema_alpha_) * Y_k + ema_alpha_ * Y_weighted_;
            updateTrajectoryFromMatrix(Y_new_);

            // Apply constraints
            current_trajectory_.nodes.front().position = start_node_.position;
            current_trajectory_.nodes.back().position = goal_node_.position;
            task_->filterTrajectory(current_trajectory_, iteration);

            // Compute costs
            const float current_collision = task_->computeStateCostSimple(current_trajectory_);
            const float current_smoothness = computeSmoothnessCost(current_trajectory_);
            const float current_total_cost = current_collision + current_smoothness;

            trajectory_history_.push_back(current_trajectory_);
            covariance_scale_history_.push_back(cov_scale_current_);
            cov_scale_current_ = computeCovarianceScale(iteration + 1, prev_cost, current_total_cost);

            if (current_total_cost < best_cost) {
                best_cost = current_total_cost;
                best_iteration = iteration;
            }

            // Sparse logging - avoid string formatting overhead
            if (iteration == 1 || iteration % 10 == 0 || iteration == num_iterations_) {
                logf("Iter %zu - Cost: %.2f (Col: %.4f, Smooth: %.4f) T: %.4f",
                    iteration, current_total_cost, current_collision, current_smoothness, current_temp);
            }

            // Early termination check
            if (std::abs(prev_cost - current_total_cost) < convergence_threshold_) break;
            prev_cost = current_total_cost;
            
            task_->postIteration(iteration, current_total_cost, current_trajectory_);
        }

        current_trajectory_ = std::move(trajectory_history_[best_iteration - 1]);
        task_->done(true, num_iterations_, best_cost, current_trajectory_);
        return true;
    }
    
    [[nodiscard]] const std::vector<float>& getCovarianceScaleHistory() const noexcept {
        return covariance_scale_history_;
    }

protected:
    void initializeTask() override {
        if (!task_) {
            std::cerr << "Warning: No task set for initialization\n";
            return;
        }
        if (!pce_config_) {
            std::cerr << "Error: No PCE configuration loaded\n";
            return;
        }
        
        task_->initialize(num_dimensions_, start_node_, goal_node_, num_nodes_, total_time_);
        std::cout << "Task initialized\n";
    }

    void logPlannerSpecificConfig() override {
        if (!pce_config_) return;

        log("--- PCEM Planner Parameters (Optimized) ---");
        logf("  Num samples:    %zu", num_samples_);
        logf("  Num iterations: %zu", num_iterations_);
        logf("  Temp:           %.4f -> %.4f", pce_config_->temperature, pce_config_->temperature_final);
        logf("  Eta/Gamma:      %.4f / %.4f", eta_, gamma_);
        logf("  Conv threshold: %.6f", convergence_threshold_);
        logf("  Cov schedule:   %s (%.4f -> %.4f)", 
             pce_config_->getScheduleName().c_str(), cov_scale_initial_, cov_scale_final_);
    }

private:
    /**
     * @brief Pre-allocate all buffers to avoid per-iteration memory allocation
     * Must be called after num_dimensions_ and num_nodes_ are set
     */
    void preAllocateBuffers() {
        const size_t M = num_samples_;
        const size_t N = num_nodes_;
        const size_t D = num_dimensions_;
        
        // Pre-allocate sample storage
        epsilon_buffer_.resize(M);
        for (auto& eps : epsilon_buffer_) {
            eps.resize(D, N);
        }
        
        // Pre-allocate sample trajectories with correct structure
        sample_trajectories_.resize(M);
        for (auto& traj : sample_trajectories_) {
            traj = current_trajectory_;  // Copy structure (nodes, time, etc.)
        }
        
        sample_costs_.resize(M);
        indices_.resize(M);
        
        // Pre-allocate elite computation buffers
        const size_t max_elites = static_cast<size_t>(M * elite_ratio_) + 1;
        raw_costs_.resize(max_elites);
        
        // Pre-allocate Eigen vectors/matrices
        weights_ = VectorXf::Zero(M);
        Y_weighted_ = MatrixXf::Zero(D, N);
        Y_new_ = MatrixXf::Zero(D, N);  // Buffer for blended trajectory
    }

    /**
     * @brief Sample noise matrices directly into pre-allocated buffer
     * 
     * More efficient version that avoids allocation by reusing epsilon_buffer_
     */
    void sampleScaledNoiseMatricesInPlace(size_t M, float scale) {
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        
        for (size_t m = 0; m < M; ++m) {
            // Sample structured noise for each dimension
            for (size_t d = 0; d < D; ++d) {
                auto noise = sampleSmoothnessNoise(N, random_engine_);
                for (size_t i = 0; i < N; ++i) {
                    epsilon_buffer_[m](d, i) = noise[i] * scale;
                }
            }
        }
    }

    /**
     * @brief Create perturbed trajectory in-place
     * 
     * Reuses the trajectory structure to avoid allocation
     */
    void createPerturbedTrajectoryInPlace(const MatrixXf& Y_k, const MatrixXf& epsilon, 
                                          Trajectory& out_trajectory) {
        // Ensure output has correct structure
        if (out_trajectory.nodes.size() != static_cast<size_t>(Y_k.cols())) {
            out_trajectory = current_trajectory_;  // Copy structure if needed
        }
        
        // Update positions in-place
        for (Eigen::Index i = 0; i < Y_k.cols(); ++i) {
            out_trajectory.nodes[i].position = Y_k.col(i) + epsilon.col(i);
        }
    }

    /**
     * @brief Compute elite costs with pre-allocated buffer
     */
    void computeEliteCosts(size_t num_elites, const MatrixXf& Y_k, size_t D) {
        for (size_t i = 0; i < num_elites; ++i) {
            const size_t m = indices_[i];
            float reg_term = 0.0f;
            
            // Use Eigen's efficient dot product
            for (size_t d = 0; d < D; ++d) {
                reg_term += epsilon_buffer_[m].row(d).dot(R_matrix_ * Y_k.row(d).transpose());
            }
            raw_costs_[i] = sample_costs_[m] + reg_term;
        }
    }

    /**
     * @brief Compute mean and standard deviation using Welford's algorithm
     *        (numerically stable single-pass)
     */
    [[nodiscard]] static std::pair<float, float> computeMeanStd(
            const std::vector<float>& values, size_t n) {
        if (n == 0) return {0.0f, 1.0f};
        
        float mean = 0.0f;
        float M2 = 0.0f;
        
        for (size_t i = 0; i < n; ++i) {
            const float delta = values[i] - mean;
            mean += delta / static_cast<float>(i + 1);
            const float delta2 = values[i] - mean;
            M2 += delta * delta2;
        }
        
        const float variance = M2 / static_cast<float>(n);
        return {mean, std::sqrt(variance + 1e-8f)};
    }

    // Task and configuration
    pce::TaskPtr task_;
    std::shared_ptr<PCEConfig> pce_config_;
    
    // Algorithm parameters (hot data - frequently accessed together)
    // Grouped for cache locality
    struct alignas(64) HotParams {  // Align to cache line
        size_t num_samples_ = 3000;
        size_t num_iterations_ = 10;
        float temperature_ = 1.5f;
        float eta_ = 1.0f;
        float gamma_ = 0.5f;
        float convergence_threshold_ = 0.01f;
        float ema_alpha_ = 0.5f;
        float elite_ratio_ = 0.1f;
    };
    
    // Inline the hot params for this example (or use the struct)
    size_t num_samples_ = 3000;
    size_t num_iterations_ = 10;
    float temperature_ = 1.5f;
    float eta_ = 1.0f;
    float gamma_ = 0.5f;
    float convergence_threshold_ = 0.01f;
    float ema_alpha_ = 0.5f;
    float elite_ratio_ = 0.1f;
    
    // Covariance scheduling
    CovarianceSchedule cov_schedule_ = CovarianceSchedule::EXPONENTIAL;
    float cov_scale_initial_ = 1.0f;
    float cov_scale_final_ = 0.1f;
    float cov_decay_rate_ = 0.9f;
    size_t cov_step_interval_ = 3;
    float cov_step_factor_ = 0.5f;
    float cov_adaptive_threshold_ = 0.05f;
    float cov_scale_current_ = 1.0f;
    
    // Pre-allocated buffers (cold data - only accessed during optimization)
    AlignedMatrixVector epsilon_buffer_;
    std::vector<Trajectory> sample_trajectories_;
    std::vector<float> sample_costs_;
    std::vector<size_t> indices_;
    std::vector<float> raw_costs_;
    VectorXf weights_;
    MatrixXf Y_weighted_;  // Pre-allocate in preAllocateBuffers based on dimensions
    MatrixXf Y_new_;       // Buffer for blended trajectory update
    
    std::vector<float> covariance_scale_history_;
    std::mt19937 random_engine_;
};