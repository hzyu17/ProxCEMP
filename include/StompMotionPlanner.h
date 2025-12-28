/**
 * @file StompMotionPlanner.h
 * @brief STOMP-based motion planner wrapper (Refactored for visualization consistency)
 * 
 * Modified to match PCEMotionPlanner's data storage patterns:
 * - Stores samples per iteration for visualization
 * - Uses trajectory_history_ naming convention
 * - Provides OptimizationHistory-compatible interface
 */
#pragma once

#include <stomp/stomp.h>
#include "StompTask.h"
#include "CollisionAvoidanceTask.h"
#include "Trajectory.h"
#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include <mutex>

namespace pce {

/**
 * @brief Data for a single optimization iteration (matches PCE visualization format)
 */
struct StompIterationData {
    std::vector<Trajectory> samples;    // Noisy rollouts for this iteration
    Trajectory mean_trajectory;          // Mean/updated trajectory
    float cost = 0.0f;                   // Total cost from STOMP (state + control)
    float collision_cost = 0.0f;         // State/collision cost component
    float smoothness_cost = 0.0f;        // Control/smoothness cost component (derived)
    int iteration_number = 0;
};

/**
 * @brief Complete optimization history (compatible with visualization)
 */
struct StompOptimizationHistory {
    std::vector<StompIterationData> iterations;
    
    // Convenience accessors matching OptimizationHistory interface
    size_t size() const { return iterations.size(); }
    bool empty() const { return iterations.empty(); }
};

/**
 * @brief STOMP Motion Planner - holds all optimization state
 * 
 * Refactored to be consistent with PCEMotionPlanner:
 * - Stores sample trajectories per iteration
 * - Provides trajectory_history_ style interface
 * - Compatible with showTrajectoryEvolution visualization
 */
class StompMotionPlanner {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXf = Eigen::VectorXf;
    
    explicit StompMotionPlanner(std::shared_ptr<CollisionAvoidanceTask> task)
        : collision_task_(task)
        , initialized_(false)
        , stomp_(nullptr)
        , noise_stddev_(20.0f)
        , noise_decay_(0.95f)
        , current_iteration_(0)
        , final_cost_(0.0f)
        , converged_(false)
    {
        rng_.seed(std::random_device{}());
    }
    
    ~StompMotionPlanner() {
        if (stomp_) {
            delete stomp_;
            stomp_ = nullptr;
        }
    }
    
    bool initialize(const StompPlannerConfig& config) {
        config_ = config;
        noise_stddev_ = config.noise_stddev;
        noise_decay_ = config.noise_decay;
        
        // Validate configuration
        if (config_.num_dimensions == 0) {
            std::cerr << "Error: num_dimensions cannot be 0\n";
            return false;
        }
        
        if (config_.num_timesteps < 10) {
            std::cerr << "Error: num_timesteps must be at least 10\n";
            return false;
        }
        
        if (config_.start_position.size() != config_.num_dimensions ||
            config_.goal_position.size() != config_.num_dimensions) {
            std::cerr << "Error: Start/goal position dimensions mismatch\n";
            return false;
        }
        
        if (config_.max_rollouts <= config_.num_rollouts) {
            config_.max_rollouts = config_.num_rollouts + 1;
        }
        
        initializeTrajectory();
        
        // Compute smoothing matrix for noise generation (uses R^-1)
        computeSmoothingMatrix();
        
        stomp::StompConfiguration stomp_config = config_.toStompConfig();
        
        std::cout << "STOMP Config Debug:\n"
                  << "  num_dimensions: " << stomp_config.num_dimensions << "\n"
                  << "  num_timesteps: " << stomp_config.num_timesteps << "\n"
                  << "  num_iterations: " << stomp_config.num_iterations << "\n"
                  << "  num_rollouts: " << stomp_config.num_rollouts << "\n"
                  << "  max_rollouts: " << stomp_config.max_rollouts << "\n";
        
        // Create minimal stateless task
        std::cout << "Creating STOMP task adapter...\n";
        stomp_task_ = std::make_shared<StompCollisionTask>();
        std::cout << "Task size: " << sizeof(StompCollisionTask) << " bytes\n";
        
        // Set back-pointer BEFORE creating STOMP
        stomp_task_->setPlanner(this);
        
        std::cout << "Creating STOMP optimizer...\n";
        try {
            stomp_ = new stomp::Stomp(stomp_config, stomp_task_);
        } catch (const std::exception& e) {
            std::cerr << "Error creating STOMP: " << e.what() << "\n";
            return false;
        }
        
        if (!stomp_) {
            std::cerr << "Error: STOMP is null\n";
            return false;
        }
        
        initialized_ = true;
        std::cout << "STOMP Planner initialized successfully\n";
        return true;
    }
    
    bool solve() {
        if (!initialized_) {
            std::cerr << "Error: Planner not initialized\n";
            return false;
        }
        
        std::cout << "\n=== Running STOMP Optimization ===\n";
        
        // Clear history and prepare for new optimization
        optimization_history_.iterations.clear();
        trajectory_history_.clear();
        current_iteration_samples_.clear();
        
        // Store initial trajectory as iteration 0
        StompIterationData initial_data;
        initial_data.mean_trajectory = current_trajectory_;
        initial_data.iteration_number = 0;
        initial_data.cost = 0.0f;
        optimization_history_.iterations.push_back(initial_data);
        trajectory_history_.push_back(current_trajectory_);
        
        VectorXd start(config_.num_dimensions);
        VectorXd goal(config_.num_dimensions);
        for (size_t d = 0; d < config_.num_dimensions; ++d) {
            start(d) = config_.start_position[d];
            goal(d) = config_.goal_position[d];
        }
        
        MatrixXd optimized_params;
        bool success = false;
        
        try {
            success = stomp_->solve(start, goal, optimized_params);
        } catch (const std::exception& e) {
            std::cerr << "Exception during solve: " << e.what() << "\n";
            return false;
        }
        
        if (success || optimized_params.size() > 0) {
            std::cout << "Converting final optimized parameters to trajectory...\n";
            std::cout << "  optimized_params size: " << optimized_params.rows() << " x " << optimized_params.cols() << "\n";
            current_trajectory_ = matrixToTrajectory(optimized_params, true);  // Enable debug
        }
        
        return success;
    }
    
    // ==================== Task callback implementations ====================
    
    /**
     * @brief Compute the smoothing matrix M from inverse control cost matrix R^-1
     * This is used to generate temporally correlated noise for STOMP
     */
    void computeSmoothingMatrix() {
        using namespace Eigen;
        
        int num_timesteps = config_.num_timesteps;
        double dt = config_.delta_t;
        
        // Finite difference rule length (same as STOMP uses)
        const int FINITE_DIFF_RULE_LENGTH = 7;
        int start_index_padded = FINITE_DIFF_RULE_LENGTH - 1;
        int num_timesteps_padded = num_timesteps + 2 * (FINITE_DIFF_RULE_LENGTH - 1);
        
        // Central difference coefficients for acceleration (from stomp/utils.h)
        const double ACC_COEFFS[7] = {-1.0/90.0, 3.0/20.0, -3.0/2.0, 49.0/18.0, -3.0/2.0, 3.0/20.0, -1.0/90.0};
        
        // Generate finite difference matrix for acceleration
        MatrixXd finite_diff_matrix_A_padded = MatrixXd::Zero(num_timesteps_padded, num_timesteps_padded);
        double multiplier = 1.0 / (dt * dt);
        
        for (int i = 0; i < num_timesteps_padded; ++i) {
            for (int j = -FINITE_DIFF_RULE_LENGTH / 2; j <= FINITE_DIFF_RULE_LENGTH / 2; ++j) {
                int index = i + j;
                if (index < 0 || index >= num_timesteps_padded) continue;
                finite_diff_matrix_A_padded(i, index) = multiplier * ACC_COEFFS[j + FINITE_DIFF_RULE_LENGTH / 2];
            }
        }
        
        // Control cost matrix R = dt * A^T * A
        MatrixXd control_cost_matrix_R_padded = dt * finite_diff_matrix_A_padded.transpose() * finite_diff_matrix_A_padded;
        MatrixXd control_cost_matrix_R = control_cost_matrix_R_padded.block(
            start_index_padded, start_index_padded, num_timesteps, num_timesteps);
        
        // Inverse of control cost matrix
        MatrixXd inv_control_cost_matrix_R = control_cost_matrix_R.fullPivLu().inverse();
        
        // Scale so max(R^-1) == 1 (same as STOMP does)
        double maxVal = std::abs(inv_control_cost_matrix_R.maxCoeff());
        inv_control_cost_matrix_R /= maxVal;
        
        // Compute projection/smoothing matrix M
        // Scale columns so that max of each column is 1/num_timesteps
        smoothing_matrix_M_ = inv_control_cost_matrix_R;
        for (int t = 0; t < num_timesteps; ++t) {
            double col_max = smoothing_matrix_M_(t, t);
            if (std::abs(col_max) > 1e-10) {
                smoothing_matrix_M_.col(t) *= (1.0 / (num_timesteps * col_max));
            }
        }
        
        std::cout << "Computed smoothing matrix M (" << smoothing_matrix_M_.rows() 
                  << " x " << smoothing_matrix_M_.cols() << ")\n";
    }
    
    bool taskGenerateNoisyParameters(const MatrixXd& parameters,
                                     std::size_t num_timesteps,
                                     int iteration_number,
                                     int rollout_number,
                                     MatrixXd& parameters_noise,
                                     MatrixXd& noise) 
    {
        // Generate smooth noise using the projection matrix M (derived from R^-1)
        // This is the proper STOMP approach for temporally correlated noise
        
        float decayed_stddev = noise_stddev_ * std::pow(noise_decay_, iteration_number);
        std::normal_distribution<double> dist(0.0, decayed_stddev);
        
        size_t num_dims = parameters.rows();
        
        // Generate white noise
        VectorXd white_noise(num_timesteps);
        
        noise.resize(num_dims, num_timesteps);
        noise.setZero();
        
        for (size_t d = 0; d < num_dims; ++d) {
            // Generate independent white noise for this dimension
            for (size_t t = 0; t < num_timesteps; ++t) {
                white_noise(t) = dist(rng_);
            }
            
            // Apply smoothing matrix: smooth_noise = M * white_noise
            // This produces temporally correlated noise
            VectorXd smooth_noise = smoothing_matrix_M_ * white_noise;
            
            // Copy to noise matrix
            noise.row(d) = smooth_noise.transpose();
            
            // Ensure start and goal have zero noise
            noise(d, 0) = 0.0;
            noise(d, num_timesteps - 1) = 0.0;
        }
        
        parameters_noise = parameters + noise;
        
        return true;
    }
    
    bool taskComputeCosts(const MatrixXd& parameters,
                          std::size_t num_timesteps,
                          int rollout_number,
                          VectorXd& costs,
                          bool& validity) 
    {
        costs.resize(num_timesteps);
        costs.setZero();
        
        if (collision_task_) {
            const auto& obs = collision_task_->getObstacleSoA();
            if (!obs.empty()) {
                const float epsilon = collision_task_->getConfig().epsilon_sdf;
                const float sigma = collision_task_->getConfig().sigma_obs;
                
                // Get dimensions from obstacle data (should match config)
                size_t obs_dims = obs.centers.rows();
                size_t param_dims = parameters.rows();
                size_t param_timesteps = parameters.cols();
                
                // Debug: Check dimensions on first call
                static bool first_call = true;
                if (first_call) {
                    std::cout << "  taskComputeCosts: parameters(" << param_dims << " x " << param_timesteps 
                              << "), obstacles dims=" << obs_dims << ", num_timesteps=" << num_timesteps << "\n";
                    first_call = false;
                }
                
                // Use minimum of parameter dims and obstacle dims for safety
                size_t num_dims = std::min(obs_dims, param_dims);
                
                for (size_t t = 1; t < num_timesteps - 1; ++t) {
                    VectorXf pos(obs_dims);
                    pos.setZero();
                    for (size_t d = 0; d < num_dims; ++d) {
                        pos(d) = static_cast<float>(parameters(d, t));
                    }
                    
                    Eigen::MatrixXf diff = obs.centers.colwise() - pos;
                    Eigen::VectorXf distances = diff.colwise().norm();
                    Eigen::VectorXf sdfs = distances - obs.combined_radii;
                    Eigen::VectorXf hinges = (epsilon - sdfs.array()).max(0.0f);
                    costs(t) = sigma * hinges.squaredNorm();
                }
            }
        }
        
        validity = (costs.sum() < 1e-3);
        return true;
    }
    
    bool taskFilterUpdates(MatrixXd& updates) {
        // STOMP convention: rows=dimensions, cols=timesteps
        // Zero out start (col 0) and goal (last col) updates
        updates.col(0).setZero();
        updates.col(updates.cols() - 1).setZero();
        return true;
    }
    
    void taskPostIteration(int iteration_number, double cost, const MatrixXd& parameters) {
        current_iteration_ = iteration_number;
        
        // Debug: Print raw matrix info on first iteration
        if (iteration_number == 0) {
            std::cout << "\n[DEBUG] taskPostIteration raw parameters matrix:\n";
            std::cout << "  Matrix dimensions: " << parameters.rows() << " x " << parameters.cols() << "\n";
            std::cout << "  First column (t=0): ";
            for (int r = 0; r < std::min(int(parameters.rows()), 5); ++r) {
                std::cout << parameters(r, 0) << " ";
            }
            std::cout << "\n  First row (d=0): ";
            for (int c = 0; c < std::min(int(parameters.cols()), 5); ++c) {
                std::cout << parameters(0, c) << " ";
            }
            std::cout << "\n  Last column (t=" << (parameters.cols()-1) << "): ";
            for (int r = 0; r < std::min(int(parameters.rows()), 5); ++r) {
                std::cout << parameters(r, parameters.cols()-1) << " ";
            }
            std::cout << "\n  Last row (d=" << (parameters.rows()-1) << "): ";
            for (int c = 0; c < std::min(int(parameters.cols()), 5); ++c) {
                std::cout << parameters(parameters.rows()-1, c) << " ";
            }
            std::cout << "\n  Expected start: (" << config_.start_position[0] << ", " << config_.start_position[1] << ")\n";
            std::cout << "  Expected goal: (" << config_.goal_position[0] << ", " << config_.goal_position[1] << ")\n";
        }
        
        // Debug first iteration's trajectory in detail
        bool debug_traj = (iteration_number == 0);
        Trajectory mean_traj = matrixToTrajectory(parameters, debug_traj);
        
        // Debug: verify mean trajectory endpoints
        if (iteration_number == 0 || iteration_number == 1) {
            if (!mean_traj.nodes.empty() && mean_traj.nodes[0].position.size() >= 2) {
                std::cout << "  Iter " << iteration_number << " mean traj: start=("
                          << mean_traj.nodes[0].position(0) << ", " 
                          << mean_traj.nodes[0].position(1) << "), goal=("
                          << mean_traj.nodes.back().position(0) << ", "
                          << mean_traj.nodes.back().position(1) << ")\n";
            }
        }
        
        // Compute collision cost for the mean trajectory (same as in taskComputeCosts)
        float collision_cost = 0.0f;
        if (collision_task_) {
            const auto& obs = collision_task_->getObstacleSoA();
            if (!obs.empty()) {
                const float epsilon = collision_task_->getConfig().epsilon_sdf;
                const float sigma = collision_task_->getConfig().sigma_obs;
                
                // Use obstacle dimensions (should match parameter dimensions)
                size_t obs_dims = obs.centers.rows();
                size_t param_dims = parameters.rows();
                size_t num_timesteps = parameters.cols();
                size_t num_dims = std::min(obs_dims, param_dims);
                
                for (size_t t = 1; t < num_timesteps - 1; ++t) {
                    VectorXf pos(obs_dims);
                    pos.setZero();
                    for (size_t d = 0; d < num_dims; ++d) {
                        pos(d) = static_cast<float>(parameters(d, t));
                    }
                    
                    Eigen::MatrixXf diff = obs.centers.colwise() - pos;
                    Eigen::VectorXf distances = diff.colwise().norm();
                    Eigen::VectorXf sdfs = distances - obs.combined_radii;
                    Eigen::VectorXf hinges = (epsilon - sdfs.array()).max(0.0f);
                    collision_cost += sigma * hinges.squaredNorm();
                }
            }
        }
        
        // Smoothness cost is derived from STOMP's total cost
        // STOMP total = state_cost + control_cost (where control_cost is the smoothness)
        float total_cost = static_cast<float>(cost);
        float smoothness_cost = std::max(0.0f, total_cost - collision_cost);
        
        // Store iteration data with samples (like PCE's trajectory_history_)
        StompIterationData iter_data;
        iter_data.samples = std::move(current_iteration_samples_);
        iter_data.mean_trajectory = mean_traj;
        iter_data.cost = total_cost;
        iter_data.collision_cost = collision_cost;
        iter_data.smoothness_cost = smoothness_cost;
        iter_data.iteration_number = iteration_number;
        
        optimization_history_.iterations.push_back(iter_data);
        trajectory_history_.push_back(mean_traj);
        
        // Clear for next iteration
        current_iteration_samples_.clear();
        
        if (iteration_number % 10 == 0) {
            std::cout << "  STOMP Iteration " << iteration_number 
                      << ": total=" << total_cost 
                      << ", collision=" << collision_cost
                      << ", smooth=" << smoothness_cost
                      << ", samples=" << iter_data.samples.size() << "\n";
        }
    }
    
    void taskDone(bool success, int total_iterations, double final_cost) {
        std::cout << "STOMP " << (success ? "succeeded" : "failed")
                  << " after " << total_iterations << " iterations"
                  << " with cost " << final_cost << "\n";
        converged_ = success;
        final_cost_ = static_cast<float>(final_cost);
    }
    
    // ==================== Accessors (consistent with PCE interface) ====================
    
    const Trajectory& getCurrentTrajectory() const { return current_trajectory_; }
    
    /** @brief Get trajectory history (mean trajectories, like PCE's trajectory_history_) */
    const std::vector<Trajectory>& getTrajectoryHistory() const { 
        return trajectory_history_; 
    }
    
    /** @brief Get full optimization history with samples (for visualization) */
    const StompOptimizationHistory& getOptimizationHistory() const { 
        return optimization_history_; 
    }
    
    /** @brief Get number of iterations completed */
    size_t getNumIterations() const { 
        return optimization_history_.iterations.size(); 
    }
    
    bool isInitialized() const { return initialized_; }
    bool hasConverged() const { return converged_; }
    float getFinalCost() const { return final_cost_; }
    
    float computeSmoothnessCost(const Trajectory& trajectory) const {
        if (trajectory.nodes.size() < 3) return 0.0f;
        
        float cost = 0.0f;
        for (size_t i = 1; i < trajectory.nodes.size() - 1; ++i) {
            VectorXf acc = trajectory.nodes[i+1].position 
                         - 2.0f * trajectory.nodes[i].position 
                         + trajectory.nodes[i-1].position;
            cost += acc.squaredNorm();
        }
        
        return cost * config_.control_cost_weight / (config_.delta_t * config_.delta_t);
    }

private:
    void initializeTrajectory() {
        current_trajectory_.nodes.resize(config_.num_timesteps);
        
        VectorXf start(config_.num_dimensions);
        VectorXf goal(config_.num_dimensions);
        for (size_t d = 0; d < config_.num_dimensions; ++d) {
            start(d) = config_.start_position[d];
            goal(d) = config_.goal_position[d];
        }
        
        std::cout << "Initializing trajectory: start=(" << start(0) << ", " << start(1) 
                  << "), goal=(" << goal(0) << ", " << goal(1) 
                  << "), timesteps=" << config_.num_timesteps << "\n";
        
        for (size_t t = 0; t < config_.num_timesteps; ++t) {
            float alpha = static_cast<float>(t) / (config_.num_timesteps - 1);
            current_trajectory_.nodes[t].position = (1.0f - alpha) * start + alpha * goal;
        }
        
        // Set start and goal indices for visualization
        current_trajectory_.start_index = 0;
        current_trajectory_.goal_index = config_.num_timesteps - 1;
        
        // Verify first and last nodes
        std::cout << "  Initialized trajectory node[0]=(" 
                  << current_trajectory_.nodes[0].position(0) << ", " 
                  << current_trajectory_.nodes[0].position(1) << ")\n";
        std::cout << "  Initialized trajectory node[" << (config_.num_timesteps-1) << "]=(" 
                  << current_trajectory_.nodes[config_.num_timesteps-1].position(0) << ", " 
                  << current_trajectory_.nodes[config_.num_timesteps-1].position(1) << ")\n";
    }
    
    Trajectory matrixToTrajectory(const MatrixXd& params, bool debug = false) const {
        Trajectory traj;
        if (params.size() == 0) return traj;
        
        // STOMP convention: rows = dimensions, cols = timesteps
        // params(d, t) gives dimension d at timestep t
        size_t num_dims = params.rows();
        size_t num_timesteps = params.cols();
        
        traj.nodes.resize(num_timesteps);
        for (size_t t = 0; t < num_timesteps; ++t) {
            traj.nodes[t].position.resize(num_dims);
            for (size_t d = 0; d < num_dims; ++d) {
                traj.nodes[t].position(d) = static_cast<float>(params(d, t));
            }
        }
        
        // Set start and goal indices for visualization
        traj.start_index = 0;
        traj.goal_index = num_timesteps - 1;
        
        // Debug: print trajectory info
        if (debug && num_timesteps > 0 && num_dims >= 2) {
            std::cout << "  Trajectory debug (dims=" << num_dims << ", timesteps=" << num_timesteps << "):\n";
            std::cout << "    Matrix size: " << params.rows() << " x " << params.cols() << "\n";
            
            // Print first 5 timesteps
            std::cout << "    First 5 nodes:\n";
            for (size_t t = 0; t < std::min(num_timesteps, size_t(5)); ++t) {
                std::cout << "      t=" << t << ": (" 
                          << traj.nodes[t].position(0) << ", " 
                          << traj.nodes[t].position(1) << ")\n";
            }
            
            // Print last 3 timesteps
            std::cout << "    Last 3 nodes:\n";
            for (size_t t = num_timesteps - 3; t < num_timesteps; ++t) {
                std::cout << "      t=" << t << ": (" 
                          << traj.nodes[t].position(0) << ", " 
                          << traj.nodes[t].position(1) << ")\n";
            }
            
            // Check if trajectory is monotonic (should increase or decrease smoothly)
            float dx_sum = 0, dy_sum = 0;
            int sign_changes_x = 0, sign_changes_y = 0;
            float prev_dx = 0, prev_dy = 0;
            for (size_t t = 1; t < num_timesteps; ++t) {
                float dx = traj.nodes[t].position(0) - traj.nodes[t-1].position(0);
                float dy = traj.nodes[t].position(1) - traj.nodes[t-1].position(1);
                dx_sum += dx;
                dy_sum += dy;
                if (t > 1) {
                    if ((dx > 0) != (prev_dx > 0) && std::abs(dx) > 0.01f && std::abs(prev_dx) > 0.01f) sign_changes_x++;
                    if ((dy > 0) != (prev_dy > 0) && std::abs(dy) > 0.01f && std::abs(prev_dy) > 0.01f) sign_changes_y++;
                }
                prev_dx = dx;
                prev_dy = dy;
            }
            std::cout << "    Total displacement: (" << dx_sum << ", " << dy_sum << ")\n";
            std::cout << "    Direction changes: x=" << sign_changes_x << ", y=" << sign_changes_y << "\n";
            if (sign_changes_x > 10 || sign_changes_y > 10) {
                std::cout << "    WARNING: Many direction changes detected - trajectory may be zig-zagging!\n";
            }
        }
        
        return traj;
    }
    
    // Core references
    std::shared_ptr<CollisionAvoidanceTask> collision_task_;
    std::shared_ptr<StompCollisionTask> stomp_task_;
    stomp::Stomp* stomp_;
    
    // Configuration
    StompPlannerConfig config_;
    
    // Optimization state
    float noise_stddev_;
    float noise_decay_;
    std::mt19937 rng_;
    int current_iteration_;
    
    // Results - consistent with PCE naming
    Trajectory current_trajectory_;
    std::vector<Trajectory> trajectory_history_;              // Mean trajectories (like PCE)
    StompOptimizationHistory optimization_history_;           // Full history with samples
    std::vector<Trajectory> current_iteration_samples_;       // Temp storage during iteration
    mutable std::mutex samples_mutex_;                        // Protects current_iteration_samples_
    
    // Smoothing matrix for noise generation (computed from R^-1)
    Eigen::MatrixXd smoothing_matrix_M_;
    
    float final_cost_;
    bool converged_;
    bool initialized_;
};

// ==================== StompCollisionTask method implementations ====================

inline bool StompCollisionTask::generateNoisyParameters(
    const MatrixXd& parameters,
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    int rollout_number,
    MatrixXd& parameters_noise,
    MatrixXd& noise) 
{
    if (!planner_) return false;
    return planner_->taskGenerateNoisyParameters(parameters, num_timesteps, iteration_number, 
                                                  rollout_number, parameters_noise, noise);
}

inline bool StompCollisionTask::computeNoisyCosts(
    const MatrixXd& parameters,
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    int rollout_number,
    VectorXd& costs,
    bool& validity) 
{
    if (!planner_) {
        costs = VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    return planner_->taskComputeCosts(parameters, num_timesteps, rollout_number, costs, validity);
}

inline bool StompCollisionTask::computeCosts(
    const MatrixXd& parameters,
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    VectorXd& costs,
    bool& validity) 
{
    return computeNoisyCosts(parameters, start_timestep, num_timesteps, 
                             iteration_number, -1, costs, validity);
}

inline bool StompCollisionTask::filterNoisyParameters(
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    int rollout_number,
    MatrixXd& parameters,
    bool& filtered) 
{
    filtered = false;
    return true;
}

inline bool StompCollisionTask::filterParameterUpdates(
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    const MatrixXd& parameters,
    MatrixXd& updates) 
{
    if (!planner_) return true;
    return planner_->taskFilterUpdates(updates);
}

inline void StompCollisionTask::postIteration(
    std::size_t start_timestep,
    std::size_t num_timesteps,
    int iteration_number,
    double cost,
    const MatrixXd& parameters) 
{
    if (planner_) {
        planner_->taskPostIteration(iteration_number, cost, parameters);
    }
}

inline void StompCollisionTask::done(
    bool success, 
    int total_iterations, 
    double final_cost,
    const MatrixXd& parameters) 
{
    if (planner_) {
        planner_->taskDone(success, total_iterations, final_cost);
    }
}

} // namespace pce