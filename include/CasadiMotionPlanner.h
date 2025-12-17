/**
 * @file CasadiMotionPlanner.h
 * @brief CasADi-based trajectory optimization using Task interface
 * 
 * This planner is fully task-agnostic like PCEMotionPlanner:
 * - NO knowledge of obstacles
 * - NO knowledge of collision detection
 * - ALL problem-specific logic in Task
 * 
 * Uses CasADi for symbolic gradient computation of smoothness cost,
 * and numerical finite differences for collision cost gradient.
 * Optimization via custom L-BFGS implementation (no CasADi callbacks).
 */
#pragma once

#include "MotionPlanner.h"
#include "Trajectory.h"
#include "task.h"
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>
#include <deque>


/**
 * @brief Configuration for CasADi-based planner
 */
struct CasADiConfig : public MotionPlannerConfig {
    // Optimization parameters
    size_t max_iterations = 500;
    float tolerance = 1e-6f;
    
    // Cost weights
    float collision_weight = 1.0f;
    
    // Solver selection (for naming only - we use L-BFGS internally)
    std::string solver = "lbfgs";
    
    // Numerical gradient settings
    float finite_diff_eps = 1e-4f;
    
    // Verbose output
    bool verbose_solver = true;
    
    // L-BFGS history size
    size_t lbfgs_history = 10;
    
    /**
     * @brief Load CasADi-specific configuration from YAML node
     */
    bool loadFromYAML(const YAML::Node& config) override {
        if (!MotionPlannerConfig::loadFromYAML(config)) {
            return false;
        }
        
        try {
            if (config["casadi_planner"]) {
                const YAML::Node& casadi = config["casadi_planner"];
                
                if (casadi["max_iterations"]) {
                    max_iterations = casadi["max_iterations"].as<size_t>();
                }
                if (casadi["tolerance"]) {
                    tolerance = casadi["tolerance"].as<float>();
                }
                if (casadi["collision_weight"]) {
                    collision_weight = casadi["collision_weight"].as<float>();
                }
                if (casadi["solver"]) {
                    solver = casadi["solver"].as<std::string>();
                }
                if (casadi["finite_diff_eps"]) {
                    finite_diff_eps = casadi["finite_diff_eps"].as<float>();
                }
                if (casadi["verbose_solver"]) {
                    verbose_solver = casadi["verbose_solver"].as<bool>();
                }
                if (casadi["lbfgs_history"]) {
                    lbfgs_history = casadi["lbfgs_history"].as<size_t>();
                }
            }
            
            print();
            return validate();
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading CasADi config: " << e.what() << "\n";
            return false;
        }
    }
    
    bool validate() const override {
        if (!MotionPlannerConfig::validate()) {
            return false;
        }
        
        if (max_iterations == 0) {
            std::cerr << "Error: max_iterations must be > 0\n";
            return false;
        }
        
        if (tolerance <= 0.0f) {
            std::cerr << "Error: tolerance must be positive\n";
            return false;
        }
        
        if (collision_weight < 0.0f) {
            std::cerr << "Error: collision_weight must be non-negative\n";
            return false;
        }
        
        return true;
    }
    
    void print() const override {
        // MotionPlannerConfig::print();
        
        std::cout << "=== CasADi Planner Configuration ===\n";
        std::cout << "Algorithm:              L-BFGS (CasADi symbolic + numerical gradients)\n";
        std::cout << "Max iterations:         " << max_iterations << "\n";
        std::cout << "Tolerance:              " << tolerance << "\n";
        std::cout << "Collision weight:       " << collision_weight << "\n";
        std::cout << "Finite diff epsilon:    " << finite_diff_eps << "\n";
        std::cout << "L-BFGS history:         " << lbfgs_history << "\n";
        std::cout << "Verbose:                " << (verbose_solver ? "yes" : "no") << "\n";
        std::cout << "\n";
    }
};


/**
 * @brief CasADi-based Motion Planner using L-BFGS optimization
 * 
 * Inherits from MotionPlanner and uses Task interface like PCEMotionPlanner.
 * 
 * Uses CasADi for symbolic differentiation of smoothness cost.
 * Uses finite differences for collision cost gradient via Task.
 * Custom L-BFGS implementation avoids CasADi callback lifetime issues.
 * 
 * Solves:
 *   min_{Y}  J_smoothness(Y) + w_c * J_collision(Y)
 *   s.t.     y_0 = start (fixed)
 *            y_N = goal (fixed)
 */
class CasADiMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    
    /**
     * @brief Constructor
     * @param task Shared pointer to task defining the optimization problem
     */
    CasADiMotionPlanner(pce::TaskPtr task = nullptr)
        : task_(task)
    {
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
     * @brief Initialize planner with CasADi configuration
     */
    bool initialize(const CasADiConfig& config) {
        try {
            casadi_config_ = std::make_shared<CasADiConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating casadi_config_: " << e.what() << std::endl;
            return false;
        }
        
        // Extract CasADi-specific parameters
        max_iterations_ = config.max_iterations;
        tolerance_ = config.tolerance;
        collision_weight_ = config.collision_weight;
        solver_name_ = config.solver;
        finite_diff_eps_ = config.finite_diff_eps;
        verbose_solver_ = config.verbose_solver;
        lbfgs_history_ = config.lbfgs_history;
        
        // Call base class initialize
        bool result = MotionPlanner::initialize(config);
        
        // if (result) {
        //     casadi_config_->print();
        // }
        
        return result;
    }
    
    std::string getPlannerName() const override {
        return "CasADi-LBFGS";
    }
    
    // Getters
    size_t getMaxIterations() const { return max_iterations_; }
    float getTolerance() const { return tolerance_; }
    float getCollisionWeight() const { return collision_weight_; }
    std::string getSolverName() const { return solver_name_; }
    
    std::shared_ptr<const CasADiConfig> getCasADiConfig() const {
        return casadi_config_;
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
     * @brief Run the L-BFGS optimization with CasADi symbolic gradients
     */
    bool optimize() override {
        if (!task_) {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }
        
        log("\n--- Starting CasADi L-BFGS Optimization ---\n");
        log("Formulation: min J_smooth(Y) + w_c * J_collision(Y)\n\n");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t N_free = N - 2;  // Exclude fixed start and goal
        const size_t n_vars = N_free * D;
        
        if (N == 0 || current_trajectory_.nodes[0].position.size() != static_cast<long>(D)) {
            std::cerr << "Error: Invalid trajectory!\n";
            return false;
        }
        
        logf("Trajectory: %zu nodes, %zu dimensions", N, D);
        logf("Free variables: %zu waypoints x %zu dims = %zu total", N_free, D, n_vars);
        
        // Clear history
        trajectory_history_.clear();
        
        // Store initial trajectory
        trajectory_history_.push_back(current_trajectory_);
        
        float initial_collision = task_->computeCollisionCostSimple(current_trajectory_);
        float initial_smoothness = computeSmoothnessCost(current_trajectory_);
        logf("Initial - Collision: %.4f, Smoothness: %.4f, Total: %.4f",
             initial_collision, initial_smoothness, 
             initial_collision * collision_weight_ + initial_smoothness);
        
        // === Build CasADi function for smoothness cost and gradient ===
        casadi::Function grad_smooth_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        // === Initialize decision variables from current trajectory ===
        std::vector<double> x(n_vars);
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                x[i * D + d] = current_trajectory_.nodes[i + 1].position(d);
            }
        }
        
        // === L-BFGS optimization ===
        bool success = runLBFGS(x, n_vars, grad_smooth_func);
        
        // === Update trajectory from final solution ===
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                current_trajectory_.nodes[i + 1].position(d) = 
                    static_cast<float>(x[i * D + d]);
            }
        }
        
        // Fix start/goal exactly
        current_trajectory_.nodes[0].position = start_node_.position;
        current_trajectory_.nodes[N - 1].position = goal_node_.position;
        
        // Apply task filtering
        bool filtered = task_->filterTrajectory(current_trajectory_, max_iterations_);
        if (filtered) {
            log("  Trajectory filtered by task\n");
        }
        
        // Store final trajectory
        trajectory_history_.push_back(current_trajectory_);
        
        // === Final Costs ===
        float final_collision = task_->computeCollisionCostSimple(current_trajectory_);
        float final_smoothness = computeSmoothnessCost(current_trajectory_);
        float final_cost = final_collision * collision_weight_ + final_smoothness;
        
        // Notify task
        task_->done(success, max_iterations_, final_cost, current_trajectory_);
        
        log("\n--- CasADi L-BFGS Optimization Complete ---");
        logf("Final Cost: %.4f (Collision: %.4f, Smoothness: %.4f)",
             final_cost, final_collision, final_smoothness);
        log("\nLog saved to: " + getLogFilename());
        
        return success;
    }

protected:
    void initializeTask() override {
        if (!task_) {
            std::cerr << "Warning: No task set for initialization\n";
            return;
        }
        
        task_->initialize(num_dimensions_, start_node_, goal_node_,
                         num_nodes_, total_time_);
        
        std::cout << "Task initialized\n";
    }
    
    void logPlannerSpecificConfig() override {
        log("--- CasADi Planner Parameters ---");
        log("  Algorithm:            L-BFGS (CasADi symbolic + numerical gradients)");
        logf("  Max iterations:       %zu", max_iterations_);
        logf("  Tolerance:            %.6f", tolerance_);
        logf("  Collision weight:     %.4f", collision_weight_);
        logf("  Finite diff epsilon:  %.6f", finite_diff_eps_);
        logf("  L-BFGS history:       %zu", lbfgs_history_);
        log("");
    }

private:
    pce::TaskPtr task_;
    std::shared_ptr<CasADiConfig> casadi_config_;
    
    size_t max_iterations_ = 500;
    float tolerance_ = 1e-6f;
    float collision_weight_ = 1.0f;
    std::string solver_name_ = "lbfgs";
    float finite_diff_eps_ = 1e-4f;
    bool verbose_solver_ = true;
    size_t lbfgs_history_ = 10;
    
    /**
     * @brief Build CasADi function for smoothness cost and its gradient
     */
    casadi::Function buildSmoothnessGradientFunction(size_t N, size_t D, size_t n_vars) {
        using namespace casadi;
        
        // Decision variables
        SX Y = SX::sym("Y", n_vars);
        
        // Build full trajectory (symbolic)
        std::vector<SX> full_traj(N * D);
        
        // Start (fixed)
        for (size_t d = 0; d < D; ++d) {
            full_traj[d] = start_node_.position(d);
        }
        
        // Free waypoints
        size_t N_free = N - 2;
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                full_traj[(i + 1) * D + d] = Y(i * D + d);
            }
        }
        
        // Goal (fixed)
        for (size_t d = 0; d < D; ++d) {
            full_traj[(N - 1) * D + d] = goal_node_.position(d);
        }
        
        // Smoothness cost: sum of squared accelerations
        SX smoothness_cost = 0;
        float dt = total_time_ / (N - 1);
        
        for (size_t i = 1; i < N - 1; ++i) {
            for (size_t d = 0; d < D; ++d) {
                SX y_prev = full_traj[(i - 1) * D + d];
                SX y_curr = full_traj[i * D + d];
                SX y_next = full_traj[(i + 1) * D + d];
                
                // Second-order finite difference (acceleration)
                SX accel = (y_prev - 2 * y_curr + y_next) / (dt * dt);
                smoothness_cost += accel * accel;
            }
        }
        
        // Compute gradient symbolically
        SX grad_smooth = gradient(smoothness_cost, Y);
        
        // Create function: input Y, output [cost, gradient]
        return Function("grad_smooth", {Y}, {smoothness_cost, grad_smooth});
    }
    
    /**
     * @brief Convert flat decision vector to Trajectory
     */
    Trajectory vectorToTrajectory(const std::vector<double>& x_vec) const {
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        
        Trajectory traj;
        traj.nodes.resize(N);
        traj.total_time = total_time_;
        traj.start_index = 0;
        traj.goal_index = N - 1;
        
        // Start node (fixed)
        traj.nodes[0] = start_node_;
        
        // Free waypoints
        for (size_t i = 0; i < N - 2; ++i) {
            traj.nodes[i + 1].position.resize(D);
            traj.nodes[i + 1].radius = start_node_.radius;
            for (size_t d = 0; d < D; ++d) {
                traj.nodes[i + 1].position(d) = 
                    static_cast<float>(x_vec[i * D + d]);
            }
        }
        
        // Goal node (fixed)
        traj.nodes[N - 1] = goal_node_;
        
        return traj;
    }
    
    /**
     * @brief Compute collision cost gradient via finite differences
     */
    void computeCollisionGradient(const std::vector<double>& x, 
                                   std::vector<double>& grad_collision) const {
        const size_t n_vars = x.size();
        grad_collision.resize(n_vars);
        
        for (size_t i = 0; i < n_vars; ++i) {
            std::vector<double> x_plus = x;
            std::vector<double> x_minus = x;
            x_plus[i] += finite_diff_eps_;
            x_minus[i] -= finite_diff_eps_;
            
            Trajectory traj_plus = vectorToTrajectory(x_plus);
            Trajectory traj_minus = vectorToTrajectory(x_minus);
            
            double f_plus = task_->computeCollisionCostSimple(traj_plus);
            double f_minus = task_->computeCollisionCostSimple(traj_minus);
            
            grad_collision[i] = (f_plus - f_minus) / (2.0 * finite_diff_eps_);
        }
    }
    
    /**
     * @brief Evaluate total cost and gradient
     */
    double evaluateCostAndGradient(const std::vector<double>& x,
                                    std::vector<double>& grad,
                                    const casadi::Function& grad_smooth_func,
                                    double& smooth_cost_out,
                                    double& collision_cost_out) {
        const size_t n_vars = x.size();
        
        // Smoothness cost and gradient (symbolic via CasADi)
        casadi::DM x_dm = casadi::DM(x);
        std::vector<casadi::DM> smooth_result = grad_smooth_func(std::vector<casadi::DM>{x_dm});
        smooth_cost_out = static_cast<double>(smooth_result[0].scalar());
        std::vector<double> grad_smooth = smooth_result[1].get_elements();
        
        // Collision cost and gradient (numerical)
        Trajectory current_traj = vectorToTrajectory(x);
        collision_cost_out = task_->computeCollisionCostSimple(current_traj);
        
        std::vector<double> grad_collision;
        computeCollisionGradient(x, grad_collision);
        
        // Combine
        grad.resize(n_vars);
        for (size_t i = 0; i < n_vars; ++i) {
            grad[i] = grad_smooth[i] + collision_weight_ * grad_collision[i];
        }
        
        return smooth_cost_out + collision_weight_ * collision_cost_out;
    }
    
    /**
     * @brief Run L-BFGS optimization
     */
    bool runLBFGS(std::vector<double>& x, size_t n_vars, 
                   const casadi::Function& grad_smooth_func) {
        
        // L-BFGS history
        std::deque<std::vector<double>> s_history;
        std::deque<std::vector<double>> y_history;
        std::deque<double> rho_history;
        
        std::vector<double> grad(n_vars);
        std::vector<double> grad_new(n_vars);
        std::vector<double> x_new(n_vars);
        
        double prev_cost = std::numeric_limits<double>::infinity();
        
        log("\nStarting L-BFGS optimization...\n");
        
        for (size_t iter = 0; iter < max_iterations_; ++iter) {
            // Evaluate cost and gradient
            double smooth_cost, collision_cost;
            double total_cost = evaluateCostAndGradient(x, grad, grad_smooth_func,
                                                         smooth_cost, collision_cost);
            
            // Compute gradient norm
            double grad_norm = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                grad_norm += grad[i] * grad[i];
            }
            grad_norm = std::sqrt(grad_norm);
            
            // Log progress
            if (verbose_solver_ && (iter % 10 == 0 || iter < 5)) {
                logf("Iter %3zu: Cost=%.4f (Smooth=%.4f, Coll=%.4f), |grad|=%.6f",
                     iter, total_cost, smooth_cost, collision_cost, grad_norm);
            }
            
            // Check convergence
            if (grad_norm < tolerance_) {
                logf("Converged at iteration %zu (gradient norm %.2e < %.2e)", 
                     iter, grad_norm, tolerance_);
                return true;
            }
            
            if (iter > 10 && std::abs(prev_cost - total_cost) < tolerance_ * 0.01) {
                logf("Converged at iteration %zu (cost change %.2e < %.2e)", 
                     iter, std::abs(prev_cost - total_cost), tolerance_ * 0.01);
                return true;
            }
            
            // === L-BFGS two-loop recursion ===
            std::vector<double> q = grad;
            std::vector<double> alpha_hist(s_history.size());
            
            // First loop (newest to oldest)
            for (int i = static_cast<int>(s_history.size()) - 1; i >= 0; --i) {
                double dot_sq = 0.0;
                for (size_t j = 0; j < n_vars; ++j) {
                    dot_sq += s_history[i][j] * q[j];
                }
                alpha_hist[i] = rho_history[i] * dot_sq;
                for (size_t j = 0; j < n_vars; ++j) {
                    q[j] -= alpha_hist[i] * y_history[i][j];
                }
            }
            
            // Initial Hessian approximation (scaled identity)
            double gamma = 1.0;
            if (!s_history.empty()) {
                double dot_yy = 0.0, dot_sy = 0.0;
                for (size_t j = 0; j < n_vars; ++j) {
                    dot_yy += y_history.back()[j] * y_history.back()[j];
                    dot_sy += s_history.back()[j] * y_history.back()[j];
                }
                if (dot_yy > 1e-10) {
                    gamma = dot_sy / dot_yy;
                }
            }
            
            std::vector<double> r(n_vars);
            for (size_t j = 0; j < n_vars; ++j) {
                r[j] = gamma * q[j];
            }
            
            // Second loop (oldest to newest)
            for (size_t i = 0; i < s_history.size(); ++i) {
                double dot_yr = 0.0;
                for (size_t j = 0; j < n_vars; ++j) {
                    dot_yr += y_history[i][j] * r[j];
                }
                double beta = rho_history[i] * dot_yr;
                for (size_t j = 0; j < n_vars; ++j) {
                    r[j] += s_history[i][j] * (alpha_hist[i] - beta);
                }
            }
            
            // Search direction
            std::vector<double> direction(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                direction[i] = -r[i];
            }
            
            // === Backtracking line search ===
            double alpha = 1.0;
            const double c1 = 1e-4;
            const double rho_ls = 0.5;
            
            double dir_grad = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                dir_grad += direction[i] * grad[i];
            }
            
            // Skip if not a descent direction
            if (dir_grad >= 0) {
                // Fall back to gradient descent
                for (size_t i = 0; i < n_vars; ++i) {
                    direction[i] = -grad[i];
                }
                dir_grad = -grad_norm * grad_norm;
                alpha = 0.01 / (grad_norm + 1e-10);
            }
            
            bool ls_success = false;
            for (int ls_iter = 0; ls_iter < 20; ++ls_iter) {
                for (size_t i = 0; i < n_vars; ++i) {
                    x_new[i] = x[i] + alpha * direction[i];
                }
                
                // Evaluate new cost (fast version - just cost, no gradient)
                Trajectory traj_new = vectorToTrajectory(x_new);
                double coll_new = task_->computeCollisionCostSimple(traj_new);
                
                casadi::DM x_new_dm = casadi::DM(x_new);
                std::vector<casadi::DM> smooth_new = grad_smooth_func(std::vector<casadi::DM>{x_new_dm});
                double smooth_new_cost = static_cast<double>(smooth_new[0].scalar());
                
                double total_new = smooth_new_cost + collision_weight_ * coll_new;
                
                if (total_new <= total_cost + c1 * alpha * dir_grad) {
                    ls_success = true;
                    break;
                }
                alpha *= rho_ls;
            }
            
            if (!ls_success) {
                // Very small step
                alpha = 1e-8;
                for (size_t i = 0; i < n_vars; ++i) {
                    x_new[i] = x[i] + alpha * direction[i];
                }
            }
            
            // Compute step and gradient difference
            std::vector<double> s(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                s[i] = x_new[i] - x[i];
            }
            
            // Evaluate gradient at new point
            double smooth_new, coll_new;
            evaluateCostAndGradient(x_new, grad_new, grad_smooth_func, smooth_new, coll_new);
            
            std::vector<double> y_vec(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                y_vec[i] = grad_new[i] - grad[i];
            }
            
            // Update L-BFGS history
            double dot_sy = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                dot_sy += s[i] * y_vec[i];
            }
            
            if (dot_sy > 1e-10) {
                if (s_history.size() >= lbfgs_history_) {
                    s_history.pop_front();
                    y_history.pop_front();
                    rho_history.pop_front();
                }
                s_history.push_back(s);
                y_history.push_back(y_vec);
                rho_history.push_back(1.0 / dot_sy);
            }
            
            x = x_new;
            prev_cost = total_cost;
            
            // Store trajectory periodically
            if (iter % 10 == 0) {
                Trajectory iter_traj = vectorToTrajectory(x);
                trajectory_history_.push_back(iter_traj);
            }
        }
        
        log("Reached maximum iterations");
        return true;  // Still return success - we made progress
    }
};
