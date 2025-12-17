/**
 * @file CasADiMotionPlanner.h
 * @brief CasADi-based trajectory optimization using Task interface
 * 
 * This planner is fully task-agnostic like PCEMotionPlanner:
 * - NO knowledge of obstacles
 * - NO knowledge of collision detection
 * - ALL problem-specific logic in Task
 * 
 * Uses CasADi's Callback mechanism to evaluate collision costs through the Task.
 */
#pragma once

#include "MotionPlanner.h"
#include "task.h"
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>


/**
 * @brief Configuration for CasADi-based planner
 */
struct CasADiConfig : public MotionPlannerConfig {
    // CasADi/IPOPT parameters
    size_t max_iterations = 500;
    float tolerance = 1e-6f;
    
    // Cost weights (smoothness uses base class, collision from task)
    float collision_weight = 1.0f;
    
    // Solver selection: "ipopt", "sqpmethod"
    std::string solver = "ipopt";
    
    // Numerical gradient settings (for task callback)
    float finite_diff_eps = 1e-4f;
    
    // Verbose IPOPT output
    bool verbose_solver = true;
    
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
        MotionPlannerConfig::print();
        
        std::cout << "=== CasADi Planner Configuration ===\n";
        std::cout << "Algorithm:              CasADi NLP (Gradient-based)\n";
        std::cout << "Solver:                 " << solver << "\n";
        std::cout << "Max iterations:         " << max_iterations << "\n";
        std::cout << "Tolerance:              " << tolerance << "\n";
        std::cout << "Collision weight:       " << collision_weight << "\n";
        std::cout << "Finite diff epsilon:    " << finite_diff_eps << "\n";
        std::cout << "Verbose solver:         " << (verbose_solver ? "yes" : "no") << "\n";
        std::cout << "\n";
    }
};


/**
 * @brief CasADi-based Motion Planner using NLP formulation
 * 
 * Inherits from MotionPlanner and uses Task interface like PCEMotionPlanner.
 * 
 * Solves:
 *   min_{Y}  J_smoothness(Y) + w_c * J_collision(Y)
 *   s.t.     y_0 = start (fixed)
 *            y_N = goal (fixed)
 * 
 * Where J_collision is evaluated through the Task interface.
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
        
        // Call base class initialize
        bool result = MotionPlanner::initialize(config);
        
        if (result) {
            casadi_config_->print();
        }
        
        return result;
    }
    
    std::string getPlannerName() const override {
        return "CasADi-" + solver_name_;
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
     * @brief Run the CasADi NLP optimization
     */
    bool optimize() override {
        if (!task_) {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }
        
        log("\n--- Starting CasADi NLP Optimization ---\n");
        log("Solver: " + solver_name_ + "\n");
        log("Formulation: min J_smooth(Y) + w_c * J_collision(Y)\n\n");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t N_free = N - 2;  // Exclude fixed start and goal
        const size_t n_vars = N_free * D;
        
        if (N == 0 || current_trajectory_.nodes[0].position.size() != D) {
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
        
        // === Build NLP using numerical approach ===
        // Since Task provides numerical collision cost, we use CasADi's 
        // finite-difference capabilities for gradient computation
        
        using namespace casadi;
        
        // Decision variables
        MX Y = MX::sym("Y", n_vars);
        
        // Build full trajectory for smoothness cost (symbolic)
        std::vector<MX> full_traj(N * D);
        
        // Start (fixed)
        for (size_t d = 0; d < D; ++d) {
            full_traj[d] = start_node_.position(d);
        }
        
        // Free waypoints
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                full_traj[(i + 1) * D + d] = Y(i * D + d);
            }
        }
        
        // Goal (fixed)
        for (size_t d = 0; d < D; ++d) {
            full_traj[(N - 1) * D + d] = goal_node_.position(d);
        }
        
        // === Smoothness Cost (symbolic - same formulation as PCE) ===
        MX smoothness_cost = 0;
        float dt = total_time_ / (N - 1);
        
        for (size_t i = 1; i < N - 1; ++i) {
            for (size_t d = 0; d < D; ++d) {
                MX y_prev = full_traj[(i - 1) * D + d];
                MX y_curr = full_traj[i * D + d];
                MX y_next = full_traj[(i + 1) * D + d];
                
                // Second-order finite difference (acceleration)
                MX accel = (y_prev - 2 * y_curr + y_next) / (dt * dt);
                smoothness_cost += accel * accel;
            }
        }
        
        // === Collision Cost (numerical via callback) ===
        // Create callback function for task collision evaluation
        Function collision_func = createCollisionCallback(N, D);
        MX collision_cost = collision_func(std::vector<MX>{Y})[0];
        
        // === Total Cost ===
        MX total_cost = smoothness_cost + collision_weight_ * collision_cost;
        
        // === Solver Setup ===
        MXDict nlp = {{"x", Y}, {"f", total_cost}};
        
        Dict opts;
        if (solver_name_ == "ipopt") {
            opts["ipopt.max_iter"] = static_cast<int>(max_iterations_);
            opts["ipopt.tol"] = static_cast<double>(tolerance_);
            opts["ipopt.print_level"] = verbose_solver_ ? 5 : 0;
            opts["print_time"] = verbose_solver_;
            opts["ipopt.hessian_approximation"] = "limited-memory";
        }
        
        Function solver;
        try {
            solver = nlpsol("solver", solver_name_, nlp, opts);
        } catch (const std::exception& e) {
            std::cerr << "Error creating solver: " << e.what() << "\n";
            return false;
        }
        
        // === Initial Guess ===
        std::vector<double> x0(n_vars);
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                x0[i * D + d] = current_trajectory_.nodes[i + 1].position(d);
            }
        }
        
        // === Solve ===
        log("\nSolving NLP...\n");
        
        DMDict arg = {{"x0", x0}};
        DMDict result;
        
        try {
            result = solver(arg);
        } catch (const std::exception& e) {
            std::cerr << "Solver failed: " << e.what() << "\n";
            return false;
        }
        
        // === Extract Solution ===
        DM x_opt = result.at("x");
        std::vector<double> x_sol = x_opt.get_elements();
        
        // Update trajectory
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                current_trajectory_.nodes[i + 1].position(d) = 
                    static_cast<float>(x_sol[i * D + d]);
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
        bool success = (final_cost < std::numeric_limits<float>::infinity());
        task_->done(success, max_iterations_, final_cost, current_trajectory_);
        
        log("\n--- CasADi Optimization Complete ---");
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
        log("  Algorithm:            CasADi NLP (Gradient-based)");
        log("  Solver:               " + solver_name_);
        logf("  Max iterations:       %zu", max_iterations_);
        logf("  Tolerance:            %.6f", tolerance_);
        logf("  Collision weight:     %.4f", collision_weight_);
        logf("  Finite diff epsilon:  %.6f", finite_diff_eps_);
        log("");
    }

private:
    pce::TaskPtr task_;
    std::shared_ptr<CasADiConfig> casadi_config_;
    
    size_t max_iterations_ = 500;
    float tolerance_ = 1e-6f;
    float collision_weight_ = 1.0f;
    std::string solver_name_ = "ipopt";
    float finite_diff_eps_ = 1e-4f;
    bool verbose_solver_ = true;
    
    /**
     * @brief Create CasADi callback for Task collision cost evaluation
     * 
     * The callback evaluates collision cost numerically through the Task,
     * with finite-difference gradients for the NLP solver.
     */
    casadi::Function createCollisionCallback(size_t N, size_t D) {
        using namespace casadi;
        
        // Capture necessary data
        pce::TaskPtr task = task_;
        TrajectoryNode start = start_node_;
        TrajectoryNode goal = goal_node_;
        float eps = finite_diff_eps_;
        size_t n_free = (N - 2) * D;
        
        /**
         * @brief Callback class for collision cost evaluation
         */
        class CollisionCostCallback : public Callback {
        public:
            CollisionCostCallback(const std::string& name,
                                  pce::TaskPtr task,
                                  size_t N, size_t D,
                                  const TrajectoryNode& start,
                                  const TrajectoryNode& goal,
                                  float eps)
                : task_(task), N_(N), D_(D)
                , start_(start), goal_(goal), eps_(eps)
            {
                n_free_ = (N_ - 2) * D_;
                construct(name);
            }
            
            // --- Callback interface ---
            casadi_int get_n_in() override { return 1; }
            casadi_int get_n_out() override { return 1; }
            
            Sparsity get_sparsity_in(casadi_int i) override {
                return Sparsity::dense(n_free_, 1);
            }
            
            Sparsity get_sparsity_out(casadi_int i) override {
                return Sparsity::dense(1, 1);
            }
            
            // Evaluate collision cost via task
            std::vector<DM> eval(const std::vector<DM>& arg) const override {
                Trajectory traj = vectorToTrajectory(arg[0]);
                float cost = task_->computeCollisionCostSimple(traj);
                return {DM(static_cast<double>(cost))};
            }
            
            // Enable gradient computation
            bool has_jacobian() const override { return true; }
            
            // Finite-difference Jacobian
            Function get_jacobian(const std::string& name,
                                 const std::vector<std::string>& inames,
                                 const std::vector<std::string>& onames,
                                 const Dict& opts) const override {
                
                // Create Jacobian callback with finite differences
                return JacobianCallback(name + "_jac", task_, N_, D_, 
                                       start_, goal_, eps_, n_free_);
            }
            
        private:
            pce::TaskPtr task_;
            size_t N_, D_, n_free_;
            TrajectoryNode start_, goal_;
            float eps_;
            
            Trajectory vectorToTrajectory(const DM& x) const {
                Trajectory traj;
                traj.nodes.resize(N_);
                traj.nodes[0] = start_;
                
                std::vector<double> x_vec = x.get_elements();
                float dt = (goal_.time - start_.time) / (N_ - 1);
                
                for (size_t i = 0; i < N_ - 2; ++i) {
                    traj.nodes[i + 1].position.resize(D_);
                    for (size_t d = 0; d < D_; ++d) {
                        traj.nodes[i + 1].position(d) = 
                            static_cast<float>(x_vec[i * D_ + d]);
                    }
                    traj.nodes[i + 1].time = start_.time + (i + 1) * dt;
                }
                
                traj.nodes[N_ - 1] = goal_;
                return traj;
            }
            
            /**
             * @brief Jacobian callback using central finite differences
             */
            class JacobianCallback : public Callback {
            public:
                JacobianCallback(const std::string& name,
                                pce::TaskPtr task,
                                size_t N, size_t D,
                                const TrajectoryNode& start,
                                const TrajectoryNode& goal,
                                float eps, size_t n_free)
                    : task_(task), N_(N), D_(D)
                    , start_(start), goal_(goal)
                    , eps_(eps), n_free_(n_free)
                {
                    construct(name);
                }
                
                casadi_int get_n_in() override { return 2; }  // x, f(x)
                casadi_int get_n_out() override { return 1; } // jacobian
                
                Sparsity get_sparsity_in(casadi_int i) override {
                    if (i == 0) return Sparsity::dense(n_free_, 1);
                    return Sparsity::dense(1, 1);
                }
                
                Sparsity get_sparsity_out(casadi_int i) override {
                    return Sparsity::dense(1, n_free_);
                }
                
                std::vector<DM> eval(const std::vector<DM>& arg) const override {
                    std::vector<double> x = arg[0].get_elements();
                    std::vector<double> grad(n_free_);
                    
                    // Central finite differences
                    for (size_t i = 0; i < n_free_; ++i) {
                        std::vector<double> x_plus = x;
                        std::vector<double> x_minus = x;
                        x_plus[i] += eps_;
                        x_minus[i] -= eps_;
                        
                        Trajectory traj_plus = vectorToTrajectory(DM(x_plus));
                        Trajectory traj_minus = vectorToTrajectory(DM(x_minus));
                        
                        float f_plus = task_->computeCollisionCostSimple(traj_plus);
                        float f_minus = task_->computeCollisionCostSimple(traj_minus);
                        
                        grad[i] = (f_plus - f_minus) / (2.0 * eps_);
                    }
                    
                    return {DM(grad).T()};
                }
                
            private:
                pce::TaskPtr task_;
                size_t N_, D_, n_free_;
                TrajectoryNode start_, goal_;
                float eps_;
                
                Trajectory vectorToTrajectory(const DM& x) const {
                    Trajectory traj;
                    traj.nodes.resize(N_);
                    traj.nodes[0] = start_;
                    
                    std::vector<double> x_vec = x.get_elements();
                    float dt = (goal_.time - start_.time) / (N_ - 1);
                    
                    for (size_t i = 0; i < N_ - 2; ++i) {
                        traj.nodes[i + 1].position.resize(D_);
                        for (size_t d = 0; d < D_; ++d) {
                            traj.nodes[i + 1].position(d) = 
                                static_cast<float>(x_vec[i * D_ + d]);
                        }
                        traj.nodes[i + 1].time = start_.time + (i + 1) * dt;
                    }
                    
                    traj.nodes[N_ - 1] = goal_;
                    return traj;
                }
            };
        };
        
        return CollisionCostCallback("collision_cost", task, N, D, start, goal, eps);
    }
};