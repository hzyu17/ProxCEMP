/**
 * @file CasadiMotionPlanner.h
 * @brief CasADi-based trajectory optimization using Task interface
 * 
 * This planner is fully task-agnostic like PCEMotionPlanner:
 * - NO knowledge of obstacles
 * - NO knowledge of collision detection
 * - ALL problem-specific logic in Task
 * 
 * Supports multiple solver backends:
 * - L-BFGS (custom implementation)
 * - IPOPT (via CasADi nlpsol with SCP trust-region)
 * - SQP (via CasADi sqpmethod)
 * - Gradient Descent (with momentum/Nesterov)
 * - Adam optimizer
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
 * @brief Supported solver types
 */
enum class CasADiSolverType {
    LBFGS,              // Custom L-BFGS implementation
    IPOPT,              // Interior point optimizer (via CasADi with SCP)
    SQP,                // Sequential Quadratic Programming (via CasADi)
    GRADIENT_DESCENT,   // Gradient descent with momentum
    ADAM                // Adam optimizer
};

/**
 * @brief Convert string to solver type
 */
inline CasADiSolverType stringToSolverType(const std::string& s) {
    if (s == "lbfgs" || s == "LBFGS" || s == "l-bfgs" || s == "L-BFGS") {
        return CasADiSolverType::LBFGS;
    } else if (s == "ipopt" || s == "IPOPT" || s == "Ipopt") {
        return CasADiSolverType::IPOPT;
    } else if (s == "sqp" || s == "SQP" || s == "sqpmethod") {
        return CasADiSolverType::SQP;
    } else if (s == "gradient_descent" || s == "gd" || s == "GD") {
        return CasADiSolverType::GRADIENT_DESCENT;
    } else if (s == "adam" || s == "Adam" || s == "ADAM") {
        return CasADiSolverType::ADAM;
    }
    std::cerr << "Warning: Unknown solver '" << s << "', defaulting to L-BFGS\n";
    return CasADiSolverType::LBFGS;
}

/**
 * @brief Convert solver type to string
 */
inline std::string solverTypeToString(CasADiSolverType type) {
    switch (type) {
        case CasADiSolverType::LBFGS: return "L-BFGS";
        case CasADiSolverType::IPOPT: return "IPOPT";
        case CasADiSolverType::SQP: return "SQP";
        case CasADiSolverType::GRADIENT_DESCENT: return "Gradient Descent";
        case CasADiSolverType::ADAM: return "Adam";
        default: return "Unknown";
    }
}


/**
 * @brief Configuration for CasADi-based planner
 */
struct CasADiConfig : public MotionPlannerConfig {
    // Common optimization parameters
    size_t max_iterations = 200;
    float tolerance = 1e-6f;
    float collision_weight = 1.0f;
    float finite_diff_eps = 1e-4f;
    bool verbose_solver = true;
    
    // Solver selection
    std::string solver = "lbfgs";
    CasADiSolverType solver_type = CasADiSolverType::LBFGS;
    
    // L-BFGS specific
    size_t lbfgs_history = 10;
    
    // IPOPT specific
    std::string ipopt_linear_solver = "mumps";
    std::string ipopt_hessian_approx = "limited-memory";
    int ipopt_print_level = 5;
    double ipopt_max_cpu_time = 60.0;
    double ipopt_acceptable_tol = 1e-4;
    int ipopt_acceptable_iter = 5;
    bool ipopt_warm_start = false;
    
    // SCP (Sequential Convex Programming) parameters for IPOPT/SQP
    int scp_max_outer_iter = 50;           // Max outer SCP iterations
    double scp_trust_region_init = 10.0;   // Initial trust region radius
    double scp_trust_region_min = 1e-4;    // Minimum trust region
    double scp_trust_region_max = 1000.0;  // Maximum trust region
    double scp_trust_expand = 2.0;         // Trust region expansion factor
    double scp_trust_shrink = 0.5;         // Trust region shrink factor
    double scp_accept_ratio = 0.1;         // Minimum ratio for step acceptance
    double scp_good_ratio = 0.75;          // Ratio for trust region expansion
    int scp_inner_max_iter = 50;           // Max iterations per inner solve
    double scp_convergence_tol = 1e-4;     // SCP convergence tolerance
    
    // SQP specific
    std::string sqp_qp_solver = "qpoases";
    int sqp_max_iter_ls = 20;
    double sqp_beta = 0.5;
    double sqp_c1 = 1e-4;
    std::string sqp_hessian_approx = "bfgs";
    
    // Gradient Descent specific
    double gd_learning_rate = 0.01;
    double gd_momentum = 0.9;
    bool gd_use_nesterov = true;
    double gd_lr_decay = 0.999;
    
    // Adam specific
    double adam_learning_rate = 0.01;
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_epsilon = 1e-8;
    
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
                
                // Common parameters
                if (casadi["max_iterations"]) {
                    max_iterations = casadi["max_iterations"].as<size_t>();
                }
                if (casadi["tolerance"]) {
                    tolerance = casadi["tolerance"].as<float>();
                }
                if (casadi["collision_weight"]) {
                    collision_weight = casadi["collision_weight"].as<float>();
                }
                if (casadi["finite_diff_eps"]) {
                    finite_diff_eps = casadi["finite_diff_eps"].as<float>();
                }
                if (casadi["verbose_solver"]) {
                    verbose_solver = casadi["verbose_solver"].as<bool>();
                }
                
                // Solver selection
                if (casadi["solver"]) {
                    solver = casadi["solver"].as<std::string>();
                    solver_type = stringToSolverType(solver);
                }
                
                // L-BFGS parameters
                if (casadi["lbfgs"]) {
                    const YAML::Node& lbfgs = casadi["lbfgs"];
                    if (lbfgs["history_size"]) {
                        lbfgs_history = lbfgs["history_size"].as<size_t>();
                    }
                }
                if (casadi["lbfgs_history"]) {
                    lbfgs_history = casadi["lbfgs_history"].as<size_t>();
                }
                
                // IPOPT parameters
                if (casadi["ipopt"]) {
                    const YAML::Node& ipopt = casadi["ipopt"];
                    if (ipopt["linear_solver"]) {
                        ipopt_linear_solver = ipopt["linear_solver"].as<std::string>();
                    }
                    if (ipopt["hessian_approximation"]) {
                        ipopt_hessian_approx = ipopt["hessian_approximation"].as<std::string>();
                    }
                    if (ipopt["print_level"]) {
                        ipopt_print_level = ipopt["print_level"].as<int>();
                    }
                    if (ipopt["max_cpu_time"]) {
                        ipopt_max_cpu_time = ipopt["max_cpu_time"].as<double>();
                    }
                    if (ipopt["acceptable_tol"]) {
                        ipopt_acceptable_tol = ipopt["acceptable_tol"].as<double>();
                    }
                    if (ipopt["acceptable_iter"]) {
                        ipopt_acceptable_iter = ipopt["acceptable_iter"].as<int>();
                    }
                    if (ipopt["warm_start_init_point"]) {
                        std::string ws = ipopt["warm_start_init_point"].as<std::string>();
                        ipopt_warm_start = (ws == "yes" || ws == "true" || ws == "1");
                    }
                }
                
                // SCP parameters
                if (casadi["scp"]) {
                    const YAML::Node& scp = casadi["scp"];
                    if (scp["max_outer_iter"]) {
                        scp_max_outer_iter = scp["max_outer_iter"].as<int>();
                    }
                    if (scp["trust_region_init"]) {
                        scp_trust_region_init = scp["trust_region_init"].as<double>();
                    }
                    if (scp["trust_region_min"]) {
                        scp_trust_region_min = scp["trust_region_min"].as<double>();
                    }
                    if (scp["trust_region_max"]) {
                        scp_trust_region_max = scp["trust_region_max"].as<double>();
                    }
                    if (scp["trust_expand"]) {
                        scp_trust_expand = scp["trust_expand"].as<double>();
                    }
                    if (scp["trust_shrink"]) {
                        scp_trust_shrink = scp["trust_shrink"].as<double>();
                    }
                    if (scp["accept_ratio"]) {
                        scp_accept_ratio = scp["accept_ratio"].as<double>();
                    }
                    if (scp["good_ratio"]) {
                        scp_good_ratio = scp["good_ratio"].as<double>();
                    }
                    if (scp["inner_max_iter"]) {
                        scp_inner_max_iter = scp["inner_max_iter"].as<int>();
                    }
                    if (scp["convergence_tol"]) {
                        scp_convergence_tol = scp["convergence_tol"].as<double>();
                    }
                }
                
                // SQP parameters
                if (casadi["sqp"]) {
                    const YAML::Node& sqp = casadi["sqp"];
                    if (sqp["qp_solver"]) {
                        sqp_qp_solver = sqp["qp_solver"].as<std::string>();
                    }
                    if (sqp["max_iter_ls"]) {
                        sqp_max_iter_ls = sqp["max_iter_ls"].as<int>();
                    }
                    if (sqp["beta"]) {
                        sqp_beta = sqp["beta"].as<double>();
                    }
                    if (sqp["c1"]) {
                        sqp_c1 = sqp["c1"].as<double>();
                    }
                    if (sqp["hessian_approximation"]) {
                        sqp_hessian_approx = sqp["hessian_approximation"].as<std::string>();
                    }
                }
                
                // Gradient Descent parameters
                if (casadi["gradient_descent"]) {
                    const YAML::Node& gd = casadi["gradient_descent"];
                    if (gd["learning_rate"]) {
                        gd_learning_rate = gd["learning_rate"].as<double>();
                    }
                    if (gd["momentum"]) {
                        gd_momentum = gd["momentum"].as<double>();
                    }
                    if (gd["use_nesterov"]) {
                        gd_use_nesterov = gd["use_nesterov"].as<bool>();
                    }
                    if (gd["lr_decay"]) {
                        gd_lr_decay = gd["lr_decay"].as<double>();
                    }
                }
                
                // Adam parameters
                if (casadi["adam"]) {
                    const YAML::Node& adam = casadi["adam"];
                    if (adam["learning_rate"]) {
                        adam_learning_rate = adam["learning_rate"].as<double>();
                    }
                    if (adam["beta1"]) {
                        adam_beta1 = adam["beta1"].as<double>();
                    }
                    if (adam["beta2"]) {
                        adam_beta2 = adam["beta2"].as<double>();
                    }
                    if (adam["epsilon"]) {
                        adam_epsilon = adam["epsilon"].as<double>();
                    }
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
        std::cout << "=== CasADi Planner Configuration ===\n";
        std::cout << "Solver:                 " << solverTypeToString(solver_type) << "\n";
        std::cout << "Max iterations:         " << max_iterations << "\n";
        std::cout << "Tolerance:              " << tolerance << "\n";
        std::cout << "Collision weight:       " << collision_weight << "\n";
        std::cout << "Finite diff epsilon:    " << finite_diff_eps << "\n";
        std::cout << "Verbose:                " << (verbose_solver ? "yes" : "no") << "\n";
        
        switch (solver_type) {
            case CasADiSolverType::LBFGS:
                std::cout << "--- L-BFGS Parameters ---\n";
                std::cout << "  History size:         " << lbfgs_history << "\n";
                break;
                
            case CasADiSolverType::IPOPT:
                std::cout << "--- IPOPT Parameters ---\n";
                std::cout << "  Linear solver:        " << ipopt_linear_solver << "\n";
                std::cout << "  Hessian approx:       " << ipopt_hessian_approx << "\n";
                std::cout << "  Print level:          " << ipopt_print_level << "\n";
                std::cout << "--- SCP Parameters ---\n";
                std::cout << "  Max outer iter:       " << scp_max_outer_iter << "\n";
                std::cout << "  Trust region init:    " << scp_trust_region_init << "\n";
                std::cout << "  Convergence tol:      " << scp_convergence_tol << "\n";
                break;
                
            case CasADiSolverType::SQP:
                std::cout << "--- SQP Parameters ---\n";
                std::cout << "  QP solver:            " << sqp_qp_solver << "\n";
                std::cout << "  Max LS iterations:    " << sqp_max_iter_ls << "\n";
                break;
                
            case CasADiSolverType::GRADIENT_DESCENT:
                std::cout << "--- Gradient Descent Parameters ---\n";
                std::cout << "  Learning rate:        " << gd_learning_rate << "\n";
                std::cout << "  Momentum:             " << gd_momentum << "\n";
                std::cout << "  Nesterov:             " << (gd_use_nesterov ? "yes" : "no") << "\n";
                break;
                
            case CasADiSolverType::ADAM:
                std::cout << "--- Adam Parameters ---\n";
                std::cout << "  Learning rate:        " << adam_learning_rate << "\n";
                std::cout << "  Beta1/Beta2:          " << adam_beta1 << "/" << adam_beta2 << "\n";
                break;
        }
        std::cout << "\n";
    }
};


/**
 * @brief CasADi-based Motion Planner with Multiple Solver Backends
 */
class CasADiMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    
    CasADiMotionPlanner(pce::TaskPtr task = nullptr)
        : task_(task)
    {
    }
    
    void setTask(pce::TaskPtr task) {
        task_ = task;
    }
    
    pce::TaskPtr getTask() const {
        return task_;
    }
    
    bool initialize(const CasADiConfig& config) {
        try {
            casadi_config_ = std::make_shared<CasADiConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating casadi_config_: " << e.what() << std::endl;
            return false;
        }
        
        // Common parameters
        max_iterations_ = config.max_iterations;
        tolerance_ = config.tolerance;
        collision_weight_ = config.collision_weight;
        solver_name_ = config.solver;
        solver_type_ = config.solver_type;
        finite_diff_eps_ = config.finite_diff_eps;
        verbose_solver_ = config.verbose_solver;
        
        // Solver-specific
        lbfgs_history_ = config.lbfgs_history;
        
        ipopt_linear_solver_ = config.ipopt_linear_solver;
        ipopt_hessian_approx_ = config.ipopt_hessian_approx;
        ipopt_print_level_ = config.ipopt_print_level;
        ipopt_max_cpu_time_ = config.ipopt_max_cpu_time;
        ipopt_acceptable_tol_ = config.ipopt_acceptable_tol;
        ipopt_acceptable_iter_ = config.ipopt_acceptable_iter;
        ipopt_warm_start_ = config.ipopt_warm_start;
        
        // SCP parameters
        scp_max_outer_iter_ = config.scp_max_outer_iter;
        scp_trust_region_init_ = config.scp_trust_region_init;
        scp_trust_region_min_ = config.scp_trust_region_min;
        scp_trust_region_max_ = config.scp_trust_region_max;
        scp_trust_expand_ = config.scp_trust_expand;
        scp_trust_shrink_ = config.scp_trust_shrink;
        scp_accept_ratio_ = config.scp_accept_ratio;
        scp_good_ratio_ = config.scp_good_ratio;
        scp_inner_max_iter_ = config.scp_inner_max_iter;
        scp_convergence_tol_ = config.scp_convergence_tol;
        
        sqp_qp_solver_ = config.sqp_qp_solver;
        sqp_max_iter_ls_ = config.sqp_max_iter_ls;
        sqp_beta_ = config.sqp_beta;
        sqp_c1_ = config.sqp_c1;
        sqp_hessian_approx_ = config.sqp_hessian_approx;
        
        gd_learning_rate_ = config.gd_learning_rate;
        gd_momentum_ = config.gd_momentum;
        gd_use_nesterov_ = config.gd_use_nesterov;
        gd_lr_decay_ = config.gd_lr_decay;
        
        adam_learning_rate_ = config.adam_learning_rate;
        adam_beta1_ = config.adam_beta1;
        adam_beta2_ = config.adam_beta2;
        adam_epsilon_ = config.adam_epsilon;
        
        return MotionPlanner::initialize(config);
    }
    
    std::string getPlannerName() const override {
        return "CasADi-" + solverTypeToString(solver_type_);
    }
    
    size_t getMaxIterations() const { return max_iterations_; }
    float getTolerance() const { return tolerance_; }
    float getCollisionWeight() const { return collision_weight_; }
    std::string getSolverName() const { return solver_name_; }
    CasADiSolverType getSolverType() const { return solver_type_; }
    
    std::shared_ptr<const CasADiConfig> getCasADiConfig() const {
        return casadi_config_;
    }
    
    float computeStateCost(const Trajectory& trajectory) const override {
        if (!task_) {
            std::cerr << "Error: No task set for collision cost computation\n";
            return std::numeric_limits<float>::infinity();
        }
        return task_->computeStateCost(trajectory);
    }
    
    bool optimize() override {
        if (!task_) {
            std::cerr << "Error: Cannot optimize without a task!\n";
            return false;
        }
        
        log("\n--- Starting CasADi Optimization ---");
        logf("Solver: %s", solverTypeToString(solver_type_).c_str());
        log("Formulation: min J_smooth(Y) + w_c * J_state(Y)\n");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t N_free = N - 2;
        const size_t n_vars = N_free * D;
        
        if (N == 0 || current_trajectory_.nodes[0].position.size() != static_cast<long>(D)) {
            std::cerr << "Error: Invalid trajectory!\n";
            return false;
        }
        
        logf("Trajectory: %zu nodes, %zu dimensions", N, D);
        logf("Free variables: %zu waypoints x %zu dims = %zu total", N_free, D, n_vars);
        
        trajectory_history_.clear();
        trajectory_history_.push_back(current_trajectory_);
        
        float initial_collision = task_->computeStateCostSimple(current_trajectory_);
        float initial_smoothness = computeSmoothnessCost(current_trajectory_);
        logf("Initial - Collision: %.4f, Smoothness: %.4f, Total: %.4f",
             initial_collision, initial_smoothness, 
             initial_collision * collision_weight_ + initial_smoothness);
        
        // Initialize decision variables
        std::vector<double> x(n_vars);
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                x[i * D + d] = current_trajectory_.nodes[i + 1].position(d);
            }
        }
        
        // Run selected solver
        bool success = false;
        switch (solver_type_) {
            case CasADiSolverType::LBFGS:
                success = runLBFGS(x, n_vars, N, D);
                break;
            case CasADiSolverType::IPOPT:
                success = runIPOPT_SCP(x, n_vars, N, D);
                break;
            case CasADiSolverType::SQP:
                success = runSQP_SCP(x, n_vars, N, D);
                break;
            case CasADiSolverType::GRADIENT_DESCENT:
                success = runGradientDescent(x, n_vars, N, D);
                break;
            case CasADiSolverType::ADAM:
                success = runAdam(x, n_vars, N, D);
                break;
        }
        
        // Update trajectory
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                current_trajectory_.nodes[i + 1].position(d) = 
                    static_cast<float>(x[i * D + d]);
            }
        }
        
        current_trajectory_.nodes[0].position = start_node_.position;
        current_trajectory_.nodes[N - 1].position = goal_node_.position;
        
        bool filtered = task_->filterTrajectory(current_trajectory_, max_iterations_);
        if (filtered) {
            log("  Trajectory filtered by task");
        }
        
        trajectory_history_.push_back(current_trajectory_);
        
        float final_collision = task_->computeStateCostSimple(current_trajectory_);
        float final_smoothness = computeSmoothnessCost(current_trajectory_);
        float final_cost = final_collision * collision_weight_ + final_smoothness;
        
        task_->done(success, max_iterations_, final_cost, current_trajectory_);
        
        log("\n--- CasADi Optimization Complete ---");
        logf("Final Cost: %.4f (Collision: %.4f, Smoothness: %.4f)",
             final_cost, final_collision, final_smoothness);
        log("Log saved to: " + getLogFilename());
        
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
        logf("  Solver:               %s", solverTypeToString(solver_type_).c_str());
        logf("  Max iterations:       %zu", max_iterations_);
        logf("  Tolerance:            %.6f", tolerance_);
        logf("  Collision weight:     %.4f", collision_weight_);
        logf("  Finite diff epsilon:  %.6f", finite_diff_eps_);
        log("");
    }


    /**
     * @brief Set a custom initial trajectory
     * 
     * @param trajectory The trajectory to use as initial guess
     * @return true if trajectory is valid and was set
     */
    bool setInitialTrajectory(const Trajectory& trajectory) {
        if (trajectory.nodes.size() != num_nodes_) {
            std::cerr << "Error: Trajectory has " << trajectory.nodes.size() 
                    << " nodes, expected " << num_nodes_ << "\n";
            return false;
        }
        
        if (trajectory.nodes[0].position.size() != static_cast<long>(num_dimensions_)) {
            std::cerr << "Error: Trajectory dimension mismatch\n";
            return false;
        }
        
        current_trajectory_ = trajectory;
        
        // Ensure boundary conditions are satisfied
        current_trajectory_.nodes[0].position = start_node_.position;
        current_trajectory_.nodes[num_nodes_ - 1].position = goal_node_.position;
        
        trajectory_history_.push_back(current_trajectory_);
        
        return true;
    }

    /**
     * @brief Get mutable reference to current trajectory (for advanced manipulation)
     */
    Trajectory& getMutableTrajectory() {
        return current_trajectory_;
    }


private:
    pce::TaskPtr task_;
    std::shared_ptr<CasADiConfig> casadi_config_;
    
    // Common parameters
    size_t max_iterations_ = 500;
    float tolerance_ = 1e-6f;
    float collision_weight_ = 1.0f;
    std::string solver_name_ = "lbfgs";
    CasADiSolverType solver_type_ = CasADiSolverType::LBFGS;
    float finite_diff_eps_ = 1e-4f;
    bool verbose_solver_ = true;
    
    // L-BFGS
    size_t lbfgs_history_ = 10;
    
    // IPOPT
    std::string ipopt_linear_solver_ = "mumps";
    std::string ipopt_hessian_approx_ = "limited-memory";
    int ipopt_print_level_ = 5;
    double ipopt_max_cpu_time_ = 60.0;
    double ipopt_acceptable_tol_ = 1e-4;
    int ipopt_acceptable_iter_ = 5;
    bool ipopt_warm_start_ = false;
    
    // SCP
    int scp_max_outer_iter_ = 50;
    double scp_trust_region_init_ = 10.0;
    double scp_trust_region_min_ = 1e-4;
    double scp_trust_region_max_ = 1000.0;
    double scp_trust_expand_ = 2.0;
    double scp_trust_shrink_ = 0.5;
    double scp_accept_ratio_ = 0.1;
    double scp_good_ratio_ = 0.75;
    int scp_inner_max_iter_ = 50;
    double scp_convergence_tol_ = 1e-4;
    
    // SQP
    std::string sqp_qp_solver_ = "qpoases";
    int sqp_max_iter_ls_ = 20;
    double sqp_beta_ = 0.5;
    double sqp_c1_ = 1e-4;
    std::string sqp_hessian_approx_ = "bfgs";
    
    // GD
    double gd_learning_rate_ = 0.01;
    double gd_momentum_ = 0.9;
    bool gd_use_nesterov_ = true;
    double gd_lr_decay_ = 0.999;
    
    // Adam
    double adam_learning_rate_ = 0.01;
    double adam_beta1_ = 0.9;
    double adam_beta2_ = 0.999;
    double adam_epsilon_ = 1e-8;
    
    // ============================================================
    // Helper Functions
    // ============================================================
    
    casadi::SX buildSmoothnessCostSymbolic(const casadi::SX& Y, size_t N, size_t D) {
        using namespace casadi;
        
        std::vector<SX> full_traj(N * D);
        
        for (size_t d = 0; d < D; ++d) {
            full_traj[d] = start_node_.position(d);
        }
        
        size_t N_free = N - 2;
        for (size_t i = 0; i < N_free; ++i) {
            for (size_t d = 0; d < D; ++d) {
                full_traj[(i + 1) * D + d] = Y(i * D + d);
            }
        }
        
        for (size_t d = 0; d < D; ++d) {
            full_traj[(N - 1) * D + d] = goal_node_.position(d);
        }
        
        SX smoothness_cost = 0;
        float dt = total_time_ / (N - 1);
        
        for (size_t i = 1; i < N - 1; ++i) {
            for (size_t d = 0; d < D; ++d) {
                SX y_prev = full_traj[(i - 1) * D + d];
                SX y_curr = full_traj[i * D + d];
                SX y_next = full_traj[(i + 1) * D + d];
                
                SX accel = (y_prev - 2 * y_curr + y_next) / (dt * dt);
                smoothness_cost += accel * accel;
            }
        }
        
        return smoothness_cost;
    }
    
    casadi::Function buildSmoothnessGradientFunction(size_t N, size_t D, size_t n_vars) {
        using namespace casadi;
        
        SX Y = SX::sym("Y", n_vars);
        SX smoothness_cost = buildSmoothnessCostSymbolic(Y, N, D);
        SX grad_smooth = gradient(smoothness_cost, Y);
        
        return Function("grad_smooth", {Y}, {smoothness_cost, grad_smooth});
    }
    
    Trajectory vectorToTrajectory(const std::vector<double>& x_vec) const {
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        
        Trajectory traj;
        traj.nodes.resize(N);
        traj.total_time = total_time_;
        traj.start_index = 0;
        traj.goal_index = N - 1;
        
        traj.nodes[0] = start_node_;
        
        for (size_t i = 0; i < N - 2; ++i) {
            traj.nodes[i + 1].position.resize(D);
            traj.nodes[i + 1].radius = start_node_.radius;
            for (size_t d = 0; d < D; ++d) {
                traj.nodes[i + 1].position(d) = static_cast<float>(x_vec[i * D + d]);
            }
        }
        
        traj.nodes[N - 1] = goal_node_;
        
        return traj;
    }
    
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
            
            double f_plus = task_->computeStateCostSimple(traj_plus);
            double f_minus = task_->computeStateCostSimple(traj_minus);
            
            grad_collision[i] = (f_plus - f_minus) / (2.0 * finite_diff_eps_);
        }
    }
    
    /**
     * @brief Compute actual total cost (for merit function evaluation)
     */
    double computeTotalCost(const std::vector<double>& x,
                            const casadi::Function& smooth_func) {
        casadi::DM x_dm = casadi::DM(x);
        std::vector<casadi::DM> result = smooth_func(std::vector<casadi::DM>{x_dm});
        double smooth_cost = static_cast<double>(result[0].scalar());
        
        Trajectory traj = vectorToTrajectory(x);
        double coll_cost = task_->computeStateCostSimple(traj);
        
        return smooth_cost + collision_weight_ * coll_cost;
    }
    
    double evaluateCostAndGradient(const std::vector<double>& x,
                                    std::vector<double>& grad,
                                    const casadi::Function& grad_smooth_func,
                                    double& smooth_cost_out,
                                    double& collision_cost_out) {
        const size_t n_vars = x.size();
        
        casadi::DM x_dm = casadi::DM(x);
        std::vector<casadi::DM> smooth_result = grad_smooth_func(std::vector<casadi::DM>{x_dm});
        smooth_cost_out = static_cast<double>(smooth_result[0].scalar());
        std::vector<double> grad_smooth = smooth_result[1].get_elements();
        
        Trajectory current_traj = vectorToTrajectory(x);
        collision_cost_out = task_->computeStateCostSimple(current_traj);
        
        std::vector<double> grad_collision;
        computeCollisionGradient(x, grad_collision);
        
        grad.resize(n_vars);
        for (size_t i = 0; i < n_vars; ++i) {
            grad[i] = grad_smooth[i] + collision_weight_ * grad_collision[i];
        }
        
        return smooth_cost_out + collision_weight_ * collision_cost_out;
    }
    
    // ============================================================
    // Solver Implementations
    // ============================================================
    
    /**
     * @brief Run IPOPT with Sequential Convex Programming (SCP) and trust regions
     * 
     * This uses a proper trust-region SCP approach:
     * 1. Linearize collision cost at current point
     * 2. Add trust region constraint ||x - x_k|| <= delta
     * 3. Solve convex subproblem with IPOPT
     * 4. Evaluate actual cost reduction vs predicted
     * 5. Accept/reject step and adjust trust region
     */
    bool runIPOPT_SCP(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        using namespace casadi;
        
        log("\nStarting IPOPT with SCP (Trust Region)...\n");
        
        // Build smoothness cost function
        Function smooth_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        // Current iterate
        std::vector<double> x_k = x;
        double trust_radius = scp_trust_region_init_;
        
        // Evaluate initial cost
        double current_cost = computeTotalCost(x_k, smooth_func);
        logf("Initial cost: %.6f, Trust radius: %.4f", current_cost, trust_radius);
        
        int consecutive_rejects = 0;
        
        for (int outer = 0; outer < scp_max_outer_iter_; ++outer) {
            // Compute collision gradient at current point
            Trajectory traj_k = vectorToTrajectory(x_k);
            double coll_k = task_->computeStateCostSimple(traj_k);
            
            std::vector<double> grad_coll_k;
            computeCollisionGradient(x_k, grad_coll_k);
            
            // Get smoothness cost at current point
            DM x_k_dm = DM(x_k);
            std::vector<DM> smooth_k = smooth_func(std::vector<DM>{x_k_dm});
            double smooth_cost_k = static_cast<double>(smooth_k[0].scalar());
            
            // Build convex subproblem:
            // min  J_smooth(x) + w_c * [c_k + grad_c' * (x - x_k)]
            // s.t. ||x - x_k||_inf <= delta  (box constraints)
            
            SX Y = SX::sym("Y", n_vars);
            SX smoothness = buildSmoothnessCostSymbolic(Y, N, D);
            
            // Linear approximation of collision: c_k + grad' * (x - x_k)
            // = grad'*x + (c_k - grad'*x_k)
            SX linear_coll = coll_k;
            for (size_t i = 0; i < n_vars; ++i) {
                linear_coll += grad_coll_k[i] * (Y(i) - x_k[i]);
            }
            
            SX total_cost = smoothness + collision_weight_ * linear_coll;
            
            // Box constraints for trust region
            std::vector<double> lbx(n_vars), ubx(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                lbx[i] = x_k[i] - trust_radius;
                ubx[i] = x_k[i] + trust_radius;
            }
            
            // IPOPT options
            Dict opts;
            opts["ipopt.max_iter"] = scp_inner_max_iter_;
            opts["ipopt.tol"] = static_cast<double>(tolerance_);
            opts["ipopt.linear_solver"] = ipopt_linear_solver_;
            opts["ipopt.hessian_approximation"] = ipopt_hessian_approx_;
            opts["ipopt.warm_start_init_point"] = "yes";
            opts["ipopt.warm_start_bound_push"] = 1e-9;
            opts["ipopt.warm_start_bound_frac"] = 1e-9;
            opts["ipopt.warm_start_slack_bound_frac"] = 1e-9;
            opts["ipopt.warm_start_slack_bound_push"] = 1e-9;
            opts["ipopt.warm_start_mult_bound_push"] = 1e-9;
            
            if (verbose_solver_ && outer % 5 == 0) {
                opts["ipopt.print_level"] = 3;
            } else {
                opts["ipopt.print_level"] = 0;
            }
            opts["print_time"] = false;
            
            // Create and solve NLP
            SXDict nlp = {{"x", Y}, {"f", total_cost}};
            
            Function solver;
            try {
                solver = nlpsol("solver", "ipopt", nlp, opts);
            } catch (const std::exception& e) {
                std::cerr << "Error creating IPOPT solver: " << e.what() << std::endl;
                return runLBFGS(x, n_vars, N, D);
            }
            
            DMDict result = solver(DMDict{
                {"x0", DM(x_k)},
                {"lbx", DM(lbx)},
                {"ubx", DM(ubx)}
            });
            
            std::vector<double> x_trial = result.at("x").get_elements();
            double predicted_cost = static_cast<double>(result.at("f").scalar());
            
            // Evaluate actual cost at trial point
            double actual_cost = computeTotalCost(x_trial, smooth_func);
            
            // Compute reduction ratio
            double predicted_reduction = current_cost - predicted_cost;
            double actual_reduction = current_cost - actual_cost;
            
            double ratio = 0.0;
            if (std::abs(predicted_reduction) > 1e-10) {
                ratio = actual_reduction / predicted_reduction;
            }
            
            // Compute step norm
            double step_norm = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                double diff = x_trial[i] - x_k[i];
                step_norm = std::max(step_norm, std::abs(diff));
            }
            
            if (verbose_solver_) {
                Trajectory traj_trial = vectorToTrajectory(x_trial);
                double coll_trial = task_->computeStateCostSimple(traj_trial);
                DM x_trial_dm = DM(x_trial);
                std::vector<DM> smooth_trial = smooth_func(std::vector<DM>{x_trial_dm});
                double smooth_trial_cost = static_cast<double>(smooth_trial[0].scalar());
                
                logf("SCP %3d: Cost=%.2f->%.2f (pred=%.2f), ratio=%.3f, delta=%.4f, |step|=%.4f",
                     outer, current_cost, actual_cost, predicted_cost, ratio, trust_radius, step_norm);
                logf("         Smooth=%.2f, Coll=%.2f", smooth_trial_cost, coll_trial);
            }
            
            // Accept or reject step based on ratio
            if (ratio >= scp_accept_ratio_ && actual_cost < current_cost) {
                // Accept step
                x_k = x_trial;
                current_cost = actual_cost;
                consecutive_rejects = 0;
                
                // Expand trust region if good step
                if (ratio >= scp_good_ratio_) {
                    trust_radius = std::min(trust_radius * scp_trust_expand_, scp_trust_region_max_);
                }
            } else {
                // Reject step, shrink trust region
                trust_radius *= scp_trust_shrink_;
                consecutive_rejects++;
                
                if (verbose_solver_) {
                    log("         Step rejected, shrinking trust region");
                }
            }
            
            // Check convergence
            if (step_norm < scp_convergence_tol_ && consecutive_rejects == 0) {
                logf("SCP converged at iteration %d (step norm %.2e < %.2e)", 
                     outer, step_norm, scp_convergence_tol_);
                x = x_k;
                return true;
            }
            
            // Check if trust region too small
            if (trust_radius < scp_trust_region_min_) {
                logf("Trust region too small (%.2e < %.2e), stopping", 
                     trust_radius, scp_trust_region_min_);
                x = x_k;
                return true;
            }
            
            // Store trajectory periodically
            if (outer % 5 == 0) {
                trajectory_history_.push_back(vectorToTrajectory(x_k));
            }
        }
        
        log("SCP reached maximum outer iterations");
        x = x_k;
        return true;
    }
    
    /**
     * @brief Run SQP with SCP approach
     */
    bool runSQP_SCP(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        using namespace casadi;
        
        log("\nStarting SQP with SCP...\n");
        
        Function smooth_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        std::vector<double> x_k = x;
        double trust_radius = scp_trust_region_init_;
        double current_cost = computeTotalCost(x_k, smooth_func);
        
        for (int outer = 0; outer < scp_max_outer_iter_; ++outer) {
            Trajectory traj_k = vectorToTrajectory(x_k);
            double coll_k = task_->computeStateCostSimple(traj_k);
            
            std::vector<double> grad_coll_k;
            computeCollisionGradient(x_k, grad_coll_k);
            
            SX Y = SX::sym("Y", n_vars);
            SX smoothness = buildSmoothnessCostSymbolic(Y, N, D);
            
            SX linear_coll = coll_k;
            for (size_t i = 0; i < n_vars; ++i) {
                linear_coll += grad_coll_k[i] * (Y(i) - x_k[i]);
            }
            
            SX total_cost = smoothness + collision_weight_ * linear_coll;
            
            std::vector<double> lbx(n_vars), ubx(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                lbx[i] = x_k[i] - trust_radius;
                ubx[i] = x_k[i] + trust_radius;
            }
            
            Dict opts;
            opts["max_iter"] = scp_inner_max_iter_;
            opts["tol_pr"] = static_cast<double>(tolerance_);
            opts["tol_du"] = static_cast<double>(tolerance_);
            opts["qpsol"] = sqp_qp_solver_;
            opts["print_time"] = false;
            opts["print_iteration"] = verbose_solver_ && (outer % 5 == 0);
            
            SXDict nlp = {{"x", Y}, {"f", total_cost}};
            
            Function solver;
            try {
                solver = nlpsol("solver", "sqpmethod", nlp, opts);
            } catch (const std::exception& e) {
                std::cerr << "Error creating SQP solver: " << e.what() << std::endl;
                return runLBFGS(x, n_vars, N, D);
            }
            
            DMDict result = solver(DMDict{
                {"x0", DM(x_k)},
                {"lbx", DM(lbx)},
                {"ubx", DM(ubx)}
            });
            
            std::vector<double> x_trial = result.at("x").get_elements();
            double predicted_cost = static_cast<double>(result.at("f").scalar());
            double actual_cost = computeTotalCost(x_trial, smooth_func);
            
            double predicted_reduction = current_cost - predicted_cost;
            double actual_reduction = current_cost - actual_cost;
            double ratio = (std::abs(predicted_reduction) > 1e-10) ? 
                           actual_reduction / predicted_reduction : 0.0;
            
            double step_norm = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                step_norm = std::max(step_norm, std::abs(x_trial[i] - x_k[i]));
            }
            
            if (verbose_solver_) {
                logf("SQP-SCP %3d: Cost=%.2f->%.2f, ratio=%.3f, delta=%.4f",
                     outer, current_cost, actual_cost, ratio, trust_radius);
            }
            
            if (ratio >= scp_accept_ratio_ && actual_cost < current_cost) {
                x_k = x_trial;
                current_cost = actual_cost;
                if (ratio >= scp_good_ratio_) {
                    trust_radius = std::min(trust_radius * scp_trust_expand_, scp_trust_region_max_);
                }
            } else {
                trust_radius *= scp_trust_shrink_;
            }
            
            if (step_norm < scp_convergence_tol_) {
                logf("SQP-SCP converged at iteration %d", outer);
                x = x_k;
                return true;
            }
            
            if (trust_radius < scp_trust_region_min_) {
                log("Trust region too small, stopping");
                x = x_k;
                return true;
            }
        }
        
        x = x_k;
        return true;
    }
    
    /**
     * @brief Run custom L-BFGS optimization
     */
    bool runLBFGS(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        casadi::Function grad_smooth_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        std::deque<std::vector<double>> s_history;
        std::deque<std::vector<double>> y_history;
        std::deque<double> rho_history;
        
        std::vector<double> grad(n_vars);
        std::vector<double> grad_new(n_vars);
        std::vector<double> x_new(n_vars);
        
        double prev_cost = std::numeric_limits<double>::infinity();
        
        log("\nStarting L-BFGS optimization...\n");
        
        for (size_t iter = 0; iter < max_iterations_; ++iter) {
            double smooth_cost, collision_cost;
            double total_cost = evaluateCostAndGradient(x, grad, grad_smooth_func,
                                                         smooth_cost, collision_cost);
            
            double grad_norm = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                grad_norm += grad[i] * grad[i];
            }
            grad_norm = std::sqrt(grad_norm);
            
            if (verbose_solver_ && (iter % 10 == 0 || iter < 5)) {
                logf("Iter %3zu: Cost=%.4f (Smooth=%.4f, Coll=%.4f), |grad|=%.6f",
                     iter, total_cost, smooth_cost, collision_cost, grad_norm);
            }
            
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
            
            // L-BFGS two-loop recursion
            std::vector<double> q = grad;
            std::vector<double> alpha_hist(s_history.size());
            
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
            
            std::vector<double> direction(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                direction[i] = -r[i];
            }
            
            // Backtracking line search
            double alpha = 1.0;
            const double c1 = 1e-4;
            const double rho_ls = 0.5;
            
            double dir_grad = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                dir_grad += direction[i] * grad[i];
            }
            
            if (dir_grad >= 0) {
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
                
                double total_new = computeTotalCost(x_new, grad_smooth_func);
                
                if (total_new <= total_cost + c1 * alpha * dir_grad) {
                    ls_success = true;
                    break;
                }
                alpha *= rho_ls;
            }
            
            if (!ls_success) {
                alpha = 1e-8;
                for (size_t i = 0; i < n_vars; ++i) {
                    x_new[i] = x[i] + alpha * direction[i];
                }
            }
            
            std::vector<double> s(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                s[i] = x_new[i] - x[i];
            }
            
            double smooth_new, coll_new;
            evaluateCostAndGradient(x_new, grad_new, grad_smooth_func, smooth_new, coll_new);
            
            std::vector<double> y_vec(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                y_vec[i] = grad_new[i] - grad[i];
            }
            
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
            
            if (iter % 10 == 0) {
                trajectory_history_.push_back(vectorToTrajectory(x));
            }
        }
        
        log("L-BFGS reached maximum iterations");
        return true;
    }
    
    /**
     * @brief Run Gradient Descent with momentum
     */
    bool runGradientDescent(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        casadi::Function grad_smooth_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        log("\nStarting Gradient Descent...\n");
        
        std::vector<double> velocity(n_vars, 0.0);
        std::vector<double> grad(n_vars);
        double lr = gd_learning_rate_;
        
        for (size_t iter = 0; iter < max_iterations_; ++iter) {
            std::vector<double> x_eval = x;
            
            if (gd_use_nesterov_) {
                for (size_t i = 0; i < n_vars; ++i) {
                    x_eval[i] = x[i] + gd_momentum_ * velocity[i];
                }
            }
            
            double smooth_cost, collision_cost;
            double total_cost = evaluateCostAndGradient(x_eval, grad, grad_smooth_func,
                                                         smooth_cost, collision_cost);
            
            double grad_norm = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                grad_norm += grad[i] * grad[i];
            }
            grad_norm = std::sqrt(grad_norm);
            
            if (verbose_solver_ && (iter % 10 == 0 || iter < 5)) {
                logf("Iter %3zu: Cost=%.4f (Smooth=%.4f, Coll=%.4f), |grad|=%.6f, lr=%.6f",
                     iter, total_cost, smooth_cost, collision_cost, grad_norm, lr);
            }
            
            if (grad_norm < tolerance_) {
                logf("Converged at iteration %zu", iter);
                return true;
            }
            
            for (size_t i = 0; i < n_vars; ++i) {
                velocity[i] = gd_momentum_ * velocity[i] - lr * grad[i];
                x[i] += velocity[i];
            }
            
            lr *= gd_lr_decay_;
            
            if (iter % 10 == 0) {
                trajectory_history_.push_back(vectorToTrajectory(x));
            }
        }
        
        log("Gradient Descent reached maximum iterations");
        return true;
    }
    
    /**
     * @brief Run Adam optimization
     */
    bool runAdam(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        casadi::Function grad_smooth_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        log("\nStarting Adam...\n");
        
        std::vector<double> m(n_vars, 0.0);
        std::vector<double> v(n_vars, 0.0);
        std::vector<double> grad(n_vars);
        
        for (size_t iter = 0; iter < max_iterations_; ++iter) {
            double smooth_cost, collision_cost;
            double total_cost = evaluateCostAndGradient(x, grad, grad_smooth_func,
                                                         smooth_cost, collision_cost);
            
            double grad_norm = 0.0;
            for (size_t i = 0; i < n_vars; ++i) {
                grad_norm += grad[i] * grad[i];
            }
            grad_norm = std::sqrt(grad_norm);
            
            if (verbose_solver_ && (iter % 10 == 0 || iter < 5)) {
                logf("Iter %3zu: Cost=%.4f (Smooth=%.4f, Coll=%.4f), |grad|=%.6f",
                     iter, total_cost, smooth_cost, collision_cost, grad_norm);
            }
            
            if (grad_norm < tolerance_) {
                logf("Converged at iteration %zu", iter);
                return true;
            }
            
            double t = static_cast<double>(iter + 1);
            double bias_correction1 = 1.0 - std::pow(adam_beta1_, t);
            double bias_correction2 = 1.0 - std::pow(adam_beta2_, t);
            
            for (size_t i = 0; i < n_vars; ++i) {
                m[i] = adam_beta1_ * m[i] + (1.0 - adam_beta1_) * grad[i];
                v[i] = adam_beta2_ * v[i] + (1.0 - adam_beta2_) * grad[i] * grad[i];
                
                double m_hat = m[i] / bias_correction1;
                double v_hat = v[i] / bias_correction2;
                
                x[i] -= adam_learning_rate_ * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
            }
            
            if (iter % 10 == 0) {
                trajectory_history_.push_back(vectorToTrajectory(x));
            }
        }
        
        log("Adam reached maximum iterations");
        return true;
    }
};
