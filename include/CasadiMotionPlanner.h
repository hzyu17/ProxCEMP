/**
 * @file CasadiMotionPlanner.h
 * @brief CasADi-based trajectory optimization using Task interface
 * 
 * Task-agnostic planner supporting multiple solver backends:
 * L-BFGS, IPOPT (with SCP), SQP (with SCP), Gradient Descent, Adam
 * 
 * Uses R-matrix based smoothness cost consistent with MotionPlanner base class.
 * Uses symbolic collision cost from CasadiCollisionTask for analytical gradients.
 */
#pragma once

#include "MotionPlanner.h"
#include "Trajectory.h"
#include "task.h"
#include "CasadiCollisionTask.h"  // For symbolic collision cost
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>
#include <deque>

enum class CasADiSolverType { LBFGS, IPOPT, SQP, GRADIENT_DESCENT, ADAM };

inline CasADiSolverType stringToSolverType(const std::string& s) {
    if (s == "lbfgs" || s == "LBFGS" || s == "l-bfgs" || s == "L-BFGS") return CasADiSolverType::LBFGS;
    if (s == "ipopt" || s == "IPOPT" || s == "Ipopt") return CasADiSolverType::IPOPT;
    if (s == "sqp" || s == "SQP" || s == "sqpmethod") return CasADiSolverType::SQP;
    if (s == "gradient_descent" || s == "gd" || s == "GD") return CasADiSolverType::GRADIENT_DESCENT;
    if (s == "adam" || s == "Adam" || s == "ADAM") return CasADiSolverType::ADAM;
    std::cerr << "Warning: Unknown solver '" << s << "', defaulting to L-BFGS\n";
    return CasADiSolverType::LBFGS;
}

inline std::string solverTypeToString(CasADiSolverType type) {
    switch (type) {
        case CasADiSolverType::LBFGS: return "L-BFGS";
        case CasADiSolverType::IPOPT: return "IPOPT";
        case CasADiSolverType::SQP: return "SQP";
        case CasADiSolverType::GRADIENT_DESCENT: return "Gradient Descent";
        case CasADiSolverType::ADAM: return "Adam";
    }
    return "Unknown";
}

struct CasADiConfig : public MotionPlannerConfig {
    // Common
    size_t max_iterations = 200;
    float tolerance = 1e-4f;
    float collision_weight = 10.0f;
    float finite_diff_eps = 1e-4f;
    bool verbose_solver = true;
    std::string solver = "lbfgs";
    CasADiSolverType solver_type = CasADiSolverType::LBFGS;
    bool use_symbolic_collision = true;  // NEW: Use symbolic collision gradients
    
    // L-BFGS
    size_t lbfgs_history = 10;
    
    // IPOPT
    std::string ipopt_linear_solver = "mumps";
    std::string ipopt_hessian_approx = "limited-memory";
    int ipopt_print_level = 0;
    double ipopt_max_cpu_time = 60.0;
    double ipopt_tol = 1e-4;
    double ipopt_acceptable_tol = 1e-2;
    int ipopt_acceptable_iter = 10;
    bool ipopt_warm_start = true;
    int ipopt_max_iter = 500;
    
    // Scaling
    bool use_objective_scaling = true;
    double objective_scale = 1e-4;
    
    // SCP
    int scp_max_outer_iter = 100;
    double scp_trust_region_init = 5.0;
    double scp_trust_region_min = 1e-6;
    double scp_trust_region_max = 100.0;
    double scp_trust_expand = 1.5;
    double scp_trust_shrink = 0.5;
    double scp_accept_ratio = 0.1;
    double scp_good_ratio = 0.75;
    int scp_inner_max_iter = 200;
    double scp_convergence_tol = 1e-3;
    double scp_cost_improvement_tol = 1e-4;
    int scp_stall_limit = 10;
    
    // SQP
    std::string sqp_qp_solver = "qrqp";  // Built-in CasADi QP solver (no external deps)
    int sqp_max_iter_ls = 20;
    double sqp_beta = 0.5;
    double sqp_c1 = 1e-4;
    std::string sqp_hessian_approx = "bfgs";
    
    // Gradient Descent
    double gd_learning_rate = 0.01;
    double gd_momentum = 0.9;
    bool gd_use_nesterov = true;
    double gd_lr_decay = 0.999;
    
    // Adam
    double adam_learning_rate = 0.01;
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_epsilon = 1e-8;

private:
    template<typename T>
    static void loadParam(const YAML::Node& node, const std::string& key, T& value) {
        if (node[key]) value = node[key].as<T>();
    }
    
public:
    bool loadFromYAML(const YAML::Node& config) override {
        if (!MotionPlannerConfig::loadFromYAML(config)) return false;
        
        try {
            if (!config["casadi_planner"]) { print(); return validate(); }
            const auto& c = config["casadi_planner"];
            
            loadParam(c, "max_iterations", max_iterations);
            loadParam(c, "tolerance", tolerance);
            loadParam(c, "collision_weight", collision_weight);
            loadParam(c, "finite_diff_eps", finite_diff_eps);
            loadParam(c, "verbose_solver", verbose_solver);
            loadParam(c, "lbfgs_history", lbfgs_history);
            loadParam(c, "use_symbolic_collision", use_symbolic_collision);
            
            if (c["solver"]) {
                solver = c["solver"].as<std::string>();
                solver_type = stringToSolverType(solver);
            }
            
            if (c["lbfgs"]) loadParam(c["lbfgs"], "history_size", lbfgs_history);
            
            if (c["ipopt"]) {
                const auto& ip = c["ipopt"];
                loadParam(ip, "linear_solver", ipopt_linear_solver);
                loadParam(ip, "hessian_approximation", ipopt_hessian_approx);
                loadParam(ip, "print_level", ipopt_print_level);
                loadParam(ip, "max_cpu_time", ipopt_max_cpu_time);
                loadParam(ip, "tol", ipopt_tol);
                loadParam(ip, "acceptable_tol", ipopt_acceptable_tol);
                loadParam(ip, "acceptable_iter", ipopt_acceptable_iter);
                loadParam(ip, "max_iter", ipopt_max_iter);
                if (ip["warm_start_init_point"]) {
                    std::string ws = ip["warm_start_init_point"].as<std::string>();
                    ipopt_warm_start = (ws == "yes" || ws == "true" || ws == "1");
                }
            }
            
            if (c["scaling"]) {
                const auto& sc = c["scaling"];
                loadParam(sc, "use_objective_scaling", use_objective_scaling);
                loadParam(sc, "objective_scale", objective_scale);
            }
            
            if (c["scp"]) {
                const auto& scp = c["scp"];
                loadParam(scp, "max_outer_iter", scp_max_outer_iter);
                loadParam(scp, "trust_region_init", scp_trust_region_init);
                loadParam(scp, "trust_region_min", scp_trust_region_min);
                loadParam(scp, "trust_region_max", scp_trust_region_max);
                loadParam(scp, "trust_expand", scp_trust_expand);
                loadParam(scp, "trust_shrink", scp_trust_shrink);
                loadParam(scp, "accept_ratio", scp_accept_ratio);
                loadParam(scp, "good_ratio", scp_good_ratio);
                loadParam(scp, "inner_max_iter", scp_inner_max_iter);
                loadParam(scp, "convergence_tol", scp_convergence_tol);
                loadParam(scp, "cost_improvement_tol", scp_cost_improvement_tol);
                loadParam(scp, "stall_limit", scp_stall_limit);
            }
            
            if (c["sqp"]) {
                const auto& sqp = c["sqp"];
                loadParam(sqp, "qp_solver", sqp_qp_solver);
                loadParam(sqp, "max_iter_ls", sqp_max_iter_ls);
                loadParam(sqp, "beta", sqp_beta);
                loadParam(sqp, "c1", sqp_c1);
                loadParam(sqp, "hessian_approximation", sqp_hessian_approx);
            }
            
            if (c["gradient_descent"]) {
                const auto& gd = c["gradient_descent"];
                loadParam(gd, "learning_rate", gd_learning_rate);
                loadParam(gd, "momentum", gd_momentum);
                loadParam(gd, "use_nesterov", gd_use_nesterov);
                loadParam(gd, "lr_decay", gd_lr_decay);
            }
            
            if (c["adam"]) {
                const auto& adam = c["adam"];
                loadParam(adam, "learning_rate", adam_learning_rate);
                loadParam(adam, "beta1", adam_beta1);
                loadParam(adam, "beta2", adam_beta2);
                loadParam(adam, "epsilon", adam_epsilon);
            }
            
            print();
            return validate();
        } catch (const std::exception& e) {
            std::cerr << "Error loading CasADi config: " << e.what() << "\n";
            return false;
        }
    }
    
    bool validate() const override {
        if (!MotionPlannerConfig::validate()) return false;
        if (max_iterations == 0) { std::cerr << "Error: max_iterations must be > 0\n"; return false; }
        if (tolerance <= 0.0f) { std::cerr << "Error: tolerance must be positive\n"; return false; }
        if (collision_weight < 0.0f) { std::cerr << "Error: collision_weight must be non-negative\n"; return false; }
        return true;
    }
    
    void print() const override {
        std::cout << "=== CasADi Planner Configuration ===\n"
                  << "Solver: " << solverTypeToString(solver_type) << "\n"
                  << "Max iterations: " << max_iterations << ", Tolerance: " << tolerance << "\n"
                  << "Collision weight: " << collision_weight << ", Finite diff eps: " << finite_diff_eps << "\n"
                  << "Verbose: " << (verbose_solver ? "yes" : "no") << "\n"
                  << "Use symbolic collision: " << (use_symbolic_collision ? "yes" : "no") << "\n";
        
        switch (solver_type) {
            case CasADiSolverType::LBFGS:
                std::cout << "L-BFGS history: " << lbfgs_history << "\n"; break;
            case CasADiSolverType::IPOPT:
                std::cout << "IPOPT - linear_solver: " << ipopt_linear_solver 
                          << ", hessian: " << ipopt_hessian_approx 
                          << ", max_iter: " << ipopt_max_iter << "\n"
                          << "IPOPT - tol: " << ipopt_tol << ", acceptable_tol: " << ipopt_acceptable_tol << "\n"
                          << "SCP - max_outer: " << scp_max_outer_iter 
                          << ", inner_max: " << scp_inner_max_iter
                          << ", trust_init: " << scp_trust_region_init << "\n"
                          << "Scaling: " << (use_objective_scaling ? "yes" : "no")
                          << ", scale: " << objective_scale << "\n"; 
                break;
            case CasADiSolverType::SQP:
                std::cout << "SQP - qp_solver: " << sqp_qp_solver << ", max_ls: " << sqp_max_iter_ls << "\n"; break;
            case CasADiSolverType::GRADIENT_DESCENT:
                std::cout << "GD - lr: " << gd_learning_rate << ", momentum: " << gd_momentum 
                          << ", nesterov: " << (gd_use_nesterov ? "yes" : "no") << "\n"; break;
            case CasADiSolverType::ADAM:
                std::cout << "Adam - lr: " << adam_learning_rate << ", beta1/2: " << adam_beta1 << "/" << adam_beta2 << "\n"; break;
        }
        std::cout << "\n";
    }
};


class CasADiMotionPlanner : public MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    CasADiMotionPlanner(pce::TaskPtr task = nullptr) : task_(task) {
        // Try to cast to CasadiCollisionTask for symbolic collision support
        casadi_task_ = std::dynamic_pointer_cast<pce::CasadiCollisionTask>(task);
    }
    
    void setTask(pce::TaskPtr task) { 
        task_ = task; 
        casadi_task_ = std::dynamic_pointer_cast<pce::CasadiCollisionTask>(task);
    }
    pce::TaskPtr getTask() const { return task_; }
    
    bool initialize(const CasADiConfig& config) {
        try {
            cfg_ = std::make_shared<CasADiConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating config: " << e.what() << std::endl;
            return false;
        }
        
        if (!MotionPlanner::initialize(config)) return false;
        
        // Convert inherited R_matrix_ to CasADi format
        buildCasadiRMatrix();
        
        // Compute initial normalization scale (fixed throughout optimization)
        computeNormalizationScale();
        
        // Build symbolic cost and gradient functions
        buildSymbolicFunctions();
        
        return true;
    }
    
    std::string getPlannerName() const override { return "CasADi-" + solverTypeToString(cfg_->solver_type); }
    
    size_t getMaxIterations() const { return cfg_->max_iterations; }
    float getTolerance() const { return cfg_->tolerance; }
    float getCollisionWeight() const { return cfg_->collision_weight; }
    std::string getSolverName() const { return cfg_->solver; }
    CasADiSolverType getSolverType() const { return cfg_->solver_type; }
    std::shared_ptr<const CasADiConfig> getCasADiConfig() const { return cfg_; }
    
    float computeStateCost(const Trajectory& trajectory) const override {
        if (!task_) {
            std::cerr << "Error: No task set for collision cost computation\n";
            return std::numeric_limits<float>::infinity();
        }
        return task_->computeStateCost(trajectory);
    }
    
    bool optimize() override {
        if (!task_) { std::cerr << "Error: Cannot optimize without a task!\n"; return false; }
        
        log("\n--- Starting CasADi Optimization ---");
        logf("Solver: %s", solverTypeToString(cfg_->solver_type).c_str());
        logf("Using symbolic collision gradients: %s", 
             (cfg_->use_symbolic_collision && casadi_task_) ? "yes" : "no (finite diff)");
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t N_free = N - 2;
        const size_t n_vars = N_free * D;
        
        if (N == 0 || current_trajectory_.nodes[0].position.size() != static_cast<long>(D)) {
            std::cerr << "Error: Invalid trajectory!\n";
            return false;
        }
        
        logf("Trajectory: %zu nodes, %zu dims, %zu free vars", N, D, n_vars);
        logf("Normalization scale: %.4f", norm_scale_);
        
        trajectory_history_.clear();
        trajectory_history_.push_back(current_trajectory_);
        
        float init_coll = task_->computeStateCostSimple(current_trajectory_);
        float init_smooth = computeSmoothnessCost(current_trajectory_);
        logf("Initial - Coll: %.4f, Smooth: %.4f, Total: %.4f", init_coll, init_smooth, 
             init_coll * cfg_->collision_weight + init_smooth);
        
        std::vector<double> x(n_vars);
        for (size_t i = 0; i < N_free; ++i)
            for (size_t d = 0; d < D; ++d)
                x[i * D + d] = current_trajectory_.nodes[i + 1].position(d);
        
        bool success = false;
        switch (cfg_->solver_type) {
            case CasADiSolverType::LBFGS: success = runLBFGS(x, n_vars, N, D); break;
            case CasADiSolverType::IPOPT: success = runIPOPT_SCP(x, n_vars, N, D); break;
            case CasADiSolverType::SQP: success = runSQP_SCP(x, n_vars, N, D); break;
            case CasADiSolverType::GRADIENT_DESCENT: success = runGradientDescent(x, n_vars, N, D); break;
            case CasADiSolverType::ADAM: success = runAdam(x, n_vars, N, D); break;
        }
        
        for (size_t i = 0; i < N_free; ++i)
            for (size_t d = 0; d < D; ++d)
                current_trajectory_.nodes[i + 1].position(d) = static_cast<float>(x[i * D + d]);
        
        current_trajectory_.nodes[0].position = start_node_.position;
        current_trajectory_.nodes[N - 1].position = goal_node_.position;
        
        if (task_->filterTrajectory(current_trajectory_, cfg_->max_iterations))
            log("  Trajectory filtered by task");
        
        trajectory_history_.push_back(current_trajectory_);
        
        float final_coll = task_->computeStateCostSimple(current_trajectory_);
        float final_smooth = computeSmoothnessCost(current_trajectory_);
        float final_cost = final_coll * cfg_->collision_weight + final_smooth;
        
        task_->done(success, cfg_->max_iterations, final_cost, current_trajectory_);
        
        log("\n--- CasADi Optimization Complete ---");
        logf("Final Cost: %.4f (Coll: %.4f, Smooth: %.4f)", final_cost, final_coll, final_smooth);
        
        return success;
    }

protected:
    void initializeTask() override {
        if (!task_) { std::cerr << "Warning: No task set for initialization\n"; return; }
        task_->initialize(num_dimensions_, start_node_, goal_node_, num_nodes_, total_time_);
        std::cout << "Task initialized\n";
    }
    
    void logPlannerSpecificConfig() override {
        log("--- CasADi Planner Parameters ---");
        logf("  Solver: %s, Max iter: %zu, Tol: %.6f", 
             solverTypeToString(cfg_->solver_type).c_str(), cfg_->max_iterations, cfg_->tolerance);
        logf("  Collision weight: %.4f, Finite diff eps: %.6f", cfg_->collision_weight, cfg_->finite_diff_eps);
        logf("  Symbolic collision: %s", (cfg_->use_symbolic_collision && casadi_task_) ? "yes" : "no");
        log("");
    }

private:
    pce::TaskPtr task_;
    std::shared_ptr<pce::CasadiCollisionTask> casadi_task_;  // For symbolic collision cost
    std::shared_ptr<CasADiConfig> cfg_;
    casadi::DM R_matrix_dm_;      // R matrix in CasADi format
    double norm_scale_sq_ = 1.0;  // Normalization scale squared (fixed from initial trajectory)
    double norm_scale_ = 1.0;     // Normalization scale
    
    // Pre-built symbolic functions
    casadi::Function total_cost_grad_func_;  // Total cost + gradient function
    bool symbolic_funcs_built_ = false;
    
    /**
     * @brief Convert inherited R_matrix_ (Eigen sparse) to CasADi DM
     */
    void buildCasadiRMatrix() {
        size_t n = R_matrix_.rows();
        R_matrix_dm_ = casadi::DM::zeros(n, n);
        
        for (int k = 0; k < R_matrix_.outerSize(); ++k) {
            for (SparseMatrixXf::InnerIterator it(R_matrix_, k); it; ++it) {
                R_matrix_dm_(it.row(), it.col()) = static_cast<double>(it.value());
            }
        }
    }
    
    /**
     * @brief Compute normalization scale from initial trajectory
     */
    void computeNormalizationScale() {
        float max_abs = 0.0f;
        for (const auto& node : current_trajectory_.nodes) {
            max_abs = std::max(max_abs, node.position.cwiseAbs().maxCoeff());
        }
        norm_scale_ = static_cast<double>(max_abs);
        norm_scale_sq_ = std::max(static_cast<double>(max_abs * max_abs), 1e-6);
    }
    
    /**
     * @brief Build symbolic cost and gradient functions for the optimization
     * 
     * Creates a CasADi function that computes:
     * - Total cost = smoothness_cost + collision_weight * collision_cost
     * - Gradient of total cost w.r.t. decision variables
     */
    void buildSymbolicFunctions() {
        using namespace casadi;
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t n_vars = (N - 2) * D;
        
        double obj_scale = cfg_->use_objective_scaling ? cfg_->objective_scale : 1.0;
        
        SX Y = SX::sym("Y", n_vars);
        
        // Build smoothness cost symbolically
        SX smooth_cost = buildSmoothnessCostSymbolic(Y, N, D, obj_scale);
        
        // Build collision cost symbolically (if CasadiCollisionTask is available)
        SX coll_cost;
        if (cfg_->use_symbolic_collision && casadi_task_) {
            coll_cost = casadi_task_->buildCollisionCostSymbolic(Y, N, obj_scale);
            log("Built symbolic collision cost function");
        } else {
            // Collision will be computed via finite differences
            coll_cost = SX::zeros(1);
            log("Using finite difference collision gradients");
        }
        
        // Total cost = smoothness + weight * collision
        SX total_cost = smooth_cost + cfg_->collision_weight * coll_cost;
        
        // Compute gradient symbolically
        SX grad = gradient(total_cost, Y);
        
        // Create function
        total_cost_grad_func_ = Function("total_cost_grad", {Y}, {total_cost, grad, smooth_cost, coll_cost},
                                         {"Y"}, {"cost", "grad", "smooth", "coll"});
        
        symbolic_funcs_built_ = true;
        logf("Built symbolic total cost function (n_vars=%zu)", n_vars);
    }
    
    /**
     * @brief Build symbolic smoothness cost using R-matrix (matches MotionPlanner)
     * 
     * Computes: sum_d(Y_d^T * R * Y_d) / norm_scale_sq_
     */
    casadi::SX buildSmoothnessCostSymbolic(const casadi::SX& Y, size_t N, size_t D, double obj_scale = 1.0) {
        using namespace casadi;
        
        SX R_sx = SX(R_matrix_dm_);
        SX cost = 0;
        
        for (size_t d = 0; d < D; ++d) {
            SX Y_d = SX::zeros(N, 1);
            
            // Start (fixed)
            Y_d(0) = static_cast<double>(start_node_.position(d));
            
            // Interior nodes (decision variables)
            for (size_t i = 0; i < N - 2; ++i) {
                Y_d(i + 1) = Y(i * D + d);
            }
            
            // Goal (fixed)
            Y_d(N - 1) = static_cast<double>(goal_node_.position(d));
            
            // Y_d^T * R * Y_d
            SX RY = mtimes(R_sx, Y_d);
            cost += mtimes(Y_d.T(), RY);
        }
        
        // Apply normalization
        cost = cost / norm_scale_sq_;
        
        return cost * obj_scale;
    }
    
    /**
     * @brief Build gradient function for smoothness cost only (fallback)
     */
    casadi::Function buildSmoothnessGradientFunction(size_t N, size_t D, size_t n_vars, double obj_scale = 1.0) {
        using namespace casadi;
        SX Y = SX::sym("Y", n_vars);
        SX cost = buildSmoothnessCostSymbolic(Y, N, D, obj_scale);
        return Function("grad_smooth", {Y}, {cost, gradient(cost, Y)});
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
            for (size_t d = 0; d < D; ++d)
                traj.nodes[i + 1].position(d) = static_cast<float>(x_vec[i * D + d]);
        }
        traj.nodes[N - 1] = goal_node_;
        return traj;
    }
    
    /**
     * @brief Compute collision gradient via finite differences (fallback when symbolic not available)
     */
    void computeCollisionGradientFiniteDiff(const std::vector<double>& x, std::vector<double>& grad) const {
        grad.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> xp = x, xm = x;
            xp[i] += cfg_->finite_diff_eps;
            xm[i] -= cfg_->finite_diff_eps;
            grad[i] = (task_->computeStateCostSimple(vectorToTrajectory(xp)) - 
                       task_->computeStateCostSimple(vectorToTrajectory(xm))) / (2.0 * cfg_->finite_diff_eps);
        }
    }
    
    /**
     * @brief Compute total cost and gradient using pre-built symbolic function
     * 
     * If symbolic collision is available, uses analytical gradients for both.
     * Otherwise, uses symbolic smoothness + finite diff collision.
     */
    void computeTotalCostAndGradient(const std::vector<double>& x, 
                                      double& cost, 
                                      std::vector<double>& grad,
                                      double& smooth_cost,
                                      double& coll_cost) {
        double obj_scale = cfg_->use_objective_scaling ? cfg_->objective_scale : 1.0;
        
        if (cfg_->use_symbolic_collision && casadi_task_ && symbolic_funcs_built_) {
            // Use fully symbolic function (both smoothness and collision)
            auto result = total_cost_grad_func_(std::vector<casadi::DM>{casadi::DM(x)});
            
            cost = static_cast<double>(result[0].scalar()) / obj_scale;
            grad = result[1].get_elements();
            smooth_cost = static_cast<double>(result[2].scalar()) / obj_scale;
            coll_cost = static_cast<double>(result[3].scalar()) / obj_scale;
            
            // Note: gradient is already scaled, but we need to scale it back for consistency
            // Actually the gradient should stay as is since cost is scaled
        } else {
            // Fallback: symbolic smoothness + finite diff collision
            const size_t N = current_trajectory_.nodes.size();
            const size_t D = num_dimensions_;
            const size_t n_vars = (N - 2) * D;
            
            auto smooth_func = buildSmoothnessGradientFunction(N, D, n_vars, obj_scale);
            auto result = smooth_func(std::vector<casadi::DM>{casadi::DM(x)});
            
            smooth_cost = static_cast<double>(result[0].scalar()) / obj_scale;
            auto grad_smooth = result[1].get_elements();
            
            // Collision via finite differences
            coll_cost = task_->computeStateCostSimple(vectorToTrajectory(x));
            std::vector<double> grad_coll;
            computeCollisionGradientFiniteDiff(x, grad_coll);
            
            cost = smooth_cost + cfg_->collision_weight * coll_cost;
            
            grad.resize(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                grad[i] = grad_smooth[i] + cfg_->collision_weight * grad_coll[i] * obj_scale;
            }
        }
    }
    
    /**
     * @brief Compute TRUE total cost (for verification/logging)
     */
    double computeTrueTotalCost(const std::vector<double>& x) const {
        Trajectory traj = vectorToTrajectory(x);
        double smooth = computeSmoothnessCost(traj);
        double coll = task_->computeStateCostSimple(traj);
        return smooth + cfg_->collision_weight * coll;
    }
    
    /**
     * @brief Get smoothness and collision costs
     */
    void computeTrueCosts(const std::vector<double>& x, double& smooth, double& coll) const {
        Trajectory traj = vectorToTrajectory(x);
        smooth = computeSmoothnessCost(traj);
        coll = task_->computeStateCostSimple(traj);
    }
    
    // ==================== Solver Implementations ====================
    
    bool runIPOPT_SCP(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        using namespace casadi;
        log("\nStarting IPOPT with SCP (Trust Region)...\n");
        
        double obj_scale = cfg_->use_objective_scaling ? cfg_->objective_scale : 1.0;
        logf("Using objective scale: %.2e", obj_scale);
        
        std::vector<double> x_k = x;
        double trust = cfg_->scp_trust_region_init;
        double current_cost = computeTrueTotalCost(x_k);
        double best_cost = current_cost;
        std::vector<double> x_best = x_k;
        
        int stall_count = 0;
        int total_inner_iters = 0;
        
        logf("Initial cost: %.2f, Trust radius: %.4f", current_cost, trust);
        
        for (int outer = 0; outer < cfg_->scp_max_outer_iter; ++outer) {
            // Build convex subproblem with BOTH costs
            SX Y = SX::sym("Y", n_vars);
            
            // Smoothness cost (fully symbolic, quadratic - convex)
            SX smooth_cost = buildSmoothnessCostSymbolic(Y, N, D, obj_scale);
            
            // Collision cost - linearized around current point
            double coll_k = task_->computeStateCostSimple(vectorToTrajectory(x_k));
            std::vector<double> grad_coll_k;
            
            if (cfg_->use_symbolic_collision && casadi_task_) {
                // Get collision gradient from symbolic function
                SX Y_temp = SX::sym("Y_temp", n_vars);
                SX coll_sym = casadi_task_->buildCollisionCostSymbolic(Y_temp, N, obj_scale);
                Function coll_grad_func = Function("coll_grad", {Y_temp}, {gradient(coll_sym, Y_temp)});
                auto grad_result = coll_grad_func(std::vector<DM>{DM(x_k)});
                grad_coll_k = grad_result[0].get_elements();
                // Unscale the gradient
                for (auto& g : grad_coll_k) g /= obj_scale;
            } else {
                computeCollisionGradientFiniteDiff(x_k, grad_coll_k);
            }
            
            // Linearized collision: c(x_k) + grad_c^T * (x - x_k)
            SX lin_coll = coll_k * obj_scale;
            for (size_t i = 0; i < n_vars; ++i)
                lin_coll += grad_coll_k[i] * obj_scale * (Y(i) - x_k[i]);
            
            // Total cost = smoothness (quadratic) + weight * linearized collision
            SX cost = smooth_cost + cfg_->collision_weight * lin_coll;
            
            // Trust region bounds
            std::vector<double> lbx(n_vars), ubx(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                lbx[i] = x_k[i] - trust;
                ubx[i] = x_k[i] + trust;
            }
            
            Dict opts;
            opts["ipopt.max_iter"] = cfg_->scp_inner_max_iter;
            opts["ipopt.tol"] = cfg_->ipopt_tol;
            opts["ipopt.acceptable_tol"] = cfg_->ipopt_acceptable_tol;
            opts["ipopt.acceptable_iter"] = cfg_->ipopt_acceptable_iter;
            opts["ipopt.linear_solver"] = cfg_->ipopt_linear_solver;
            opts["ipopt.hessian_approximation"] = cfg_->ipopt_hessian_approx;
            
            if (cfg_->ipopt_warm_start && outer > 0) {
                opts["ipopt.warm_start_init_point"] = "yes";
                opts["ipopt.warm_start_bound_push"] = 1e-9;
                opts["ipopt.warm_start_bound_frac"] = 1e-9;
                opts["ipopt.warm_start_slack_bound_frac"] = 1e-9;
                opts["ipopt.warm_start_slack_bound_push"] = 1e-9;
                opts["ipopt.warm_start_mult_bound_push"] = 1e-9;
                opts["ipopt.mu_init"] = 1e-6;
            }
            
            opts["ipopt.print_level"] = (cfg_->verbose_solver && outer % 10 == 0) ? 3 : 0;
            opts["print_time"] = false;
            
            Function solver;
            try {
                solver = nlpsol("solver", "ipopt", SXDict{{"x", Y}, {"f", cost}}, opts);
            } catch (const std::exception& e) {
                std::cerr << "\n=== IPOPT Solver Error ===\n";
                std::cerr << "Error: " << e.what() << "\n";
                std::cerr << "Make sure IPOPT is properly installed.\n";
                std::cerr << "========================\n\n";
                return false;  // Fail explicitly - don't silently fall back
            }
            
            auto result = solver(DMDict{{"x0", DM(x_k)}, {"lbx", DM(lbx)}, {"ubx", DM(ubx)}});
            auto x_trial = result.at("x").get_elements();
            
            Dict stats = solver.stats();
            int inner_iters = 0;
            if (stats.find("iter_count") != stats.end()) {
                inner_iters = static_cast<int>(stats.at("iter_count"));
                total_inner_iters += inner_iters;
            }
            
            double actual_cost = computeTrueTotalCost(x_trial);
            double pred_cost_scaled = static_cast<double>(result.at("f").scalar());
            double pred_cost = pred_cost_scaled / obj_scale;
            
            double pred_red = current_cost - pred_cost;
            double actual_red = current_cost - actual_cost;
            double ratio = (std::abs(pred_red) > 1e-10) ? actual_red / pred_red : 0.0;
            
            double step = 0;
            for (size_t i = 0; i < n_vars; ++i)
                step = std::max(step, std::abs(x_trial[i] - x_k[i]));
            
            double rel_improvement = (current_cost > 1e-10) ? actual_red / current_cost : 0.0;
            
            if (cfg_->verbose_solver) {
                double smooth_k, coll_k_true;
                computeTrueCosts(x_k, smooth_k, coll_k_true);
                logf("SCP %3d: Cost=%.2f (S=%.4f, C=%.4f), trust=%.2f, step=%.4f, inner=%d",
                     outer, current_cost, smooth_k, coll_k_true, trust, step, inner_iters);
            }
            
            if (ratio >= cfg_->scp_accept_ratio && actual_cost < current_cost) {
                x_k = x_trial;
                current_cost = actual_cost;
                
                // Save trajectory history for visualization (every accepted step)
                trajectory_history_.push_back(vectorToTrajectory(x_k));
                
                if (current_cost < best_cost) {
                    best_cost = current_cost;
                    x_best = x_k;
                    stall_count = 0;
                } else {
                    stall_count++;
                }
                
                if (ratio >= cfg_->scp_good_ratio)
                    trust = std::min(trust * cfg_->scp_trust_expand, cfg_->scp_trust_region_max);
                
                if (rel_improvement < cfg_->scp_cost_improvement_tol && step < cfg_->scp_convergence_tol) {
                    logf("SCP converged at iter %d (rel_impr=%.2e, step=%.2e)", outer, rel_improvement, step);
                    x = x_k;
                    return true;
                }
            } else {
                trust *= cfg_->scp_trust_shrink;
                stall_count++;
                if (cfg_->verbose_solver)
                    log("         Step rejected");
            }
            
            if (stall_count >= cfg_->scp_stall_limit) {
                logf("SCP stalled after %d iterations without improvement", stall_count);
                x = x_best;
                return true;
            }
            
            if (trust < cfg_->scp_trust_region_min) {
                logf("Trust region too small (%.2e), stopping", trust);
                x = x_best;
                return true;
            }
        }
        
        logf("SCP reached max iterations, total inner: %d", total_inner_iters);
        x = x_best;
        return true;
    }
    
    bool runSQP_SCP(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        using namespace casadi;
        log("\nStarting SQP with SCP...\n");
        
        double obj_scale = cfg_->use_objective_scaling ? cfg_->objective_scale : 1.0;
        std::vector<double> x_k = x;
        double trust = cfg_->scp_trust_region_init;
        double current_cost = computeTrueTotalCost(x_k);
        
        for (int outer = 0; outer < cfg_->scp_max_outer_iter; ++outer) {
            double coll_k = task_->computeStateCostSimple(vectorToTrajectory(x_k));
            std::vector<double> grad_coll_k;
            
            if (cfg_->use_symbolic_collision && casadi_task_) {
                SX Y_temp = SX::sym("Y_temp", n_vars);
                SX coll_sym = casadi_task_->buildCollisionCostSymbolic(Y_temp, N, obj_scale);
                Function coll_grad_func = Function("coll_grad", {Y_temp}, {gradient(coll_sym, Y_temp)});
                auto grad_result = coll_grad_func(std::vector<DM>{DM(x_k)});
                grad_coll_k = grad_result[0].get_elements();
                for (auto& g : grad_coll_k) g /= obj_scale;
            } else {
                computeCollisionGradientFiniteDiff(x_k, grad_coll_k);
            }
            
            SX Y = SX::sym("Y", n_vars);
            SX cost = buildSmoothnessCostSymbolic(Y, N, D, obj_scale);
            SX lin_coll = coll_k * obj_scale;
            for (size_t i = 0; i < n_vars; ++i)
                lin_coll += grad_coll_k[i] * obj_scale * (Y(i) - x_k[i]);
            cost += cfg_->collision_weight * lin_coll;
            
            std::vector<double> lbx(n_vars), ubx(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                lbx[i] = x_k[i] - trust;
                ubx[i] = x_k[i] + trust;
            }
            
            Dict opts;
            opts["max_iter"] = cfg_->scp_inner_max_iter;
            opts["tol_pr"] = opts["tol_du"] = cfg_->ipopt_tol;
            opts["qpsol"] = cfg_->sqp_qp_solver;
            opts["print_time"] = false;
            opts["print_iteration"] = cfg_->verbose_solver && (outer % 10 == 0);
            
            // For nlpsol QP solver, set IPOPT as the NLP solver
            if (cfg_->sqp_qp_solver == "nlpsol") {
                opts["qpsol_options.nlpsol"] = "ipopt";
                opts["qpsol_options.nlpsol_options.ipopt.print_level"] = 0;
                opts["qpsol_options.nlpsol_options.print_time"] = false;
            }
            
            Function solver;
            std::string qp_solver_used = cfg_->sqp_qp_solver;
            
            try {
                solver = nlpsol("solver", "sqpmethod", SXDict{{"x", Y}, {"f", cost}}, opts);
            } catch (const std::exception& e) {
                // If primary QP solver fails, try nlpsol fallback (uses IPOPT)
                if (cfg_->sqp_qp_solver != "nlpsol") {
                    log("Primary QP solver failed, trying nlpsol fallback...");
                    try {
                        opts["qpsol"] = "nlpsol";
                        opts["qpsol_options.nlpsol"] = "ipopt";
                        opts["qpsol_options.nlpsol_options.ipopt.print_level"] = 0;
                        opts["qpsol_options.nlpsol_options.print_time"] = false;
                        solver = nlpsol("solver", "sqpmethod", SXDict{{"x", Y}, {"f", cost}}, opts);
                        qp_solver_used = "nlpsol";
                        if (outer == 0) log("Using nlpsol (IPOPT) as QP solver");
                    } catch (const std::exception& e2) {
                        std::cerr << "\n=== SQP Solver Error ===\n";
                        std::cerr << "Primary error: " << e.what() << "\n";
                        std::cerr << "Fallback error: " << e2.what() << "\n";
                        std::cerr << "Available QP solvers: qrqp (built-in), nlpsol\n";
                        std::cerr << "External (require installation): qpoases, osqp, hpipm\n";
                        std::cerr << "========================\n\n";
                        return false;
                    }
                } else {
                    std::cerr << "\n=== SQP Solver Error ===\n";
                    std::cerr << "Error: " << e.what() << "\n";
                    std::cerr << "========================\n\n";
                    return false;
                }
            }
            
            if (outer == 0) logf("Using QP solver: %s", qp_solver_used.c_str());
            
            auto result = solver(DMDict{{"x0", DM(x_k)}, {"lbx", DM(lbx)}, {"ubx", DM(ubx)}});
            auto x_trial = result.at("x").get_elements();
            double pred_cost = static_cast<double>(result.at("f").scalar()) / obj_scale;
            double actual_cost = computeTrueTotalCost(x_trial);
            
            double ratio = 0;
            if (double pred_red = current_cost - pred_cost; std::abs(pred_red) > 1e-10)
                ratio = (current_cost - actual_cost) / pred_red;
            
            double step = 0;
            for (size_t i = 0; i < n_vars; ++i)
                step = std::max(step, std::abs(x_trial[i] - x_k[i]));
            
            if (cfg_->verbose_solver) {
                double smooth_k, coll_k_true;
                computeTrueCosts(x_k, smooth_k, coll_k_true);
                logf("SQP-SCP %3d: Cost=%.2f (S=%.4f, C=%.4f), trust=%.4f", 
                     outer, current_cost, smooth_k, coll_k_true, trust);
            }
            
            if (ratio >= cfg_->scp_accept_ratio && actual_cost < current_cost) {
                x_k = x_trial;
                current_cost = actual_cost;
                if (ratio >= cfg_->scp_good_ratio)
                    trust = std::min(trust * cfg_->scp_trust_expand, cfg_->scp_trust_region_max);
                
                // Save trajectory history for visualization (every accepted step)
                trajectory_history_.push_back(vectorToTrajectory(x_k));
            } else {
                trust *= cfg_->scp_trust_shrink;
            }
            
            if (step < cfg_->scp_convergence_tol) { logf("SQP-SCP converged at iter %d", outer); x = x_k; return true; }
            if (trust < cfg_->scp_trust_region_min) { log("Trust region too small"); x = x_k; return true; }
        }
        
        x = x_k;
        return true;
    }
    
    bool runLBFGS(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        log("\nStarting L-BFGS...\n");
        
        std::deque<std::vector<double>> s_hist, y_hist;
        std::deque<double> rho_hist;
        std::vector<double> grad(n_vars), grad_new(n_vars), x_new(n_vars);
        double prev_cost = std::numeric_limits<double>::infinity();
        double smooth_cost, coll_cost;
        
        for (size_t iter = 0; iter < cfg_->max_iterations; ++iter) {
            // Compute cost and gradient using unified function
            double cost;
            computeTotalCostAndGradient(x, cost, grad, smooth_cost, coll_cost);
            
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm);
            
            if (cfg_->verbose_solver && (iter % 10 == 0 || iter < 5))
                logf("Iter %3zu: Cost=%.4f (S=%.4f, C=%.4f), |g|=%.6f", 
                     iter, smooth_cost + cfg_->collision_weight * coll_cost, smooth_cost, coll_cost, gnorm);
            
            if (gnorm < cfg_->tolerance) { logf("Converged (grad) at iter %zu", iter); return true; }
            if (iter > 10 && std::abs(prev_cost - cost) < cfg_->tolerance * 0.01) {
                logf("Converged (cost) at iter %zu", iter);
                return true;
            }
            
            // Two-loop recursion
            std::vector<double> q = grad, alpha(s_hist.size());
            for (int i = s_hist.size() - 1; i >= 0; --i) {
                double dot = 0;
                for (size_t j = 0; j < n_vars; ++j) dot += s_hist[i][j] * q[j];
                alpha[i] = rho_hist[i] * dot;
                for (size_t j = 0; j < n_vars; ++j) q[j] -= alpha[i] * y_hist[i][j];
            }
            
            double gamma = 1.0;
            if (!s_hist.empty()) {
                double yy = 0, sy = 0;
                for (size_t j = 0; j < n_vars; ++j) {
                    yy += y_hist.back()[j] * y_hist.back()[j];
                    sy += s_hist.back()[j] * y_hist.back()[j];
                }
                if (yy > 1e-10) gamma = sy / yy;
            }
            
            std::vector<double> r(n_vars);
            for (size_t j = 0; j < n_vars; ++j) r[j] = gamma * q[j];
            
            for (size_t i = 0; i < s_hist.size(); ++i) {
                double dot = 0;
                for (size_t j = 0; j < n_vars; ++j) dot += y_hist[i][j] * r[j];
                double beta = rho_hist[i] * dot;
                for (size_t j = 0; j < n_vars; ++j) r[j] += s_hist[i][j] * (alpha[i] - beta);
            }
            
            std::vector<double> dir(n_vars);
            for (size_t i = 0; i < n_vars; ++i) dir[i] = -r[i];
            
            // Line search
            double a = 1.0;
            double dg = 0;
            for (size_t i = 0; i < n_vars; ++i) dg += dir[i] * grad[i];
            
            if (dg >= 0) {
                for (size_t i = 0; i < n_vars; ++i) dir[i] = -grad[i];
                dg = -gnorm * gnorm;
                a = 0.01 / (gnorm + 1e-10);
            }
            
            bool ls_ok = false;
            for (int ls = 0; ls < 20; ++ls) {
                for (size_t i = 0; i < n_vars; ++i) x_new[i] = x[i] + a * dir[i];
                if (computeTrueTotalCost(x_new) <= cost + 1e-4 * a * dg) { ls_ok = true; break; }
                a *= 0.5;
            }
            if (!ls_ok) { a = 1e-8; for (size_t i = 0; i < n_vars; ++i) x_new[i] = x[i] + a * dir[i]; }
            
            std::vector<double> s(n_vars);
            for (size_t i = 0; i < n_vars; ++i) s[i] = x_new[i] - x[i];
            
            // Compute new gradient
            double new_cost, new_smooth, new_coll;
            computeTotalCostAndGradient(x_new, new_cost, grad_new, new_smooth, new_coll);
            
            std::vector<double> y(n_vars);
            double sy = 0;
            for (size_t i = 0; i < n_vars; ++i) {
                y[i] = grad_new[i] - grad[i];
                sy += s[i] * y[i];
            }
            
            if (sy > 1e-10) {
                if (s_hist.size() >= cfg_->lbfgs_history) {
                    s_hist.pop_front(); y_hist.pop_front(); rho_hist.pop_front();
                }
                s_hist.push_back(s);
                y_hist.push_back(y);
                rho_hist.push_back(1.0 / sy);
            }
            
            x = x_new;
            prev_cost = cost;
            if (iter % 10 == 0) trajectory_history_.push_back(vectorToTrajectory(x));
        }
        
        log("L-BFGS reached max iterations");
        return true;
    }
    
    bool runGradientDescent(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        log("\nStarting Gradient Descent...\n");
        
        std::vector<double> vel(n_vars, 0), grad(n_vars);
        double lr = cfg_->gd_learning_rate;
        double smooth_cost, coll_cost;
        
        for (size_t iter = 0; iter < cfg_->max_iterations; ++iter) {
            std::vector<double> x_eval = x;
            if (cfg_->gd_use_nesterov)
                for (size_t i = 0; i < n_vars; ++i) x_eval[i] = x[i] + cfg_->gd_momentum * vel[i];
            
            double cost;
            computeTotalCostAndGradient(x_eval, cost, grad, smooth_cost, coll_cost);
            
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm);
            
            if (cfg_->verbose_solver && (iter % 10 == 0 || iter < 5))
                logf("Iter %3zu: Cost=%.4f (S=%.4f, C=%.4f), |g|=%.6f, lr=%.6f", 
                     iter, smooth_cost + cfg_->collision_weight * coll_cost, smooth_cost, coll_cost, gnorm, lr);
            
            if (gnorm < cfg_->tolerance) { logf("Converged at iter %zu", iter); return true; }
            
            for (size_t i = 0; i < n_vars; ++i) {
                vel[i] = cfg_->gd_momentum * vel[i] - lr * grad[i];
                x[i] += vel[i];
            }
            lr *= cfg_->gd_lr_decay;
            if (iter % 10 == 0) trajectory_history_.push_back(vectorToTrajectory(x));
        }
        
        log("GD reached max iterations");
        return true;
    }
    
    bool runAdam(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        log("\nStarting Adam...\n");
        
        std::vector<double> m(n_vars, 0), v(n_vars, 0), grad(n_vars);
        double smooth_cost, coll_cost;
        
        for (size_t iter = 0; iter < cfg_->max_iterations; ++iter) {
            double cost;
            computeTotalCostAndGradient(x, cost, grad, smooth_cost, coll_cost);
            
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm);
            
            if (cfg_->verbose_solver && (iter % 10 == 0 || iter < 5))
                logf("Iter %3zu: Cost=%.4f (S=%.4f, C=%.4f), |g|=%.6f", 
                     iter, smooth_cost + cfg_->collision_weight * coll_cost, smooth_cost, coll_cost, gnorm);
            
            if (gnorm < cfg_->tolerance) { logf("Converged at iter %zu", iter); return true; }
            
            double t = iter + 1;
            double bc1 = 1 - std::pow(cfg_->adam_beta1, t);
            double bc2 = 1 - std::pow(cfg_->adam_beta2, t);
            
            for (size_t i = 0; i < n_vars; ++i) {
                m[i] = cfg_->adam_beta1 * m[i] + (1 - cfg_->adam_beta1) * grad[i];
                v[i] = cfg_->adam_beta2 * v[i] + (1 - cfg_->adam_beta2) * grad[i] * grad[i];
                x[i] -= cfg_->adam_learning_rate * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + cfg_->adam_epsilon);
            }
            if (iter % 10 == 0) trajectory_history_.push_back(vectorToTrajectory(x));
        }
        
        log("Adam reached max iterations");
        return true;
    }
};