/**
 * @file CasadiMotionPlanner.h
 * @brief CasADi-based trajectory optimization using Task interface
 * 
 * Task-agnostic planner supporting multiple solver backends:
 * L-BFGS, IPOPT (with SCP), SQP (with SCP), Gradient Descent, Adam
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
    float tolerance = 1e-4f;          // Relaxed from 1e-6
    float collision_weight = 10.0f;
    float finite_diff_eps = 1e-4f;
    bool verbose_solver = true;
    std::string solver = "lbfgs";
    CasADiSolverType solver_type = CasADiSolverType::LBFGS;
    
    // L-BFGS
    size_t lbfgs_history = 10;
    
    // IPOPT - improved defaults
    std::string ipopt_linear_solver = "mumps";
    std::string ipopt_hessian_approx = "limited-memory";
    int ipopt_print_level = 0;           // Reduce verbosity (was 5)
    double ipopt_max_cpu_time = 60.0;
    double ipopt_tol = 1e-4;             // Main convergence tolerance
    double ipopt_acceptable_tol = 1e-2;  // Relaxed acceptable tolerance
    int ipopt_acceptable_iter = 10;      // Accept after 10 acceptable iterations
    bool ipopt_warm_start = true;        // Enable warm starting
    int ipopt_max_iter = 500;            // Increased from 100
    
    // Scaling - NEW
    bool use_objective_scaling = true;
    double objective_scale = 1e-4;       // Scale down large objectives
    
    // SCP - improved defaults
    int scp_max_outer_iter = 100;        // Increased from 50
    double scp_trust_region_init = 5.0;  // Smaller initial trust (was 10)
    double scp_trust_region_min = 1e-6;  // Smaller minimum (was 1e-4)
    double scp_trust_region_max = 100.0; // Smaller maximum (was 1000)
    double scp_trust_expand = 1.5;       // More conservative expansion (was 2.0)
    double scp_trust_shrink = 0.5;
    double scp_accept_ratio = 0.1;
    double scp_good_ratio = 0.75;
    int scp_inner_max_iter = 200;        // Increased from 50
    double scp_convergence_tol = 1e-3;   // Relaxed from 1e-4
    double scp_cost_improvement_tol = 1e-4; // NEW: minimum relative cost improvement
    int scp_stall_limit = 10;            // NEW: stop after N iterations without improvement
    
    // SQP
    std::string sqp_qp_solver = "qpoases";
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
                  << "Verbose: " << (verbose_solver ? "yes" : "no") << "\n";
        
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
    
    CasADiMotionPlanner(pce::TaskPtr task = nullptr) : task_(task) {}
    
    void setTask(pce::TaskPtr task) { task_ = task; }
    pce::TaskPtr getTask() const { return task_; }
    
    bool initialize(const CasADiConfig& config) {
        try {
            cfg_ = std::make_shared<CasADiConfig>(config);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception creating config: " << e.what() << std::endl;
            return false;
        }
        return MotionPlanner::initialize(config);
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
        
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        const size_t N_free = N - 2;
        const size_t n_vars = N_free * D;
        
        if (N == 0 || current_trajectory_.nodes[0].position.size() != static_cast<long>(D)) {
            std::cerr << "Error: Invalid trajectory!\n";
            return false;
        }
        
        logf("Trajectory: %zu nodes, %zu dims, %zu free vars", N, D, n_vars);
        
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
        log("");
    }

    bool setInitialTrajectory(const Trajectory& trajectory) {
        if (trajectory.nodes.size() != num_nodes_) {
            std::cerr << "Error: Trajectory node count mismatch\n";
            return false;
        }
        if (trajectory.nodes[0].position.size() != static_cast<long>(num_dimensions_)) {
            std::cerr << "Error: Trajectory dimension mismatch\n";
            return false;
        }
        current_trajectory_ = trajectory;
        current_trajectory_.nodes[0].position = start_node_.position;
        current_trajectory_.nodes[num_nodes_ - 1].position = goal_node_.position;
        trajectory_history_.push_back(current_trajectory_);
        return true;
    }

    Trajectory& getMutableTrajectory() { return current_trajectory_; }

private:
    pce::TaskPtr task_;
    std::shared_ptr<CasADiConfig> cfg_;
    
    // Helper functions
    casadi::SX buildSmoothnessCostSymbolic(const casadi::SX& Y, size_t N, size_t D, double scale = 1.0) {
        using namespace casadi;
        std::vector<SX> traj(N * D);
        
        for (size_t d = 0; d < D; ++d) traj[d] = start_node_.position(d);
        for (size_t i = 0; i < N - 2; ++i)
            for (size_t d = 0; d < D; ++d)
                traj[(i + 1) * D + d] = Y(i * D + d);
        for (size_t d = 0; d < D; ++d) traj[(N - 1) * D + d] = goal_node_.position(d);
        
        SX cost = 0;
        float dt = total_time_ / (N - 1);
        for (size_t i = 1; i < N - 1; ++i)
            for (size_t d = 0; d < D; ++d) {
                SX accel = (traj[(i-1)*D+d] - 2*traj[i*D+d] + traj[(i+1)*D+d]) / (dt*dt);
                cost += accel * accel;
            }
        return cost * scale;
    }
    
    casadi::Function buildSmoothnessGradientFunction(size_t N, size_t D, size_t n_vars, double scale = 1.0) {
        using namespace casadi;
        SX Y = SX::sym("Y", n_vars);
        SX cost = buildSmoothnessCostSymbolic(Y, N, D, scale);
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
    
    void computeCollisionGradient(const std::vector<double>& x, std::vector<double>& grad) const {
        grad.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> xp = x, xm = x;
            xp[i] += cfg_->finite_diff_eps;
            xm[i] -= cfg_->finite_diff_eps;
            grad[i] = (task_->computeStateCostSimple(vectorToTrajectory(xp)) - 
                       task_->computeStateCostSimple(vectorToTrajectory(xm))) / (2.0 * cfg_->finite_diff_eps);
        }
    }
    
    double computeTotalCost(const std::vector<double>& x, const casadi::Function& smooth_func, double scale = 1.0) {
        auto result = smooth_func(std::vector<casadi::DM>{casadi::DM(x)});
        double smooth = static_cast<double>(result[0].scalar()) / scale; // Unscale for true cost
        return smooth + cfg_->collision_weight * task_->computeStateCostSimple(vectorToTrajectory(x));
    }
    
    double evaluateCostAndGradient(const std::vector<double>& x, std::vector<double>& grad,
                                   const casadi::Function& grad_func, double& smooth, double& coll,
                                   double scale = 1.0) {
        auto result = grad_func(std::vector<casadi::DM>{casadi::DM(x)});
        smooth = static_cast<double>(result[0].scalar()) / scale; // Unscale
        auto grad_smooth = result[1].get_elements();
        
        coll = task_->computeStateCostSimple(vectorToTrajectory(x));
        
        std::vector<double> grad_coll;
        computeCollisionGradient(x, grad_coll);
        
        grad.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            grad[i] = grad_smooth[i] + cfg_->collision_weight * grad_coll[i] * scale;
        
        return smooth + cfg_->collision_weight * coll;
    }
    
    // Solver implementations
    bool runIPOPT_SCP(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        using namespace casadi;
        log("\nStarting IPOPT with SCP (Trust Region)...\n");
        
        // Compute scaling factor based on initial cost
        Trajectory init_traj = vectorToTrajectory(x);
        double init_coll = task_->computeStateCostSimple(init_traj);
        double scale = cfg_->use_objective_scaling ? cfg_->objective_scale : 1.0;
        
        logf("Using objective scale: %.2e", scale);
        
        Function smooth_func = buildSmoothnessGradientFunction(N, D, n_vars, scale);
        std::vector<double> x_k = x;
        double trust = cfg_->scp_trust_region_init;
        double current_cost = computeTotalCost(x_k, smooth_func, scale);
        double best_cost = current_cost;
        std::vector<double> x_best = x_k;
        
        int stall_count = 0;
        int total_inner_iters = 0;
        
        logf("Initial cost: %.2f, Trust radius: %.4f", current_cost, trust);
        
        for (int outer = 0; outer < cfg_->scp_max_outer_iter; ++outer) {
            // Compute collision cost and gradient at current point
            double coll_k = task_->computeStateCostSimple(vectorToTrajectory(x_k));
            std::vector<double> grad_coll_k;
            computeCollisionGradient(x_k, grad_coll_k);
            
            // Build convex subproblem with scaled objective
            SX Y = SX::sym("Y", n_vars);
            SX cost = buildSmoothnessCostSymbolic(Y, N, D, scale);
            
            // Linearized collision (scaled)
            SX lin_coll = coll_k * scale;
            for (size_t i = 0; i < n_vars; ++i)
                lin_coll += grad_coll_k[i] * scale * (Y(i) - x_k[i]);
            cost += cfg_->collision_weight * lin_coll;
            
            // Trust region bounds
            std::vector<double> lbx(n_vars), ubx(n_vars);
            for (size_t i = 0; i < n_vars; ++i) {
                lbx[i] = x_k[i] - trust;
                ubx[i] = x_k[i] + trust;
            }
            
            // IPOPT options - tuned for better convergence
            Dict opts;
            opts["ipopt.max_iter"] = cfg_->scp_inner_max_iter;
            opts["ipopt.tol"] = cfg_->ipopt_tol;
            opts["ipopt.acceptable_tol"] = cfg_->ipopt_acceptable_tol;
            opts["ipopt.acceptable_iter"] = cfg_->ipopt_acceptable_iter;
            opts["ipopt.linear_solver"] = cfg_->ipopt_linear_solver;
            opts["ipopt.hessian_approximation"] = cfg_->ipopt_hessian_approx;
            
            // Warm start settings
            if (cfg_->ipopt_warm_start && outer > 0) {
                opts["ipopt.warm_start_init_point"] = "yes";
                opts["ipopt.warm_start_bound_push"] = 1e-9;
                opts["ipopt.warm_start_bound_frac"] = 1e-9;
                opts["ipopt.warm_start_slack_bound_frac"] = 1e-9;
                opts["ipopt.warm_start_slack_bound_push"] = 1e-9;
                opts["ipopt.warm_start_mult_bound_push"] = 1e-9;
                opts["ipopt.mu_init"] = 1e-6;
            }
            
            // Verbosity control
            opts["ipopt.print_level"] = (cfg_->verbose_solver && outer % 10 == 0) ? 3 : 0;
            opts["print_time"] = false;
            
            // Create and solve
            Function solver;
            try {
                solver = nlpsol("solver", "ipopt", SXDict{{"x", Y}, {"f", cost}}, opts);
            } catch (const std::exception& e) {
                std::cerr << "IPOPT error: " << e.what() << ", falling back to L-BFGS\n";
                return runLBFGS(x, n_vars, N, D);
            }
            
            auto result = solver(DMDict{{"x0", DM(x_k)}, {"lbx", DM(lbx)}, {"ubx", DM(ubx)}});
            auto x_trial = result.at("x").get_elements();
            
            // Get solver stats
            Dict stats = solver.stats();
            int inner_iters = 0;
            if (stats.find("iter_count") != stats.end()) {
                inner_iters = static_cast<int>(stats.at("iter_count"));
                total_inner_iters += inner_iters;
            }
            
            // Evaluate actual cost (unscaled)
            double actual_cost = computeTotalCost(x_trial, smooth_func, scale);
            double pred_cost = static_cast<double>(result.at("f").scalar()) / scale; // Unscale
            
            double pred_red = current_cost - pred_cost;
            double actual_red = current_cost - actual_cost;
            double ratio = (std::abs(pred_red) > 1e-10) ? actual_red / pred_red : 0.0;
            
            // Step size
            double step = 0;
            for (size_t i = 0; i < n_vars; ++i)
                step = std::max(step, std::abs(x_trial[i] - x_k[i]));
            
            // Relative improvement
            double rel_improvement = (current_cost > 1e-10) ? actual_red / current_cost : 0.0;
            
            if (cfg_->verbose_solver) {
                logf("SCP %3d: %.2f->%.2f (pred %.2f), ratio=%.3f, trust=%.2f, step=%.4f, inner=%d",
                     outer, current_cost, actual_cost, pred_cost, ratio, trust, step, inner_iters);
            }
            
            // Accept or reject
            if (ratio >= cfg_->scp_accept_ratio && actual_cost < current_cost) {
                x_k = x_trial;
                double old_cost = current_cost;
                current_cost = actual_cost;
                
                // Track best solution
                if (current_cost < best_cost) {
                    best_cost = current_cost;
                    x_best = x_k;
                    stall_count = 0;
                } else {
                    stall_count++;
                }
                
                // Expand trust region if good step
                if (ratio >= cfg_->scp_good_ratio)
                    trust = std::min(trust * cfg_->scp_trust_expand, cfg_->scp_trust_region_max);
                
                // Check for convergence
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
            
            // Check stall
            if (stall_count >= cfg_->scp_stall_limit) {
                logf("SCP stalled after %d iterations without improvement", stall_count);
                x = x_best;
                return true;
            }
            
            // Check trust region
            if (trust < cfg_->scp_trust_region_min) {
                logf("Trust region too small (%.2e), stopping", trust);
                x = x_best;
                return true;
            }
            
            // Store trajectory periodically
            if (outer % 10 == 0)
                trajectory_history_.push_back(vectorToTrajectory(x_k));
        }
        
        logf("SCP reached max iterations, total inner: %d", total_inner_iters);
        x = x_best;
        return true;
    }
    
    bool runSQP_SCP(std::vector<double>& x, size_t n_vars, size_t N, size_t D) {
        using namespace casadi;
        log("\nStarting SQP with SCP...\n");
        
        double scale = cfg_->use_objective_scaling ? cfg_->objective_scale : 1.0;
        Function smooth_func = buildSmoothnessGradientFunction(N, D, n_vars, scale);
        std::vector<double> x_k = x;
        double trust = cfg_->scp_trust_region_init;
        double current_cost = computeTotalCost(x_k, smooth_func, scale);
        
        for (int outer = 0; outer < cfg_->scp_max_outer_iter; ++outer) {
            double coll_k = task_->computeStateCostSimple(vectorToTrajectory(x_k));
            std::vector<double> grad_coll_k;
            computeCollisionGradient(x_k, grad_coll_k);
            
            SX Y = SX::sym("Y", n_vars);
            SX cost = buildSmoothnessCostSymbolic(Y, N, D, scale);
            SX lin_coll = coll_k * scale;
            for (size_t i = 0; i < n_vars; ++i)
                lin_coll += grad_coll_k[i] * scale * (Y(i) - x_k[i]);
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
            
            Function solver;
            try {
                solver = nlpsol("solver", "sqpmethod", SXDict{{"x", Y}, {"f", cost}}, opts);
            } catch (const std::exception& e) {
                std::cerr << "SQP error: " << e.what() << ", falling back to L-BFGS\n";
                return runLBFGS(x, n_vars, N, D);
            }
            
            auto result = solver(DMDict{{"x0", DM(x_k)}, {"lbx", DM(lbx)}, {"ubx", DM(ubx)}});
            auto x_trial = result.at("x").get_elements();
            double pred_cost = static_cast<double>(result.at("f").scalar()) / scale;
            double actual_cost = computeTotalCost(x_trial, smooth_func, scale);
            
            double ratio = 0;
            if (double pred_red = current_cost - pred_cost; std::abs(pred_red) > 1e-10)
                ratio = (current_cost - actual_cost) / pred_red;
            
            double step = 0;
            for (size_t i = 0; i < n_vars; ++i)
                step = std::max(step, std::abs(x_trial[i] - x_k[i]));
            
            if (cfg_->verbose_solver)
                logf("SQP-SCP %3d: %.2f->%.2f, ratio=%.3f, trust=%.4f", outer, current_cost, actual_cost, ratio, trust);
            
            if (ratio >= cfg_->scp_accept_ratio && actual_cost < current_cost) {
                x_k = x_trial;
                current_cost = actual_cost;
                if (ratio >= cfg_->scp_good_ratio)
                    trust = std::min(trust * cfg_->scp_trust_expand, cfg_->scp_trust_region_max);
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
        casadi::Function grad_func = buildSmoothnessGradientFunction(N, D, n_vars);
        
        std::deque<std::vector<double>> s_hist, y_hist;
        std::deque<double> rho_hist;
        std::vector<double> grad(n_vars), grad_new(n_vars), x_new(n_vars);
        double prev_cost = std::numeric_limits<double>::infinity();
        
        log("\nStarting L-BFGS...\n");
        
        for (size_t iter = 0; iter < cfg_->max_iterations; ++iter) {
            double smooth, coll;
            double cost = evaluateCostAndGradient(x, grad, grad_func, smooth, coll);
            
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm);
            
            if (cfg_->verbose_solver && (iter % 10 == 0 || iter < 5))
                logf("Iter %3zu: Cost=%.4f (S=%.4f, C=%.4f), |g|=%.6f", iter, cost, smooth, coll, gnorm);
            
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
                if (computeTotalCost(x_new, grad_func) <= cost + 1e-4 * a * dg) { ls_ok = true; break; }
                a *= 0.5;
            }
            if (!ls_ok) { a = 1e-8; for (size_t i = 0; i < n_vars; ++i) x_new[i] = x[i] + a * dir[i]; }
            
            std::vector<double> s(n_vars);
            for (size_t i = 0; i < n_vars; ++i) s[i] = x_new[i] - x[i];
            
            double sn, cn;
            evaluateCostAndGradient(x_new, grad_new, grad_func, sn, cn);
            
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
        casadi::Function grad_func = buildSmoothnessGradientFunction(N, D, n_vars);
        log("\nStarting Gradient Descent...\n");
        
        std::vector<double> vel(n_vars, 0), grad(n_vars);
        double lr = cfg_->gd_learning_rate;
        
        for (size_t iter = 0; iter < cfg_->max_iterations; ++iter) {
            std::vector<double> x_eval = x;
            if (cfg_->gd_use_nesterov)
                for (size_t i = 0; i < n_vars; ++i) x_eval[i] = x[i] + cfg_->gd_momentum * vel[i];
            
            double smooth, coll;
            evaluateCostAndGradient(x_eval, grad, grad_func, smooth, coll);
            
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm);
            
            if (cfg_->verbose_solver && (iter % 10 == 0 || iter < 5))
                logf("Iter %3zu: Cost=%.4f, |g|=%.6f, lr=%.6f", iter, smooth + cfg_->collision_weight * coll, gnorm, lr);
            
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
        casadi::Function grad_func = buildSmoothnessGradientFunction(N, D, n_vars);
        log("\nStarting Adam...\n");
        
        std::vector<double> m(n_vars, 0), v(n_vars, 0), grad(n_vars);
        
        for (size_t iter = 0; iter < cfg_->max_iterations; ++iter) {
            double smooth, coll;
            evaluateCostAndGradient(x, grad, grad_func, smooth, coll);
            
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm);
            
            if (cfg_->verbose_solver && (iter % 10 == 0 || iter < 5))
                logf("Iter %3zu: Cost=%.4f, |g|=%.6f", iter, smooth + cfg_->collision_weight * coll, gnorm);
            
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