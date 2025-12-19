/**
 * @file test_all_solvers.cpp
 * @brief Test all solvers (PCE, NGD, CasADi backends) with convex cost
 * 
 * Uses QuadraticBowlTask where:
 *   - J_state optimal = straight line
 *   - J_smooth optimal = straight line  
 *   - Combined optimal = straight line (cost ≈ 0)
 * 
 * All solvers start from perturbed trajectory and must converge to optimal.
 * Pass criteria: final < initial AND final < 1.0
 */

#include "TestTask.h"
#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CasadiMotionPlanner.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>

// Silence cout and cerr during solver runs
class SilentScope {
    std::streambuf* old_cout_;
    std::streambuf* old_cerr_;
    std::ostringstream sink_;
public:
    SilentScope() : old_cout_(std::cout.rdbuf(sink_.rdbuf())),
                    old_cerr_(std::cerr.rdbuf(sink_.rdbuf())) {}
    ~SilentScope() { 
        std::cout.rdbuf(old_cout_); 
        std::cerr.rdbuf(old_cerr_);
    }
};

struct TestResult {
    std::string solver;
    double initial_cost = 0;
    double final_cost = 0;
    double time_ms = 0;
    bool passed = false;
};

void printResult(const TestResult& r) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::setw(10) << r.solver << " │ "
              << std::setw(12) << r.initial_cost << " → "
              << std::setw(12) << (std::isnan(r.final_cost) ? -1.0 : r.final_cost) << " │ "
              << std::setw(10) << r.time_ms << " ms │ "
              << (r.passed ? "✓ PASS" : "✗ FAIL") << "\n";
}

// ============================================================
// PCE Configuration (from config.yaml)
// ============================================================

PCEConfig makePCEConfig(size_t nodes = 20, size_t dims = 2) {
    PCEConfig config;
    
    config.num_dimensions = dims;
    config.num_discretization = nodes;
    config.total_time = 10.0f;
    config.node_collision_radius = 0.1f;
    config.start_position.resize(dims, 0.0f);
    config.goal_position.resize(dims, 10.0f);
    
    // PCE parameters from config.yaml
    config.num_samples = 100;
    config.num_iterations = 100;
    config.elite_ratio = 0.1f;
    config.eta = 1.5f;
    config.temperature = 1.5f;
    config.temperature_final = 0.05f;
    config.cov_scale_initial = 1.0f;
    config.cov_scale_final = 0.01f;
    config.cov_schedule = CovarianceSchedule::COSINE;
    config.ema_alpha = 0.4f;
    
    return config;
}

// ============================================================
// NGD Configuration (from config.yaml)
// ============================================================

NGDConfig makeNGDConfig(size_t nodes = 20, size_t dims = 2) {
    NGDConfig config;
    
    config.num_dimensions = dims;
    config.num_discretization = nodes;
    config.total_time = 10.0f;
    config.node_collision_radius = 0.1f;
    config.start_position.resize(dims, 0.0f);
    config.goal_position.resize(dims, 10.0f);
    
    // NGD parameters from config.yaml
    config.num_samples = 100;
    config.num_iterations = 100;
    config.learning_rate = 0.0001f;
    config.temperature = 1.5f;
    config.convergence_threshold = 0.01f;
    
    return config;
}

// ============================================================
// CasADi Configuration (from config.yaml)
// ============================================================

CasADiConfig makeCasADiConfig(const std::string& solver, size_t nodes = 20, size_t dims = 2) {
    CasADiConfig config;
    
    config.num_dimensions = dims;
    config.num_discretization = nodes;
    config.total_time = 10.0f;
    config.node_collision_radius = 0.1f;
    config.start_position.resize(dims, 0.0f);
    config.goal_position.resize(dims, 10.0f);
    
    config.solver = solver;
    config.solver_type = stringToSolverType(solver);
    config.collision_weight = 10.0f;       // From config.yaml
    config.tolerance = 1e-6f;
    config.verbose_solver = false;         // Quiet for test
    
    if (solver == "lbfgs") {
        config.max_iterations = 500;
        config.lbfgs_history = 10;
    } 
    else if (solver == "ipopt") {
        config.max_iterations = 100;       // scp.inner_max_iter
        config.ipopt_print_level = 0;
        config.scp_max_outer_iter = 50;
        config.scp_trust_region_init = 10.0;
    }
    else if (solver == "sqp") {
        config.max_iterations = 100;
        config.scp_max_outer_iter = 50;
        config.scp_trust_region_init = 10.0;
    }
    else if (solver == "gd") {
        config.max_iterations = 5000;
        config.gd_learning_rate = 0.001f;  // From config.yaml
        config.gd_momentum = 0.9f;
        config.gd_use_nesterov = true;
        config.gd_lr_decay = 1.0f;         // No decay
    }
    else if (solver == "adam") {
        config.max_iterations = 5000;      // max_iterations from config.yaml
        config.adam_learning_rate = 0.01f; // From config.yaml
    }
    
    return config;
}

// ============================================================
// PCE Test (with perturbation)
// ============================================================

TestResult runPCETest(float perturb_amplitude = 1.0f) {
    TestResult result;
    result.solver = "PCE";
    
    auto task = std::make_shared<pce::QuadraticBowlTask>();
    ProximalCrossEntropyMotionPlanner planner(task);
    PCEConfig config = makePCEConfig();
    
    {
        SilentScope quiet;
        
        if (!planner.initialize(config)) {
            return result;
        }
        
        // Perturb trajectory away from optimal
        planner.perturbTrajectory(perturb_amplitude, 42);
        
        // Compute initial total cost = J_state + J_smooth
        const auto& traj = planner.getCurrentTrajectory();
        float state_cost = task->computeStateCostSimple(traj);
        float smoothness = planner.computeSmoothnessCost(traj);
        result.initial_cost = state_cost + smoothness;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        planner.solve();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // Compute final total cost
        const auto& final_traj = planner.getCurrentTrajectory();
        state_cost = task->computeStateCostSimple(final_traj);
        smoothness = planner.computeSmoothnessCost(final_traj);
        result.final_cost = state_cost + smoothness;
    }
    
    // Pass: must improve AND reach reasonable cost
    result.passed = (result.final_cost < result.initial_cost) && 
                    (result.final_cost < 1.0) && 
                    !std::isnan(result.final_cost);
    return result;
}

// ============================================================
// NGD Test (with perturbation)
// ============================================================

TestResult runNGDTest(float perturb_amplitude = 1.0f) {
    TestResult result;
    result.solver = "NGD";
    
    auto task = std::make_shared<pce::QuadraticBowlTask>();
    NGDMotionPlanner planner(task);
    NGDConfig config = makeNGDConfig();
    
    {
        SilentScope quiet;
        
        if (!planner.initialize(config)) {
            return result;
        }
        
        // Perturb trajectory away from optimal
        planner.perturbTrajectory(perturb_amplitude, 42);
        
        // Compute initial total cost = J_state + J_smooth
        const auto& traj = planner.getCurrentTrajectory();
        float state_cost = task->computeStateCostSimple(traj);
        float smoothness = planner.computeSmoothnessCost(traj);
        result.initial_cost = state_cost + smoothness;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        planner.solve();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // Compute final total cost
        const auto& final_traj = planner.getCurrentTrajectory();
        state_cost = task->computeStateCostSimple(final_traj);
        smoothness = planner.computeSmoothnessCost(final_traj);
        result.final_cost = state_cost + smoothness;
    }
    
    // Pass: must improve AND reach reasonable cost
    result.passed = (result.final_cost < result.initial_cost) && 
                    (result.final_cost < 1.0) && 
                    !std::isnan(result.final_cost);
    return result;
}

// ============================================================
// CasADi Test (with perturbation)
// ============================================================

TestResult runCasADiTest(const std::string& solver, float perturb_amplitude = 1.0f) {
    TestResult result;
    result.solver = solver;
    
    auto task = std::make_shared<pce::QuadraticBowlTask>();
    CasADiMotionPlanner planner(task);
    CasADiConfig config = makeCasADiConfig(solver);
    
    {
        SilentScope quiet;
        
        if (!planner.initialize(config)) {
            return result;
        }
        
        // Perturb trajectory away from optimal
        planner.perturbTrajectory(perturb_amplitude, 42);
        
        // Compute initial total cost = J_state + J_smooth
        const auto& traj = planner.getCurrentTrajectory();
        float state_cost = task->computeStateCostSimple(traj);
        float smoothness = planner.computeSmoothnessCost(traj);
        result.initial_cost = state_cost + smoothness;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        planner.solve();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // Compute final total cost
        const auto& final_traj = planner.getCurrentTrajectory();
        state_cost = task->computeStateCostSimple(final_traj);
        smoothness = planner.computeSmoothnessCost(final_traj);
        result.final_cost = state_cost + smoothness;
    }
    
    // Pass: must improve AND reach reasonable cost
    result.passed = (result.final_cost < result.initial_cost) && 
                    (result.final_cost < 1.0) && 
                    !std::isnan(result.final_cost);
    return result;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Solver Benchmark Test (QuadraticBowlTask)                    ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Cost = J_state + J_smooth (both convex, optimal = straight line) ║\n";
    std::cout << "║  Start: perturbed │ Pass: improve AND final < 1.0                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::vector<TestResult> results;
    
    std::cout << "  Solver     │  Initial Total →  Final Total │       Time   │ Result\n";
    std::cout << "  ───────────┼────────────────────────────────┼──────────────┼────────\n";
    
    // Test PCE
    {
        TestResult r = runPCETest();
        results.push_back(r);
        printResult(r);
    }
    
    // Test NGD
    {
        TestResult r = runNGDTest();
        results.push_back(r);
        printResult(r);
    }
    
    // Test CasADi solvers
    std::vector<std::string> casadi_solvers = {"lbfgs", "ipopt", "gd", "adam"};
    
    for (const auto& solver : casadi_solvers) {
        TestResult r = runCasADiTest(solver);
        results.push_back(r);
        printResult(r);
    }
    
    std::cout << "  ───────────┴────────────────────────────────┴──────────────┴────────\n";
    
    // Count passes
    int passed = 0;
    for (const auto& r : results) {
        if (r.passed) passed++;
    }
    
    std::cout << "\n  Result: " << passed << "/" << results.size() << " passed";
    
    if (passed == (int)results.size()) {
        std::cout << " ✓ All tests passed!\n";
    } else {
        std::cout << " ✗ Some tests failed\n";
    }
    
    // Find best solver
    double best_cost = std::numeric_limits<double>::infinity();
    std::string best_solver;
    double best_time = 0;
    
    for (const auto& r : results) {
        if (r.passed && r.final_cost < best_cost) {
            best_cost = r.final_cost;
            best_solver = r.solver;
            best_time = r.time_ms;
        }
    }
    
    if (!best_solver.empty()) {
        std::cout << "\n  ★ Best: " << best_solver 
                  << " (cost=" << std::scientific << std::setprecision(2) << best_cost 
                  << ", time=" << std::fixed << std::setprecision(1) << best_time << " ms)\n";
    }
    
    std::cout << "\n";
    
    return (passed == (int)results.size()) ? 0 : 1;
}
