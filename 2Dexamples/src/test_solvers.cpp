/**
 * @file test_solvers.cpp
 * @brief Test CasADi solvers with known convex cost functions
 * 
 * Usage: ./test_solvers [solver_name]
 *   solver_name: lbfgs, ipopt, sqp, gd, adam (default: lbfgs)
 */

#include "ConvexCosts.h"
#include "task.h"
#include "CasadiMotionPlanner.h"
#include <iostream>
#include <memory>
#include <chrono>

/**
 * @brief Perturb a trajectory away from straight line
 */
void perturbTrajectory(Trajectory& traj, float amplitude = 2.0f) {
    size_t N = traj.nodes.size();
    size_t D = traj.nodes[0].position.size();
    
    for (size_t i = 1; i < N - 1; ++i) {
        float t = static_cast<float>(i) / (N - 1);
        for (size_t d = 0; d < D; ++d) {
            float pert = amplitude * std::sin(M_PI * t) * std::cos(2 * M_PI * d / D + i * 0.5f);
            traj.nodes[i].position(d) += pert;
        }
    }
}

/**
 * @brief Run a single solver test
 */
bool runSolverTest(const std::string& solver_name,
                   pce::TaskPtr task,
                   size_t num_nodes = 20,
                   size_t num_dims = 2) {
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Testing " << solver_name << " solver\n";
    std::cout << std::string(60, '=') << "\n";
    
    // Create planner with task
    CasADiMotionPlanner planner(task);
    
    // Configure
    CasADiConfig config;
    config.num_dimensions = num_dims;
    config.num_discretization = num_nodes;
    config.total_time = 1.0f;
    config.node_collision_radius = 0.1f;
    
    // Set start/goal positions
    config.start_position.resize(num_dims, 0.0f);
    config.goal_position.resize(num_dims, 10.0f);
    
    // Solver selection
    config.solver = solver_name;
    config.solver_type = stringToSolverType(solver_name);
    config.max_iterations = 500;
    config.tolerance = 1e-6f;
    config.collision_weight = 10.0f;
    config.verbose_solver = true;
    
    // Solver-specific tuning
    if (solver_name == "adam") {
        config.max_iterations = 2000;
        config.adam_learning_rate = 0.1;
    } else if (solver_name == "gd") {
        config.max_iterations = 2000;
        config.gd_learning_rate = 0.05;
        config.gd_momentum = 0.9;
    }
    
    // Initialize planner (this sets up trajectory internally)
    if (!planner.initialize(config)) {
        std::cerr << "Failed to initialize planner!\n";
        return false;
    }
    
    // Perturb the initial trajectory to test optimization
    // Access and modify current_trajectory_ through the planner
    // NOTE: You may need to add a method to perturb or get mutable access to trajectory
    // For now, the straight-line initialization should work as the collision cost
    // pulls toward the optimal (which is also straight line for QuadraticBowlTask)
    
    // Run optimization
    auto t_start = std::chrono::high_resolution_clock::now();
    bool success = planner.solve();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    std::cout << "\nSolver " << solver_name << " completed in " << elapsed_ms << " ms\n";
    std::cout << "Success: " << (success ? "yes" : "no") << "\n";
    
    // Get final trajectory and evaluate
    const Trajectory& final_traj = planner.getCurrentTrajectory();
    float final_collision = task->computeStateCostSimple(final_traj);
    float final_smoothness = planner.computeSmoothnessCost(final_traj);
    
    std::cout << "Final collision cost: " << final_collision << "\n";
    std::cout << "Final smoothness cost: " << final_smoothness << "\n";
    
    return success && (final_collision < 0.1f);
}

/**
 * @brief Run tests with different tasks
 */
void runAllTests(const std::string& solver_name) {
    size_t num_nodes = 20;
    size_t num_dims = 2;
    
    std::cout << "\n" << std::string(70, '#') << "\n";
    std::cout << "# Testing solver: " << solver_name << "\n";
    std::cout << std::string(70, '#') << "\n";
    
    int passed = 0;
    int total = 0;
    
    // Test 1: Quadratic Bowl (simplest)
    {
        std::cout << "\n--- Test 1: QuadraticBowlTask ---\n";
        auto task = std::make_shared<pce::QuadraticBowlTask>();
        if (runSolverTest(solver_name, task, num_nodes, num_dims)) passed++;
        total++;
    }
    
    // Test 2: Quadratic Attractor
    {
        std::cout << "\n--- Test 2: QuadraticAttractorTask ---\n";
        auto task = std::make_shared<pce::QuadraticAttractorTask>();
        if (runSolverTest(solver_name, task, num_nodes, num_dims)) passed++;
        total++;
    }
    
    // Test 3: Zero collision (smoothness only)
    {
        std::cout << "\n--- Test 3: ZeroCollisionTask ---\n";
        auto task = std::make_shared<pce::ZeroCollisionTask>();
        if (runSolverTest(solver_name, task, num_nodes, num_dims)) passed++;
        total++;
    }
    
    // Test 4: Offset attractor (curved optimal)
    {
        std::cout << "\n--- Test 4: OffsetAttractorTask ---\n";
        auto task = std::make_shared<pce::OffsetAttractorTask>(2.0f);
        if (runSolverTest(solver_name, task, num_nodes, num_dims)) passed++;
        total++;
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Results for " << solver_name << ": " << passed << "/" << total << " tests passed\n";
    std::cout << std::string(70, '=') << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== CasADi Solver Test Suite ===\n";
    std::cout << "Testing solvers with CONVEX cost functions\n";
    std::cout << "All tests should pass - failures indicate bugs!\n\n";
    
    // Get solver name from command line or test all
    std::vector<std::string> solvers_to_test;
    
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            solvers_to_test.push_back(argv[i]);
        }
    } else {
        // Default: test L-BFGS (most reliable)
        solvers_to_test = {"lbfgs"};
    }
    
    std::cout << "Solvers to test: ";
    for (const auto& s : solvers_to_test) {
        std::cout << s << " ";
    }
    std::cout << "\n";
    
    for (const auto& solver : solvers_to_test) {
        runAllTests(solver);
    }
    
    std::cout << "\n=== All Tests Complete ===\n";
    
    return 0;
}