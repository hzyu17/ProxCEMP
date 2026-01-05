#include "CasadiMotionPlanner.h"
#include "CasadiCollisionTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include "IterationData.h"
#include "OptimizationVisualizer.h"
#include <filesystem>
#include <map>
#include <iomanip>
#include <chrono>

// Helper function to check if trajectory is 3D
bool is3DTrajectory(const Trajectory& traj) {
    if (traj.nodes.empty()) return false;
    return traj.nodes[0].position.size() >= 3;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "   CasADi 3D Motion Planning Experiments\n";
    std::cout << "=================================================\n\n";

    // Load 3D configuration
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../configs/config_3d.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    std::cout << "Loading 3D configuration from: " << config_file << "\n\n";

    YAML::Node config = YAML::LoadFile(config_file);

    // Verify 3D configuration
    size_t num_dims = config["motion_planning"]["num_dimensions"].as<size_t>(2);
    if (num_dims != 3) {
        std::cerr << "Warning: Configuration has num_dimensions = " << num_dims 
                  << ", expected 3 for 3D planning.\n";
    }

    bool visualize = config["experiment"] && 
                     config["experiment"]["visualize_initial_state"] &&
                     config["experiment"]["visualize_initial_state"].as<bool>();

    // Print 3D workspace info
    std::cout << "=== 3D Workspace Configuration ===\n";
    std::cout << "Dimensions: " << num_dims << "D\n";
    if (config["motion_planning"]["start_position"]) {
        auto start = config["motion_planning"]["start_position"].as<std::vector<float>>();
        std::cout << "Start: [" << start[0] << ", " << start[1] << ", " << start[2] << "]\n";
    }
    if (config["motion_planning"]["goal_position"]) {
        auto goal = config["motion_planning"]["goal_position"].as<std::vector<float>>();
        std::cout << "Goal:  [" << goal[0] << ", " << goal[1] << ", " << goal[2] << "]\n";
    }
    std::cout << "Map size: " << config["environment"]["map_width"].as<float>(800) << " x "
              << config["environment"]["map_height"].as<float>(600) << " x "
              << config["environment"]["map_depth"].as<float>(400) << "\n";
    std::cout << "==================================\n\n";

    // =========================================================================
    // Define all valid CasADi solvers to test
    // =========================================================================
    // Valid solver types (from CasADiSolverType enum):
    //   - "lbfgs"  : L-BFGS quasi-Newton method
    //   - "ipopt"  : IPOPT with SCP (Sequential Convex Programming)
    //   - "sqp"    : SQP with SCP (requires qpoases library)
    //   - "gd"     : Gradient Descent with momentum
    //   - "adam"   : Adam optimizer
    // =========================================================================
    std::vector<std::string> casadi_solvers = {
        "lbfgs",    // L-BFGS quasi-Newton
        "ipopt",    // IPOPT with SCP trust region
        "sqp",      // SQP with SCP (uncomment if qpoases is available)
        "gd",       // Gradient Descent with Nesterov momentum
        "adam"      // Adam optimizer
    };

    std::cout << "Solvers to test: ";
    for (size_t i = 0; i < casadi_solvers.size(); ++i) {
        std::cout << casadi_solvers[i];
        if (i < casadi_solvers.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    // Store results for each solver
    std::map<std::string, OptimizationHistory> casadi_histories;
    std::map<std::string, bool> casadi_success;
    std::map<std::string, std::shared_ptr<CasADiMotionPlanner>> casadi_planners;
    std::map<std::string, std::shared_ptr<pce::CasadiCollisionTask>> casadi_tasks;
    std::map<std::string, double> casadi_times;  // Track computation time

    for (const auto& solver_name : casadi_solvers) {
        std::cout << "\n=========================================\n";
        std::cout << "=== CasADi-" << solver_name << " (3D) ===\n";
        std::cout << "=========================================\n";

        // Create CasadiCollisionTask (extends CollisionAvoidanceTask with symbolic costs)
        auto task = std::make_shared<pce::CasadiCollisionTask>(config);
        casadi_tasks[solver_name] = task;

        // Create planner with CasADi-aware task
        auto planner = std::make_shared<CasADiMotionPlanner>(task);
        casadi_planners[solver_name] = planner;

        // Load configuration
        CasADiConfig casadi_config;
        if (!casadi_config.loadFromFile(config_file)) {
            std::cerr << "Failed to load CasADi configuration from file\n";
            casadi_success[solver_name] = false;
            casadi_times[solver_name] = 0.0;
            continue;
        }

        // Override solver type
        casadi_config.solver = solver_name;
        casadi_config.solver_type = stringToSolverType(solver_name);

        // Verify solver type was correctly parsed
        std::cout << "Configured solver type: " << solverTypeToString(casadi_config.solver_type) << "\n";

        std::cout << "Initializing CasADi-" << solver_name << " (3D)...\n";
        if (!planner->initialize(casadi_config)) {
            std::cerr << "Error: CasADi-" << solver_name << " initialization failed\n";
            casadi_success[solver_name] = false;
            casadi_times[solver_name] = 0.0;
            continue;
        }

        // Verify 3D trajectory
        if (!is3DTrajectory(planner->getCurrentTrajectory())) {
            std::cerr << "Warning: CasADi-" << solver_name << " trajectory is not 3D!\n";
        }

        if (visualize) {
            visualizeInitialState(task->getObstacles(), planner->getCurrentTrajectory(),
                                  "CasADi-" + solver_name + " 3D - Initial State");
        }

        // Collect optimization history
        OptimizationHistory history;
        history.clear();

        // Store initial state
        {
            IterationData init_data;
            init_data.iteration = 0;
            init_data.mean_trajectory = planner->getCurrentTrajectory();
            init_data.collision_cost = task->computeStateCost(init_data.mean_trajectory);
            init_data.smoothness_cost = planner->computeSmoothnessCost(init_data.mean_trajectory);
            init_data.total_cost = init_data.collision_cost + init_data.smoothness_cost;
            history.addIteration(init_data);
        }

        // Run optimization with timing
        std::cout << "Running CasADi-" << solver_name << " 3D optimization...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = planner->solve();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        casadi_success[solver_name] = success;
        casadi_times[solver_name] = elapsed_ms;

        // Convert trajectory history
        auto traj_history = planner->getTrajectoryHistory();
        for (size_t i = 0; i < traj_history.size(); ++i) {
            IterationData iter_data;
            iter_data.iteration = i;
            iter_data.mean_trajectory = traj_history[i];
            iter_data.collision_cost = task->computeStateCost(traj_history[i]);
            iter_data.smoothness_cost = planner->computeSmoothnessCost(traj_history[i]);
            iter_data.total_cost = iter_data.collision_cost + iter_data.smoothness_cost;
            history.addIteration(iter_data);
        }

        history.final_trajectory = planner->getCurrentTrajectory();
        float final_collision = task->computeStateCost(history.final_trajectory);
        float final_smoothness = planner->computeSmoothnessCost(history.final_trajectory);
        history.final_cost = final_collision + final_smoothness;
        history.converged = success;
        history.total_iterations = traj_history.size();
        casadi_histories[solver_name] = history;

        std::cout << (success ? "✓" : "✗") << " CasADi-" << solver_name << " (3D)"
                  << (success ? " completed successfully" : " failed")
                  << " in " << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
        std::cout << "  Final cost: " << history.final_cost 
                  << " (collision: " << final_collision 
                  << ", smoothness: " << final_smoothness << ")\n";
    }

    // =========================================================================
    // 3D VISUALIZATION WITH MULTI-VIEW PROJECTIONS
    // =========================================================================
    std::cout << "\n\n=== Interactive 3D Optimization Visualization ===\n";
    std::cout << "Using multi-view projection (XY, XZ, YZ planes)\n\n";

    // Use first successful task's obstacle map for visualization
    std::shared_ptr<ObstacleMap> obstacle_map_ptr;
    for (const auto& [name, task] : casadi_tasks) {
        if (task) {
            obstacle_map_ptr = task->getObstacleMap();
            break;
        }
    }

    if (obstacle_map_ptr) {
        OptimizationVisualizer visualizer(800, 600);  // Base size, 3D will use 1200x600

        for (const auto& solver_name : casadi_solvers) {
            // Check if this solver has valid results
            if (casadi_histories.find(solver_name) == casadi_histories.end()) {
                std::cout << "Skipping " << solver_name << " - no history found\n";
                continue;
            }
            if (!casadi_success[solver_name]) {
                std::cout << "Skipping " << solver_name << " - optimization failed\n";
                continue;
            }

            const auto& history = casadi_histories[solver_name];

            // Use 3D visualization (multi-view projection)
            std::cout << "\nDisplaying CasADi-" << solver_name << " 3D trajectory evolution "
                      << "(iters: " << history.total_iterations 
                      << ", cost: " << std::fixed << std::setprecision(2) << history.final_cost << ")...\n";
            visualizer.setOutputPrefix("casadi_" + solver_name + "_3d");
            visualizer.showTrajectoryEvolution3D(*obstacle_map_ptr, history,
                                                  "CasADi-" + solver_name + " 3D - Trajectory Evolution");

            std::cout << "Displaying CasADi-" << solver_name << " cost convergence...\n";
            visualizer.showCostPlot(history, "CasADi-" + solver_name + " 3D - Cost Convergence");
        }
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   CasADi 3D Motion Planning Results Summary\n";
    std::cout << "=================================================\n\n";

    // Use first task/planner for consistent evaluation
    std::shared_ptr<pce::CasadiCollisionTask> ref_task;
    std::shared_ptr<CasADiMotionPlanner> ref_planner;
    for (const auto& solver_name : casadi_solvers) {
        if (casadi_tasks.find(solver_name) != casadi_tasks.end() && casadi_tasks[solver_name]) {
            ref_task = casadi_tasks[solver_name];
            ref_planner = casadi_planners[solver_name];
            break;
        }
    }

    if (!ref_task || !ref_planner) {
        std::cerr << "Error: No valid task/planner for evaluation\n";
        return 1;
    }

    auto evaluate_final = [&](const Trajectory& traj) {
        float collision = ref_task->computeStateCost(traj);
        float smoothness = ref_planner->computeSmoothnessCost(traj);
        return std::make_tuple(collision + smoothness, collision, smoothness);
    };

    // Print header
    std::cout << std::left 
              << std::setw(20) << "Solver"
              << std::setw(10) << "Status"
              << std::setw(8) << "Iters"
              << std::setw(12) << "Time (ms)"
              << std::setw(14) << "Total Cost"
              << std::setw(12) << "Collision"
              << "Smoothness\n";
    std::cout << std::string(94, '-') << "\n";

    std::string best_planner;
    float best_cost = std::numeric_limits<float>::max();
    std::string fastest_planner;
    double fastest_time = std::numeric_limits<double>::max();

    for (const auto& solver_name : casadi_solvers) {
        if (casadi_planners.find(solver_name) == casadi_planners.end()) continue;
        if (!casadi_planners[solver_name]) continue;

        auto [total, collision, smoothness] = evaluate_final(casadi_planners[solver_name]->getCurrentTrajectory());
        const auto& history = casadi_histories[solver_name];
        bool success = casadi_success[solver_name];
        double time_ms = casadi_times[solver_name];

        std::string planner_name = "CasADi-" + solver_name + " (3D)";
        
        std::cout << std::left 
                  << std::setw(20) << planner_name
                  << std::setw(10) << (success ? "SUCCESS" : "FAILED")
                  << std::setw(8) << history.total_iterations
                  << std::setw(12) << std::fixed << std::setprecision(1) << time_ms
                  << std::setw(14) << std::setprecision(4) << total
                  << std::setw(12) << collision
                  << smoothness << "\n";

        if (success && total < best_cost && !std::isnan(total)) {
            best_cost = total;
            best_planner = planner_name;
        }
        
        if (success && time_ms < fastest_time) {
            fastest_time = time_ms;
            fastest_planner = planner_name;
        }
    }

    std::cout << std::string(94, '-') << "\n";
    
    // Summary statistics
    std::cout << "\n★ Best quality:  " << best_planner 
              << " (total cost = " << std::setprecision(4) << best_cost << ")\n";
    std::cout << "★ Fastest:       " << fastest_planner 
              << " (time = " << std::setprecision(1) << fastest_time << " ms)\n";

    // Solver comparison notes
    std::cout << "\n--- Solver Characteristics ---\n";
    std::cout << "L-BFGS:  Quasi-Newton, good for smooth problems, fast convergence\n";
    std::cout << "IPOPT:   Interior-point + SCP, handles constraints well, robust\n";
    std::cout << "SQP:     Sequential QP + SCP, good for constrained optimization\n";
    std::cout << "GD:      First-order, simple but slow convergence\n";
    std::cout << "Adam:    Adaptive learning rate, good for noisy gradients\n";

    // Print final 3D trajectory endpoints for verification
    std::cout << "\n=== Final 3D Trajectory Endpoints ===\n";
    for (const auto& solver_name : casadi_solvers) {
        if (casadi_planners.find(solver_name) == casadi_planners.end()) continue;
        if (!casadi_planners[solver_name]) continue;
        if (!casadi_success[solver_name]) continue;

        const auto& final_traj = casadi_planners[solver_name]->getCurrentTrajectory();
        if (!final_traj.nodes.empty() && is3DTrajectory(final_traj)) {
            const auto& start = final_traj.nodes[final_traj.start_index].position;
            const auto& goal = final_traj.nodes[final_traj.goal_index].position;
            std::cout << "CasADi-" << solver_name << " Start: [" 
                      << start(0) << ", " << start(1) << ", " << start(2) << "]\n";
            std::cout << "CasADi-" << solver_name << " Goal:  [" 
                      << goal(0) << ", " << goal(1) << ", " << goal(2) << "]\n";
            break;  // Just show one example since they should all have same start/goal
        }
    }

    std::cout << "\n=================================================\n";
    std::cout << "   CasADi 3D Motion Planning Experiments Complete\n";
    std::cout << "=================================================\n";

    return 0;
}