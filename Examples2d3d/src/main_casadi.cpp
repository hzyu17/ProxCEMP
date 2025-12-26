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

int main() {
    std::cout << "=================================================\n";
    std::cout << "   CasADi Motion Planning Experiments\n";
    std::cout << "=================================================\n\n";

    // Load configuration
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path config_path = source_path.parent_path() / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"]["visualize_initial_state"].as<bool>(false);

    // Define solvers to test
    // Note: "sqp" requires qpoases library, exclude if not available
    std::vector<std::string> casadi_solvers = {"ipopt", "gd", "adam", "lbfgs"};

    // Store results for each solver
    std::map<std::string, OptimizationHistory> casadi_histories;
    std::map<std::string, bool> casadi_success;
    std::map<std::string, std::shared_ptr<CasADiMotionPlanner>> casadi_planners;
    std::map<std::string, std::shared_ptr<pce::CasadiCollisionTask>> casadi_tasks;

    for (const auto& solver_name : casadi_solvers) {
        std::cout << "\n=== CasADi Planner (" << solver_name << ") ===\n";

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
            continue;
        }

        // Override solver type
        casadi_config.solver = solver_name;
        casadi_config.solver_type = stringToSolverType(solver_name);

        std::cout << "Initializing CasADi-" << solver_name << "...\n";
        if (!planner->initialize(casadi_config)) {
            std::cerr << "Error: CasADi-" << solver_name << " initialization failed\n";
            casadi_success[solver_name] = false;
            continue;
        }

        if (visualize) {
            visualizeInitialState(task->getObstacles(), planner->getCurrentTrajectory(),
                                  "CasADi-" + solver_name + " - Initial State");
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

        // Run optimization
        std::cout << "Running CasADi-" << solver_name << " optimization...\n";
        bool success = planner->solve();
        casadi_success[solver_name] = success;

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

        std::cout << (success ? "✓" : "✗") << " CasADi-" << solver_name 
                  << (success ? " completed successfully\n" : " failed\n");
    }

    // =========================================================================
    // VISUALIZATION
    // =========================================================================
    std::cout << "\n=== Interactive Optimization Visualization ===\n";

    // Use first successful task's obstacle map for visualization
    std::shared_ptr<ObstacleMap> obstacle_map_ptr;
    for (const auto& [name, task] : casadi_tasks) {
        if (task) {
            obstacle_map_ptr = task->getObstacleMap();
            break;
        }
    }

    if (obstacle_map_ptr) {
        OptimizationVisualizer visualizer(900, 700);

        for (const auto& solver_name : casadi_solvers) {
            if (casadi_histories.find(solver_name) == casadi_histories.end()) continue;

            const auto& history = casadi_histories[solver_name];

            std::cout << "\nDisplaying CasADi-" << solver_name << " trajectory evolution...\n";
            visualizer.showTrajectoryEvolution(*obstacle_map_ptr, history,
                                               "CasADi-" + solver_name + " - Trajectory Evolution");

            std::cout << "Displaying CasADi-" << solver_name << " cost convergence...\n";
            visualizer.showCostPlot(history, "CasADi-" + solver_name + " - Cost Convergence");
        }
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   CasADi Results Summary\n";
    std::cout << "=================================================\n";

    // Use first task/planner for consistent evaluation
    auto ref_task = casadi_tasks.begin()->second;
    auto ref_planner = casadi_planners.begin()->second;

    auto evaluate_final = [&](const Trajectory& traj) {
        float collision = ref_task->computeStateCost(traj);
        float smoothness = ref_planner->computeSmoothnessCost(traj);
        return std::make_tuple(collision + smoothness, collision, smoothness);
    };

    std::cout << std::left << std::setw(18) << "Planner"
              << std::setw(12) << "Status"
              << std::setw(10) << "Iters"
              << std::setw(15) << "Total Cost"
              << std::setw(15) << "Collision"
              << "Smoothness\n";
    std::cout << std::string(85, '-') << "\n";

    std::string best_planner;
    float best_cost = std::numeric_limits<float>::max();

    for (const auto& solver_name : casadi_solvers) {
        if (casadi_planners.find(solver_name) == casadi_planners.end()) continue;

        auto [total, collision, smoothness] = evaluate_final(casadi_planners[solver_name]->getCurrentTrajectory());
        const auto& history = casadi_histories[solver_name];
        bool success = casadi_success[solver_name];

        std::string planner_name = "CasADi-" + solver_name;
        std::cout << std::left << std::setw(18) << planner_name
                  << std::setw(12) << (success ? "SUCCESS" : "FAILED")
                  << std::setw(10) << history.total_iterations
                  << std::setw(15) << total
                  << std::setw(15) << collision
                  << smoothness << "\n";

        if (total < best_cost && !std::isnan(total)) {
            best_cost = total;
            best_planner = planner_name;
        }
    }

    std::cout << std::string(85, '-') << "\n";
    std::cout << "\n★ Best result: " << best_planner << " with total cost = " << best_cost << "\n";
    std::cout << "\n=================================================\n";

    return 0;
}