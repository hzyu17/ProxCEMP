#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include "IterationData.h"
#include "OptimizationVisualizer.h"
#include <filesystem>
#include <iomanip>

int main() {
    std::cout << "=================================================\n";
    std::cout << "   Motion Planning Experiments (PCE & NGD)\n";
    std::cout << "=================================================\n\n";

    // Load configuration
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path config_path = source_path.parent_path() / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"]["visualize_initial_state"].as<bool>(false);

    // =========================================================================
    // PCE PLANNER
    // =========================================================================
    std::cout << "\n=== PCE Planner ===\n";

    auto task_pce = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_pce = std::make_shared<ProximalCrossEntropyMotionPlanner>(task_pce);

    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return -1;
    }

    if (!planner_pce->initialize(pce_config)) {
        std::cerr << "Error: PCE Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        visualizeInitialState(task_pce->getObstacles(), planner_pce->getCurrentTrajectory(),
                              "PCEM - Initial State");
    }

    // PCE optimization with data collection
    std::cout << "\n=== Running PCE Optimization ===\n";

    OptimizationHistory pce_history;
    {
        IterationData init_data;
        init_data.iteration = 0;
        init_data.mean_trajectory = planner_pce->getCurrentTrajectory();
        init_data.collision_cost = task_pce->computeStateCost(init_data.mean_trajectory);
        init_data.smoothness_cost = planner_pce->computeSmoothnessCost(init_data.mean_trajectory);
        init_data.total_cost = init_data.collision_cost + init_data.smoothness_cost;
        pce_history.addIteration(init_data);
    }

    bool success_pce = planner_pce->solve();

    for (size_t i = 0; i < planner_pce->getTrajectoryHistory().size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i + 1;
        iter_data.mean_trajectory = planner_pce->getTrajectoryHistory()[i];
        iter_data.collision_cost = task_pce->computeStateCost(iter_data.mean_trajectory);
        iter_data.smoothness_cost = planner_pce->computeSmoothnessCost(iter_data.mean_trajectory);
        iter_data.total_cost = iter_data.collision_cost + iter_data.smoothness_cost;
        pce_history.addIteration(iter_data);
    }

    pce_history.final_trajectory = planner_pce->getCurrentTrajectory();
    pce_history.final_cost = task_pce->computeStateCost(pce_history.final_trajectory) +
                             planner_pce->computeSmoothnessCost(pce_history.final_trajectory);
    pce_history.converged = success_pce;
    pce_history.total_iterations = planner_pce->getTrajectoryHistory().size();

    std::cout << (success_pce ? "\n✓ PCE optimization completed successfully\n"
                              : "\n✗ PCE optimization failed\n");

    // =========================================================================
    // NGD PLANNER
    // =========================================================================
    std::cout << "\n=== NGD Planner ===\n";

    auto task_ngd = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_ngd = std::make_shared<NGDMotionPlanner>(task_ngd);

    NGDConfig ngd_config;
    if (!ngd_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load NGD configuration from file\n";
        return -1;
    }

    if (!planner_ngd->initialize(ngd_config)) {
        std::cerr << "Error: NGD Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        visualizeInitialState(task_ngd->getObstacles(), planner_ngd->getCurrentTrajectory(),
                              "NGD - Initial State");
    }

    std::cout << "\n=== Running NGD Optimization ===\n";

    OptimizationHistory ngd_history;
    {
        IterationData init_data;
        init_data.iteration = 0;
        init_data.mean_trajectory = planner_ngd->getCurrentTrajectory();
        init_data.collision_cost = task_ngd->computeStateCost(init_data.mean_trajectory);
        init_data.smoothness_cost = planner_ngd->computeSmoothnessCost(init_data.mean_trajectory);
        init_data.total_cost = init_data.collision_cost + init_data.smoothness_cost;
        ngd_history.addIteration(init_data);
    }

    bool success_ngd = planner_ngd->solve();

    for (size_t i = 0; i < planner_ngd->getTrajectoryHistory().size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i + 1;
        iter_data.mean_trajectory = planner_ngd->getTrajectoryHistory()[i];
        iter_data.collision_cost = task_ngd->computeStateCost(iter_data.mean_trajectory);
        iter_data.smoothness_cost = planner_ngd->computeSmoothnessCost(iter_data.mean_trajectory);
        iter_data.total_cost = iter_data.collision_cost + iter_data.smoothness_cost;
        ngd_history.addIteration(iter_data);
    }

    ngd_history.final_trajectory = planner_ngd->getCurrentTrajectory();
    ngd_history.final_cost = task_ngd->computeStateCost(ngd_history.final_trajectory) +
                             planner_ngd->computeSmoothnessCost(ngd_history.final_trajectory);
    ngd_history.converged = success_ngd;
    ngd_history.total_iterations = planner_ngd->getTrajectoryHistory().size();

    std::cout << (success_ngd ? "\n✓ NGD optimization completed successfully\n"
                              : "\n✗ NGD optimization failed\n");

    // =========================================================================
    // VISUALIZATION
    // =========================================================================
    std::cout << "\n=== Interactive Optimization Visualization ===\n";

    auto obstacle_map_ptr = task_pce->getObstacleMap();
    OptimizationVisualizer visualizer(900, 700);

    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, pce_history, "PCEM - Trajectory Evolution");
    visualizer.showCostPlot(pce_history, "PCEM - Cost Convergence");

    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, ngd_history, "NGD - Trajectory Evolution");
    visualizer.showCostPlot(ngd_history, "NGD - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   Results Summary\n";
    std::cout << "=================================================\n";

    auto evaluate_final = [&](const Trajectory& traj, auto& planner) {
        float collision = task_pce->computeStateCost(traj);
        float smoothness = planner->computeSmoothnessCost(traj);
        return std::make_tuple(collision + smoothness, collision, smoothness);
    };

    auto [pce_total, pce_coll, pce_smooth] = evaluate_final(planner_pce->getCurrentTrajectory(), planner_pce);
    auto [ngd_total, ngd_coll, ngd_smooth] = evaluate_final(planner_ngd->getCurrentTrajectory(), planner_ngd);

    std::cout << std::left << std::setw(12) << "Planner"
              << std::setw(12) << "Status"
              << std::setw(10) << "Iters"
              << std::setw(15) << "Total Cost"
              << std::setw(15) << "Collision"
              << "Smoothness\n";
    std::cout << std::string(75, '-') << "\n";

    std::cout << std::left << std::setw(12) << "PCE"
              << std::setw(12) << (success_pce ? "SUCCESS" : "FAILED")
              << std::setw(10) << pce_history.total_iterations
              << std::setw(15) << pce_total 
              << std::setw(15) << pce_coll 
              << pce_smooth << "\n";

    std::cout << std::left << std::setw(12) << "NGD"
              << std::setw(12) << (success_ngd ? "SUCCESS" : "FAILED")
              << std::setw(10) << ngd_history.total_iterations
              << std::setw(15) << ngd_total 
              << std::setw(15) << ngd_coll 
              << ngd_smooth << "\n";

    std::cout << std::string(75, '-') << "\n";

    // Find best
    std::string best = "PCE";
    float best_cost = pce_total;
    if (ngd_total < best_cost) { 
        best_cost = ngd_total; 
        best = "NGD"; 
    }

    std::cout << "\n★ Best result: " << best << " with total cost = " << best_cost << "\n";
    std::cout << "\n=================================================\n";

    return 0;
}