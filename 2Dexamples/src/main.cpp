#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include "IterationData.h"
#include "OptimizationVisualizer.h"
#include <filesystem>

int main() {
    std::cout << "=================================================\n";
    std::cout << "   Motion Planning Experiments\n";
    std::cout << "=================================================\n\n";

    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    // Load configuration
    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"] && 
                     config["experiment"]["visualize_initial_state"] &&
                     config["experiment"]["visualize_initial_state"].as<bool>();

    // =========================================================================
    // PCE PLANNER
    // =========================================================================
    std::cout << "\n=== PCE Planner ===\n";

    // Create task for PCE
    auto task_pce = std::make_shared<pce::CollisionAvoidanceTask>(config);

    // Create PCE planner and set task
    auto planner_pce = std::make_shared<ProximalCrossEntropyMotionPlanner>(task_pce);

    // Load PCE configuration
    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return -1;
    }

    std::cout << "\n=== Initializing PCE Planner ===\n";
    if (!planner_pce->initialize(pce_config)) {
        std::cerr << "Error: PCE Planner initialization failed\n";
        return 1;
    }

    // Visualize initial state if requested
    if (visualize) {
        std::cout << "Showing PCE initial state visualization...\n";
        visualizeInitialState(task_pce->getObstacles(),
                            planner_pce->getCurrentTrajectory(),
                            "PCEM - Initial State");
    }

    // =========================================================================
    // PCE OPTIMIZATION WITH DATA COLLECTION
    // =========================================================================
    std::cout << "\n=== Running PCE Optimization (with data collection) ===\n";
    
    OptimizationHistory pce_history;
    pce_history.clear();
    
    // Store initial state
    {
        IterationData init_data;
        init_data.iteration = 0;
        init_data.mean_trajectory = planner_pce->getCurrentTrajectory();
        init_data.total_cost = task_pce->computeCollisionCost(init_data.mean_trajectory);
        init_data.collision_cost = init_data.total_cost;
        init_data.smoothness_cost = 0.0f;  // Or compute if available
        pce_history.addIteration(init_data);
    }
    
    // Option B: Use existing solve() with trajectory history
    bool success_pce = planner_pce->solve();
    
    // Convert trajectory history to OptimizationHistory format
    auto traj_history_pce = planner_pce->getTrajectoryHistory();
    for (size_t i = 0; i < traj_history_pce.size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = traj_history_pce[i];
        iter_data.total_cost = task_pce->computeCollisionCost(traj_history_pce[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        // Note: samples not available from history, leave empty
        pce_history.addIteration(iter_data);
    }
    
    pce_history.final_trajectory = planner_pce->getCurrentTrajectory();
    pce_history.final_cost = task_pce->computeCollisionCost(pce_history.final_trajectory);
    pce_history.converged = success_pce;
    pce_history.total_iterations = traj_history_pce.size();

    if (success_pce) {
        std::cout << "\n✓ PCE optimization completed successfully\n";
    } else {
        std::cout << "\n✗ PCE optimization failed\n";
    }

    // =========================================================================
    // NGD PLANNER (similar approach)
    // =========================================================================
    std::cout << "\n=== Running NGD Optimization (with data collection) ===\n";

    auto task_ngd = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_ngd = std::make_shared<NGDMotionPlanner>(task_ngd);

    NGDConfig ngd_config;
    if (!ngd_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load NGD configuration from file\n";
        return -1;
    }

    std::cout << "\n=== Initializing NGD Planner ===\n";
    if (!planner_ngd->initialize(ngd_config)) {
        std::cerr << "Error: NGD Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        std::cout << "Showing NGD initial state visualization...\n";
        visualizeInitialState(task_ngd->getObstacles(),
                            planner_ngd->getCurrentTrajectory(),
                            "NGD - Initial State");
    }

    // Collect NGD history
    OptimizationHistory ngd_history;
    ngd_history.clear();

    // NGD optimization with data collection
    std::cout << "\n=== Running NGD Optimization ===\n";
    bool success_ngd = planner_ngd->solve();
    
    auto traj_history_ngd = planner_ngd->getTrajectoryHistory();
    for (size_t i = 0; i < traj_history_ngd.size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = traj_history_ngd[i];
        iter_data.total_cost = task_ngd->computeCollisionCost(traj_history_ngd[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        ngd_history.addIteration(iter_data);
    }

    ngd_history.final_trajectory = planner_ngd->getCurrentTrajectory();
    ngd_history.final_cost = task_ngd->computeCollisionCost(ngd_history.final_trajectory);
    ngd_history.converged = success_ngd;
    ngd_history.total_iterations = traj_history_ngd.size();

    // =========================================================================
    // VISUALIZATION WITH COST PLOTS
    // =========================================================================
    std::cout << "\n\n=== Interactive Optimization Visualization ===\n";

    auto obstacle_map_ptr = task_pce->getObstacleMap();
    OptimizationVisualizer visualizer(900, 700);

    // Show PCE trajectory evolution
    std::cout << "\nDisplaying PCE trajectory evolution...\n";
    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, pce_history, "PCEM - Trajectory Evolution");

    // Show PCE cost convergence
    std::cout << "\nDisplaying PCE cost convergence...\n";
    visualizer.showCostPlot(pce_history, "PCEM - Cost Convergence");

    // Show NGD trajectory evolution
    std::cout << "\nDisplaying NGD trajectory evolution...\n";
    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, ngd_history, "NGD - Trajectory Evolution");

    // Show NGD cost convergence
    std::cout << "\nDisplaying NGD cost convergence...\n";
    visualizer.showCostPlot(ngd_history, "NGD - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   Results Summary\n";
    std::cout << "=================================================\n";
    std::cout << "PCE Planner:\n";
    std::cout << "  Status: " << (success_pce ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Iterations: " << pce_history.total_iterations << "\n";
    std::cout << "  Initial cost: " << pce_history.iterations.front().total_cost << "\n";
    std::cout << "  Final cost: " << pce_history.final_cost << "\n";
    
    std::cout << "\nNGD Planner:\n";
    std::cout << "  Status: " << (success_ngd ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Iterations: " << ngd_history.total_iterations << "\n";
    std::cout << "  Initial cost: " << ngd_history.iterations.front().total_cost << "\n";
    std::cout << "  Final cost: " << ngd_history.final_cost << "\n";
    
    std::cout << "\nUse arrow keys to navigate iterations in the visualizer.\n";
    std::cout << "Press 'P' to save high-resolution PNG.\n";
    std::cout << "=================================================\n";

    return 0;
}