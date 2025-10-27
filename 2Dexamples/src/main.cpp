#include "PCEMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
// #include "NGDMotionPlanner.h"
#include "visualization.h"
#include "visualization_base.h"
#include <filesystem>

int main() {
    std::cout << "=================================================\n";
    std::cout << "   Motion Planning Experiments\n";
    std::cout << "=================================================\n\n";

    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    // Handle visualization in main() BEFORE calling solve()
    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"] && 
                    config["experiment"]["visualize_initial_state"] &&
                    config["experiment"]["visualize_initial_state"].as<bool>();

    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);
        
    // 4. Create PCE planner and set task
    auto planner_pce = std::make_shared<ProximalCrossEntropyMotionPlanner>(task);

    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return -1;
    }

    std::cout << "\n=== Initializing Planner ===\n";
    if (!planner_pce->initialize(pce_config)) {
        std::cerr << "Error: Planner initialization failed\n";
        return 1;
    }
        
    if (visualize) {        
        // Show visualization
        std::cout << "Showing initial state visualization...\n";
        visualizeInitialState(task->getObstacles(),
                              planner_pce->getCurrentTrajectory(),
                             "PCEM - Initial State");
    }
    
    // Now run optimization
    bool success_pce = planner_pce->solve();
    
    // // --- Run NGD ---
    // std::cout << "\n=== NGD Planner ===\n";
    // NGDMotionPlanner planner_ngd;
    
    // if (visualize) {
    //     planner_ngd.initializeOnly(config_file);
    //     visualizeInitialState(planner_ngd.getObstacles(),
    //                          planner_ngd.getCurrentTrajectory(),
    //                          "NGD - Initial State");
    // }
    
    // bool success_ngd = planner_ngd.solve(config_file);
    
    // --- Final visualizations ---
    // std::cout << "\n--- Showing Final Results ---\n";
    auto traj_history = planner_pce->getTrajectoryHistory();

    auto obstacle_map_ptr = task->getObstacleMap();
    
    showTrajectoryHistoryWindow(*obstacle_map_ptr, traj_history, "PCEM Results");
    // showPlannerWindow(planner_ngd, "NGD Results");

    saveTrajectoryHistoryToFile(*obstacle_map_ptr, traj_history, "pcem_final.png");
    
    // savePlannerToFile(*planner_pce, "pcem_final.png");
    // savePlannerToFile(planner_ngd, "ngd_final.png");
    
    return 0;
}