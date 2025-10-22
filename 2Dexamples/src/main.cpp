#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include <filesystem>
#include "TrajectoryComparison.h"

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

    // Run PCE optimization
    std::cout << "\n=== Running PCE Optimization ===\n";
    bool success_pce = planner_pce->solve();

    if (success_pce) {
        std::cout << "\n✓ PCE optimization completed successfully\n";
    } else {
        std::cout << "\n✗ PCE optimization failed\n";
    }

    // =========================================================================
    // NGD PLANNER
    // =========================================================================
    std::cout << "\n\n=== NGD Planner ===\n";

    // Create task for NGD (can reuse same task type)
    auto task_ngd = std::make_shared<pce::CollisionAvoidanceTask>(config);

    // Create NGD planner and set task
    auto planner_ngd = std::make_shared<NGDMotionPlanner>(task_ngd);

    // Load NGD configuration
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

    // Visualize initial state if requested
    if (visualize) {
        std::cout << "Showing NGD initial state visualization...\n";
        visualizeInitialState(task_ngd->getObstacles(),
                            planner_ngd->getCurrentTrajectory(),
                            "NGD - Initial State");
    }

    // Run NGD optimization
    std::cout << "\n=== Running NGD Optimization ===\n";
    bool success_ngd = planner_ngd->optimize();

    if (success_ngd) {
        std::cout << "\n✓ NGD optimization completed successfully\n";
    } else {
        std::cout << "\n✗ NGD optimization failed\n";
    }

    // =========================================================================
    // FINAL VISUALIZATIONS
    // =========================================================================
    std::cout << "\n\n=== Showing Final Results ===\n";

    // Get trajectory histories
    auto traj_history_pce = planner_pce->getTrajectoryHistory();
    auto traj_history_ngd = planner_ngd->getTrajectoryHistory();

    // Get obstacle map
    auto obstacle_map_ptr = task_pce->getObstacleMap();

    // Show PCE results
    std::cout << "Displaying PCE results...\n";
    showTrajectoryHistoryWindow(*obstacle_map_ptr, traj_history_pce, "PCEM Results");
    saveTrajectoryHistoryToFile(*obstacle_map_ptr, traj_history_pce, "pcem_final.png");

    // Show NGD results
    std::cout << "Displaying NGD results...\n";
    showTrajectoryHistoryWindow(*obstacle_map_ptr, traj_history_ngd, "NGD Results");
    saveTrajectoryHistoryToFile(*obstacle_map_ptr, traj_history_ngd, "ngd_final.png");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   Results Summary\n";
    std::cout << "=================================================\n";
    std::cout << "PCE Planner: " << (success_pce ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "NGD Planner: " << (success_ngd ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "\nResults saved:\n";
    std::cout << "  - pcem_final.png\n";
    std::cout << "  - ngd_final.png\n";
    std::cout << "\nLog files saved to logs/ directory\n";
    std::cout << "=================================================\n";

    return 0;
}
