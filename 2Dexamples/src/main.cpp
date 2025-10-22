#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "visualization.h"  // Only include in main, not in MotionPlanner.h
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

    // --- Run PCEM ---
    std::cout << "\n=== PCEM Planner ===\n";
    ProximalCrossEntropyMotionPlanner planner_pce;
    
    // ✓ Handle visualization in main() BEFORE calling solve()
    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"] && 
                    config["experiment"]["visualize_initial_state"] &&
                    config["experiment"]["visualize_initial_state"].as<bool>();
    
    if (visualize) {
        // Initialize without running optimization
        planner_pce.initializeOnly(config_file);
        
        // Show visualization
        std::cout << "Showing initial state visualization...\n";
        visualizeInitialState(planner_pce.getObstacles(),
                             planner_pce.getCurrentTrajectory(),
                             "PCEM - Initial State");
    }
    
    // Now run optimization
    bool success_pce = planner_pce.solve(config_file);
    
    // --- Run NGD ---
    std::cout << "\n=== NGD Planner ===\n";
    NGDMotionPlanner planner_ngd;
    
    if (visualize) {
        planner_ngd.initializeOnly(config_file);
        visualizeInitialState(planner_ngd.getObstacles(),
                             planner_ngd.getCurrentTrajectory(),
                             "NGD - Initial State");
    }
    
    bool success_ngd = planner_ngd.solve(config_file);
    
    // --- Final visualizations ---
    std::cout << "\n--- Showing Final Results ---\n";
    showPlannerWindow(planner_pce, "PCEM Results");
    showPlannerWindow(planner_ngd, "NGD Results");

    savePlannerToFile(planner_pce, "pcem_final.png");
    savePlannerToFile(planner_ngd, "ngd_final.png");
    
    return 0;
}