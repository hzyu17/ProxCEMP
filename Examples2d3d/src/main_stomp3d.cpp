#include "StompMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include "IterationData.h"
#include "OptimizationVisualizer.h"
#include <filesystem>
#include <iomanip>

// Helper function to check if trajectory is 3D
bool is3DTrajectory(const Trajectory& traj) {
    if (traj.nodes.empty()) return false;
    return traj.nodes[0].position.size() >= 3;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "   STOMP 3D Motion Planning\n";
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
    std::cout << "==================================\n";

    // =========================================================================
    // STOMP PLANNER (3D)
    // =========================================================================
    std::cout << "\n=== STOMP Planner (3D) ===\n";

    auto task_stomp = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_stomp = std::make_shared<pce::StompMotionPlanner>(task_stomp);

    pce::StompPlannerConfig stomp_config;
    if (!stomp_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load STOMP configuration, setting defaults for 3D...\n";
        if (const auto& mp = config["motion_planning"]) {
            stomp_config.num_timesteps = mp["num_nodes"].as<size_t>(50);
            stomp_config.num_dimensions = mp["num_dimensions"].as<size_t>(3);
            if (mp["start_position"]) stomp_config.start_position = mp["start_position"].as<std::vector<float>>();
            if (mp["goal_position"]) stomp_config.goal_position = mp["goal_position"].as<std::vector<float>>();
        }
    }

    std::cout << "\n=== Initializing STOMP Planner (3D) ===\n";
    bool stomp_initialized = planner_stomp->initialize(stomp_config);
    if (!stomp_initialized) {
        std::cerr << "Error: STOMP Planner initialization failed\n";
        return 1;
    }

    // Verify 3D trajectory
    if (!is3DTrajectory(planner_stomp->getCurrentTrajectory())) {
        std::cerr << "Warning: STOMP trajectory is not 3D!\n";
    }

    if (visualize) {
        visualizeInitialState(task_stomp->getObstacles(), planner_stomp->getCurrentTrajectory(),
                              "STOMP 3D - Initial State");
    }

    std::cout << "\n=== Running STOMP 3D Optimization ===\n";
    bool success_stomp = planner_stomp->solve();

    // =========================================================================
    // Convert STOMP history using StompAdapter
    // =========================================================================
    const auto& planner_history = planner_stomp->getOptimizationHistory();
    std::cout << "\n[DEBUG] Planner history has " << planner_history.iterations.size() << " iterations\n";
    if (!planner_history.iterations.empty()) {
        const auto& last_planner_iter = planner_history.iterations.back();
        std::cout << "[DEBUG] Last planner iteration: cost=" << last_planner_iter.cost
                  << ", collision=" << last_planner_iter.collision_cost
                  << ", smooth=" << last_planner_iter.smoothness_cost << "\n";
    }

    OptimizationHistory stomp_history = StompAdapter::convert(planner_history);

    std::cout << "[DEBUG] Converted history has " << stomp_history.iterations.size() << " iterations\n";
    if (!stomp_history.iterations.empty()) {
        const auto& last_vis_iter = stomp_history.iterations.back();
        std::cout << "[DEBUG] Last converted iteration: total=" << last_vis_iter.total_cost
                  << ", collision=" << last_vis_iter.collision_cost
                  << ", smooth=" << last_vis_iter.smoothness_cost << "\n";
    }

    stomp_history.final_trajectory = planner_stomp->getCurrentTrajectory();
    stomp_history.final_cost = planner_stomp->getFinalCost();
    stomp_history.converged = success_stomp;
    stomp_history.total_iterations = planner_stomp->getNumIterations();

    // Verify 3D trajectory structure before visualization
    std::cout << "\n[DEBUG] Verifying final 3D trajectory:\n";
    const auto& final_traj = stomp_history.final_trajectory;
    std::cout << "  Number of nodes: " << final_traj.nodes.size() << "\n";
    std::cout << "  start_index: " << final_traj.start_index << ", goal_index: " << final_traj.goal_index << "\n";
    if (!final_traj.nodes.empty()) {
        std::cout << "  Position dimensions: " << final_traj.nodes[0].position.size() << "\n";
        if (final_traj.nodes[0].position.size() >= 3) {
            std::cout << "  First 5 nodes:\n";
            for (size_t i = 0; i < std::min(size_t(5), final_traj.nodes.size()); ++i) {
                std::cout << "    [" << i << "] (" << final_traj.nodes[i].position(0) 
                          << ", " << final_traj.nodes[i].position(1) 
                          << ", " << final_traj.nodes[i].position(2) << ")\n";
            }
            std::cout << "  Last 3 nodes:\n";
            size_t n = final_traj.nodes.size();
            for (size_t i = n - 3; i < n; ++i) {
                std::cout << "    [" << i << "] (" << final_traj.nodes[i].position(0) 
                          << ", " << final_traj.nodes[i].position(1) 
                          << ", " << final_traj.nodes[i].position(2) << ")\n";
            }
        }
    }
    
    // Also verify the first iteration's trajectory
    if (!stomp_history.iterations.empty()) {
        std::cout << "\n[DEBUG] Verifying first iteration 3D trajectory:\n";
        const auto& first_traj = stomp_history.iterations[0].mean_trajectory;
        std::cout << "  Number of nodes: " << first_traj.nodes.size() << "\n";
        if (!first_traj.nodes.empty() && first_traj.nodes[0].position.size() >= 3) {
            std::cout << "  First 3 nodes:\n";
            for (size_t i = 0; i < std::min(size_t(3), first_traj.nodes.size()); ++i) {
                std::cout << "    [" << i << "] (" << first_traj.nodes[i].position(0) 
                          << ", " << first_traj.nodes[i].position(1) 
                          << ", " << first_traj.nodes[i].position(2) << ")\n";
            }
        }
    }

    // Print sample statistics
    std::cout << "  Iterations collected: " << stomp_history.iterations.size() << "\n";
    if (!stomp_history.iterations.empty()) {
        size_t total_samples = 0;
        for (const auto& iter : stomp_history.iterations) {
            total_samples += iter.samples.size();
        }
        std::cout << "  Total samples captured: " << total_samples << "\n";
        if (total_samples > 0) {
            std::cout << "  Average samples/iteration: " 
                      << (total_samples / stomp_history.iterations.size()) << "\n";
        }
    }

    std::cout << (success_stomp ? "\n✓ STOMP 3D optimization completed successfully\n"
                                : "\n✗ STOMP 3D optimization failed\n");

    // =========================================================================
    // 3D VISUALIZATION WITH MULTI-VIEW PROJECTIONS
    // =========================================================================
    std::cout << "\n\n=== Interactive 3D Optimization Visualization ===\n";
    std::cout << "Using multi-view projection (XY, XZ, YZ planes)\n\n";

    auto obstacle_map_ptr = task_stomp->getObstacleMap();
    OptimizationVisualizer visualizer(800, 600);  // Base size, 3D will use 1200x600

    // Show hint about sample visualization if samples are available
    if (!stomp_history.iterations.empty() && !stomp_history.iterations[0].samples.empty()) {
        std::cout << "  Samples available - press 'A' to toggle sample visualization\n";
    }

    // Use 3D visualization (multi-view projection)
    std::cout << "Displaying STOMP 3D trajectory evolution...\n";
    visualizer.setOutputPrefix("stomp_3d");
    visualizer.showTrajectoryEvolution3D(*obstacle_map_ptr, stomp_history, 
                                          "STOMP 3D - Trajectory Evolution");

    std::cout << "Displaying STOMP 3D cost convergence...\n";
    visualizer.showCostPlot(stomp_history, "STOMP 3D - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   STOMP 3D Results Summary\n";
    std::cout << "=================================================\n";

    // Get costs directly from planner's internal optimization history
    const auto& internal_history = planner_stomp->getOptimizationHistory();
    
    float final_collision = 0.0f;
    float final_smoothness = 0.0f;
    float final_total = planner_stomp->getFinalCost();
    float initial_cost = final_total;
    
    if (!internal_history.iterations.empty()) {
        const auto& final_iter = internal_history.iterations.back();
        final_collision = final_iter.collision_cost;
        final_smoothness = final_iter.smoothness_cost;
        final_total = final_iter.cost;
        
        // Get initial cost for improvement calculation
        initial_cost = internal_history.iterations.front().cost;
        if (initial_cost < 1e-6f) {
            if (internal_history.iterations.size() > 1) {
                initial_cost = internal_history.iterations[1].cost;
            }
        }
    }

    std::cout << std::left << std::setw(20) << "Status:" 
              << (success_stomp ? "SUCCESS" : "FAILED") << "\n";
    std::cout << std::left << std::setw(20) << "Iterations:" 
              << stomp_history.total_iterations << "\n";
    std::cout << std::left << std::setw(20) << "Collision Cost:" 
              << std::fixed << std::setprecision(4) << final_collision << "\n";
    std::cout << std::left << std::setw(20) << "Smoothness Cost:" 
              << std::fixed << std::setprecision(4) << final_smoothness << "\n";
    std::cout << std::left << std::setw(20) << "Total Cost:" 
              << std::fixed << std::setprecision(4) << final_total << "\n";

    // Show cost improvement
    if (initial_cost > 1e-6f) {
        float improvement = (initial_cost - final_total) / initial_cost * 100.0f;
        std::cout << std::left << std::setw(20) << "Cost Reduction:" 
                  << std::fixed << std::setprecision(1) << improvement << "%\n";
    }

    // Print final 3D trajectory endpoints
    std::cout << "\n=== Final 3D Trajectory Endpoints ===\n";
    if (!final_traj.nodes.empty() && is3DTrajectory(final_traj)) {
        const auto& start = final_traj.nodes[final_traj.start_index].position;
        const auto& goal = final_traj.nodes[final_traj.goal_index].position;
        std::cout << "Start: [" << start(0) << ", " << start(1) << ", " << start(2) << "]\n";
        std::cout << "Goal:  [" << goal(0) << ", " << goal(1) << ", " << goal(2) << "]\n";
    }

    std::cout << "\n=================================================\n";
    std::cout << "   STOMP 3D Motion Planning Complete\n";
    std::cout << "=================================================\n";

    return success_stomp ? 0 : 1;
}