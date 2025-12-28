#include "StompMotionPlanner.h"
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
    std::cout << "   STOMP Motion Planning\n";
    std::cout << "=================================================\n\n";

    // Load configuration
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path config_path = source_path.parent_path() / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"]["visualize_initial_state"].as<bool>(false);

    // =========================================================================
    // STOMP PLANNER
    // =========================================================================
    std::cout << "\n=== STOMP Planner ===\n";

    auto task_stomp = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_stomp = std::make_shared<pce::StompMotionPlanner>(task_stomp);

    pce::StompPlannerConfig stomp_config;
    if (!stomp_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load STOMP configuration, setting defaults...\n";
        if (const auto& mp = config["motion_planning"]) {
            stomp_config.num_timesteps = mp["num_nodes"].as<size_t>(50);
            stomp_config.num_dimensions = mp["num_dimensions"].as<size_t>(2);
            if (mp["start_position"]) stomp_config.start_position = mp["start_position"].as<std::vector<float>>();
            if (mp["goal_position"]) stomp_config.goal_position = mp["goal_position"].as<std::vector<float>>();
        }
    }

    bool stomp_initialized = planner_stomp->initialize(stomp_config);
    if (!stomp_initialized) {
        std::cerr << "Error: STOMP Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        visualizeInitialState(task_stomp->getObstacles(), planner_stomp->getCurrentTrajectory(),
                              "STOMP - Initial State");
    }

    std::cout << "\n=== Running STOMP Optimization ===\n";
    bool success_stomp = planner_stomp->solve();

    // =========================================================================
    // Convert STOMP history using StompAdapter
    // =========================================================================
    // Debug: Print costs from planner's history before conversion
    const auto& planner_history = planner_stomp->getOptimizationHistory();
    std::cout << "\n[DEBUG] Planner history has " << planner_history.iterations.size() << " iterations\n";
    if (!planner_history.iterations.empty()) {
        const auto& last_planner_iter = planner_history.iterations.back();
        std::cout << "[DEBUG] Last planner iteration: cost=" << last_planner_iter.cost
                  << ", collision=" << last_planner_iter.collision_cost
                  << ", smooth=" << last_planner_iter.smoothness_cost << "\n";
    }

    OptimizationHistory stomp_history = StompAdapter::convert(planner_history);

    // Debug: Print costs after conversion
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

    // Verify trajectory structure before visualization
    std::cout << "\n[DEBUG] Verifying final trajectory:\n";
    const auto& final_traj = stomp_history.final_trajectory;
    std::cout << "  Number of nodes: " << final_traj.nodes.size() << "\n";
    std::cout << "  start_index: " << final_traj.start_index << ", goal_index: " << final_traj.goal_index << "\n";
    if (!final_traj.nodes.empty()) {
        std::cout << "  First node position size: " << final_traj.nodes[0].position.size() << "\n";
        if (final_traj.nodes[0].position.size() >= 2) {
            std::cout << "  First 5 nodes:\n";
            for (size_t i = 0; i < std::min(size_t(5), final_traj.nodes.size()); ++i) {
                std::cout << "    [" << i << "] (" << final_traj.nodes[i].position(0) 
                          << ", " << final_traj.nodes[i].position(1) << ")\n";
            }
            std::cout << "  Last 3 nodes:\n";
            size_t n = final_traj.nodes.size();
            for (size_t i = n - 3; i < n; ++i) {
                std::cout << "    [" << i << "] (" << final_traj.nodes[i].position(0) 
                          << ", " << final_traj.nodes[i].position(1) << ")\n";
            }
        }
    }
    
    // Also verify the first iteration's trajectory
    if (!stomp_history.iterations.empty()) {
        std::cout << "\n[DEBUG] Verifying first iteration trajectory:\n";
        const auto& first_traj = stomp_history.iterations[0].mean_trajectory;
        std::cout << "  Number of nodes: " << first_traj.nodes.size() << "\n";
        if (!first_traj.nodes.empty() && first_traj.nodes[0].position.size() >= 2) {
            std::cout << "  First 3 nodes:\n";
            for (size_t i = 0; i < std::min(size_t(3), first_traj.nodes.size()); ++i) {
                std::cout << "    [" << i << "] (" << first_traj.nodes[i].position(0) 
                          << ", " << first_traj.nodes[i].position(1) << ")\n";
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

    std::cout << (success_stomp ? "\n✓ STOMP optimization completed successfully\n"
                                : "\n✗ STOMP optimization failed\n");

    // =========================================================================
    // VISUALIZATION
    // =========================================================================
    std::cout << "\n=== Interactive Optimization Visualization ===\n";

    auto obstacle_map_ptr = task_stomp->getObstacleMap();
    OptimizationVisualizer visualizer(900, 700);

    // Show hint about sample visualization if samples are available
    if (!stomp_history.iterations.empty() && !stomp_history.iterations[0].samples.empty()) {
        std::cout << "  Samples available - press 'A' to toggle sample visualization\n";
    }

    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, stomp_history, "STOMP - Trajectory Evolution");
    visualizer.showCostPlot(stomp_history, "STOMP - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   STOMP Results Summary\n";
    std::cout << "=================================================\n";

    // Get costs directly from planner's internal optimization history (known to be correct)
    const auto& internal_history = planner_stomp->getOptimizationHistory();
    
    float final_collision = 0.0f;
    float final_smoothness = 0.0f;
    float final_total = planner_stomp->getFinalCost();
    float initial_cost = final_total;  // Default to final if no history
    
    if (!internal_history.iterations.empty()) {
        const auto& final_iter = internal_history.iterations.back();
        final_collision = final_iter.collision_cost;
        final_smoothness = final_iter.smoothness_cost;
        final_total = final_iter.cost;  // Use .cost not .total_cost (pce::StompIterationData field name)
        
        // Get initial cost for improvement calculation
        initial_cost = internal_history.iterations.front().cost;
        if (initial_cost < 1e-6f) {
            // Initial iteration might have cost=0, use second iteration if available
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

    std::cout << "\n=================================================\n";

    return success_stomp ? 0 : 1;
}