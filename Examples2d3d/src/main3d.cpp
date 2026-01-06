#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CasadiMotionPlanner.h"
#include "StompMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include "IterationData.h"
#include "OptimizationVisualizer.h"
#include <filesystem>
#include <map>

// Helper function to check if trajectory is 3D
bool is3DTrajectory(const Trajectory& traj) {
    if (traj.nodes.empty()) return false;
    return traj.nodes[0].position.size() >= 3;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "   3D Motion Planning Experiments\n";
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
    // PCE PLANNER (3D)
    // =========================================================================
    std::cout << "\n=== PCE Planner (3D) ===\n";

    auto task_pce = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_pce = std::make_shared<ProximalCrossEntropyMotionPlanner>(task_pce);

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

    // Verify 3D trajectory
    if (!is3DTrajectory(planner_pce->getCurrentTrajectory())) {
        std::cerr << "Warning: PCE trajectory is not 3D!\n";
    }

    // PCE Optimization with data collection
    std::cout << "\n=== Running PCE 3D Optimization ===\n";
    
    OptimizationHistory pce_history;
    pce_history.clear();
    
    // Store initial state
    {
        IterationData init_data;
        init_data.iteration = 0;
        init_data.mean_trajectory = planner_pce->getCurrentTrajectory();
        init_data.total_cost = task_pce->computeStateCost(init_data.mean_trajectory);
        init_data.collision_cost = init_data.total_cost;
        init_data.smoothness_cost = 0.0f;
        pce_history.addIteration(init_data);
    }
    
    bool success_pce = planner_pce->solve();
    
    auto traj_history_pce = planner_pce->getTrajectoryHistory();
    for (size_t i = 0; i < traj_history_pce.size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = traj_history_pce[i];
        iter_data.total_cost = task_pce->computeStateCost(traj_history_pce[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = planner_pce->computeSmoothnessCost(traj_history_pce[i]);
        pce_history.addIteration(iter_data);
    }
    
    pce_history.final_trajectory = planner_pce->getCurrentTrajectory();
    pce_history.final_cost = task_pce->computeStateCost(pce_history.final_trajectory) + planner_pce->computeSmoothnessCost(pce_history.final_trajectory);
    pce_history.converged = success_pce;
    pce_history.total_iterations = traj_history_pce.size();

    if (success_pce) {
        std::cout << "\n✓ PCE 3D optimization completed successfully\n";
    } else {
        std::cout << "\n✗ PCE 3D optimization failed\n";
    }

    // =========================================================================
    // NGD PLANNER (3D)
    // =========================================================================
    std::cout << "\n=== NGD Planner (3D) ===\n";

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

    OptimizationHistory ngd_history;
    ngd_history.clear();

    std::cout << "\n=== Running NGD 3D Optimization ===\n";
    bool success_ngd = planner_ngd->solve();
    
    auto traj_history_ngd = planner_ngd->getTrajectoryHistory();
    for (size_t i = 0; i < traj_history_ngd.size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = traj_history_ngd[i];
        iter_data.total_cost = task_ngd->computeStateCost(traj_history_ngd[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = planner_ngd->computeSmoothnessCost(traj_history_ngd[i]);
        ngd_history.addIteration(iter_data);
    }

    ngd_history.final_trajectory = planner_ngd->getCurrentTrajectory();
    ngd_history.final_cost = task_ngd->computeStateCost(ngd_history.final_trajectory) + planner_ngd->computeSmoothnessCost(ngd_history.final_trajectory);
    ngd_history.converged = success_ngd;
    ngd_history.total_iterations = traj_history_ngd.size();

    if (success_ngd) {
        std::cout << "\n✓ NGD 3D optimization completed successfully\n";
    } else {
        std::cout << "\n✗ NGD 3D optimization failed\n";
    }

    // =========================================================================
    // 3D VISUALIZATION WITH MULTI-VIEW PROJECTIONS
    // =========================================================================
    std::cout << "\n\n=== Interactive 3D Optimization Visualization ===\n";
    std::cout << "Using multi-view projection (XY, XZ, YZ planes)\n\n";

    auto obstacle_map_ptr = task_pce->getObstacleMap();
    OptimizationVisualizer visualizer(800, 600);  // Base size, 3D will use 1200x600

    // Show PCE 3D trajectory evolution (multi-view)
    std::cout << "Displaying PCE 3D trajectory evolution...\n";
    visualizer.setOutputPrefix("pce_3d");
    visualizer.showTrajectoryEvolution3D(*obstacle_map_ptr, pce_history, 
                                          "PCEM 3D - Trajectory Evolution");

    // Show PCE cost convergence
    std::cout << "Displaying PCE cost convergence...\n";
    visualizer.showCostPlot(pce_history, "PCEM 3D - Cost Convergence");

    // Show NGD 3D trajectory evolution
    std::cout << "Displaying NGD 3D trajectory evolution...\n";
    visualizer.setOutputPrefix("ngd_3d");
    visualizer.showTrajectoryEvolution3D(*obstacle_map_ptr, ngd_history, 
                                          "NGD 3D - Trajectory Evolution");

    std::cout << "Displaying NGD cost convergence...\n";
    visualizer.showCostPlot(ngd_history, "NGD 3D - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   3D Results Summary (Unified Cost Evaluation)\n";
    std::cout << "=================================================\n";

    auto evaluate_final = [&](const Trajectory& traj) {
        float collision = task_pce->computeStateCost(traj);
        float smoothness = planner_pce->computeSmoothnessCost(traj);
        return std::make_pair(collision + smoothness, collision);
    };

    auto pce_final_res = evaluate_final(planner_pce->getCurrentTrajectory());
    auto ngd_final_res = evaluate_final(planner_ngd->getCurrentTrajectory());

    std::cout << std::left << std::setw(18) << "Planner" 
              << std::setw(12) << "Status" 
              << std::setw(10) << "Iters" 
              << std::setw(15) << "Total Cost" 
              << "Collision Cost\n";
    std::cout << std::string(70, '-') << "\n";

    std::cout << std::left << std::setw(18) << "PCE (3D)" 
              << std::setw(12) << (success_pce ? "SUCCESS" : "FAILED")
              << std::setw(10) << pce_history.total_iterations
              << std::setw(15) << pce_final_res.first
              << pce_final_res.second << "\n";

    std::cout << std::left << std::setw(18) << "NGD (3D)" 
              << std::setw(12) << (success_ngd ? "SUCCESS" : "FAILED")
              << std::setw(10) << ngd_history.total_iterations
              << std::setw(15) << ngd_final_res.first
              << ngd_final_res.second << "\n";

    std::cout << std::string(70, '-') << "\n";

    // Find best result
    std::string best_planner = "PCE (3D)";
    float best_cost = pce_final_res.first;
    
    if (ngd_final_res.first < best_cost) {
        best_cost = ngd_final_res.first;
        best_planner = "NGD (3D)";
    }
    
    std::cout << "\n★ Best 3D result: " << best_planner << " with total cost = " << best_cost << "\n";

    // Print final 3D trajectory endpoints for verification
    std::cout << "\n=== Final Trajectory Endpoints ===\n";
    const auto& final_traj = planner_pce->getCurrentTrajectory();
    if (!final_traj.nodes.empty() && is3DTrajectory(final_traj)) {
        const auto& start = final_traj.nodes[final_traj.start_index].position;
        const auto& goal = final_traj.nodes[final_traj.goal_index].position;
        std::cout << "PCE Start: [" << start(0) << ", " << start(1) << ", " << start(2) << "]\n";
        std::cout << "PCE Goal:  [" << goal(0) << ", " << goal(1) << ", " << goal(2) << "]\n";
    }

    std::cout << "\n=================================================\n";
    std::cout << "   3D Motion Planning Experiments Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
