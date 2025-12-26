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

    OptimizationHistory stomp_history;
    
    // Record initial state
    {
        IterationData init_data;
        init_data.iteration = 0;
        init_data.mean_trajectory = planner_stomp->getCurrentTrajectory();
        init_data.collision_cost = task_stomp->computeStateCost(init_data.mean_trajectory);
        init_data.smoothness_cost = planner_stomp->computeSmoothnessCost(init_data.mean_trajectory);
        init_data.total_cost = init_data.collision_cost + init_data.smoothness_cost;
        stomp_history.addIteration(init_data);
    }

    std::cout << "\n=== Running STOMP Optimization ===\n";
    bool success_stomp = planner_stomp->solve();

    // Record iteration history
    for (size_t i = 0; i < planner_stomp->getTrajectoryHistory().size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i + 1;
        iter_data.mean_trajectory = planner_stomp->getTrajectoryHistory()[i];
        iter_data.collision_cost = task_stomp->computeStateCost(iter_data.mean_trajectory);
        iter_data.smoothness_cost = planner_stomp->computeSmoothnessCost(iter_data.mean_trajectory);
        iter_data.total_cost = iter_data.collision_cost + iter_data.smoothness_cost;
        stomp_history.addIteration(iter_data);
    }

    stomp_history.final_trajectory = planner_stomp->getCurrentTrajectory();
    stomp_history.final_cost = task_stomp->computeStateCost(stomp_history.final_trajectory) +
                               planner_stomp->computeSmoothnessCost(stomp_history.final_trajectory);
    stomp_history.converged = success_stomp;
    stomp_history.total_iterations = planner_stomp->getTrajectoryHistory().size();

    std::cout << (success_stomp ? "\n✓ STOMP optimization completed successfully\n"
                                : "\n✗ STOMP optimization failed\n");

    // =========================================================================
    // VISUALIZATION
    // =========================================================================
    std::cout << "\n=== Interactive Optimization Visualization ===\n";

    auto obstacle_map_ptr = task_stomp->getObstacleMap();
    OptimizationVisualizer visualizer(900, 700);

    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, stomp_history, "STOMP - Trajectory Evolution");
    visualizer.showCostPlot(stomp_history, "STOMP - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   STOMP Results Summary\n";
    std::cout << "=================================================\n";

    float final_collision = task_stomp->computeStateCost(planner_stomp->getCurrentTrajectory());
    float final_smoothness = planner_stomp->computeSmoothnessCost(planner_stomp->getCurrentTrajectory());
    float final_total = final_collision + final_smoothness;

    std::cout << std::left << std::setw(20) << "Status:" 
              << (success_stomp ? "SUCCESS" : "FAILED") << "\n";
    std::cout << std::left << std::setw(20) << "Iterations:" 
              << stomp_history.total_iterations << "\n";
    std::cout << std::left << std::setw(20) << "Collision Cost:" 
              << final_collision << "\n";
    std::cout << std::left << std::setw(20) << "Smoothness Cost:" 
              << final_smoothness << "\n";
    std::cout << std::left << std::setw(20) << "Total Cost:" 
              << final_total << "\n";

    std::cout << "\n=================================================\n";

    return success_stomp ? 0 : 1;
}