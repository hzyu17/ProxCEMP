#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CasadiMotionPlanner.h"
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
        init_data.total_cost = task_pce->computeStateCost(init_data.mean_trajectory);
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
        iter_data.total_cost = task_pce->computeStateCost(traj_history_pce[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        // Note: samples not available from history, leave empty
        pce_history.addIteration(iter_data);
    }
    
    pce_history.final_trajectory = planner_pce->getCurrentTrajectory();
    pce_history.final_cost = task_pce->computeStateCost(pce_history.final_trajectory);
    pce_history.converged = success_pce;
    pce_history.total_iterations = traj_history_pce.size();

    if (success_pce) {
        std::cout << "\n✓ PCE optimization completed successfully\n";
    } else {
        std::cout << "\n✗ PCE optimization failed\n";
    }

    // =========================================================================
    // NGD PLANNER
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
        iter_data.total_cost = task_ngd->computeStateCost(traj_history_ngd[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        ngd_history.addIteration(iter_data);
    }

    ngd_history.final_trajectory = planner_ngd->getCurrentTrajectory();
    ngd_history.final_cost = task_ngd->computeStateCost(ngd_history.final_trajectory);
    ngd_history.converged = success_ngd;
    ngd_history.total_iterations = traj_history_ngd.size();

    if (success_ngd) {
        std::cout << "\n✓ NGD optimization completed successfully\n";
    } else {
        std::cout << "\n✗ NGD optimization failed\n";
    }

    // =========================================================================
    // CASADI PLANNER
    // =========================================================================
    std::cout << "\n=== CasADi Planner ===\n";

    // Create task for CasADi
    auto task_casadi = std::make_shared<pce::CollisionAvoidanceTask>(config);

    // Create CasADi planner and set task
    auto planner_casadi = std::make_shared<CasADiMotionPlanner>(task_casadi);

    // Load CasADi configuration
    CasADiConfig casadi_config;
    if (!casadi_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load CasADi configuration from file\n";
        return -1;
    }

    std::cout << "\n=== Initializing CasADi Planner ===\n";
    if (!planner_casadi->initialize(casadi_config)) {
        std::cerr << "Error: CasADi Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        std::cout << "Showing CasADi initial state visualization...\n";
        visualizeInitialState(task_casadi->getObstacles(),
                            planner_casadi->getCurrentTrajectory(),
                            "CasADi - Initial State");
    }

    // =========================================================================
    // CASADI OPTIMIZATION WITH DATA COLLECTION
    // =========================================================================
    std::cout << "\n=== Running CasADi Optimization (with data collection) ===\n";

    OptimizationHistory casadi_history;
    casadi_history.clear();

    // Store initial state
    {
        IterationData init_data;
        init_data.iteration = 0;
        init_data.mean_trajectory = planner_casadi->getCurrentTrajectory();
        init_data.total_cost = task_casadi->computeStateCost(init_data.mean_trajectory);
        init_data.collision_cost = init_data.total_cost;
        init_data.smoothness_cost = 0.0f;
        casadi_history.addIteration(init_data);
    }

    // Run CasADi optimization
    bool success_casadi = planner_casadi->solve();

    // Convert trajectory history to OptimizationHistory format
    auto traj_history_casadi = planner_casadi->getTrajectoryHistory();
    for (size_t i = 0; i < traj_history_casadi.size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = traj_history_casadi[i];
        iter_data.total_cost = task_casadi->computeStateCost(traj_history_casadi[i]);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        casadi_history.addIteration(iter_data);
    }

    casadi_history.final_trajectory = planner_casadi->getCurrentTrajectory();
    casadi_history.final_cost = task_casadi->computeStateCost(casadi_history.final_trajectory);
    casadi_history.converged = success_casadi;
    casadi_history.total_iterations = traj_history_casadi.size();

    if (success_casadi) {
        std::cout << "\n✓ CasADi optimization completed successfully\n";
    } else {
        std::cout << "\n✗ CasADi optimization failed\n";
    }

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

    // Show CasADi trajectory evolution
    std::cout << "\nDisplaying CasADi trajectory evolution...\n";
    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, casadi_history, "CasADi - Trajectory Evolution");

    // Show CasADi cost convergence
    std::cout << "\nDisplaying CasADi cost convergence...\n";
    visualizer.showCostPlot(casadi_history, "CasADi - Cost Convergence");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   Results Summary (Unified Cost Evaluation)\n";
    std::cout << "=================================================\n";

    // Define a lambda for consistent evaluation using the same task logic
    auto evaluate_final = [&](const Trajectory& traj) {
        float collision = task_pce->computeStateCost(traj);
        float smoothness = planner_pce->computeSmoothnessCost(traj); // Smoothness logic is usually in the base planner
        return std::make_pair(collision + smoothness, collision);
    };

    auto pce_final_res = evaluate_final(planner_pce->getCurrentTrajectory());
    auto ngd_final_res = evaluate_final(planner_ngd->getCurrentTrajectory());
    auto casadi_final_res = evaluate_final(planner_casadi->getCurrentTrajectory());

    std::cout << std::left << std::setw(15) << "Planner" 
              << std::setw(12) << "Status" 
              << std::setw(12) << "Iters" 
              << std::setw(15) << "Total Cost" 
              << "Collision Cost\n";
    std::cout << std::string(65, '-') << "\n";

    std::cout << std::left << std::setw(15) << "PCE" 
              << std::setw(12) << (success_pce ? "SUCCESS" : "FAILED")
              << std::setw(12) << pce_history.total_iterations
              << std::setw(15) << pce_final_res.first
              << pce_final_res.second << "\n";

    std::cout << std::left << std::setw(15) << "NGD" 
              << std::setw(12) << (success_ngd ? "SUCCESS" : "FAILED")
              << std::setw(12) << ngd_history.total_iterations
              << std::setw(15) << ngd_final_res.first
              << ngd_final_res.second << "\n";

    std::cout << std::left << std::setw(15) << "CasADi" 
              << std::setw(12) << (success_casadi ? "SUCCESS" : "FAILED")
              << std::setw(12) << casadi_history.total_iterations
              << std::setw(15) << casadi_final_res.first
              << casadi_final_res.second << "\n";

    std::cout << "\n=================================================\n";

    return 0;
}