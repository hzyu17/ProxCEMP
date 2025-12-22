#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CasadiMotionPlanner.h"
#include "StompMotionPlanner.h"  // NEW: STOMP planner
#include "CollisionAvoidanceTask.h"
#include "ObstacleMap.h"
#include "visualization.h"
#include "visualization_base.h"
#include "IterationData.h"
#include "OptimizationVisualizer.h"
#include <filesystem>
#include <map>

int main() {

    {
        std::cout << "\n========================================\n";
        std::cout << "EARLY STOMP TEST (before any other code)\n";
        std::cout << "========================================\n";
        
        // Size checks
        std::cout << "Size checks:\n";
        std::cout << "  sizeof(stomp::StompConfiguration): " << sizeof(stomp::StompConfiguration) << "\n";
        std::cout << "  sizeof(std::shared_ptr<stomp::Task>): " << sizeof(std::shared_ptr<stomp::Task>) << "\n";
        
        // Create minimal config
        stomp::StompConfiguration cfg;
        cfg.num_timesteps = 20;
        cfg.num_iterations = 5;
        cfg.num_dimensions = 2;
        cfg.num_rollouts = 3;
        cfg.max_rollouts = 5;
        cfg.num_iterations_after_valid = 0;
        cfg.delta_t = 0.1;
        cfg.control_cost_weight = 0.0;
        cfg.exponentiated_cost_sensitivity = 10.0;
        cfg.initialization_method = stomp::TrajectoryInitializations::LINEAR_INTERPOLATION;
        
        // Create our task
        auto early_task = std::make_shared<pce::StompCollisionTask>();
        std::cout << "Task created, size=" << sizeof(pce::StompCollisionTask) << "\n";
        
        // Create STOMP
        std::cout << "Creating STOMP in early test...\n";
        stomp::Stomp* early_stomp = nullptr;
        try {
            early_stomp = new stomp::Stomp(cfg, early_task);
            std::cout << "EARLY TEST: STOMP created successfully!\n";
            
            // Quick solve test
            Eigen::VectorXd start(2), goal(2);
            start << 0.0, 0.0;
            goal << 100.0, 100.0;
            Eigen::MatrixXd result;
            bool success = early_stomp->solve(start, goal, result);
            std::cout << "EARLY TEST: solve() " << (success ? "SUCCESS" : "FAILED") << "\n";
            
            delete early_stomp;
            std::cout << "EARLY TEST: Cleanup successful\n";
        } catch (const std::exception& e) {
            std::cerr << "EARLY TEST FAILED: " << e.what() << "\n";
            return 1;
        }
        
        std::cout << "========================================\n";
        std::cout << "EARLY TEST PASSED - continuing with program\n";
        std::cout << "========================================\n\n";
    }

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
        init_data.smoothness_cost = 0.0f;
        pce_history.addIteration(init_data);
    }
    
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
    // STOMP PLANNER (NEW)
    // =========================================================================
    std::cout << "\n=== STOMP Planner ===\n";

    auto task_stomp = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_stomp = std::make_shared<pce::StompMotionPlanner>(task_stomp);

    pce::StompPlannerConfig stomp_config;
    bool stomp_config_loaded = stomp_config.loadFromFile(config_file);
    
    if (!stomp_config_loaded) {
        std::cerr << "Failed to load STOMP configuration, setting defaults manually...\n";
        
        // Set defaults from motion_planning section
        if (const auto& mp = config["motion_planning"]) {
            stomp_config.num_timesteps = mp["num_nodes"].as<size_t>(50);
            stomp_config.num_dimensions = mp["num_dimensions"].as<size_t>(2);
            if (mp["start_position"]) {
                stomp_config.start_position = mp["start_position"].as<std::vector<float>>();
            }
            if (mp["goal_position"]) {
                stomp_config.goal_position = mp["goal_position"].as<std::vector<float>>();
            }
        }
        
        // Verify we have valid start/goal
        if (stomp_config.start_position.empty() || stomp_config.goal_position.empty()) {
            std::cerr << "Error: Could not determine start/goal positions for STOMP\n";
            std::cerr << "Skipping STOMP planner...\n";
            // We'll handle this by checking initialized_ later
        }
    }
    
    // Print config for debugging
    std::cout << "STOMP Config:\n"
              << "  num_dimensions: " << stomp_config.num_dimensions << "\n"
              << "  num_timesteps: " << stomp_config.num_timesteps << "\n"
              << "  start_position size: " << stomp_config.start_position.size() << "\n"
              << "  goal_position size: " << stomp_config.goal_position.size() << "\n";

    std::cout << "\n=== Initializing STOMP Planner ===\n";
    bool stomp_initialized = planner_stomp->initialize(stomp_config);
    
    if (!stomp_initialized) {
        std::cerr << "Error: STOMP Planner initialization failed\n";
        // Continue with other planners instead of returning
    }

    if (stomp_initialized && visualize) {
        std::cout << "Showing STOMP initial state visualization...\n";
        visualizeInitialState(task_stomp->getObstacles(),
                            planner_stomp->getCurrentTrajectory(),
                            "STOMP - Initial State");
    }

    // Collect STOMP history
    OptimizationHistory stomp_history;
    stomp_history.clear();
    bool success_stomp = false;

    if (stomp_initialized) {
        // Store initial state
        {
            IterationData init_data;
            init_data.iteration = 0;
            init_data.mean_trajectory = planner_stomp->getCurrentTrajectory();
            init_data.total_cost = task_stomp->computeStateCost(init_data.mean_trajectory);
            init_data.collision_cost = init_data.total_cost;
            init_data.smoothness_cost = 0.0f;
            stomp_history.addIteration(init_data);
        }

        std::cout << "\n=== Running STOMP Optimization ===\n";
        success_stomp = planner_stomp->solve();

        auto traj_history_stomp = planner_stomp->getTrajectoryHistory();
        for (size_t i = 0; i < traj_history_stomp.size(); ++i) {
            IterationData iter_data;
            iter_data.iteration = i;
            iter_data.mean_trajectory = traj_history_stomp[i];
            iter_data.total_cost = task_stomp->computeStateCost(traj_history_stomp[i]);
            iter_data.collision_cost = iter_data.total_cost;
            iter_data.smoothness_cost = 0.0f;
            stomp_history.addIteration(iter_data);
        }

        stomp_history.final_trajectory = planner_stomp->getCurrentTrajectory();
        stomp_history.final_cost = task_stomp->computeStateCost(stomp_history.final_trajectory);
        stomp_history.converged = success_stomp;
        stomp_history.total_iterations = traj_history_stomp.size();

        if (success_stomp) {
            std::cout << "\n✓ STOMP optimization completed successfully\n";
        } else {
            std::cout << "\n✗ STOMP optimization failed\n";
        }
    } else {
        std::cout << "\n✗ STOMP skipped due to initialization failure\n";
    }

    // // =========================================================================
    // // CASADI PLANNERS - ALL SOLVERS
    // // =========================================================================
    
    // // Define all solvers to test (sqp excluded - needs qpoases library)
    // // std::vector<std::string> casadi_solvers = {"lbfgs", "ipopt", "gd", "adam"};
    // std::vector<std::string> casadi_solvers = {"lbfgs"};

    // // Store results for each solver
    // std::map<std::string, OptimizationHistory> casadi_histories;
    // std::map<std::string, bool> casadi_success;
    // std::map<std::string, std::shared_ptr<CasADiMotionPlanner>> casadi_planners;
    // std::map<std::string, std::shared_ptr<pce::CollisionAvoidanceTask>> casadi_tasks;
    
    // for (const auto& solver_name : casadi_solvers) {
    //     std::cout << "\n=== CasADi Planner (" << solver_name << ") ===\n";

    //     // Create fresh task for each solver
    //     auto task_casadi = std::make_shared<pce::CollisionAvoidanceTask>(config);
    //     casadi_tasks[solver_name] = task_casadi;

    //     // Create planner
    //     auto planner_casadi = std::make_shared<CasADiMotionPlanner>(task_casadi);
    //     casadi_planners[solver_name] = planner_casadi;

    //     // Load configuration
    //     CasADiConfig casadi_config;
    //     if (!casadi_config.loadFromFile(config_file)) {
    //         std::cerr << "Failed to load CasADi configuration from file\n";
    //         casadi_success[solver_name] = false;
    //         continue;
    //     }
        
    //     // Override solver type
    //     casadi_config.solver = solver_name;
    //     casadi_config.solver_type = stringToSolverType(solver_name);

    //     std::cout << "Initializing CasADi-" << solver_name << "...\n";
    //     if (!planner_casadi->initialize(casadi_config)) {
    //         std::cerr << "Error: CasADi-" << solver_name << " initialization failed\n";
    //         casadi_success[solver_name] = false;
    //         continue;
    //     }

    //     if (visualize) {
    //         std::cout << "Showing CasADi-" << solver_name << " initial state...\n";
    //         visualizeInitialState(task_casadi->getObstacles(),
    //                             planner_casadi->getCurrentTrajectory(),
    //                             "CasADi-" + solver_name + " - Initial State");
    //     }

    //     // Optimization with data collection
    //     std::cout << "Running CasADi-" << solver_name << " optimization...\n";

    //     OptimizationHistory history;
    //     history.clear();

    //     // Store initial state
    //     {
    //         IterationData init_data;
    //         init_data.iteration = 0;
    //         init_data.mean_trajectory = planner_casadi->getCurrentTrajectory();
    //         init_data.total_cost = task_casadi->computeStateCost(init_data.mean_trajectory);
    //         init_data.collision_cost = init_data.total_cost;
    //         init_data.smoothness_cost = 0.0f;
    //         history.addIteration(init_data);
    //     }

    //     // Run optimization
    //     bool success = planner_casadi->solve();
    //     casadi_success[solver_name] = success;

    //     // Convert trajectory history
    //     auto traj_history = planner_casadi->getTrajectoryHistory();
    //     for (size_t i = 0; i < traj_history.size(); ++i) {
    //         IterationData iter_data;
    //         iter_data.iteration = i;
    //         iter_data.mean_trajectory = traj_history[i];
    //         iter_data.total_cost = task_casadi->computeStateCost(traj_history[i]);
    //         iter_data.collision_cost = iter_data.total_cost;
    //         iter_data.smoothness_cost = 0.0f;
    //         history.addIteration(iter_data);
    //     }

    //     history.final_trajectory = planner_casadi->getCurrentTrajectory();
    //     history.final_cost = task_casadi->computeStateCost(history.final_trajectory);
    //     history.converged = success;
    //     history.total_iterations = traj_history.size();

    //     casadi_histories[solver_name] = history;

    //     if (success) {
    //         std::cout << "✓ CasADi-" << solver_name << " completed successfully\n";
    //     } else {
    //         std::cout << "✗ CasADi-" << solver_name << " failed\n";
    //     }
    // }

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

    // Show STOMP trajectory evolution (NEW)
    if (stomp_initialized) {
        std::cout << "\nDisplaying STOMP trajectory evolution...\n";
        visualizer.showTrajectoryEvolution(*obstacle_map_ptr, stomp_history, "STOMP - Trajectory Evolution");

        // Show STOMP cost convergence (NEW)
        std::cout << "\nDisplaying STOMP cost convergence...\n";
        visualizer.showCostPlot(stomp_history, "STOMP - Cost Convergence");
    }

    // // Show all CasADi solver results
    // for (const auto& solver_name : casadi_solvers) {
    //     if (casadi_histories.find(solver_name) == casadi_histories.end()) continue;
        
    //     const auto& history = casadi_histories[solver_name];
        
    //     std::cout << "\nDisplaying CasADi-" << solver_name << " trajectory evolution...\n";
    //     visualizer.showTrajectoryEvolution(*obstacle_map_ptr, history, 
    //                                        "CasADi-" + solver_name + " - Trajectory Evolution");

    //     std::cout << "Displaying CasADi-" << solver_name << " cost convergence...\n";
    //     visualizer.showCostPlot(history, "CasADi-" + solver_name + " - Cost Convergence");
    // }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   Results Summary (Unified Cost Evaluation)\n";
    std::cout << "=================================================\n";

    // Define a lambda for consistent evaluation
    auto evaluate_final = [&](const Trajectory& traj) {
        float collision = task_pce->computeStateCost(traj);
        float smoothness = planner_pce->computeSmoothnessCost(traj);
        return std::make_pair(collision + smoothness, collision);
    };

    auto pce_final_res = evaluate_final(planner_pce->getCurrentTrajectory());
    auto ngd_final_res = evaluate_final(planner_ngd->getCurrentTrajectory());
    
    // STOMP results (only if initialized)
    std::pair<float, float> stomp_final_res = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    if (stomp_initialized) {
        stomp_final_res = evaluate_final(planner_stomp->getCurrentTrajectory());
    }

    std::cout << std::left << std::setw(18) << "Planner" 
              << std::setw(12) << "Status" 
              << std::setw(10) << "Iters" 
              << std::setw(15) << "Total Cost" 
              << "Collision Cost\n";
    std::cout << std::string(70, '-') << "\n";

    std::cout << std::left << std::setw(18) << "PCE" 
              << std::setw(12) << (success_pce ? "SUCCESS" : "FAILED")
              << std::setw(10) << pce_history.total_iterations
              << std::setw(15) << pce_final_res.first
              << pce_final_res.second << "\n";

    std::cout << std::left << std::setw(18) << "NGD" 
              << std::setw(12) << (success_ngd ? "SUCCESS" : "FAILED")
              << std::setw(10) << ngd_history.total_iterations
              << std::setw(15) << ngd_final_res.first
              << ngd_final_res.second << "\n";

    // NEW: STOMP results
    if (stomp_initialized) {
        std::cout << std::left << std::setw(18) << "STOMP" 
                  << std::setw(12) << (success_stomp ? "SUCCESS" : "FAILED")
                  << std::setw(10) << stomp_history.total_iterations
                  << std::setw(15) << stomp_final_res.first
                  << stomp_final_res.second << "\n";
    } else {
        std::cout << std::left << std::setw(18) << "STOMP" 
                  << std::setw(12) << "SKIPPED"
                  << std::setw(10) << "-"
                  << std::setw(15) << "-"
                  << "-\n";
    }

    // // Print all CasADi solver results
    // for (const auto& solver_name : casadi_solvers) {
    //     if (casadi_planners.find(solver_name) == casadi_planners.end()) continue;
        
    //     auto casadi_final_res = evaluate_final(casadi_planners[solver_name]->getCurrentTrajectory());
    //     const auto& history = casadi_histories[solver_name];
    //     bool success = casadi_success[solver_name];
        
    //     std::string planner_name = "CasADi-" + solver_name;
    //     std::cout << std::left << std::setw(18) << planner_name
    //               << std::setw(12) << (success ? "SUCCESS" : "FAILED")
    //               << std::setw(10) << history.total_iterations
    //               << std::setw(15) << casadi_final_res.first
    //               << casadi_final_res.second << "\n";
    // }

    std::cout << std::string(70, '-') << "\n";

    // Find best result
    std::string best_planner = "PCE";
    float best_cost = pce_final_res.first;
    
    if (ngd_final_res.first < best_cost) {
        best_cost = ngd_final_res.first;
        best_planner = "NGD";
    }
    
    // NEW: Include STOMP in comparison (only if initialized and successful)
    if (stomp_initialized && stomp_final_res.first < best_cost && !std::isnan(stomp_final_res.first)) {
        best_cost = stomp_final_res.first;
        best_planner = "STOMP";
    }
    
    // for (const auto& solver_name : casadi_solvers) {
    //     if (casadi_planners.find(solver_name) == casadi_planners.end()) continue;
    //     auto res = evaluate_final(casadi_planners[solver_name]->getCurrentTrajectory());
    //     if (res.first < best_cost && !std::isnan(res.first)) {
    //         best_cost = res.first;
    //         best_planner = "CasADi-" + solver_name;
    //     }
    // }
    
    std::cout << "\n★ Best result: " << best_planner << " with total cost = " << best_cost << "\n";
    std::cout << "\n=================================================\n";

    return 0;
}