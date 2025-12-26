#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
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
    // Early STOMP test
    {
        std::cout << "\n========================================\n";
        std::cout << "EARLY STOMP TEST\n";
        std::cout << "========================================\n";

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

        auto early_task = std::make_shared<pce::StompCollisionTask>();

        try {
            stomp::Stomp early_stomp(cfg, early_task);
            Eigen::VectorXd start(2), goal(2);
            start << 0.0, 0.0;
            goal << 100.0, 100.0;
            Eigen::MatrixXd result;
            bool success = early_stomp.solve(start, goal, result);
            std::cout << "EARLY TEST: solve() " << (success ? "SUCCESS" : "FAILED") << "\n";
        } catch (const std::exception& e) {
            std::cerr << "EARLY TEST FAILED: " << e.what() << "\n";
            return 1;
        }

        std::cout << "========================================\n\n";
    }

    std::cout << "=================================================\n";
    std::cout << "   Motion Planning Experiments\n";
    std::cout << "=================================================\n\n";

    // Load configuration
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path config_path = source_path.parent_path() / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();

    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"]["visualize_initial_state"].as<bool>(false);

    // =========================================================================
    // PCE PLANNER
    // =========================================================================
    std::cout << "\n=== PCE Planner ===\n";

    auto task_pce = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_pce = std::make_shared<ProximalCrossEntropyMotionPlanner>(task_pce);

    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return -1;
    }

    if (!planner_pce->initialize(pce_config)) {
        std::cerr << "Error: PCE Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        visualizeInitialState(task_pce->getObstacles(), planner_pce->getCurrentTrajectory(),
                              "PCEM - Initial State");
    }

    // PCE optimization with data collection
    std::cout << "\n=== Running PCE Optimization ===\n";

    OptimizationHistory pce_history;
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

    for (size_t i = 0; i < planner_pce->getTrajectoryHistory().size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = planner_pce->getTrajectoryHistory()[i];
        iter_data.total_cost = task_pce->computeStateCost(iter_data.mean_trajectory);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        pce_history.addIteration(iter_data);
    }

    pce_history.final_trajectory = planner_pce->getCurrentTrajectory();
    pce_history.final_cost = task_pce->computeStateCost(pce_history.final_trajectory);
    pce_history.converged = success_pce;
    pce_history.total_iterations = planner_pce->getTrajectoryHistory().size();

    std::cout << (success_pce ? "\n✓ PCE optimization completed successfully\n"
                              : "\n✗ PCE optimization failed\n");

    // =========================================================================
    // NGD PLANNER
    // =========================================================================
    std::cout << "\n=== NGD Planner ===\n";

    auto task_ngd = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto planner_ngd = std::make_shared<NGDMotionPlanner>(task_ngd);

    NGDConfig ngd_config;
    if (!ngd_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load NGD configuration from file\n";
        return -1;
    }

    if (!planner_ngd->initialize(ngd_config)) {
        std::cerr << "Error: NGD Planner initialization failed\n";
        return 1;
    }

    if (visualize) {
        visualizeInitialState(task_ngd->getObstacles(), planner_ngd->getCurrentTrajectory(),
                              "NGD - Initial State");
    }

    std::cout << "\n=== Running NGD Optimization ===\n";

    OptimizationHistory ngd_history;
    bool success_ngd = planner_ngd->solve();

    for (size_t i = 0; i < planner_ngd->getTrajectoryHistory().size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = i;
        iter_data.mean_trajectory = planner_ngd->getTrajectoryHistory()[i];
        iter_data.total_cost = task_ngd->computeStateCost(iter_data.mean_trajectory);
        iter_data.collision_cost = iter_data.total_cost;
        iter_data.smoothness_cost = 0.0f;
        ngd_history.addIteration(iter_data);
    }

    ngd_history.final_trajectory = planner_ngd->getCurrentTrajectory();
    ngd_history.final_cost = task_ngd->computeStateCost(ngd_history.final_trajectory);
    ngd_history.converged = success_ngd;
    ngd_history.total_iterations = planner_ngd->getTrajectoryHistory().size();

    std::cout << (success_ngd ? "\n✓ NGD optimization completed successfully\n"
                              : "\n✗ NGD optimization failed\n");

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
    }

    if (stomp_initialized && visualize) {
        visualizeInitialState(task_stomp->getObstacles(), planner_stomp->getCurrentTrajectory(),
                              "STOMP - Initial State");
    }

    OptimizationHistory stomp_history;
    bool success_stomp = false;

    if (stomp_initialized) {
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

        for (size_t i = 0; i < planner_stomp->getTrajectoryHistory().size(); ++i) {
            IterationData iter_data;
            iter_data.iteration = i;
            iter_data.mean_trajectory = planner_stomp->getTrajectoryHistory()[i];
            iter_data.total_cost = task_stomp->computeStateCost(iter_data.mean_trajectory);
            iter_data.collision_cost = iter_data.total_cost;
            iter_data.smoothness_cost = 0.0f;
            stomp_history.addIteration(iter_data);
        }

        stomp_history.final_trajectory = planner_stomp->getCurrentTrajectory();
        stomp_history.final_cost = task_stomp->computeStateCost(stomp_history.final_trajectory);
        stomp_history.converged = success_stomp;
        stomp_history.total_iterations = planner_stomp->getTrajectoryHistory().size();

        std::cout << (success_stomp ? "\n✓ STOMP optimization completed successfully\n"
                                    : "\n✗ STOMP optimization failed\n");
    } else {
        std::cout << "\n✗ STOMP skipped due to initialization failure\n";
    }

    // =========================================================================
    // VISUALIZATION
    // =========================================================================
    std::cout << "\n=== Interactive Optimization Visualization ===\n";

    auto obstacle_map_ptr = task_pce->getObstacleMap();
    OptimizationVisualizer visualizer(900, 700);

    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, pce_history, "PCEM - Trajectory Evolution");
    visualizer.showCostPlot(pce_history, "PCEM - Cost Convergence");

    visualizer.showTrajectoryEvolution(*obstacle_map_ptr, ngd_history, "NGD - Trajectory Evolution");
    visualizer.showCostPlot(ngd_history, "NGD - Cost Convergence");

    if (stomp_initialized) {
        visualizer.showTrajectoryEvolution(*obstacle_map_ptr, stomp_history, "STOMP - Trajectory Evolution");
        visualizer.showCostPlot(stomp_history, "STOMP - Cost Convergence");
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n=================================================\n";
    std::cout << "   Results Summary\n";
    std::cout << "=================================================\n";

    auto evaluate_final = [&](const Trajectory& traj) {
        float collision = task_pce->computeStateCost(traj);
        float smoothness = planner_pce->computeSmoothnessCost(traj);
        return std::make_pair(collision + smoothness, collision);
    };

    auto pce_res = evaluate_final(planner_pce->getCurrentTrajectory());
    auto ngd_res = evaluate_final(planner_ngd->getCurrentTrajectory());
    auto stomp_res = stomp_initialized ? evaluate_final(planner_stomp->getCurrentTrajectory())
                                       : std::make_pair(std::numeric_limits<float>::max(),
                                                        std::numeric_limits<float>::max());

    std::cout << std::left << std::setw(12) << "Planner"
              << std::setw(12) << "Status"
              << std::setw(10) << "Iters"
              << std::setw(15) << "Total Cost"
              << "Collision Cost\n";
    std::cout << std::string(60, '-') << "\n";

    std::cout << std::left << std::setw(12) << "PCE"
              << std::setw(12) << (success_pce ? "SUCCESS" : "FAILED")
              << std::setw(10) << pce_history.total_iterations
              << std::setw(15) << pce_res.first << pce_res.second << "\n";

    std::cout << std::left << std::setw(12) << "NGD"
              << std::setw(12) << (success_ngd ? "SUCCESS" : "FAILED")
              << std::setw(10) << ngd_history.total_iterations
              << std::setw(15) << ngd_res.first << ngd_res.second << "\n";

    if (stomp_initialized) {
        std::cout << std::left << std::setw(12) << "STOMP"
                  << std::setw(12) << (success_stomp ? "SUCCESS" : "FAILED")
                  << std::setw(10) << stomp_history.total_iterations
                  << std::setw(15) << stomp_res.first << stomp_res.second << "\n";
    } else {
        std::cout << std::left << std::setw(12) << "STOMP"
                  << std::setw(12) << "SKIPPED"
                  << std::setw(10) << "-"
                  << std::setw(15) << "-" << "-\n";
    }

    std::cout << std::string(60, '-') << "\n";

    // Find best
    std::string best = "PCE";
    float best_cost = pce_res.first;
    if (ngd_res.first < best_cost) { best_cost = ngd_res.first; best = "NGD"; }
    if (stomp_initialized && stomp_res.first < best_cost && !std::isnan(stomp_res.first)) {
        best_cost = stomp_res.first; best = "STOMP";
    }

    std::cout << "\n★ Best result: " << best << " with total cost = " << best_cost << "\n";
    std::cout << "\n=================================================\n";

    return 0;
}