#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "CasadiMotionPlanner.h"
#include "CollisionAvoidanceTask.h"
#include "OptimizationVisualizer.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <memory>

/**
 * @file main_mute.cpp
 * @brief Headless statistical analysis and automated plotting utility.
 */

int main() {
    // 1. Path Setup
    // Assuming binary is in build/ and config is in configs/
    std::filesystem::path config_path = "../configs/config.yaml";
    if (!std::filesystem::exists(config_path)) {
        std::cerr << "CRITICAL ERROR: Config file not found at " << config_path << std::endl;
        return 1;
    }
    
    // 2. Load Global Configuration
    YAML::Node config = YAML::LoadFile(config_path.string());
    int current_seed = config["experiment"]["random_seed"].as<int>();

    // Determine if we should generate PNGs (Set by Python script for Best/Worst cases)
    bool should_save_img = config["experiment"]["visualize_initial_state"] && 
                           config["experiment"]["visualize_initial_state"].as<bool>();

    // 3. Initialize Shared Task for Evaluation
    auto task_eval = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto obstacle_map_ptr = task_eval->getObstacleMap();

    // 4. Execution Lambda
    // Captures task and timing logic to reduce code duplication
    auto run_planner = [&](auto planner, const auto& planner_config, std::string planner_name) {
        auto start = std::chrono::high_resolution_clock::now();
        
        bool init_ok = planner->initialize(planner_config);
        bool solve_ok = false;
        
        if (init_ok) {
            solve_ok = planner->solve();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        
        float total_cost = 0.0f;
        if (solve_ok) {
            const auto& final_traj = planner->getCurrentTrajectory();
            float collision = task_eval->computeCollisionCost(final_traj);
            float smoothness = planner->computeSmoothnessCost(final_traj);
            total_cost = collision + smoothness;

            // Automated Headless Plotting
            if (should_save_img) {
                OptimizationVisualizer visualizer(800, 600);
                std::string filename = "traj_result_" + planner_name + "_seed_" + std::to_string(current_seed) + ".png";
                
                // This calls the non-interactive RenderTexture method
                visualizer.saveStaticPlot(*obstacle_map_ptr, final_traj, filename);
            }
        } else {
            total_cost = 1e10f; // Failure penalty
        }
        
        return std::make_pair(total_cost, duration);
    };

    // 5. PCE Planner Execution
    PCEConfig pce_cfg;
    pce_cfg.loadFromYAML(config);
    auto pce_planner = std::make_shared<ProximalCrossEntropyMotionPlanner>(task_eval);
    auto pce_res = run_planner(pce_planner, pce_cfg, "PCE");

    // 6. NGD Planner Execution
    NGDConfig ngd_cfg;
    ngd_cfg.loadFromYAML(config);
    auto ngd_planner = std::make_shared<NGDMotionPlanner>(task_eval);
    auto ngd_res = run_planner(ngd_planner, ngd_cfg, "NGD");

    // 7. CasADi Planner Execution
    CasADiConfig cas_cfg;
    cas_cfg.loadFromYAML(config);
    auto cas_planner = std::make_shared<CasADiMotionPlanner>(task_eval);
    auto cas_res = run_planner(cas_planner, cas_cfg, "CasADi");

    // 8. Machine-Readable Summary (Captured by run_stats.py)
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "DATA_RESULT," 
              << current_seed << ","
              << pce_res.first << "," << pce_res.second << ","
              << ngd_res.first << "," << ngd_res.second << ","
              << cas_res.first << "," << cas_res.second 
              << std::endl;

    return 0;
}