#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector> 
#include <random> 
#include <cmath> 
#include <memory>

#include "../include/PCEMotionPlanner.h" 
#include "../include/CollisionAvoidanceTask.h"
#include "../include/visualization.h"
#include <string>

int main() {
    // --- 1. Load Configuration ---
    std::string config_file = "../configs/config.yaml";
    YAML::Node config;
    
    std::cout << "=== Load from YAML ===\n\n";
    
    try {
        config = YAML::LoadFile(config_file);
        std::cout << "Loaded configuration from: " << config_file << "\n";
    } catch (const YAML::BadFile& e) {
        std::cerr << "Error: Could not open config file: " << config_file << "\n";
        return 1;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing config file: " << e.what() << "\n";
        return 1;
    }

    // --- 2. Create Task (handles obstacle generation and management) ---
    std::cout << "\n=== Creating Collision Avoidance Task ===\n";
    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);
    
    // Get obstacle map from task
    auto obstacle_map = task->getObstacleMap();
    std::cout << "Task created with " << obstacle_map->size() << " obstacles\n";

    // --- 3. Create and Initialize Planner ---
    std::cout << "\n=== Creating PCEM Planner ===\n";
    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return -1;
    }
    ProximalCrossEntropyMotionPlanner planner(task);
    
    // Initialize planner (loads config and sets up trajectory)
    std::cout << "\n=== Initializing Planner ===\n";
    if (!planner.initialize(pce_config)) {
        std::cerr << "Error: Planner initialization failed\n";
        return 1;
    }
    
    std::cout << "Initial trajectory has " << planner.getCurrentTrajectory().nodes.size() 
              << " nodes\n";

    // --- 4. Visualization Setup ---
    unsigned int map_width = 800;
    unsigned int map_height = 600;
    
    if (config["environment"]) {
        map_width = config["environment"]["map_width"].as<unsigned int>(map_width);
        map_height = config["environment"]["map_height"].as<unsigned int>(map_height);
    }
    
    sf::RenderWindow window(
        sf::VideoMode({map_width, map_height}), 
        "Trajectory Visualization - Collision Status Display", 
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    std::cout << "\n=== Controls ===\n";
    std::cout << "SPACE:  Start optimization\n";
    std::cout << "R:      Reset (regenerate trajectory)\n";
    std::cout << "C:      Toggle collision spheres display\n";
    std::cout << "ESC:    Quit\n";
    std::cout << "\nCollision sphere colors:\n";
    std::cout << "  GREEN  = Safe (no collision)\n";
    std::cout << "  ORANGE = Near collision\n";
    std::cout << "  RED    = In collision\n";
    std::cout << "================\n\n";

    bool show_collision_spheres = true;
    float collision_threshold = 10.0f;

    // --- 5. Main Loop ---
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            if (event->is<sf::Event::KeyPressed>()) {
                const auto& key_event = event->getIf<sf::Event::KeyPressed>();
                
                if (key_event->code == sf::Keyboard::Key::Escape) {
                    window.close();
                }
                
                if (key_event->code == sf::Keyboard::Key::C) {
                    show_collision_spheres = !show_collision_spheres;
                    std::cout << "Collision spheres display: " 
                              << (show_collision_spheres ? "ON" : "OFF") << "\n";
                }
                
                if (key_event->code == sf::Keyboard::Key::R) {
                    std::cout << "\n--- Resetting Trajectory ---\n";
                    
                    // Re-initialize to get a fresh trajectory
                    if (!planner.initialize(pce_config)) {
                        std::cerr << "Error: Reset failed\n";
                    } else {
                        std::cout << "Trajectory reset with " 
                                  << planner.getCurrentTrajectory().nodes.size() 
                                  << " nodes\n";
                    }
                }
                
                if (key_event->code == sf::Keyboard::Key::Space) {
                    std::cout << "\n--- Starting Optimization ---\n";
                    
                    // Run optimization (no parameters needed)
                    if (planner.optimize()) {
                        std::cout << "Optimization completed successfully\n";
                    } else {
                        std::cout << "Optimization failed or interrupted\n";
                    }
                }
            }
        }
        
        // --- Render current state ---
        const Trajectory& current_traj = planner.getCurrentTrajectory();
        
        if (show_collision_spheres) {
            // Draw trajectory with collision spheres showing collision status
            visualizeTrajectoryWithCollisionSpheres(
                window, 
                *obstacle_map, 
                current_traj, 
                collision_threshold,
                true  // clear background
            );
        } else {
            // Draw simple trajectory without collision spheres
            visualizeTrajectory(
                window, 
                *obstacle_map, 
                current_traj, 
                sf::Color(0, 0, 255, 255),  // Blue trajectory
                true  // clear background
            );
        }
        
        window.display();
    }

    // --- 6. Save Results ---
    // Get random seed from task config
    unsigned int seed = 999;
    if (config["experiment"] && config["experiment"]["random_seed"]) {
        seed = config["experiment"]["random_seed"].as<unsigned int>();
    }
    
    obstacle_map->saveToJSON("obstacle_map_seed_" + std::to_string(seed) + ".json");
    std::cout << "\nSaved obstacle map to obstacle_map_seed_" << seed << ".json\n";

    return 0;
}