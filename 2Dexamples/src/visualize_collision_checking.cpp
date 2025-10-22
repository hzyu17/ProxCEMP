#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector> 
#include <random> 
#include <cmath> 

#include "../include/PCEMotionPlanner.h" 
#include "../include/visualization.h"


int main() {
    // --- 1. Setup ---
    int numInitialNodes = 100;
    float initialTotalTime = 8.0f;
    float nodeCollisionRadius = 15.0f;
    unsigned int randomSeed = std::random_device{}();  // Default: random seed
    YAML::Node config;

    // Read values from config.yaml
    try {
        config = YAML::LoadFile("../configs/config.yaml");

        if (config["experiment"]) {
            const YAML::Node& experimentConfig = config["experiment"];
            // Read random seed from config
            if (experimentConfig["random_seed"]) {
                randomSeed = experimentConfig["random_seed"].as<unsigned int>();
                std::cout << "Using seed from config: " << randomSeed << "\n";
            } else {
                std::cout << "No seed in config. Using random seed: " << randomSeed << "\n";
            }
        }

        if (config["motion_planning"]) {
            const YAML::Node& plannerConfig = config["motion_planning"];
            if (plannerConfig["num_discretization"]) {
                numInitialNodes = plannerConfig["num_discretization"].as<int>();
            }
            if (plannerConfig["total_time"]) {
                initialTotalTime = plannerConfig["total_time"].as<float>();
            }
            if (plannerConfig["node_collision_radius"]) {
                nodeCollisionRadius = plannerConfig["node_collision_radius"].as<float>();
            }
        }
    } catch (const YAML::BadFile& e) {
        std::cerr << "Warning: Could not open config.yaml. Using default values.\n";
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing config.yaml: " << e.what() << ". Using default values.\n";
    }

    std::cout << "Visualization parameters:\n"
              << "  num_discretization: " << numInitialNodes << "\n"
              << "  total_time: " << initialTotalTime << "\n"
              << "  node_collision_radius: " << nodeCollisionRadius << "\n"
              << "  random_seed: " << randomSeed << "\n";

    // --- 2. Create Obstacle Map with Seed ---
    ObstacleMap obstacle_map(2);  // 2D obstacle map
    obstacle_map.setMapSize(MAP_WIDTH, MAP_HEIGHT);
    
    // Set the seed before generating obstacles
    obstacle_map.setSeed(randomSeed);
    obstacle_map.generateRandom2D(NUM_OBSTACLES, OBSTACLE_RADIUS);
    
    std::cout << "Generated " << obstacle_map.size() << " obstacles\n";

    // Define start/goal
    PathNode start(50.0f, 550.0f, nodeCollisionRadius);   // Bottom left
    PathNode goal(750.0f, 50.0f, nodeCollisionRadius);    // Top right

    std::cout << "Start: [" << start.position(0) << ", " << start.position(1) << "], radius=" << start.radius << "\n";
    std::cout << "Goal:  [" << goal.position(0) << ", " << goal.position(1) << "], radius=" << goal.radius << "\n";

    // --- 3. Motion Planning Initialization ---
    float clearance_dist = 100.0f;
    
    // Initialize the PCEM planner with ObstacleMap
    ProximalCrossEntropyMotionPlanner planner;
    planner.initialize(start, goal, numInitialNodes, initialTotalTime, 
                      InterpolationMethod::LINEAR, obstacle_map, clearance_dist);

    std::cout << "Planner initialized in " << planner.getNumDimensions() << "D space\n";
    std::cout << "Obstacles after clearance: " << obstacle_map.size() << "\n";

    // --- 4. Visualization Setup ---
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), 
                           "Trajectory Visualization - Collision Status Display", 
                           sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    std::cout << "\n=== Controls ===\n";
    std::cout << "SPACE:  Start optimization\n";
    std::cout << "R:      Reset\n";
    std::cout << "C:      Toggle collision radius display\n";
    std::cout << "ESC:    Quit\n";
    std::cout << "\nCollision radius colors:\n";
    std::cout << "  GREEN = No collision\n";
    std::cout << "  RED   = In collision\n";
    std::cout << "================\n\n";

    bool optimized = false;
    bool show_collision_radius = true;

    // Count initial collisions
    size_t initial_collisions = 0;
    for (const auto& node : planner.getCurrentTrajectory().nodes) {
        if (isNodeInCollision(node, planner.getObstacles())) {
            initial_collisions++;
        }
    }
    std::cout << "Initial trajectory has " << initial_collisions << " nodes in collision\n";

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
                    show_collision_radius = !show_collision_radius;
                    std::cout << "Collision radius display: " 
                              << (show_collision_radius ? "ON" : "OFF") << "\n";
                }
                
                if (key_event->code == sf::Keyboard::Key::R) {
                    std::cout << "\n--- Resetting ---\n";
                    
                    // Regenerate obstacles with the same seed
                    obstacle_map.clear();
                    obstacle_map.setSeed(randomSeed);
                    obstacle_map.generateRandom2D(NUM_OBSTACLES, OBSTACLE_RADIUS);
                    std::cout << "Regenerated " << obstacle_map.size() << " obstacles\n";
                    
                    // Reinitialize planner
                    planner.initialize(start, goal, numInitialNodes, initialTotalTime, 
                                     InterpolationMethod::LINEAR, obstacle_map, clearance_dist);
                    
                    std::cout << "Obstacles after clearance: " << obstacle_map.size() << "\n";
                    
                    // Count collisions after reset
                    size_t reset_collisions = 0;
                    for (const auto& node : planner.getCurrentTrajectory().nodes) {
                        if (isNodeInCollision(node, planner.getObstacles())) {
                            reset_collisions++;
                        }
                    }
                    std::cout << "Reset trajectory has " << reset_collisions << " nodes in collision\n";
                    
                    optimized = false;
                }
            }
        }
        
        // --- Render current state ---
        window.clear(sf::Color(240, 240, 240));
        
        // 1. Draw obstacles from the map
        drawObstacleMap(window, obstacle_map);
        
        // 2. Draw current trajectory with collision-colored radii
        const Trajectory& current_traj = planner.getCurrentTrajectory();
        drawTrajectoryWithCollisionRadius(window, current_traj, planner.getObstacles(), 
                                         show_collision_radius);
        
        window.display();
    }

    obstacle_map.saveToJSON("obstacle_map_seed_999.json");

    return 0;
}
