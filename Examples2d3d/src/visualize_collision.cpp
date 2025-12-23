#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector> 
#include <random> 
#include <cmath> 
#include <memory>

#include "../include/PCEMotionPlanner.h" 
#include "../examples/CollisionAvoidanceTask.h"
#include "../examples/visualization.h"
#include <string>

namespace {
    constexpr float SAVE_SCALE = 4.0f;  // 4x scale for ~400 DPI print quality
}

int main() {
    // --- 1. Load Configuration ---
    std::string config_file = "../Examples2d3d/configs/config.yaml"; // Running from the root build directory
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
    std::cout << "S:      Save high-res PNG (4x)\n";
    std::cout << "ESC:    Quit\n";
    std::cout << "\nCollision sphere colors:\n";
    std::cout << "  GREEN  = Safe (no collision)\n";
    std::cout << "  ORANGE = Near collision\n";
    std::cout << "  RED    = In collision\n";
    std::cout << "================\n\n";

    bool show_collision_spheres = true;
    float collision_threshold = 10.0f;
    int save_counter = 0;

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
                
                if (key_event->code == sf::Keyboard::Key::S) {
                    // Save high resolution PNG (4x scale)
                    unsigned int save_width = static_cast<unsigned int>(map_width * SAVE_SCALE);
                    unsigned int save_height = static_cast<unsigned int>(map_height * SAVE_SCALE);
                    
                    sf::RenderTexture render_texture;
                    if (render_texture.resize({save_width, save_height})) {
                        render_texture.clear(sf::Color(240, 240, 240));
                        
                        // Use scaled view to render at higher resolution
                        sf::View scaled_view(sf::FloatRect({0.f, 0.f}, 
                                            {static_cast<float>(map_width), static_cast<float>(map_height)}));
                        scaled_view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
                        render_texture.setView(scaled_view);
                        
                        const Trajectory& current_traj = planner.getCurrentTrajectory();
                        
                        if (show_collision_spheres) {
                            visualizeTrajectoryWithCollisionSpheres(
                                render_texture, 
                                *obstacle_map, 
                                current_traj, 
                                collision_threshold,
                                false  // don't clear - already cleared
                            );
                        } else {
                            visualizeTrajectory(
                                render_texture, 
                                *obstacle_map, 
                                current_traj, 
                                sf::Color(0, 0, 255, 255),
                                false
                            );
                        }
                        
                        render_texture.display();
                        
                        sf::Image screenshot = render_texture.getTexture().copyToImage();
                        std::string filename = "collision_2d_" + std::to_string(save_counter++) + "_highres.png";
                        
                        if (screenshot.saveToFile(filename)) {
                            std::cout << "Saved: " << filename 
                                      << " (" << save_width << "x" << save_height << " pixels)\n";
                        } else {
                            std::cerr << "Failed to save image!\n";
                        }
                    } else {
                        std::cerr << "Failed to create render texture for saving!\n";
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