#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector> 
#include <random> 
#include <cmath> 

#include "../include/ObstacleMap.h" 
#include "../include/Trajectory.h"
#include "../include/MotionPlanner.h" 
#include "../include/PCEMotionPlanner.h" 

// --- Drawing Utilities (Copied from main.cpp) ---

void drawNode(sf::RenderWindow& window, const PathNode& node, float radius, const sf::Color& color) {
    sf::CircleShape circle(radius);
    circle.setFillColor(color);
    circle.setOrigin(sf::Vector2f(radius, radius));
    circle.setPosition({node.x, node.y}); 
    window.draw(circle);
}

void drawTrajectorySegments(sf::RenderWindow& window, const Trajectory& trajectory, const sf::Color& color = sf::Color(50, 50, 200, 200)) {
    if (trajectory.nodes.size() < 2) return;
    sf::VertexArray lines(sf::PrimitiveType::LineStrip, trajectory.nodes.size());
    for (size_t i = 0; i < trajectory.nodes.size(); ++i) {
        lines[i].position = sf::Vector2f(trajectory.nodes[i].x, trajectory.nodes[i].y);
        lines[i].color = color; 
    }
    window.draw(lines);
}

/**
 * @brief Visualizes the initial trajectory, including the collision checking spheres
 * for each state node, and the obstacles.
 * @param window The SFML window for drawing.
 * @param planner The MotionPlanner object containing the initial path and obstacles.
 */
void visualizeInitialTrajectoryWithSpheres(sf::RenderWindow& window, const MotionPlanner& planner) {
    window.clear(sf::Color(240, 240, 240));

    // Draw obstacles
    const std::vector<Obstacle>& obstacles = planner.getObstacles();
    for (const auto& obs : obstacles) {
        sf::CircleShape circle(obs.radius);
        circle.setFillColor(sf::Color(100, 100, 100, 180)); // Dark grey obstacle
        circle.setOrigin(sf::Vector2f(obs.radius, obs.radius));
        circle.setPosition({obs.x, obs.y}); 
        window.draw(circle);
    }
    
    // 2. Draw the trajectory segments (thin line for the path)
    const Trajectory& initial_path = planner.getCurrentTrajectory();
    drawTrajectorySegments(window, initial_path, sf::Color(50, 50, 255, 150));

    // 3. Draw the trajectory states as collision spheres
    for (size_t i = 0; i < initial_path.nodes.size(); ++i) {
        const auto& node = initial_path.nodes[i];
        
        // Calculate collision status
        // NOTE: calculateSDF must be available from ObstacleMap.h/cpp
        float sdf_value = calculateSDF(node.x, node.y, obstacles);
        float effective_sdf = sdf_value - node.radius;

        unsigned char alpha_fill;
        sf::Color node_color;
        if (effective_sdf < 0.0f) {
            // Collision: Red (Solid fill for clear indication)
            node_color = sf::Color(255, 50, 50); 
            alpha_fill = 255;
        } else if (effective_sdf < 10.0f) { 
            // Near-Collision (within 10 units of the buffer margin): Orange
            node_color = sf::Color(255, 165, 0); 
            alpha_fill = 150;
        } else {
            // Free Space: Transparent Blue/Green
            node_color = sf::Color(50, 200, 50); 
            alpha_fill = 80;
        }
        
        // Set fill color with computed transparency
        sf::Color final_fill_color = node_color;
        final_fill_color.a = alpha_fill;

        sf::CircleShape circle(node.radius);
        circle.setFillColor(final_fill_color);
        circle.setOutlineThickness(1.0f);
        circle.setOutlineColor(sf::Color(0, 0, 0, 50)); // Light outline
        circle.setOrigin(sf::Vector2f(node.radius, node.radius));
        circle.setPosition({node.x, node.y}); 
        window.draw(circle);
        
        // Optionally draw a small dot for the center of the node
        if (i == initial_path.start_index || i == initial_path.goal_index) {
            drawNode(window, node, 5.0f, (i == initial_path.start_index) ? sf::Color::Green : sf::Color::Red);
        } else {
            drawNode(window, node, 1.5f, sf::Color(0, 0, 0, 200));
        }
    }

    window.display();
}


int main() {
    // --- 1. Setup ---
    // Get initial motion planning parameters
    int numInitialNodes = 100;
    float initialTotalTime = 8.0f;
    YAML::Node config;
    float nodeCollisionRadius = 15.0f;

    // Read values from config.yaml
    try {
        config = YAML::LoadFile("../configs/config.yaml");

        if (config["motion_planning"]) {
            const YAML::Node& plannerConfig = config["motion_planning"];
            if (plannerConfig["num_discretization"]) {
                numInitialNodes = plannerConfig["num_discretization"].as<int>();
            }
            if (plannerConfig["initial_total_time"]) {
                initialTotalTime = plannerConfig["initial_total_time"].as<float>();
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

    std::cout << "Obstacle Map hyperparameters:\n"
                  << "  num_discretization: " << numInitialNodes << "\n"
                  << "  initial_total_time: " << initialTotalTime << "\n"
                  << "  node_collision_radius: " << nodeCollisionRadius << "\n";

    // Generate obstacles
    std::vector<Obstacle> obstacles = generateObstacles(NUM_OBSTACLES, OBSTACLE_RADIUS, MAP_WIDTH, MAP_HEIGHT);

    // Define a challenging start/goal that requires optimization
    PathNode start = PathNode{50.0f, 550.0f, nodeCollisionRadius}; // Bottom left
    PathNode goal = PathNode{750.0f, 50.0f, nodeCollisionRadius};  // Top right

    // --- 2. Motion Planning ---
    float clearance_dist = 100.0;
    // 2a. Initialize the PCEM planner with the obstacle map
    ProximalCrossEntropyMotionPlanner planner(obstacles, config);
    planner.initialize(start, goal, numInitialNodes, initialTotalTime, InterpolationMethod::LINEAR, obstacles, clearance_dist);

    // --- 3. Initial State and Collision Check Visualization ---
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), "Initial Trajectory and Collision Spheres", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    // Call the new visualization function
    visualizeInitialTrajectoryWithSpheres(window, planner);

    // --- 4. Main Loop (Static Display) ---
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
    }

    return 0;
}
