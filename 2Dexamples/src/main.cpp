#include <SFML/Graphics.hpp>
#include <iostream>
#include <variant> 
#include <random> 


#include "../include/ObstacleMap.h" // Includes Obstacle and Map constants
#include "../include/Trajectory.h"
#include "../include/MotionPlanner.h" // Includes the base planner class and enums
#include "../include/PCEMotionPlanner.h" // Includes the PCEM planner class
#include "NGDMotionPlanner.h" // Includes the Natural Gradient Planner

// --- Drawing Utilities ---

/**
 * @brief Draws a single node (circle) of the trajectory.
 */
void drawNode(sf::RenderWindow& window, const PathNode& node, float radius, const sf::Color& color) {
    sf::CircleShape circle(radius);
    circle.setFillColor(color);
    // Set origin to center for correct positioning
    circle.setOrigin(sf::Vector2f(radius, radius));
    // Draw without offset, since each planner now has its own window
    circle.setPosition({node.x, node.y}); 
    window.draw(circle);
}

/**
 * @brief Draws the trajectory segments (lines connecting nodes).
 */
void drawTrajectorySegments(sf::RenderWindow& window, const Trajectory& trajectory, const sf::Color& color) {
    if (trajectory.nodes.size() < 2) return;

    // Use sf::PrimitiveType::LineStrip for SFML 3.x compatibility
    sf::VertexArray lines(sf::PrimitiveType::LineStrip, trajectory.nodes.size());
    for (size_t i = 0; i < trajectory.nodes.size(); ++i) {
        lines[i].position = sf::Vector2f(trajectory.nodes[i].x, trajectory.nodes[i].y);
        lines[i].color = color; 
    }
    window.draw(lines);
}

/**
 * @brief Draws the entire optimization history for a single planner in the current viewport.
 * @param window The SFML window for drawing.
 * @param planner The MotionPlanner object containing the history.
 */
void visualizeOptimizationHistory(sf::RenderWindow& window, const MotionPlanner& planner) {
    
    // Draw obstacles
    const std::vector<Obstacle>& obstacles = planner.getObstacles();
    for (const auto& obs : obstacles) {
        sf::CircleShape circle(obs.radius);
        circle.setFillColor(sf::Color(100, 100, 100, 180)); // Dark grey obstacle
        circle.setOrigin(sf::Vector2f(obs.radius, obs.radius));
        circle.setPosition({obs.x, obs.y}); 
        window.draw(circle);
    }
    
    const auto& history = planner.getTrajectoryHistory();
    size_t num_iterations = history.size();
    
    if (num_iterations == 0) {
        return;
    }

    // Draw historical trajectories with increasing transparency (except the last one)
    for (size_t i = 0; i < num_iterations - 1; ++i) {
        // Calculate a transparency factor: older paths are more transparent.
        float alpha_float = 5.0f + (145.0f * (float)i / (num_iterations - 2));
        unsigned char alpha = static_cast<unsigned char>(alpha_float); 
        
        sf::Color history_color(50, 50, 255, alpha); // Fading blue
        drawTrajectorySegments(window, history[i], history_color);
    }

    // Draw the Final (or Current) Trajectory brightly
    const Trajectory& final_path = history.back();
    drawTrajectorySegments(window, final_path, sf::Color(255, 0, 0, 255)); // Bright Red

    // Draw Start/Goal Points
    if (!final_path.nodes.empty()) {
        const PathNode& start_node = final_path.nodes[final_path.start_index];
        const PathNode& goal_node = final_path.nodes[final_path.goal_index];
        drawNode(window, start_node, 5.0f, sf::Color::Green);
        drawNode(window, goal_node, 5.0f, sf::Color::Red);
    }
}

/**
 * @brief Opens an SFML window to display the results for a single planner.
 * This function is blocking and will return when the window is closed.
 * @param planner The motion planner whose history to display.
 * @param title The title of the window.
 */
void showPlannerWindow(const MotionPlanner& planner, const std::string& title) {
    // Setup SFML window
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), title, sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    // Main Loop (Static Display)
    while (window.isOpen()) {
        // Event handling
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
        
        window.clear(sf::Color(240, 240, 240));
        visualizeOptimizationHistory(window, planner);
        window.display();
    }
}


int main() {
    // --- 1. Setup ---

    // Define the visualization constants (no longer needed, but keeping MAP_HEIGHT_CONST unused for clarity)
    const float MAP_HEIGHT_CONST = MAP_HEIGHT;

    // Get initial motion planning parameters
    int numInitialNodes = 100;
    float initialTotalTime = 8.0f;
    YAML::Node config;
    float nodeCollisionRadius = 15.0f;

    // Read values from config.yaml
    try {
        config = YAML::LoadFile("../src/config.yaml");

        if (config["motion_planner"]) {
            const YAML::Node& plannerConfig = config["motion_planner"];
            if (plannerConfig["initial_nodes"]) {
                numInitialNodes = plannerConfig["initial_nodes"].as<int>();
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

    // Generate obstacles once for both planners
    std::vector<Obstacle> obstacles = generateObstacles(NUM_OBSTACLES, OBSTACLE_RADIUS, MAP_WIDTH, MAP_HEIGHT);

    // Define the start/goal points
    PathNode start = PathNode{50.0f, 550.0f, nodeCollisionRadius}; // Bottom left
    PathNode goal = PathNode{750.0f, 50.0f, nodeCollisionRadius};  // Top right

    
    // --- 2. Motion Planning - PCEM ---
    std::cout << "\n--- Starting PCEM Planner ---\n";
    ProximalCrossEntropyMotionPlanner planner_pce(obstacles, config);
    planner_pce.initialize(start, goal, numInitialNodes, initialTotalTime, InterpolationMethod::LINEAR);
    bool success_pce = planner_pce.optimize();
    // FIX: Explicitly call the base class version to avoid name hiding issues.
    std::cout << "PCEM Optimization finished. Cost: " << planner_pce.MotionPlanner::computeCollisionCost(obstacles) + planner_pce.computeSmoothnessCost() << "\n";

    // --- 3. Motion Planning - NGD ---
    std::cout << "\n--- Starting NGD Planner ---\n";
    NGDMotionPlanner planner_ngd(obstacles, config);
    // Initialize NGD with the same initial trajectory
    planner_ngd.initialize(start, goal, numInitialNodes, initialTotalTime, InterpolationMethod::LINEAR);
    bool success_ngd = planner_ngd.optimize();
    // FIX: Explicitly call the base class version to avoid name hiding issues.
    std::cout << "NGD Optimization finished. Cost: " << planner_ngd.MotionPlanner::computeCollisionCost(obstacles) + planner_ngd.computeSmoothnessCost() << "\n";


    // --- 4. Optimization History Visualization (Sequential Windows) ---
    
    // Show PCEM results first. This window will block until closed.
    showPlannerWindow(planner_pce, "PCEM Motion Planner Results");

    // Show NGD results second. This window will open after the first is closed.
    showPlannerWindow(planner_ngd, "NGD Motion Planner Results");

    return 0;
}
