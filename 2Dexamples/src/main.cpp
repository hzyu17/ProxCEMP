#include <SFML/Graphics.hpp>
#include <iostream>
#include <variant> 
#include <random> // Added for RNG in visualization
#include <yaml-cpp/yaml.h> // Include the YAML-CPP header


#include "../include/ObstacleMap.h" // Includes Obstacle and Map constants
#include "../include/Trajectory.h"
#include "../include/MotionPlanner.h" // Includes the base planner class and enums
#include "../include/PCEMotionPlanner.h" // Includes the new PCEM planner class

// --- Drawing Utilities ---

// Function to draw a single node (circle) of the trajectory
void drawNode(sf::RenderWindow& window, const PathNode& node, float radius, const sf::Color& color) {
    sf::CircleShape circle(radius);
    circle.setFillColor(color);
    // Set origin to center for correct positioning
    circle.setOrigin(sf::Vector2f(radius, radius));
    circle.setPosition({node.x, node.y}); 
    window.draw(circle);
}

// Function to draw the trajectory segments (lines connecting nodes)
void drawTrajectorySegments(sf::RenderWindow& window, const Trajectory& trajectory, const sf::Color& color = sf::Color(50, 50, 200, 200)) {
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
 * @brief Visualizes the entire optimization history stored in the MotionPlanner.
 * It draws older trajectories transparently and the final trajectory vividly.
 * @param window The SFML window for drawing.
 * @param planner The MotionPlanner object containing the history.
 */
void visualizeOptimizationHistory(sf::RenderWindow& window, const MotionPlanner& planner) {
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
    
    const auto& history = planner.getTrajectoryHistory();
    size_t num_iterations = history.size();
    
    if (num_iterations == 0) {
        std::cout << "No trajectory history to display.\n";
        window.display();
        return;
    }

    // Draw all historical trajectories (except the last one) with increasing transparency
    for (size_t i = 0; i < num_iterations - 1; ++i) {
        // Calculate a transparency factor: older paths are more transparent.
        // Alpha ranges from 5 (oldest) to 150 (second-to-last).
        float alpha_float = 5.0f + (145.0f * (float)i / (num_iterations - 2));
        // FIX: Use unsigned char for the 8-bit color component, which is compatible with sf::Color.
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

    window.display();
}


int main() {
    // --- 1. Setup ---

    // Get initial motion planning parameters
    int numInitialNodes = 100;
    float initialStepSize = 8.0f;
    YAML::Node config;

    // Read values from config.yaml
    try {
        config = YAML::LoadFile("../src/config.yaml");

        if (config["motion_planner"]) {
            const YAML::Node& plannerConfig = config["motion_planner"];
            if (plannerConfig["initial_nodes"]) {
                numInitialNodes = plannerConfig["initial_nodes"].as<int>();
            }
            if (plannerConfig["initial_step_size"]) {
                initialStepSize = plannerConfig["initial_step_size"].as<float>();
            }
        }
    } catch (const YAML::BadFile& e) {
        std::cerr << "Warning: Could not open config.yaml. Using default values.\n";
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing config.yaml: " << e.what() << ". Using default values.\n";
    }

    // Generate obstacles
    std::vector<Obstacle> obstacles = generateObstacles(NUM_OBSTACLES, OBSTACLE_RADIUS, MAP_WIDTH, MAP_HEIGHT);
    
    // Define explicit start and goal nodes for interpolation
    constexpr float NODE_COLLISION_RADIUS = 15.0f;

    // Define a challenging start/goal that requires optimization
    PathNode start = PathNode{50.0f, 550.0f, NODE_COLLISION_RADIUS}; // Bottom left
    PathNode goal = PathNode{750.0f, 50.0f, NODE_COLLISION_RADIUS};  // Top right

    // --- 2. Motion Planning ---
    
    // 2a. Initialize the PCEM planner with the obstacle map
    ProximalCrossEntropyMotionPlanner planner(obstacles, config);

    // 2b. Initialize a *Bezier* base trajectory (smoother start)
    // Use the values read from the config file
    planner.initialize(start, goal, numInitialNodes, initialStepSize, InterpolationMethod::LINEAR);
    
    std::cout << "\nStarting optimization...\n";
    // 2c. Run the optimization loop (which is simulated in PCEMotionPlanner.h)
    bool success = planner.optimize();

    std::cout << "Optimization finished. Success: " << (success ? "True" : "False") << "\n";

    // --- 3. Optimization History Visualization ---
    
    // Setup SFML window
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), "2D Motion Planning History", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    // Call the history visualization function once
    visualizeOptimizationHistory(window, planner);

    // --- 4. Main Loop (Static Display) ---
    
    // The main loop keeps the window open until the user closes it.
    while (window.isOpen()) {
        // Event handling (SFML 3.x returns std::optional<sf::Event>)
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
    }

    return 0;
}