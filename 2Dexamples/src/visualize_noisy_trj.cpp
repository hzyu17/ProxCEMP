#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include <yaml-cpp/yaml.h> // New include for configuration parsing

// Include necessary headers from your project structure
#include "../include/ObstacleMap.h" // For MAP_WIDTH, MAP_HEIGHT, Obstacle, generateObstacles, NUM_OBSTACLES, OBSTACLE_RADIUS
#include "../include/Trajectory.h" // For PathNode, Trajectory, generateInterpolatedTrajectoryLinear
#include "../include/PCEMotionPlanner.h" // For MotionPlanner base class, sampleSmoothnessNoise, BasicPlanner

// --- Drawing Utilities (Copied and adapted from main.cpp) ---

void drawTrajectorySegments(sf::RenderWindow& window, const Trajectory& trajectory, const sf::Color& color) {
    if (trajectory.nodes.size() < 2) return;

    sf::VertexArray lines(sf::PrimitiveType::LineStrip, trajectory.nodes.size());
    for (size_t i = 0; i < trajectory.nodes.size(); ++i) {
        lines[i].position = sf::Vector2f(trajectory.nodes[i].x, trajectory.nodes[i].y);
        lines[i].color = color; 
    }
    window.draw(lines);
}

void drawPathNodes(sf::RenderWindow& window, const Trajectory& trajectory, float radius, const sf::Color& color) {
    for (const auto& node : trajectory.nodes) {
        sf::CircleShape circle(radius);
        circle.setFillColor(color);
        circle.setOrigin(sf::Vector2f(radius, radius));
        circle.setPosition({node.x, node.y}); 
        window.draw(circle);
    }
}

/**
 * @brief Draws the obstacles from the environment.
 */
void drawObstacles(sf::RenderWindow& window, const std::vector<Obstacle>& obstacles) {
    for (const auto& obs : obstacles) {
        sf::CircleShape circle(obs.radius);
        circle.setFillColor(sf::Color(100, 100, 100, 180)); // Dark grey obstacle
        circle.setOrigin(sf::Vector2f(obs.radius, obs.radius));
        circle.setPosition({obs.x, obs.y}); 
        window.draw(circle);
    }
}


// --- Main Visualization Function ---

void visualizeNoise(const MotionPlanner& planner, const Trajectory& base_trajectory, const std::vector<Trajectory>& noisy_samples) {
    
    // Setup SFML window
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), "Smoothness Noise Visualization (N(0, R^-1))", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    // Get the obstacles from the planner (which now holds the generated list)
    const std::vector<Obstacle>& obstacles = planner.getObstacles();

    // Main Loop (Static Display)
    while (window.isOpen()) {
        // Event handling
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
        
        window.clear(sf::Color(240, 240, 240));

        // 1. Draw obstacles first
        drawObstacles(window, obstacles);

        // 2. Draw all noisy samples (faded blue)
        // This shows the distribution boundary of the noise
        for (const auto& sample : noisy_samples) {
            drawTrajectorySegments(window, sample, sf::Color(50, 50, 255, 30)); // Low alpha for many paths
        }

        // 3. Draw the base trajectory (thick red line, high visibility)
        drawTrajectorySegments(window, base_trajectory, sf::Color(255, 0, 0, 255));
        drawPathNodes(window, base_trajectory, 3.0f, sf::Color(255, 100, 100));
        
        // 4. Draw Start/Goal nodes (Fixed points)
        if (!base_trajectory.nodes.empty()) {
            const PathNode& start_node = base_trajectory.nodes[base_trajectory.start_index];
            const PathNode& goal_node = base_trajectory.nodes[base_trajectory.goal_index];
            
            // Start node (Green)
            sf::CircleShape start_circle(6.0f);
            start_circle.setFillColor(sf::Color::Green);
            start_circle.setOrigin({6.0f, 6.0f});
            start_circle.setPosition({start_node.x, start_node.y});
            window.draw(start_circle);

            // Goal node (Red)
            sf::CircleShape goal_circle(6.0f);
            goal_circle.setFillColor(sf::Color::Red);
            goal_circle.setOrigin({6.0f, 6.0f});
            goal_circle.setPosition({goal_node.x, goal_node.y});
            window.draw(goal_circle);
        }

        window.display();
    }
}


int main() {
    std::cout << "--- Trajectory Noise Visualization ---\n";

    // --- 1. Setup Parameters (Defaults) ---
    int numNodes = 100;
    float totalTime = 10.0f; 
    float nodeRadius = 5.0f;
    const size_t numSamples = 100; // Fixed visualization parameter

    YAML::Node config;

    // --- 1b. Read Config File ---
    try {
        config = YAML::LoadFile("../src/config.yaml");

        if (config["motion_planning"]) {
            const YAML::Node& mp_config = config["motion_planning"];
            if (mp_config["initial_nodes"]) {
                numNodes = mp_config["initial_nodes"].as<int>();
            }
            // NOTE: Interpreting 'initial_step_size' as 'totalTime' (trajectory duration)
            if (mp_config["initial_step_size"]) {
                totalTime = mp_config["initial_step_size"].as<float>();
            }
            if (mp_config["node_collision_radius"]) {
                nodeRadius = mp_config["node_collision_radius"].as<float>();
            }
        } else {
            std::cerr << "Warning: 'motion_planning' section not found in config.yaml. Using defaults.\n";
        }
    } catch (const YAML::BadFile& e) {
        std::cerr << "Warning: Could not open config.yaml. Using default values.\n";
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing config.yaml: " << e.what() << ". Using default values.\n";
    }

    // Log the read values
    std::cout << "Config loaded:\n";
    std::cout << "  Initial Nodes: " << numNodes << "\n";
    std::cout << "  Total Time (Initial Step Size): " << totalTime << "\n";
    std::cout << "  Node Radius: " << nodeRadius << "\n";
    
    // --- 1c. Define Start/Goal ---
    PathNode start = {50.0f, 550.0f, nodeRadius};
    PathNode goal = {750.0f, 50.0f, nodeRadius};
    
    // Generate actual obstacles
    std::vector<Obstacle> obstacles = generateObstacles(NUM_OBSTACLES, OBSTACLE_RADIUS, MAP_WIDTH, MAP_HEIGHT); 

    // --- 2. Initialize Motion Planner (used only for noise sampling and R matrix) ---
    // Pass the generated obstacles to the planner and use configured parameters
    ProximalCrossEntropyMotionPlanner planner(obstacles, config);
    planner.initialize(start, goal, numNodes, totalTime, InterpolationMethod::LINEAR);
    
    const Trajectory& base_trajectory = planner.getCurrentTrajectory();
    const size_t N = base_trajectory.nodes.size();
    
    // --- 3. Generate Noise Samples (epsilon) ---
    std::mt19937 rng; // Standard RNG engine
    std::vector<std::vector<float>> epsilon_x_samples(numSamples);
    std::vector<std::vector<float>> epsilon_y_samples(numSamples);
    
    std::cout << "Generating " << numSamples << " smoothness noise samples for " << N << " nodes...\n";

    for (size_t m = 0; m < numSamples; ++m) {
        epsilon_x_samples[m] = planner.sampleSmoothnessNoise(N, rng);
        epsilon_y_samples[m] = planner.sampleSmoothnessNoise(N, rng);
    }

    // --- 4. Create Noisy Sample Trajectories (Y + epsilon) ---
    std::vector<Trajectory> noisy_samples;
    noisy_samples.reserve(numSamples);

    std::vector<float> Y_base_x(N);
    std::vector<float> Y_base_y(N);
    for (size_t i = 0; i < N; ++i) {
        Y_base_x[i] = base_trajectory.nodes[i].x;
        Y_base_y[i] = base_trajectory.nodes[i].y;
    }
    
    for (size_t m = 0; m < numSamples; ++m) {
        Trajectory perturbed_traj = base_trajectory;
        for (size_t i = 0; i < N; ++i) {
            perturbed_traj.nodes[i].x = Y_base_x[i] + epsilon_x_samples[m][i];
            perturbed_traj.nodes[i].y = Y_base_y[i] + epsilon_y_samples[m][i];
        }
        noisy_samples.push_back(perturbed_traj);
    }
    
    std::cout << "Generated " << noisy_samples.size() << " perturbed trajectories.\n";

    // --- 5. Visualize ---
    visualizeNoise(planner, base_trajectory, noisy_samples);

    return 0;
}
