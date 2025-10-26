#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "../include/PCEMotionPlanner.h"
#include "../include/CollisionAvoidanceTask.h"
#include "../include/visualization.h"

// --- Visualization Function ---

/**
 * @brief Visualizes smoothness noise distribution N(0, R^-1) in workspace
 */
void visualizeNoise(const std::vector<ObstacleND>& obstacles,
                    const Trajectory& workspace_base_trajectory, 
                    const std::vector<Trajectory>& workspace_noisy_samples) {
    
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), 
                           "Smoothness Noise Visualization N(0, R^-1)", 
                           sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    std::cout << "\n=== Controls ===\n";
    std::cout << "ESC: Exit\n";
    std::cout << "================\n\n";

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
            }
        }
        
        window.clear(sf::Color(240, 240, 240));

        // 1. Draw obstacles
        drawObstacles(window, obstacles);

        // 2. Draw all noisy samples (faded blue cloud)
        for (const auto& sample : workspace_noisy_samples) {
            drawTrajectorySegments(window, sample, sf::Color(50, 50, 255, 30));
        }

        // 3. Draw base trajectory (red line)
        drawTrajectorySegments(window, workspace_base_trajectory, sf::Color(255, 0, 0, 255));
        
        for (const auto& node : workspace_base_trajectory.nodes) {
            drawNode(window, node, 3.0f, sf::Color(255, 100, 100));
        }
        
        // 4. Draw Start/Goal
        if (!workspace_base_trajectory.nodes.empty()) {
            drawNode(window, workspace_base_trajectory.nodes[workspace_base_trajectory.start_index], 
                    8.0f, sf::Color::Green);
            drawNode(window, workspace_base_trajectory.nodes[workspace_base_trajectory.goal_index], 
                    8.0f, sf::Color::Red);
        }

        window.display();
    }
}


int main() {
    std::cout << "========================================\n";
    std::cout << "  Trajectory Noise Visualization\n";
    std::cout << "  (Smoothness Distribution N(0, R^-1))\n";
    std::cout << "========================================\n\n";

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
    
    // Get trajectory and environment
    const Trajectory& config_trajectory = planner.getCurrentTrajectory();
    const size_t N = config_trajectory.nodes.size();
    const size_t D = config_trajectory.dimensions();
    
    // Get obstacles from task
    std::vector<ObstacleND> obstacles;
    if (task) {
        obstacles = task->getObstacles();
    }
    
    std::cout << "Configuration:\n";
    std::cout << "  Trajectory nodes: " << N << "\n";
    std::cout << "  Dimensions: " << D << "\n";
    std::cout << "  Obstacles: " << obstacles.size() << "\n\n";

    // --- 6. Setup Random Number Generator ---
    std::mt19937 rng;
    unsigned int seed = pce_config.random_seed;
    rng.seed(seed);
    std::cout << "Random seed: " << seed << "\n\n";

    int num_samples = pce_config.num_samples;

    // --- 7. Generate Noise Samples ---
    std::cout << "Generating " << num_samples << " noise samples from N(0, R^-1)...\n";
    
    // Access base class method for sampling noise
    std::vector<Eigen::MatrixXf> epsilon_samples;
    
    // For now, assuming sampleNoiseMatrices is made public or accessible
    // If it's protected, you'll need to add a public wrapper method in the planner
    epsilon_samples = planner.sampleNoiseMatrices(num_samples, N, D);
    
    std::cout << "Noise sampling complete!\n\n";

    // --- 8. Create Perturbed Trajectories ---
    std::cout << "Creating perturbed trajectories...\n";
    
    // Get base trajectory as matrix
    Eigen::MatrixXf Y_base(D, N);
    for (size_t i = 0; i < N; ++i) {
        Y_base.col(i) = config_trajectory.nodes[i].position;
    }
    
    std::vector<Trajectory> config_noisy_samples;
    config_noisy_samples.reserve(num_samples);
    
    for (size_t m = 0; m < num_samples; ++m) {
        // Create perturbed trajectory
        Trajectory perturbed_traj = config_trajectory; // Copy base
        Eigen::MatrixXf Y_perturbed = Y_base + epsilon_samples[m];
        
        // Update positions
        for (size_t i = 0; i < N; ++i) {
            perturbed_traj.nodes[i].position = Y_perturbed.col(i);
        }
        
        config_noisy_samples.push_back(perturbed_traj);
    }
    
    std::cout << "Perturbed trajectories created!\n\n";

    // --- 9. Apply Forward Kinematics (Config → Workspace) ---
    std::cout << "Applying forward kinematics to workspace...\n";
    
    auto fk = planner.getForwardKinematics();
    Trajectory workspace_base = fk->apply(config_trajectory);
    
    std::vector<Trajectory> workspace_noisy_samples;
    workspace_noisy_samples.reserve(num_samples);
    
    for (size_t m = 0; m < num_samples; ++m) {
        workspace_noisy_samples.push_back(fk->apply(config_noisy_samples[m]));
    }
    std::cout << "Workspace transformation complete!\n\n";

    // --- 10. Compute Noise Statistics ---
    float total_perturbation = 0.0f;
    float max_perturbation = 0.0f;
    
    for (size_t m = 0; m < num_samples; ++m) {
        for (size_t i = 0; i < N; ++i) {
            Eigen::VectorXf diff = workspace_noisy_samples[m].nodes[i].position 
                                  - workspace_base.nodes[i].position;
            float perturbation = diff.norm();
            
            total_perturbation += perturbation;
            max_perturbation = std::max(max_perturbation, perturbation);
        }
    }
    
    float avg_perturbation = total_perturbation / (num_samples * N);
    
    std::cout << "=== Noise Statistics (Workspace) ===\n";
    std::cout << "  Average perturbation: " << avg_perturbation << " units\n";
    std::cout << "  Maximum perturbation: " << max_perturbation << " units\n";
    std::cout << "  Total samples: " << num_samples << "\n";
    std::cout << "  Nodes per trajectory: " << N << "\n\n";
    
    std::cout << "Visualization Legend:\n";
    std::cout << "  Blue cloud    = Noise distribution N(0, R^-1)\n";
    std::cout << "  Red line      = Base trajectory (mean)\n";
    std::cout << "  Green dot     = Start position\n";
    std::cout << "  Red dot       = Goal position\n";
    std::cout << "  Gray circles  = Obstacles\n\n";
    
    std::cout << "Opening visualization window...\n";

    // --- 11. Visualize ---
    visualizeNoise(obstacles, workspace_base, workspace_noisy_samples);

    std::cout << "\nVisualization closed. Exiting.\n";
    return 0;
}
