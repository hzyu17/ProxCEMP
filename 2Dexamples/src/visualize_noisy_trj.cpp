#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "../include/ObstacleMap.h"
#include "../include/Trajectory.h"
#include "../include/ForwardKinematics.h"
#include "../include/PCEMotionPlanner.h"
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

    // --- 1. Get Config File Path ---
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();
    
    std::cout << "Loading config from: " << config_file << "\n\n";

    // --- 2. Read Visualization Parameters ---
    YAML::Node config;
    size_t numSamples = 500;
    
    try {
        config = YAML::LoadFile(config_file);
        
        // Optional: add visualization section to config.yaml
        if (config["visualization"] && config["visualization"]["num_noise_samples"]) {
            numSamples = config["visualization"]["num_noise_samples"].as<size_t>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not read visualization config: " << e.what() << "\n";
        std::cerr << "Using default: " << numSamples << " samples\n\n";
    }
    
    std::cout << "Visualization: " << numSamples << " noise samples\n\n";

    // --- 3. Initialize Planner ---
    std::cout << "Initializing planner (no optimization)...\n";
    ProximalCrossEntropyMotionPlanner planner;
    
    // Initialize without running optimization
    // We'll use a helper method that only does setup
    if (!planner.initializeOnly(config_file)) {
        std::cerr << "Failed to initialize planner!\n";
        return 1;
    }
    
    std::cout << "Planner initialized successfully!\n\n";

    // Get trajectory and environment
    const Trajectory& config_trajectory = planner.getCurrentTrajectory();
    const std::vector<ObstacleND>& obstacles = planner.getObstacles();
    const size_t N = config_trajectory.nodes.size();
    const size_t D = config_trajectory.dimensions();
    
    std::cout << "Configuration:\n";
    std::cout << "  Trajectory nodes: " << N << "\n";
    std::cout << "  Dimensions: " << D << "\n";
    std::cout << "  Obstacles: " << obstacles.size() << "\n\n";

    // --- 4. Setup Random Number Generator ---
    std::mt19937 rng;
    unsigned int seed = 42;
    if (config["experiment"] && config["experiment"]["random_seed"]) {
        seed = config["experiment"]["random_seed"].as<unsigned int>();
    }
    rng.seed(seed);
    std::cout << "Random seed: " << seed << "\n\n";

    // --- 5. Generate Noise Samples ---
    std::cout << "Generating " << numSamples << " noise samples from N(0, R^-1)...\n";
    
    std::vector<Eigen::MatrixXf> epsilon_samples = planner.sampleNoiseMatrices(numSamples, N, D);
    
    std::cout << "Noise sampling complete!\n\n";

    // --- 6. Create Perturbed Trajectories ---
    std::cout << "Creating perturbed trajectories...\n";
    
    Eigen::MatrixXf Y_base = planner.trajectoryToMatrix();
    std::vector<Trajectory> config_noisy_samples;
    config_noisy_samples.reserve(numSamples);
    
    for (size_t m = 0; m < numSamples; ++m) {
        Trajectory perturbed_traj = planner.createPerturbedTrajectory(Y_base, epsilon_samples[m]);
        config_noisy_samples.push_back(perturbed_traj);
        
        if ((m + 1) % 100 == 0) {
            std::cout << "  Created " << (m + 1) << "/" << numSamples << " trajectories\n";
        }
    }
    
    std::cout << "Perturbed trajectories created!\n\n";

    // --- 7. Apply Forward Kinematics (Config → Workspace) ---
    std::cout << "Applying forward kinematics to workspace...\n";
    
    auto fk = planner.getForwardKinematics();
    Trajectory workspace_base = fk->apply(config_trajectory);
    
    std::vector<Trajectory> workspace_noisy_samples;
    workspace_noisy_samples.reserve(numSamples);
    
    for (size_t m = 0; m < numSamples; ++m) {
        workspace_noisy_samples.push_back(fk->apply(config_noisy_samples[m]));
        
        if ((m + 1) % 100 == 0) {
            std::cout << "  Transformed " << (m + 1) << "/" << numSamples << " samples\n";
        }
    }
    std::cout << "Workspace transformation complete!\n\n";

    // --- 8. Compute Noise Statistics ---
    float total_perturbation = 0.0f;
    float max_perturbation = 0.0f;
    
    for (size_t m = 0; m < numSamples; ++m) {
        for (size_t i = 0; i < N; ++i) {
            Eigen::VectorXf diff = workspace_noisy_samples[m].nodes[i].position 
                                  - workspace_base.nodes[i].position;
            float perturbation = diff.norm();
            
            total_perturbation += perturbation;
            max_perturbation = std::max(max_perturbation, perturbation);
        }
    }
    
    float avg_perturbation = total_perturbation / (numSamples * N);
    
    std::cout << "=== Noise Statistics (Workspace) ===\n";
    std::cout << "  Average perturbation: " << avg_perturbation << " units\n";
    std::cout << "  Maximum perturbation: " << max_perturbation << " units\n";
    std::cout << "  Total samples: " << numSamples << "\n";
    std::cout << "  Nodes per trajectory: " << N << "\n\n";
    
    std::cout << "Visualization Legend:\n";
    std::cout << "  Blue cloud    = Noise distribution N(0, R^-1)\n";
    std::cout << "  Red line      = Base trajectory (mean)\n";
    std::cout << "  Green dot     = Start position\n";
    std::cout << "  Red dot       = Goal position\n";
    std::cout << "  Gray circles  = Obstacles\n\n";
    
    std::cout << "Opening visualization window...\n";

    // --- 9. Visualize ---
    visualizeNoise(obstacles, workspace_base, workspace_noisy_samples);

    std::cout << "\nVisualization closed. Exiting.\n";
    return 0;
}
