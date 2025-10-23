#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "../include/ObstacleMap.h"
#include "../include/Trajectory.h"
#include "../include/ForwardKinematics.h"
#include "../include/PCEMotionPlanner.h"
#include "../include/visualization3d.h"

/**
 * @brief Visualizes smoothness noise distribution N(0, R^-1) in 3D workspace
 */
void visualizeNoise3D(const std::vector<ObstacleND>& obstacles,
                      const Trajectory& workspace_base_trajectory, 
                      const std::vector<Trajectory>& workspace_noisy_samples) {
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    GLFWwindow* window = glfwCreateWindow(1200, 900,
                                         "3D Smoothness Noise Visualization N(0, R^-1)",
                                         nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        return;
    }
    
    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
    // Create renderer
    Renderer3D renderer;
    glfwSetWindowUserPointer(window, &renderer);
    
    // Adjust camera to center of workspace
    renderer.camera.target = glm::vec3(400.0f, 300.0f, 400.0f);
    renderer.camera.distance = 1200.0f;
    renderer.camera.yaw = -45.0f;
    renderer.camera.pitch = 30.0f;
    renderer.camera.rotate(0.0f, 0.0f);  // Update position
    
    std::cout << "\n=== 3D Controls ===\n";
    std::cout << "Left Mouse Drag:  Rotate camera\n";
    std::cout << "Right Mouse Drag: Pan camera\n";
    std::cout << "Mouse Scroll:     Zoom in/out\n";
    std::cout << "G:                Toggle grid\n";
    std::cout << "ESC:              Exit\n";
    std::cout << "===================\n\n";
    
    bool showGrid = true;
    bool gKeyWasPressed = false;
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Process keyboard input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        // Toggle grid with G key (with debouncing)
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            if (!gKeyWasPressed) {
                showGrid = !showGrid;
                std::cout << "Grid: " << (showGrid ? "ON" : "OFF") << "\n";
                gKeyWasPressed = true;
            }
        } else {
            gKeyWasPressed = false;
        }
        
        // Clear screen
        glClearColor(0.94f, 0.94f, 0.94f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // 1. Draw grid (optional)
        if (showGrid) {
            renderer.drawGrid(800.0f, 10);
        }
        
        // 2. Draw obstacles
        renderer.drawObstacles(obstacles);
        
        // 3. Draw noisy samples (semi-transparent blue)
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        for (const auto& sample : workspace_noisy_samples) {
            renderer.drawTrajectorySegments(sample, glm::vec4(0.2f, 0.2f, 1.0f, 0.08f));
        }
        
        // 4. Draw base trajectory (red)
        renderer.drawTrajectorySegments(workspace_base_trajectory, 
                                       glm::vec4(1.0f, 0.0f, 0.0f, 0.9f));
        
        // 5. Draw nodes on base trajectory
        for (const auto& node : workspace_base_trajectory.nodes) {
            renderer.drawNode(node, 3.0f, glm::vec3(1.0f, 0.4f, 0.4f));
        }
        
        // 6. Draw Start/Goal
        if (!workspace_base_trajectory.nodes.empty()) {
            renderer.drawNode(workspace_base_trajectory.nodes[workspace_base_trajectory.start_index],
                            10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
            renderer.drawNode(workspace_base_trajectory.nodes[workspace_base_trajectory.goal_index],
                            10.0f, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();
}


int main() {
    std::cout << "========================================\n";
    std::cout << "  3D Trajectory Noise Visualization\n";
    std::cout << "  (Smoothness Distribution N(0, R^-1))\n";
    std::cout << "========================================\n\n";

    // --- 1. Get Config File Path ---
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();
    
    std::cout << "Loading config from: " << config_file << "\n\n";

    // --- 2. Read Configuration ---
    YAML::Node config;
    
    // Default values
    size_t numSamples = 500;
    size_t numNodes = 100;
    size_t numObstacles = 50;
    size_t numDimensions = 3;
    float obstacleRadius = 30.0f;
    float mapWidth = 800.0f;
    float mapHeight = 600.0f;
    float mapDepth = 800.0f;
    float nodeRadius = 15.0f;
    float totalTime = 8.0f;
    float clearanceDist = 100.0f;
    unsigned int seed = 999;
    
    std::vector<float> start = {50.0f, 50.0f, 50.0f};
    std::vector<float> goal = {750.0f, 550.0f, 750.0f};
    
    try {
        config = YAML::LoadFile(config_file);
        
        // Read experiment settings
        if (config["experiment"] && config["experiment"]["random_seed"]) {
            seed = config["experiment"]["random_seed"].as<unsigned int>();
        }
        
        // Read environment settings
        if (config["environment"]) {
            auto env = config["environment"];
            
            if (env["map_width"]) mapWidth = env["map_width"].as<float>();
            if (env["map_height"]) mapHeight = env["map_height"].as<float>();
            if (env["map_depth"]) mapDepth = env["map_depth"].as<float>();
            
            if (env["num_obstacles"]) numObstacles = env["num_obstacles"].as<size_t>();
            if (env["obstacle_radius"]) obstacleRadius = env["obstacle_radius"].as<float>();
            if (env["clearance_distance"]) clearanceDist = env["clearance_distance"].as<float>();
            
            if (env["start_position"] && env["start_position"].IsSequence()) {
                start.clear();
                for (const auto& val : env["start_position"]) {
                    start.push_back(val.as<float>());
                }
            }
            
            if (env["goal_position"] && env["goal_position"].IsSequence()) {
                goal.clear();
                for (const auto& val : env["goal_position"]) {
                    goal.push_back(val.as<float>());
                }
            }
        }
        
        // Read motion planning settings
        if (config["motion_planning"]) {
            auto mp = config["motion_planning"];
            if (mp["num_dimensions"]) numDimensions = mp["num_dimensions"].as<size_t>();
            if (mp["num_discretization"]) numNodes = mp["num_discretization"].as<size_t>();
            if (mp["total_time"]) totalTime = mp["total_time"].as<float>();
            if (mp["node_collision_radius"]) nodeRadius = mp["node_collision_radius"].as<float>();
        }
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML Error: " << e.what() << "\n";
        std::cerr << "Using default values\n\n";
    }
    
    std::cout << "Configuration:\n";
    std::cout << "  Dimensions: " << numDimensions << "\n";
    std::cout << "  Map: " << mapWidth << " x " << mapHeight << " x " << mapDepth << "\n";
    std::cout << "  Start: [" << start[0] << ", " << start[1] << ", " << start[2] << "]\n";
    std::cout << "  Goal: [" << goal[0] << ", " << goal[1] << ", " << goal[2] << "]\n";
    std::cout << "  Trajectory nodes: " << numNodes << "\n";
    std::cout << "  Node radius: " << nodeRadius << "\n";
    std::cout << "  Obstacles: " << numObstacles << " (radius=" << obstacleRadius << ")\n";
    std::cout << "  Clearance distance: " << clearanceDist << "\n";
    std::cout << "  Noise samples: " << numSamples << "\n";
    std::cout << "  Random seed: " << seed << "\n\n";

    // --- 3. Generate Obstacles Dynamically ---
    std::cout << "Generating obstacles dynamically...\n";
    
    ObstacleMap obstacle_map(numDimensions);  // 3D
    obstacle_map.setMapSize(mapWidth, mapHeight, mapDepth);
    obstacle_map.setSeed(seed);
    obstacle_map.generateRandom(numObstacles, obstacleRadius);
    
    std::cout << "Generated " << obstacle_map.size() << " 3D obstacles\n\n";

    // --- 4. Initialize Planner ---
    std::cout << "Initializing planner...\n";
    
    PathNode startNode(start[0], start[1], start[2], nodeRadius);
    PathNode goalNode(goal[0], goal[1], goal[2], nodeRadius);
    
    ProximalCrossEntropyMotionPlanner planner;
    planner.initialize(numDimensions, startNode, goalNode, numNodes, totalTime,
                      InterpolationMethod::LINEAR, obstacle_map, clearanceDist);
    
    std::cout << "Planner initialized successfully!\n";
    std::cout << "Obstacles after clearance: " << obstacle_map.size() << "\n\n";

    // Get trajectory
    const Trajectory& config_trajectory = planner.getCurrentTrajectory();
    const std::vector<ObstacleND>& obstacles = planner.getObstacles();
    const size_t N = config_trajectory.nodes.size();
    const size_t D = config_trajectory.dimensions();
    
    if (D != 3) {
        std::cerr << "Error: Expected 3D trajectory, got " << D << "D\n";
        return 1;
    }
    
    std::cout << "Trajectory:\n";
    std::cout << "  Nodes: " << N << "\n";
    std::cout << "  Dimensions: " << D << "D\n";
    std::cout << "  Obstacles: " << obstacles.size() << "\n\n";

    // --- 5. Setup Random Number Generator ---
    std::mt19937 rng(seed);

    // --- 6. Generate Noise Samples ---
    std::cout << "Generating " << numSamples << " noise samples from N(0, R^-1)...\n";
    
    std::vector<Eigen::MatrixXf> epsilon_samples = planner.sampleNoiseMatrices(numSamples, N, D);
    std::cout << "Noise sampling complete!\n\n";

    // --- 7. Create Perturbed Trajectories ---
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

    // --- 8. Apply Forward Kinematics ---
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

    // --- 9. Compute Statistics ---
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
    
    std::cout << "=== Noise Statistics (3D Workspace) ===\n";
    std::cout << "  Average perturbation: " << avg_perturbation << " units\n";
    std::cout << "  Maximum perturbation: " << max_perturbation << " units\n";
    std::cout << "  Total samples: " << numSamples << "\n";
    std::cout << "  Nodes per trajectory: " << N << "\n\n";
    
    std::cout << "Visualization Legend:\n";
    std::cout << "  Blue cloud    = Noise distribution N(0, R^-1)\n";
    std::cout << "  Red line      = Base trajectory (mean)\n";
    std::cout << "  Green sphere  = Start position\n";
    std::cout << "  Red sphere    = Goal position\n";
    std::cout << "  Gray spheres  = Obstacles\n\n";
    
    std::cout << "Opening 3D visualization window...\n";

    // --- 10. Visualize ---
    visualizeNoise3D(obstacles, workspace_base, workspace_noisy_samples);

    std::cout << "\nVisualization closed. Exiting.\n";
    return 0;
}
