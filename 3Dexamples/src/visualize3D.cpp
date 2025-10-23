#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "../include/PCEMotionPlanner.h"
#include "../include/visualization3d.h"

// Constants for 3D space
const float MAP_WIDTH_3D = 800.0f;
const float MAP_HEIGHT_3D = 600.0f;
const float MAP_DEPTH_3D = 800.0f;
const int NUM_OBSTACLES_3D = 50;
const float OBSTACLE_RADIUS_3D = 30.0f;

int main() {
    float mapWidth = 800.0f;
    float mapHeight = 600.0f;
    float mapDepth = 800.0f;
    int numObstacles = 50;

    size_t numDimensions = 3;
    float obstacleRadius = 30.0f;
    float clearanceDistance = 100.0f;
    
    int numInitialNodes = 100;
    float initialTotalTime = 8.0f;
    float nodeCollisionRadius = 15.0f;
    
    std::vector<float> startPosition = {50.0f, 50.0f, 50.0f};
    std::vector<float> goalPosition = {750.0f, 550.0f, 750.0f};
    
    unsigned int randomSeed = std::random_device{}();
    YAML::Node config;

    // Read values from config.yaml
    try {
        config = YAML::LoadFile("../configs/config.yaml");
        
        std::cout << "Successfully loaded config.yaml\n";

        // Read experiment settings
        if (config["experiment"]) {
            const YAML::Node& experimentConfig = config["experiment"];
            if (experimentConfig["random_seed"]) {
                randomSeed = experimentConfig["random_seed"].as<unsigned int>();
                std::cout << "Using seed from config: " << randomSeed << "\n";
            } else {
                std::cout << "No seed in config. Using random seed: " << randomSeed << "\n";
            }
        }

        // Read environment settings
        if (config["environment"]) {
            const YAML::Node& envConfig = config["environment"];
            
            if (envConfig["map_width"]) {
                mapWidth = envConfig["map_width"].as<float>();
            }
            if (envConfig["map_height"]) {
                mapHeight = envConfig["map_height"].as<float>();
            }
            if (envConfig["map_depth"]) {
                mapDepth = envConfig["map_depth"].as<float>();
            }
            
            if (envConfig["num_obstacles"]) {
                numObstacles = envConfig["num_obstacles"].as<int>();
            }
            if (envConfig["obstacle_radius"]) {
                obstacleRadius = envConfig["obstacle_radius"].as<float>();
            }
            if (envConfig["clearance_distance"]) {
                clearanceDistance = envConfig["clearance_distance"].as<float>();
            }
            
            // Read start position
            if (envConfig["start_position"] && envConfig["start_position"].IsSequence()) {
                startPosition.clear();
                for (const auto& val : envConfig["start_position"]) {
                    startPosition.push_back(val.as<float>());
                }
                if (startPosition.size() != 3) {
                    std::cerr << "Warning: start_position should have 3 values. Using defaults.\n";
                    startPosition = {50.0f, 50.0f, 50.0f};
                }
            }
            
            // Read goal position
            if (envConfig["goal_position"] && envConfig["goal_position"].IsSequence()) {
                goalPosition.clear();
                for (const auto& val : envConfig["goal_position"]) {
                    goalPosition.push_back(val.as<float>());
                }
                if (goalPosition.size() != 3) {
                    std::cerr << "Warning: goal_position should have 3 values. Using defaults.\n";
                    goalPosition = {750.0f, 550.0f, 750.0f};
                }
            }
        }

        // Read motion planning settings
        if (config["motion_planning"]) {
            const YAML::Node& plannerConfig = config["motion_planning"];

            if (plannerConfig["num_dimensions"]) numDimensions = plannerConfig["num_dimensions"].as<size_t>();
            
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

    // --- 3. Print Configuration ---
    std::cout << "\n=== 3D Visualization Configuration ===\n";
    std::cout << "Environment:\n";
    std::cout << "  Map size: " << mapWidth << " x " << mapHeight << " x " << mapDepth << "\n";
    std::cout << "  Obstacles: " << numObstacles << " (radius=" << obstacleRadius << ")\n";
    std::cout << "  Clearance distance: " << clearanceDistance << "\n";
    std::cout << "  Start: [" << startPosition[0] << ", " << startPosition[1] 
              << ", " << startPosition[2] << "]\n";
    std::cout << "  Goal: [" << goalPosition[0] << ", " << goalPosition[1] 
              << ", " << goalPosition[2] << "]\n";
    std::cout << "Motion Planning:\n";
    std::cout << "  Discretization nodes: " << numInitialNodes << "\n";
    std::cout << "  Total time: " << initialTotalTime << "\n";
    std::cout << "  Node collision radius: " << nodeCollisionRadius << "\n";
    std::cout << "  Random seed: " << randomSeed << "\n";
    std::cout << "=====================================\n\n";

    // --- 2. Initialize GLFW and OpenGL ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(800, 600, 
                                         "3D Trajectory Visualization - Collision Status Display",
                                         nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";

    // --- 3. Create 3D Obstacle Map with Seed ---
    ObstacleMap obstacle_map(numDimensions);  // 3D obstacle map
    obstacle_map.setMapSize(MAP_WIDTH_3D, MAP_HEIGHT_3D, MAP_DEPTH_3D);
    
    obstacle_map.setSeed(randomSeed);
    obstacle_map.generateRandom(NUM_OBSTACLES_3D, OBSTACLE_RADIUS_3D);
    
    std::cout << "Generated " << obstacle_map.size() << " 3D obstacles\n";

    // Define start/goal in 3D space
    PathNode start(50.0f, 50.0f, 50.0f, nodeCollisionRadius);
    PathNode goal(750.0f, 550.0f, 750.0f, nodeCollisionRadius);

    std::cout << "Start: [" << start.position(0) << ", " << start.position(1) 
              << ", " << start.position(2) << "], radius=" << start.radius << "\n";
    std::cout << "Goal:  [" << goal.position(0) << ", " << goal.position(1) 
              << ", " << goal.position(2) << "], radius=" << goal.radius << "\n";

    // --- 4. Motion Planning Initialization ---
    float clearance_dist = 100.0f;
    
    ProximalCrossEntropyMotionPlanner planner;
    planner.initialize(numDimensions, start, goal, numInitialNodes, initialTotalTime,
                      InterpolationMethod::LINEAR, obstacle_map, clearance_dist);

    std::cout << "Planner initialized in " << planner.getNumDimensions() << "D space\n";
    std::cout << "Obstacles after clearance: " << obstacle_map.size() << "\n";

    // --- 5. Setup 3D Renderer ---
    Renderer3D renderer;
    glfwSetWindowUserPointer(window, &renderer);  // Store renderer pointer for callbacks
    
    // Position camera to view the entire space
    renderer.camera.target = glm::vec3(MAP_WIDTH_3D / 2, MAP_HEIGHT_3D / 2, MAP_DEPTH_3D / 2);
    renderer.camera.distance = 1200.0f;
    renderer.camera.yaw = -45.0f;
    renderer.camera.pitch = 30.0f;
    renderer.camera.zoom(0); // Update position

    std::cout << "\n=== 3D Visualization Controls ===\n";
    std::cout << "Mouse Controls:\n";
    std::cout << "  LEFT DRAG:   Rotate camera\n";
    std::cout << "  MIDDLE DRAG: Pan camera\n";
    std::cout << "  SCROLL:      Zoom in/out\n";
    std::cout << "\nKeyboard Controls:\n";
    std::cout << "  SPACE:  Start optimization\n";
    std::cout << "  R:      Reset\n";
    std::cout << "  C:      Toggle collision radius display\n";
    std::cout << "  G:      Toggle grid display\n";
    std::cout << "  ESC:    Quit\n";
    std::cout << "\nCollision radius colors:\n";
    std::cout << "  GREEN = No collision\n";
    std::cout << "  RED   = In collision\n";
    std::cout << "=================================\n\n";

    bool optimized = false;
    bool show_collision_radius = true;
    bool show_grid = true;

    // Count initial collisions
    size_t initial_collisions = 0;
    for (const auto& node : planner.getCurrentTrajectory().nodes) {
        if (isNodeInCollision(node, planner.getObstacles())) {
            initial_collisions++;
        }
    }
    std::cout << "Initial trajectory has " << initial_collisions << " nodes in collision\n";

    // --- 6. Main Render Loop ---
    while (!glfwWindowShouldClose(window)) {
        // Process input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            static bool c_key_was_pressed = false;
            if (!c_key_was_pressed) {
                show_collision_radius = !show_collision_radius;
                std::cout << "Collision radius display: " 
                         << (show_collision_radius ? "ON" : "OFF") << "\n";
                c_key_was_pressed = true;
            }
        } else {
            static bool c_key_was_pressed = false;
            c_key_was_pressed = false;
        }
        
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            static bool g_key_was_pressed = false;
            if (!g_key_was_pressed) {
                show_grid = !show_grid;
                std::cout << "Grid display: " << (show_grid ? "ON" : "OFF") << "\n";
                g_key_was_pressed = true;
            }
        } else {
            static bool g_key_was_pressed = false;
            g_key_was_pressed = false;
        }
        
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            static bool r_key_was_pressed = false;
            if (!r_key_was_pressed) {
                std::cout << "\n--- Resetting ---\n";
                
                // Regenerate obstacles with the same seed
                obstacle_map.clear();
                obstacle_map.setSeed(randomSeed);
                obstacle_map.generateRandom(NUM_OBSTACLES_3D, OBSTACLE_RADIUS_3D);
                std::cout << "Regenerated " << obstacle_map.size() << " 3D obstacles\n";
                
                // Reinitialize planner
                planner.initialize(numDimensions, start, goal, numInitialNodes, initialTotalTime,
                                 InterpolationMethod::LINEAR, obstacle_map, clearance_dist);
                
                std::cout << "Obstacles after clearance: " << obstacle_map.size() << "\n";
                
                // Count collisions after reset
                size_t reset_collisions = 0;
                for (const auto& node : planner.getCurrentTrajectory().nodes) {
                    if (isNodeInCollision(node, planner.getObstacles())) {
                        reset_collisions++;
                    }
                }
                std::cout << "Reset trajectory has " << reset_collisions 
                         << " nodes in collision\n";
                
                optimized = false;
                r_key_was_pressed = true;
            }
        } else {
            static bool r_key_was_pressed = false;
            r_key_was_pressed = false;
        }
        
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            static bool space_key_was_pressed = false;
            if (!space_key_was_pressed && !optimized) {
                std::cout << "\n--- Starting Optimization ---\n";
                // Add optimization code here when ready
                optimized = true;
                space_key_was_pressed = true;
            }
        } else {
            static bool space_key_was_pressed = false;
            space_key_was_pressed = false;
        }

        // --- Render ---
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw grid
        if (show_grid) {
            renderer.drawGrid(MAP_WIDTH_3D, 10);
        }

        // Draw obstacles
        renderer.drawObstacleMap(obstacle_map);

        // Draw trajectory with collision visualization
        const Trajectory& current_traj = planner.getCurrentTrajectory();
        renderer.drawTrajectoryWithCollisionRadius(current_traj, planner.getObstacles(),
                                                   show_collision_radius);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --- 7. Cleanup ---
    
    // Save obstacle map
    obstacle_map.saveToJSON("obstacle_map_3d_seed_" + std::to_string(randomSeed) + ".json");
    std::cout << "\nSaved obstacle map to JSON\n";

    glfwTerminate();
    return 0;
}
