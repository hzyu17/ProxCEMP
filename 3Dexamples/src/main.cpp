#include "PCEMotionPlanner.h"
#include "NGDMotionPlanner.h"
#include "visualization3d.h"
#include <filesystem>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <yaml-cpp/yaml.h>

/**
 * @brief Visualizes initial trajectory state in 3D
 */
void visualizeInitialState3D(const std::vector<ObstacleND>& obstacles,
                             const Trajectory& trajectory,
                             const std::string& window_title) {
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    GLFWwindow* window = glfwCreateWindow(1200, 900, window_title.c_str(), nullptr, nullptr);
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
    std::cout << "SPACE/ENTER:      Continue to optimization\n";
    std::cout << "ESC:              Exit\n";
    std::cout << "===================\n\n";
    
    bool showGrid = true;
    bool gKeyWasPressed = false;
    bool shouldContinue = false;
    
    // Main render loop
    while (!glfwWindowShouldClose(window) && !shouldContinue) {
        // Process keyboard input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS || 
            glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
            shouldContinue = true;
            std::cout << "Continuing to optimization...\n";
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
        
        // 3. Draw initial trajectory (blue/cyan)
        renderer.drawTrajectorySegments(trajectory, glm::vec4(0.2f, 0.6f, 1.0f, 0.8f));
        
        // 4. Draw nodes
        for (const auto& node : trajectory.nodes) {
            renderer.drawNode(node, 3.0f, glm::vec3(0.3f, 0.7f, 1.0f));
        }
        
        // 5. Draw Start/Goal
        if (!trajectory.nodes.empty()) {
            renderer.drawNode(trajectory.nodes[trajectory.start_index],
                            10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
            renderer.drawNode(trajectory.nodes[trajectory.goal_index],
                            10.0f, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();
}

/**
 * @brief Visualizes final planner results in 3D with history navigation
 */
void showPlannerWindow3D(MotionPlanner& planner, const std::string& window_title) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    GLFWwindow* window = glfwCreateWindow(1200, 900, window_title.c_str(), nullptr, nullptr);
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
    
    // Setup camera
    renderer.camera.target = glm::vec3(400.0f, 300.0f, 400.0f);
    renderer.camera.distance = 1200.0f;
    renderer.camera.yaw = -45.0f;
    renderer.camera.pitch = 30.0f;
    renderer.camera.rotate(0.0f, 0.0f);
    
    const auto& obstacles = planner.getObstacles();
    const auto& trajectory = planner.getCurrentTrajectory();
    const auto& history = planner.getTrajectoryHistory();
    
    size_t current_iteration = history.empty() ? 0 : history.size() - 1;
    bool showGrid = true;
    bool showHistory = true;
    bool showCollisionRadius = false;
    bool gKeyWasPressed = false;
    bool hKeyWasPressed = false;
    bool cKeyWasPressed = false;
    
    std::cout << "\n=== 3D Visualization Controls ===\n";
    std::cout << "Left Mouse Drag:  Rotate camera\n";
    std::cout << "Right Mouse Drag: Pan camera\n";
    std::cout << "Mouse Scroll:     Zoom in/out\n";
    std::cout << "Left/Right Arrow: Navigate trajectory history\n";
    std::cout << "G:                Toggle grid\n";
    std::cout << "H:                Toggle history\n";
    std::cout << "C:                Toggle collision radius\n";
    std::cout << "ESC:              Close window\n";
    std::cout << "==================================\n\n";
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Keyboard input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        // Navigate history
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS && !history.empty() && current_iteration > 0) {
            current_iteration--;
            std::cout << "Iteration: " << current_iteration << "/" << (history.size() - 1) << "\n";
            glfwWaitEventsTimeout(0.1);  // Debounce
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS && !history.empty() && current_iteration < history.size() - 1) {
            current_iteration++;
            std::cout << "Iteration: " << current_iteration << "/" << (history.size() - 1) << "\n";
            glfwWaitEventsTimeout(0.1);  // Debounce
        }
        
        // Toggle grid
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            if (!gKeyWasPressed) {
                showGrid = !showGrid;
                std::cout << "Grid: " << (showGrid ? "ON" : "OFF") << "\n";
                gKeyWasPressed = true;
            }
        } else {
            gKeyWasPressed = false;
        }
        
        // Toggle history
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
            if (!hKeyWasPressed) {
                showHistory = !showHistory;
                std::cout << "History: " << (showHistory ? "ON" : "OFF") << "\n";
                hKeyWasPressed = true;
            }
        } else {
            hKeyWasPressed = false;
        }
        
        // Toggle collision radius
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            if (!cKeyWasPressed) {
                showCollisionRadius = !showCollisionRadius;
                std::cout << "Collision Radius: " << (showCollisionRadius ? "ON" : "OFF") << "\n";
                cKeyWasPressed = true;
            }
        } else {
            cKeyWasPressed = false;
        }
        
        // Clear screen
        glClearColor(0.94f, 0.94f, 0.94f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Draw scene
        if (showGrid) {
            renderer.drawGrid(800.0f, 10);
        }
        
        renderer.drawObstacles(obstacles);
        
        // Draw trajectory history (semi-transparent)
        if (showHistory && !history.empty()) {
            for (size_t i = 0; i < current_iteration; ++i) {
                float alpha = 0.1f + 0.3f * (float(i) / std::max(1.0f, float(current_iteration)));
                renderer.drawTrajectorySegments(history[i], 
                    glm::vec4(0.5f, 0.5f, 0.5f, alpha));
            }
        }
        
        // Draw current trajectory
        const Trajectory& current_traj = history.empty() ? trajectory : history[current_iteration];
        
        if (showCollisionRadius) {
            renderer.drawTrajectoryWithCollisionRadius(current_traj, obstacles, true);
        } else {
            renderer.drawTrajectorySegments(current_traj, glm::vec4(0.0f, 0.8f, 0.2f, 0.9f));
            
            // Draw nodes
            for (const auto& node : current_traj.nodes) {
                renderer.drawNode(node, 3.0f, glm::vec3(0.2f, 0.9f, 0.3f));
            }
            
            // Draw start/goal
            if (!current_traj.nodes.empty()) {
                renderer.drawNode(current_traj.nodes[current_traj.start_index],
                                10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
                renderer.drawNode(current_traj.nodes[current_traj.goal_index],
                                10.0f, glm::vec3(1.0f, 0.0f, 0.0f));
            }
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();
}

int main() {
    std::cout << "=================================================\n";
    std::cout << " 3D Motion Planning Experiments\n";
    std::cout << "=================================================\n\n";
    
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../configs/config.yaml";
    std::string config_file = std::filesystem::canonical(config_path).string();
    
    // Load config to check visualization setting
    YAML::Node config = YAML::LoadFile(config_file);
    bool visualize = config["experiment"] &&
                     config["experiment"]["visualize_initial_state"] &&
                     config["experiment"]["visualize_initial_state"].as<bool>();
    
    // --- Run PCEM ---
    std::cout << "\n=== PCEM Planner (3D) ===\n";
    ProximalCrossEntropyMotionPlanner planner_pce;
    
    if (visualize) {
        // Initialize without running optimization
        planner_pce.initializeOnly(config_file);

        // Show 3D visualization
        std::cout << "Showing initial state visualization (3D)...\n";
        visualizeInitialState3D(planner_pce.getObstacles(),
                               planner_pce.getCurrentTrajectory(),
                               "PCEM - Initial State (3D)");
    }
    
    // Now run optimization
    bool success_pce = planner_pce.solve(config_file);
    
    // // --- Run NGD ---
    // std::cout << "\n=== NGD Planner (3D) ===\n";
    // NGDMotionPlanner planner_ngd;
    
    // if (visualize) {
    //     planner_ngd.initializeOnly(config_file);
    //     visualizeInitialState3D(planner_ngd.getObstacles(),
    //                            planner_ngd.getCurrentTrajectory(),
    //                            "NGD - Initial State (3D)");
    // }
    
    // bool success_ngd = planner_ngd.solve(config_file);
    
    // --- Final 3D visualizations ---
    std::cout << "\n--- Showing Final Results (3D) ---\n";
    showPlannerWindow3D(planner_pce, "PCEM Results (3D)");
    // showPlannerWindow3D(planner_ngd, "NGD Results (3D)");
    
    std::cout << "\n=== Experiment Complete ===\n";
    std::cout << "PCEM: " << (success_pce ? "SUCCESS" : "FAILED") << "\n";
    // std::cout << "NGD:  " << (success_ngd ? "SUCCESS" : "FAILED") << "\n";
    
    return 0;
}
