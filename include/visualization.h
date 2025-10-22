#pragma once

#include "visualization_base.h"

// Forward declarations for types assumed to be in MotionPlanner.h
// These structures must be fully defined where Visualization.h is included.
class MotionPlanner; 


/**
 * @brief Visualizes the optimization history to any SFML RenderTarget (Window or Texture).
 * It relies on the templated functions (drawNode, drawTrajectorySegments, drawObstacles) 
 * from visualization.h, assuming they correctly plot N-D data onto the first 2 dimensions.
 */
template<typename RenderTarget>
void visualizeOptimizationHistoryToTarget(RenderTarget& target, const MotionPlanner& planner) {
    
    target.clear(sf::Color(240, 240, 240)); 

    // 1. Draw obstacles (planner_obstacles_nd is the filtered list)
    const std::vector<ObstacleND>& obstacles_nd = planner.getObstacles(); 
    drawObstacles(target, obstacles_nd);
    
    const auto& history = planner.getTrajectoryHistory();
    size_t num_iterations = history.size();
    
    if (num_iterations == 0) {
        // If history is empty, draw the current initial trajectory
        const Trajectory& current_traj = planner.getCurrentTrajectory();
        drawTrajectorySegments(target, current_traj, sf::Color(0, 0, 255, 255));
        num_iterations = 1;
    }

    // 2. Draw historical paths (fading)
    for (size_t i = 0; i < num_iterations - 1; ++i) {
        // Calculate alpha based on iteration index
        float alpha_float = 5.0f + (145.0f * (float)i / std::max(1.0f, (float)num_iterations - 2));
        unsigned char alpha = static_cast<unsigned char>(alpha_float); 
        sf::Color history_color(50, 50, 255, alpha); 
        drawTrajectorySegments(target, history[i], history_color);
    }

    // 3. Draw the Final Trajectory (bright red)
    const Trajectory& final_path = history.back();
    drawTrajectorySegments(target, final_path, sf::Color(255, 0, 0, 255)); 

    // 4. Draw Start/Goal Points
    if (!final_path.nodes.empty()) {
        const PathNode& start_node = final_path.nodes[final_path.start_index];
        const PathNode& goal_node = final_path.nodes[final_path.goal_index];
        
        // Use a fixed radius for visibility, otherwise the sphere radius
        drawNode(target, start_node, 6.0f, sf::Color::Green); 
        drawNode(target, goal_node, 6.0f, sf::Color::Red);
    }
}


/**
 * @brief Calculates the signed distance function (SDF) for 2D obstacles (projection).
 */
inline float calculateSDF2D(float x, float y, const std::vector<ObstacleND>& obstacles) {
    float min_sdf = std::numeric_limits<float>::max();
    
    for (const auto& obs : obstacles) {
        if (obs.center.size() < 2) continue;
        
        float dx = x - obs.center(0);
        float dy = y - obs.center(1);
        float dist_to_center = std::sqrt(dx*dx + dy*dy);
        float signed_distance = dist_to_center - obs.radius;
        
        if (signed_distance < min_sdf) {
            min_sdf = signed_distance;
        }
    }
    
    return min_sdf;
}

// --- Visualization Scenes (Templatized for RenderTarget) ---

/**
 * @brief Visualizes the initial trajectory, including the collision checking spheres.
 */
template<typename RenderTarget>
inline void visualizeInitialTrajectoryWithSpheres(RenderTarget& target, const MotionPlanner& planner) {
    target.clear(sf::Color(240, 240, 240));

    const std::vector<ObstacleND>& obstacles = planner.getObstacles();
    drawObstacles(target, obstacles); 
    
    const Trajectory& initial_path = planner.getCurrentTrajectory();
    drawTrajectorySegments(target, initial_path, sf::Color(50, 50, 255, 150)); 

    for (size_t i = 0; i < initial_path.nodes.size(); ++i) {
        const auto& node = initial_path.nodes[i];
        
        float node_x = node.position(0);
        float node_y = node.position.size() > 1 ? node.position(1) : 0.0f;

        float sdf_value = calculateSDF2D(node_x, node_y, obstacles);
        float effective_sdf = sdf_value - node.radius;

        unsigned char alpha_fill;
        sf::Color node_color;
        
        if (effective_sdf < 0.0f) {
            node_color = sf::Color(255, 50, 50); 
            alpha_fill = 255;
        } else if (effective_sdf < 10.0f) { 
            node_color = sf::Color(255, 165, 0); 
            alpha_fill = 150;
        } else {
            node_color = sf::Color(50, 200, 50); 
            alpha_fill = 80;
        }
        
        sf::Color final_fill_color = node_color;
        final_fill_color.a = alpha_fill;

        sf::CircleShape circle(node.radius);
        circle.setFillColor(final_fill_color);
        circle.setOutlineThickness(1.0f);
        circle.setOutlineColor(sf::Color(0, 0, 0, 50));
        circle.setOrigin(sf::Vector2f(node.radius, node.radius));
        circle.setPosition({node_x, node_y}); 
        target.draw(circle);
        
        if (i == initial_path.start_index || i == initial_path.goal_index) {
            drawNode(target, node, 5.0f, 
                    (i == initial_path.start_index) ? sf::Color::Green : sf::Color::Red); 
        } else {
            drawNode(target, node, 1.5f, sf::Color(0, 0, 0, 200)); 
        }
    }
    
    // Only display if the target is a window, not a texture
    if constexpr (std::is_same_v<RenderTarget, sf::RenderWindow>) {
        target.display();
    }
}


/**
 * @brief Visualizes optimization progress with trajectory history (fading lines).
 */
template<typename RenderTarget>
inline void visualizeOptimizationProgress(RenderTarget& target, const MotionPlanner& planner) {
    target.clear(sf::Color(240, 240, 240));

    const std::vector<ObstacleND>& obstacles = planner.getObstacles();
    drawObstacles(target, obstacles); 
    
    const std::vector<Trajectory>& history = planner.getTrajectoryHistory();
    
    for (size_t iter = 0; iter < history.size(); ++iter) {
        float alpha = 50.0f + (200.0f * iter / std::max(1ul, history.size() - 1));
        sf::Color traj_color(100, 100, 255, static_cast<unsigned char>(alpha));
        
        drawTrajectorySegments(target, history[iter], traj_color); 
    }
    
    if (!history.empty()) {
        const Trajectory& final_traj = history.back();
        drawTrajectorySegments(target, final_traj, sf::Color(0, 0, 255, 255)); 
        
        if (!final_traj.nodes.empty()) {
            drawNode(target, final_traj.nodes.front(), 8.0f, sf::Color::Green); 
            drawNode(target, final_traj.nodes.back(), 8.0f, sf::Color::Red);     
        }
    }
    
    // Only display if the target is a window
    if constexpr (std::is_same_v<RenderTarget, sf::RenderWindow>) {
        target.display();
    }
}


/**
 * @brief Visualizes robot arm configuration with workspace path.
 */
template<typename RenderTarget>
inline void visualizeRobotArmWithFK(RenderTarget& target, 
                             const MotionPlanner& planner,
                             const Eigen::Vector2f& base_position = Eigen::Vector2f(400.0f, 400.0f),
                             float scale = 100.0f) {
    target.clear(sf::Color(240, 240, 240));

    const std::vector<ObstacleND>& obstacles = planner.getObstacles();
    drawObstacles(target, obstacles);
    
    const Trajectory& joint_traj = planner.getCurrentTrajectory();
    
    // NOTE: This relies on MotionPlanner::applyForwardKinematics 
    Trajectory workspace_traj = planner.applyForwardKinematics(joint_traj);
    
    if (workspace_traj.nodes.size() >= 2) {
        sf::VertexArray ee_path(sf::PrimitiveType::LineStrip, workspace_traj.nodes.size());
        for (size_t i = 0; i < workspace_traj.nodes.size(); ++i) {
            float ee_x = base_position(0) + workspace_traj.nodes[i].position(0) * scale;
            float ee_y = base_position(1) - workspace_traj.nodes[i].position(1) * scale;
            ee_path[i].position = sf::Vector2f(ee_x, ee_y);
            ee_path[i].color = sf::Color(255, 100, 0, 200);
        }
        target.draw(ee_path);
    }
    
    sf::CircleShape base(8.0f);
    base.setFillColor(sf::Color(50, 50, 50, 255));
    base.setOrigin(sf::Vector2f(8.0f, 8.0f));
    base.setPosition({base_position[0], base_position[1]});
    target.draw(base);
    
    // Only display if the target is a window
    if constexpr (std::is_same_v<RenderTarget, sf::RenderWindow>) {
        target.display();
    }
}


// --- Utility Functions (Non-Templatized) ---

/**
 * @brief Utility function to save the current planner state visualization to a file.
 */
inline bool savePlannerToFile(const MotionPlanner& planner, const std::string& filename) {
    // Determine the path to save the file
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path figures_dir = source_dir / "figures";
    
    try {
        if (!std::filesystem::exists(figures_dir)) {
            std::filesystem::create_directories(figures_dir);
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating figures directory: " << e.what() << std::endl;
        return false;
    }

    std::filesystem::path output_path = figures_dir / filename;

    // Use RenderTexture for off-screen rendering
    sf::RenderTexture renderTexture;
    
    // FIX: Reverting to resize, as 'create' is not available in your SFML version with these arguments.
    // The resize method is preferred for default-constructed sf::RenderTexture in newer SFML versions.
    if (!renderTexture.resize({(unsigned int)MAP_WIDTH, (unsigned int)MAP_HEIGHT})) { 
        std::cerr << "Error: Could not create render texture!" << std::endl;
        return false;
    }

    renderTexture.clear(sf::Color(240, 240, 240));
    
    // Calls the now-templated function with sf::RenderTexture
    visualizeOptimizationProgress(renderTexture, planner); 
    
    renderTexture.display();

    // Save to file
    sf::Image screenshot = renderTexture.getTexture().copyToImage();
    if (screenshot.saveToFile(output_path.string())) {
        std::cout << "Saved visualization to: " << output_path << std::endl;
        return true;
    } else {
        std::cerr << "Error: Could not save image to " << output_path << std::endl;
        return false;
    }
}

/**
 * @brief Opens an SFML window to display the results for a single planner.
 */
inline void showPlannerWindow(const MotionPlanner& planner, const std::string& title) {
    sf::RenderWindow window(sf::VideoMode({(unsigned int)MAP_WIDTH, (unsigned int)MAP_HEIGHT}), title, sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    // Calls the now-templated function with sf::RenderWindow
    visualizeInitialTrajectoryWithSpheres(window, planner);

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
        
        // Calls the now-templated function with sf::RenderWindow
        visualizeOptimizationProgress(window, planner); 
    }
}
