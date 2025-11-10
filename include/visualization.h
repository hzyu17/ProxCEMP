#pragma once

#include "visualization_base.h"
#include "ObstacleMap.h"
#include "Trajectory.h"
#include <vector>
#include <memory>
#include <filesystem>

/**
 * @brief Visualizes a trajectory with obstacles to any SFML RenderTarget.
 * Works directly with ObstacleMap and Trajectory - no MotionPlanner needed!
 */
template<typename RenderTarget>
void visualizeTrajectory(
    RenderTarget& target,
    const std::vector<ObstacleND>& obstacles,
    const Trajectory& trajectory,
    const sf::Color& trajectory_color = sf::Color(0, 0, 255, 255),
    bool clear_background = true)
{
    if (clear_background) {
        target.clear(sf::Color(240, 240, 240));
    }

    // 1. Draw obstacles
    drawObstacles(target, obstacles);
    
    // 2. Draw trajectory
    drawTrajectorySegments(target, trajectory, trajectory_color);
    
    // 3. Draw start/goal points
    if (!trajectory.nodes.empty()) {
        const PathNode& start_node = trajectory.nodes[trajectory.start_index];
        const PathNode& goal_node = trajectory.nodes[trajectory.goal_index];
        
        drawNode(target, start_node, 6.0f, sf::Color::Green);
        drawNode(target, goal_node, 6.0f, sf::Color::Red);
    }
}


/**
 * @brief Overload that takes ObstacleMap directly
 */
template<typename RenderTarget>
void visualizeTrajectory(
    RenderTarget& target,
    const ObstacleMap& obstacle_map,
    const Trajectory& trajectory,
    const sf::Color& trajectory_color = sf::Color(0, 0, 255, 255),
    bool clear_background = true)
{
    visualizeTrajectory(target, obstacle_map.getObstacles(), trajectory, 
                       trajectory_color, clear_background);
}


/**
 * @brief Visualizes multiple trajectories (e.g., optimization history)
 */
template<typename RenderTarget>
void visualizeTrajectoryHistory(
    RenderTarget& target,
    const std::vector<ObstacleND>& obstacles,
    const std::vector<Trajectory>& history,
    bool fade_older = true,
    bool clear_background = true)
{
    if (clear_background) {
        target.clear(sf::Color(240, 240, 240));
    }

    // 1. Draw obstacles
    drawObstacles(target, obstacles);
    
    if (history.empty()) {
        return;
    }

    // 2. Draw historical trajectories with fading
    size_t num_iterations = history.size();
    
    for (size_t i = 0; i < num_iterations - 1; ++i) {
        if (fade_older) {
            // Calculate alpha based on iteration index (older = more faded)
            float alpha_float = 5.0f + (145.0f * (float)i / std::max(1.0f, (float)num_iterations - 2));
            unsigned char alpha = static_cast<unsigned char>(alpha_float);
            sf::Color history_color(50, 50, 255, alpha);
            drawTrajectorySegments(target, history[i], history_color);
        } else {
            // All history trajectories same color
            drawTrajectorySegments(target, history[i], sf::Color(50, 50, 255, 100));
        }
    }

    // 3. Draw final trajectory (brightest)
    const Trajectory& final_traj = history.back();
    drawTrajectorySegments(target, final_traj, sf::Color(255, 0, 0, 255));

    // 4. Draw start/goal points
    if (!final_traj.nodes.empty()) {
        const PathNode& start_node = final_traj.nodes[final_traj.start_index];
        const PathNode& goal_node = final_traj.nodes[final_traj.goal_index];
        
        drawNode(target, start_node, 6.0f, sf::Color::Green);
        drawNode(target, goal_node, 6.0f, sf::Color::Red);
    }
}


/**
 * @brief Overload that takes ObstacleMap directly
 */
template<typename RenderTarget>
void visualizeTrajectoryHistory(
    RenderTarget& target,
    const ObstacleMap& obstacle_map,
    const std::vector<Trajectory>& history,
    bool fade_older = true,
    bool clear_background = true)
{
    visualizeTrajectoryHistory(target, obstacle_map.getObstacles(), history, 
                              fade_older, clear_background);
}


/**
 * @brief Calculates the signed distance function (SDF) for 2D obstacles
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


/**
 * @brief Visualizes trajectory with collision checking spheres
 * Shows which nodes are in collision (red), near collision (orange), or safe (green)
 */
template<typename RenderTarget>
void visualizeTrajectoryWithCollisionSpheres(
    RenderTarget& target,
    const std::vector<ObstacleND>& obstacles,
    const Trajectory& trajectory,
    float collision_threshold = 10.0f,
    bool clear_background = true)
{
    if (clear_background) {
        target.clear(sf::Color(240, 240, 240));
    }

    // 1. Draw obstacles
    drawObstacles(target, obstacles);
    
    // 2. Draw trajectory segments
    drawTrajectorySegments(target, trajectory, sf::Color(50, 50, 255, 150));

    // 3. Draw collision spheres for each node
    for (size_t i = 0; i < trajectory.nodes.size(); ++i) {
        const auto& node = trajectory.nodes[i];
        
        float node_x = node.position(0);
        float node_y = node.position.size() > 1 ? node.position(1) : 0.0f;

        // Calculate SDF at this node
        float sdf_value = calculateSDF2D(node_x, node_y, obstacles);
        float effective_sdf = sdf_value - node.radius;

        // Color based on collision status
        unsigned char alpha_fill;
        sf::Color node_color;
        
        if (effective_sdf < 0.0f) {
            // In collision
            node_color = sf::Color(255, 50, 50);
            alpha_fill = 255;
        } else if (effective_sdf < collision_threshold) {
            // Near collision
            node_color = sf::Color(255, 165, 0);
            alpha_fill = 150;
        } else {
            // Safe
            node_color = sf::Color(50, 200, 50);
            alpha_fill = 80;
        }
        
        sf::Color final_fill_color = node_color;
        final_fill_color.a = alpha_fill;

        // Draw collision sphere
        sf::CircleShape circle(node.radius);
        circle.setFillColor(final_fill_color);
        circle.setOutlineThickness(1.0f);
        circle.setOutlineColor(sf::Color(0, 0, 0, 50));
        circle.setOrigin(sf::Vector2f(node.radius, node.radius));
        circle.setPosition({node_x, node_y});
        target.draw(circle);
        
        // Draw node center
        if (i == trajectory.start_index || i == trajectory.goal_index) {
            drawNode(target, node, 5.0f, 
                    (i == trajectory.start_index) ? sf::Color::Green : sf::Color::Red);
        } else {
            drawNode(target, node, 1.5f, sf::Color(0, 0, 0, 200));
        }
    }
}


/**
 * @brief Overload that takes ObstacleMap directly
 */
template<typename RenderTarget>
void visualizeTrajectoryWithCollisionSpheres(
    RenderTarget& target,
    const ObstacleMap& obstacle_map,
    const Trajectory& trajectory,
    float collision_threshold = 10.0f,
    bool clear_background = true)
{
    visualizeTrajectoryWithCollisionSpheres(target, obstacle_map.getObstacles(), 
                                           trajectory, collision_threshold, clear_background);
}


/**
 * @brief Visualizes just the obstacles (useful for environment visualization)
 */
template<typename RenderTarget>
void visualizeObstacles(
    RenderTarget& target,
    const std::vector<ObstacleND>& obstacles,
    bool clear_background = true)
{
    if (clear_background) {
        target.clear(sf::Color(240, 240, 240));
    }
    
    drawObstacles(target, obstacles);
}


/**
 * @brief Overload that takes ObstacleMap directly
 */
template<typename RenderTarget>
void visualizeObstacles(
    RenderTarget& target,
    const ObstacleMap& obstacle_map,
    bool clear_background = true)
{
    visualizeObstacles(target, obstacle_map.getObstacles(), clear_background);
}


// --- File Save Utilities ---

/**
 * @brief Saves a visualization to a PNG file
 * @param width Image width
 * @param height Image height
 * @param filename Output filename (e.g., "trajectory.png")
 * @param draw_func Lambda that draws to the render texture
 * @return true if successful
 */
inline bool saveVisualizationToFile(
    unsigned int width,
    unsigned int height,
    const std::string& filename,
    std::function<void(sf::RenderTexture&)> draw_func)
{
    // Determine output path
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

    // Create render texture
    sf::RenderTexture renderTexture;
    if (!renderTexture.resize({width, height})) {
        std::cerr << "Error: Could not create render texture!" << std::endl;
        return false;
    }

    renderTexture.clear(sf::Color(240, 240, 240));
    
    // Draw using provided function
    draw_func(renderTexture);
    
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
 * @brief Convenience function to save trajectory visualization
 */
inline bool saveTrajectoryToFile(
    const ObstacleMap& obstacle_map,
    const Trajectory& trajectory,
    const std::string& filename,
    unsigned int width = 800,
    unsigned int height = 600)
{
    return saveVisualizationToFile(width, height, filename,
        [&](sf::RenderTexture& texture) {
            visualizeTrajectory(texture, obstacle_map, trajectory);
        }
    );
}


/**
 * @brief Convenience function to save trajectory history visualization
 */
inline bool saveTrajectoryHistoryToFile(
    const ObstacleMap& obstacle_map,
    const std::vector<Trajectory>& history,
    const std::string& filename,
    unsigned int width = 800,
    unsigned int height = 600)
{
    return saveVisualizationToFile(width, height, filename,
        [&](sf::RenderTexture& texture) {
            visualizeTrajectoryHistory(texture, obstacle_map, history);
        }
    );
}


/**
 * @brief Convenience function to save trajectory with collision spheres
 */
inline bool saveTrajectoryWithCollisionSpheresToFile(
    const ObstacleMap& obstacle_map,
    const Trajectory& trajectory,
    const std::string& filename,
    unsigned int width = 800,
    unsigned int height = 600)
{
    return saveVisualizationToFile(width, height, filename,
        [&](sf::RenderTexture& texture) {
            visualizeTrajectoryWithCollisionSpheres(texture, obstacle_map, trajectory);
        }
    );
}


// --- Interactive Window Display ---

/**
 * @brief Opens an interactive window to display a trajectory
 */
inline void showTrajectoryWindow(
    const ObstacleMap& obstacle_map,
    const Trajectory& trajectory,
    const std::string& title = "Trajectory Visualization",
    unsigned int width = 800,
    unsigned int height = 600)
{
    sf::RenderWindow window(
        sf::VideoMode({width, height}), 
        title, 
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    // Initial draw
    visualizeTrajectory(window, obstacle_map, trajectory);
    window.display();

    // Event loop
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
    }
}


/**
 * @brief Opens an interactive window to display trajectory history
 */
inline void showTrajectoryHistoryWindow(
    const ObstacleMap& obstacle_map,
    const std::vector<Trajectory>& history,
    const std::string& title = "Optimization History",
    unsigned int width = 800,
    unsigned int height = 600)
{
    sf::RenderWindow window(
        sf::VideoMode({width, height}), 
        title, 
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    // Initial draw
    visualizeTrajectoryHistory(window, obstacle_map, history);
    window.display();

    // Event loop
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
    }
}


/**
 * @brief Opens an interactive window to display trajectory with collision spheres
 */
inline void showTrajectoryWithCollisionSpheresWindow(
    const ObstacleMap& obstacle_map,
    const Trajectory& trajectory,
    const std::string& title = "Collision Visualization",
    unsigned int width = 800,
    unsigned int height = 600)
{
    sf::RenderWindow window(
        sf::VideoMode({width, height}), 
        title, 
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    // Initial draw
    visualizeTrajectoryWithCollisionSpheres(window, obstacle_map, trajectory);
    window.display();

    // Event loop
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
    }
}


// --- Backward Compatibility (for MotionPlanner) ---

// Forward declaration
class MotionPlanner;

/**
 * @brief Legacy function for backward compatibility with MotionPlanner
 * @deprecated Use visualizeTrajectoryHistory with ObstacleMap instead
 */
template<typename RenderTarget>
inline void visualizeOptimizationHistoryToTarget(
    const std::vector<ObstacleND>& target, 
    const std::vector<ObstacleND>& obstacles,
    const std::vector<Trajectory>& history)
{
    visualizeTrajectoryHistory(target, obstacles, history, true, true);
}
