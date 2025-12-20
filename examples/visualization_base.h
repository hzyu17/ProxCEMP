#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>

#include "Trajectory.h"
#include "ObstacleMap.h"
#include "collision_utils.h"

// --- Basic Drawing Functions (templated for RenderWindow or RenderTexture) ---

template<typename RenderTarget>
inline void drawNode(RenderTarget& target, const TrajectoryNode& node, float radius, const sf::Color& color) {
    sf::CircleShape circle(radius);
    circle.setFillColor(color);
    circle.setOrigin(sf::Vector2f(radius, radius));
    circle.setPosition({node.position(0), node.position(1)}); 
    target.draw(circle);
}

template<typename RenderTarget>
inline void drawNodeCollisionRadius(RenderTarget& target, const TrajectoryNode& node, bool in_collision) {
    sf::Color status_color = in_collision ? sf::Color::Red : sf::Color::Green;
    
    sf::CircleShape circle(node.radius);
    circle.setFillColor(sf::Color(status_color.r, status_color.g, status_color.b, 30));
    circle.setOutlineColor(sf::Color(status_color.r, status_color.g, status_color.b, 150));
    circle.setOutlineThickness(1.5f);
    circle.setOrigin(sf::Vector2f(node.radius, node.radius));
    circle.setPosition({node.position(0), node.position(1)}); 
    target.draw(circle);
}

template<typename RenderTarget>
inline void drawTrajectorySegments(RenderTarget& target, const Trajectory& trajectory, 
                                   const sf::Color& color = sf::Color(50, 50, 200, 200)) {
    if (trajectory.nodes.size() < 2) return;
    
    sf::VertexArray lines(sf::PrimitiveType::LineStrip, trajectory.nodes.size());
    for (size_t i = 0; i < trajectory.nodes.size(); ++i) {
        lines[i].position = sf::Vector2f(trajectory.nodes[i].position(0), trajectory.nodes[i].position(1));
        lines[i].color = color; 
    }
    target.draw(lines);
}

template<typename RenderTarget>
inline void drawObstacles(RenderTarget& target, const std::vector<ObstacleND>& obstacles) {
    for (const auto& obs : obstacles) {
        float obs_x = obs.center(0);
        float obs_y = obs.center.size() > 1 ? obs.center(1) : 0.0f;
        
        sf::CircleShape circle(obs.radius);
        circle.setFillColor(sf::Color(100, 100, 100, 180));
        circle.setOrigin(sf::Vector2f(obs.radius, obs.radius));
        circle.setPosition({obs_x, obs_y}); 
        target.draw(circle);
        
        sf::CircleShape center_dot(2.0f);
        center_dot.setFillColor(sf::Color(50, 50, 50, 255));
        center_dot.setOrigin(sf::Vector2f(2.0f, 2.0f));
        center_dot.setPosition({obs_x, obs_y});
        target.draw(center_dot);
    }
}

template<typename RenderTarget>
inline void drawObstacleMap(RenderTarget& target, const ObstacleMap& obstacle_map) {
    drawObstacles(target, obstacle_map.getObstacles());
}

template<typename RenderTarget>
inline void drawTrajectoryWithCollisionRadius(RenderTarget& target, 
                                              const Trajectory& trajectory,
                                              const std::vector<ObstacleND>& obstacles,
                                              bool show_collision_radius = true,
                                              const sf::Color& line_color = sf::Color(50, 50, 200, 200)) {
    if (show_collision_radius) {
        for (const auto& node : trajectory.nodes) {
            bool in_collision = isNodeInCollision(node, obstacles);
            drawNodeCollisionRadius(target, node, in_collision);
        }
    }
    
    drawTrajectorySegments(target, trajectory, line_color);
    
    for (const auto& node : trajectory.nodes) {
        drawNode(target, node, 2.0f, sf::Color(100, 100, 255, 180));
    }
    
    if (!trajectory.nodes.empty()) {
        drawNode(target, trajectory.nodes[trajectory.start_index], 6.0f, sf::Color::Green);
        drawNode(target, trajectory.nodes[trajectory.goal_index], 6.0f, sf::Color::Red);
    }
}

// --- Special function for initial state visualization (only works with RenderWindow) ---

inline void visualizeInitialState(const std::vector<ObstacleND>& obstacles,
                                  const Trajectory& trajectory,
                                  const std::string& window_title = "Initial Trajectory - Collision Check") {
    
    sf::RenderWindow window(sf::VideoMode({static_cast<unsigned int>(MAP_WIDTH), 
                                          static_cast<unsigned int>(MAP_HEIGHT)}), 
                           window_title, 
                           sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);
    
    std::cout << "\n=== Initial State Visualization ===\n";
    std::cout << "Trajectory nodes: " << trajectory.nodes.size() << "\n";
    std::cout << "\nVisualization Controls:\n";
    std::cout << "  SPACE/ENTER: Continue to optimization\n";
    std::cout << "  C:           Toggle collision radius display\n";
    std::cout << "  ESC:         Exit program\n";
    std::cout << "\nColors:\n";
    std::cout << "  GREEN radius = No collision\n";
    std::cout << "  RED radius   = In collision\n";
    std::cout << "  Gray circles = Obstacles\n";
    std::cout << "====================================\n\n";

    bool show_collision_radius = true;
    bool should_continue = false;

    while (window.isOpen() && !should_continue) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                std::cout << "Window closed. Exiting program.\n";
                std::exit(0);
            }
            
            if (event->is<sf::Event::KeyPressed>()) {
                const auto& key_event = event->getIf<sf::Event::KeyPressed>();
                
                if (key_event->code == sf::Keyboard::Key::Escape) {
                    std::cout << "ESC pressed. Exiting program.\n";
                    std::exit(0);
                }
                
                if (key_event->code == sf::Keyboard::Key::Space || 
                    key_event->code == sf::Keyboard::Key::Enter) {
                    std::cout << "Continuing to optimization...\n";
                    should_continue = true;
                }
                
                if (key_event->code == sf::Keyboard::Key::C) {
                    show_collision_radius = !show_collision_radius;
                    std::cout << "Collision radius display: " 
                              << (show_collision_radius ? "ON" : "OFF") << "\n";
                }
            }
        }
        
        window.clear(sf::Color(240, 240, 240));
        drawObstacles(window, obstacles);
        drawTrajectoryWithCollisionRadius(window, trajectory, obstacles, show_collision_radius);
        window.display();
    }
    
    window.close();
}
