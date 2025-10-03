#include "../include/ObstacleMap.h"

// Global variable for easier constant referencing in this file
constexpr float MIN_SPACING_SQ = (OBSTACLE_RADIUS * 2.0f) * (OBSTACLE_RADIUS * 2.0f);

// --- Implementation ---

float distanceSquared(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

std::vector<Obstacle> generateObstacles(int count, float radius, int width, int height) {
    std::vector<Obstacle> obstacles;
    std::random_device rd;
    std::mt19937 gen(rd());
    // Define the range for obstacle centers, accounting for borders
    std::uniform_real_distribution<> distrib_x(radius + OBSTACLE_RADIUS, width - radius - OBSTACLE_RADIUS);
    std::uniform_real_distribution<> distrib_y(radius + OBSTACLE_RADIUS, height - radius - OBSTACLE_RADIUS);

    int attempts = 0;
    const int MAX_ATTEMPTS_PER_OBSTACLE = 200;

    while (obstacles.size() < count && attempts < count * MAX_ATTEMPTS_PER_OBSTACLE) {
        float new_x = static_cast<float>(distrib_x(gen));
        float new_y = static_cast<float>(distrib_y(gen));
        bool overlap = false;

        // Check against existing obstacles
        for (const auto& obs : obstacles) {
            if (distanceSquared(new_x, new_y, obs.x, obs.y) < MIN_SPACING_SQ) {
                overlap = true;
                break;
            }
        }

        if (!overlap) {
            obstacles.push_back({new_x, new_y, radius});
        }
        attempts++;
    }

    if (obstacles.size() < count) {
        // Log a warning if not all obstacles could be placed
        // std::cerr << "Warning: Could only place " << obstacles.size() << " out of " << count << " obstacles.\n";
    }

    return obstacles;
}

float calculateSDF(float qx, float qy, const std::vector<Obstacle>& obstacles) {
    float min_dist_sq = std::numeric_limits<float>::max();

    // 1. Find the distance to the center of the nearest obstacle
    for (const auto& obs : obstacles) {
        float dist_sq = distanceSquared(qx, qy, obs.x, obs.y);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }

    // 2. Compute the signed distance
    float distance_to_center = std::sqrt(min_dist_sq);
    
    // SDF = (Distance to center) - (Radius)
    return distance_to_center - OBSTACLE_RADIUS;
}

