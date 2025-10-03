#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <limits> // Needed for std::numeric_limits


// --- Constants ---
constexpr int MAP_WIDTH = 800;
constexpr int MAP_HEIGHT = 600;
constexpr int NUM_OBSTACLES = 20;
constexpr float OBSTACLE_RADIUS = 20.0f;
constexpr float MIN_BORDER_DIST = OBSTACLE_RADIUS;
constexpr float TRAJECTORY_RADIUS = 10.0f; // New constant for collision checking radius

// --- Structures ---
struct Obstacle {
    float x;
    float y;
    float radius;
};

// --- Function Prototypes ---

/**
 * @brief Calculates the squared Euclidean distance between two points.
 */
float distanceSquared(float x1, float y1, float x2, float y2);

/**
 * @brief Generates a list of non-overlapping circular obstacles within the map boundaries.
 */
std::vector<Obstacle> generateObstacles(int count, float radius, int width, int height);

/**
 * @brief Calculates the Signed Distance Function (SDF) from a query point (qx, qy)
 * to the nearest obstacle.
 */
float calculateSDF(float qx, float qy, const std::vector<Obstacle>& obstacles);
