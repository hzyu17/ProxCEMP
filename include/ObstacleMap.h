#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>  

using json = nlohmann::json;

// --- Constants ---
constexpr int MAP_WIDTH = 800;
constexpr int MAP_HEIGHT = 600;
constexpr int MAP_DEPTH = 800;
constexpr int NUM_OBSTACLES = 20;
constexpr float OBSTACLE_RADIUS = 20.0f;
constexpr float TRAJECTORY_RADIUS = 10.0f;

// --- Legacy 2D Obstacle (for backward compatibility) ---
struct Obstacle {
    float x;
    float y;
    float radius;
};

// --- N-Dimensional Obstacle ---
struct ObstacleND {
    Eigen::VectorXf center;  // N-dimensional center position
    float radius;            // Obstacle radius
    
    // Constructors
    ObstacleND() : center(Eigen::VectorXf::Zero(2)), radius(0.0f) {}
    
    ObstacleND(const Eigen::VectorXf& c, float r) : center(c), radius(r) {}
    
    // 2D convenience constructor
    ObstacleND(float x, float y, float r) : center(2), radius(r) {
        center << x, y;
    }
    
    // 3D convenience constructor
    ObstacleND(float x, float y, float z, float r) : center(3), radius(r) {
        center << x, y, z;
    }
    
    // Backward compatibility accessors
    float x() const { return center(0); }
    float y() const { return center.size() > 1 ? center(1) : 0.0f; }
    float z() const { return center.size() > 2 ? center(2) : 0.0f; }
    
    size_t dimensions() const { return center.size(); }
};

/**
 * @brief ObstacleMap - Unified obstacle management for N-dimensional spaces
 */
class ObstacleMap {
public:
    // Constructor with optional seed
    ObstacleMap(size_t dimensions = 2, unsigned int seed = std::random_device{}()) 
        : dimensions_(dimensions), 
          map_width_(MAP_WIDTH), 
          map_height_(MAP_HEIGHT),
          map_depth_(MAP_DEPTH),
          gen_(seed)  // ✓ Initialize with seed
    {
        std::cout << "ObstacleMap initialized with seed: " << seed << "\n";
    }

    /**
     * @brief Sets a new random seed
     */
    void setSeed(unsigned int seed) {
        gen_.seed(seed);
        std::cout << "ObstacleMap seed set to: " << seed << "\n";
    }
    
    // --- Obstacle Management ---
    
    /**
     * @brief Adds an obstacle to the map
     */
    void addObstacle(const ObstacleND& obstacle) {
        if (obstacle.dimensions() == dimensions_) {
            obstacles_.push_back(obstacle);
        } else {
            std::cerr << "Warning: Obstacle dimension mismatch. Expected " 
                      << dimensions_ << "D, got " << obstacle.dimensions() << "D\n";
        }
    }

    // --- JSON Save/Load Functions ---
    
    /**
     * @brief Saves the obstacle map to a JSON file
     * @param filename Path to the output JSON file
     * @return True if successful, false otherwise
     */
    bool saveToJSON(const std::string& filename) const {
        try {
            json j;
            
            // Store metadata
            j["metadata"] = {
                {"dimensions", dimensions_},
                {"map_width", map_width_},
                {"map_height", map_height_},
                {"map_depth", map_depth_},
                {"num_obstacles", obstacles_.size()}
            };
            
            // Store obstacles
            json obstacles_array = json::array();
            for (const auto& obs : obstacles_) {
                json obs_json;
                obs_json["radius"] = obs.radius;
                
                // Store center coordinates as array
                json center_array = json::array();
                for (size_t d = 0; d < obs.center.size(); ++d) {
                    center_array.push_back(obs.center(d));
                }
                obs_json["center"] = center_array;
                
                obstacles_array.push_back(obs_json);
            }
            j["obstacles"] = obstacles_array;
            
            // Write to file with pretty printing
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filename << " for writing\n";
                return false;
            }
            
            file << j.dump(2);  // 2-space indentation for readability
            file.close();
            
            std::cout << "Saved " << obstacles_.size() << " obstacles to " << filename << "\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error saving to JSON: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * @brief Loads obstacle map from a JSON file
     * @param filename Path to the input JSON file
     * @param clear_existing If true, clears existing obstacles before loading
     * @return True if successful, false otherwise
     */
    bool loadFromJSON(const std::string& filename, bool clear_existing = true) {
        try {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filename << " for reading\n";
                return false;
            }
            
            json j;
            file >> j;
            file.close();
            
            // Clear existing obstacles if requested
            if (clear_existing) {
                obstacles_.clear();
            }
            
            // Load metadata
            if (j.contains("metadata")) {
                dimensions_ = j["metadata"]["dimensions"].get<size_t>();
                map_width_ = j["metadata"]["map_width"].get<float>();
                map_height_ = j["metadata"]["map_height"].get<float>();
                
                if (j["metadata"].contains("map_depth")) {
                    map_depth_ = j["metadata"]["map_depth"].get<float>();
                }
                
            }
            
            // Load obstacles
            if (!j.contains("obstacles")) {
                std::cerr << "Error: JSON file does not contain 'obstacles' field\n";
                return false;
            }
            
            for (const auto& obs_json : j["obstacles"]) {
                float radius = obs_json["radius"].get<float>();
                
                // Load center coordinates
                const auto& center_array = obs_json["center"];
                Eigen::VectorXf center(center_array.size());
                for (size_t d = 0; d < center_array.size(); ++d) {
                    center(d) = center_array[d].get<float>();
                }
                
                // Verify dimension consistency
                if (center.size() != dimensions_) {
                    std::cerr << "Warning: Obstacle dimension mismatch. Expected " 
                              << dimensions_ << "D, got " << center.size() << "D\n";
                    continue;
                }
                
                obstacles_.emplace_back(center, radius);
            }
            
            std::cout << "Loaded " << obstacles_.size() << " obstacles from " 
                      << filename << "\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading from JSON: " << e.what() << "\n";
            return false;
        }
    }
    
    
    /**
     * @brief Adds a 2D obstacle (convenience method)
     */
    void addObstacle(float x, float y, float radius) {
        if (dimensions_ == 2) {
            obstacles_.emplace_back(x, y, radius);
        } else {
            std::cerr << "Warning: Cannot add 2D obstacle to " << dimensions_ << "D map\n";
        }
    }

    /**
     * @brief Adds a 3D obstacle (convenience method)
     */
    void addObstacle(float x, float y, float z, float radius) {
        if (dimensions_ == 3) {
            obstacles_.emplace_back(x, y, z, radius);
        } else {
            std::cerr << "Warning: Cannot add 3D obstacle to " << dimensions_ << "D map\n";
        }
    }
    
    /**
     * @brief Removes obstacles within a sphere of given radius around a point
     */
    void clearRegion(const Eigen::VectorXf& center, float clearance_radius) {
        if (center.size() != dimensions_) {
            std::cerr << "Warning: Point dimension mismatch in clearRegion\n";
            return;
        }
        
        size_t initial_count = obstacles_.size();
        
        obstacles_.erase(
            std::remove_if(obstacles_.begin(), obstacles_.end(),
                [&](const ObstacleND& obs) {
                    float dist = (obs.center - center).norm();
                    return dist <= clearance_radius;
                }),
            obstacles_.end()
        );
        
        size_t removed = initial_count - obstacles_.size();
        if (removed > 0) {
            std::cout << "Cleared " << removed << " obstacle(s) from region\n";
        }
    }
    
    /**
     * @brief Clears obstacles near start and goal positions
     */
    void clearStartGoalRegions(const Eigen::VectorXf& start, 
                               const Eigen::VectorXf& goal, 
                               float clearance_radius) {
        size_t initial_count = obstacles_.size();
        
        obstacles_.erase(
            std::remove_if(obstacles_.begin(), obstacles_.end(),
                [&](const ObstacleND& obs) {
                    float dist_start = (obs.center - start).norm();
                    float dist_goal = (obs.center - goal).norm();
                    return (dist_start <= clearance_radius) || 
                           (dist_goal <= clearance_radius);
                }),
            obstacles_.end()
        );
        
        size_t removed = initial_count - obstacles_.size();
        if (removed > 0) {
            std::cout << "Cleared " << removed 
                      << " obstacle(s) near start/goal\n";
        }
    }
    
    /**
     * @brief Removes all obstacles
     */
    void clear() {
        obstacles_.clear();
    }
    
    /**
     * @brief Generates random obstacles with minimum spacing (works for 2D, 3D, or N-D)
     */
    void generateRandom(size_t count, float radius, float min_spacing_factor = 2.0f) {
        Eigen::VectorXf min_bounds(dimensions_);
        Eigen::VectorXf max_bounds(dimensions_);
        
        // Set bounds based on dimensions
        min_bounds(0) = OBSTACLE_RADIUS;
        max_bounds(0) = map_width_ - OBSTACLE_RADIUS;
        
        if (dimensions_ >= 2) {
            min_bounds(1) = OBSTACLE_RADIUS;
            max_bounds(1) = map_height_ - OBSTACLE_RADIUS;
        }
        
        if (dimensions_ >= 3) {
            min_bounds(2) = OBSTACLE_RADIUS;
            max_bounds(2) = map_depth_ - OBSTACLE_RADIUS;
        }
        
        // For dimensions > 3, bounds would need to be set appropriately
        for (size_t d = 3; d < dimensions_; ++d) {
            min_bounds(d) = OBSTACLE_RADIUS;
            max_bounds(d) = 100.0f - OBSTACLE_RADIUS; // Default value, adjust as needed
        }
        
        generateRandomND(count, radius, min_bounds, max_bounds, min_spacing_factor);
    }

    /**
     * @brief Generates random N-dimensional obstacles with minimum spacing
     */
    void generateRandomND(size_t count, float radius,
                        const Eigen::VectorXf& min_bounds,
                        const Eigen::VectorXf& max_bounds,
                        float min_spacing_factor = 2.0f) {
        if (min_bounds.size() != dimensions_ || max_bounds.size() != dimensions_) {
            std::cerr << "Error: Bounds dimension mismatch\n";
            return;
        }
        
        const float min_spacing = radius * min_spacing_factor;
        const float min_spacing_sq = min_spacing * min_spacing;
        
        // Create distributions for each dimension
        std::vector<std::uniform_real_distribution<float>> distributions;
        for (size_t d = 0; d < dimensions_; ++d) {
            distributions.emplace_back(
                min_bounds(d) + radius,
                max_bounds(d) - radius
            );
        }
        
        int attempts = 0;
        const int MAX_ATTEMPTS = static_cast<int>(count) * 200;
        
        while (obstacles_.size() < count && attempts < MAX_ATTEMPTS) {
            Eigen::VectorXf new_center(dimensions_);
            for (size_t d = 0; d < dimensions_; ++d) {
                new_center(d) = distributions[d](gen_);
            }
            
            bool overlap = false;
            for (const auto& obs : obstacles_) {
                if ((new_center - obs.center).squaredNorm() < min_spacing_sq) {
                    overlap = true;
                    break;
                }
            }
            
            if (!overlap) {
                obstacles_.emplace_back(new_center, radius);
            }
            attempts++;
        }
        
        if (obstacles_.size() < count) {
            std::cout << "Warning: Could only place " << obstacles_.size()
                    << " out of " << count << " obstacles\n";
        } else {
            std::cout << "Generated " << obstacles_.size() << " " 
                    << dimensions_ << "D obstacles\n";
        }
    }
    
    // --- Distance Queries ---
    
    /**
     * @brief Computes signed distance to nearest obstacle
     * @param query_point N-dimensional query point
     * @return Signed distance (positive = free space, negative = inside obstacle)
     */
    float computeSDF(const Eigen::VectorXf& query_point) const {
        if (query_point.size() != dimensions_) {
            std::cerr << "Warning: Query point dimension mismatch\n";
            return std::numeric_limits<float>::max();
        }
        
        if (obstacles_.empty()) {
            return std::numeric_limits<float>::max();
        }
        
        float min_dist = std::numeric_limits<float>::max();
        float nearest_radius = 0.0f;
        
        for (const auto& obs : obstacles_) {
            float dist_to_center = (query_point - obs.center).norm();
            float signed_dist = dist_to_center - obs.radius;
            
            if (std::abs(signed_dist) < std::abs(min_dist)) {
                min_dist = signed_dist;
                nearest_radius = obs.radius;
            }
        }
        
        return min_dist;
    }
    
    /**
     * @brief Checks if a point is in collision with any obstacle
     */
    bool isInCollision(const Eigen::VectorXf& query_point, 
                      float safety_margin = 0.0f) const {
        return computeSDF(query_point) < safety_margin;
    }
    
    // --- Getters ---
    
    const std::vector<ObstacleND>& getObstacles() const {
        return obstacles_;
    }
    
    std::vector<ObstacleND>& getObstacles() {
        return obstacles_;
    }
    
    size_t size() const {
        return obstacles_.size();
    }
    
    bool empty() const {
        return obstacles_.empty();
    }
    
    size_t getDimensions() const {
        return dimensions_;
    }
    
    float getMapWidth() const { return map_width_; }
    float getMapHeight() const { return map_height_; }
    
    void setMapSize(float width, float height) {
        map_width_ = width;
        map_height_ = height;
    }

    void setMapSize(float width, float height, float depth) {
        map_width_ = width;
        map_height_ = height;
        map_depth_ = depth;
    }
    
    // --- Legacy Support ---
    
    /**
     * @brief Converts to legacy 2D obstacle format
     */
    std::vector<Obstacle> toLegacy2D() const {
        std::vector<Obstacle> legacy;
        legacy.reserve(obstacles_.size());
        
        for (const auto& obs : obstacles_) {
            if (obs.dimensions() >= 2) {
                legacy.push_back({obs.x(), obs.y(), obs.radius});
            }
        }
        
        return legacy;
    }
    
    /**
     * @brief Loads from legacy 2D obstacle format
     */
    void loadFromLegacy2D(const std::vector<Obstacle>& legacy_obstacles) {
        clear();
        dimensions_ = 2;
        
        for (const auto& obs : legacy_obstacles) {
            obstacles_.emplace_back(obs.x, obs.y, obs.radius);
        }
    }

private:
    std::vector<ObstacleND> obstacles_;
    size_t dimensions_;
    float map_width_;
    float map_height_;
    float map_depth_;
    std::mt19937 gen_;
};

// --- Global Helper Functions (for backward compatibility) ---

inline float distanceSquared(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

inline float calculateSDF(float qx, float qy, const std::vector<Obstacle>& obstacles) {
    float min_dist_sq = std::numeric_limits<float>::max();
    
    for (const auto& obs : obstacles) {
        float dist_sq = distanceSquared(qx, qy, obs.x, obs.y);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }
    
    float distance_to_center = std::sqrt(min_dist_sq);
    return distance_to_center - OBSTACLE_RADIUS;
}

inline std::vector<Obstacle> generateObstacles(int count, float radius, 
                                              int width, int height) {
    ObstacleMap map(2);
    map.setMapSize(static_cast<float>(width), static_cast<float>(height));
    map.generateRandom(count, radius);
    return map.toLegacy2D();
}
