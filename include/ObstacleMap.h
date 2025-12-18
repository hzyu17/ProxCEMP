#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>
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

// --- Legacy 2D Obstacle ---
struct Obstacle {
    float x;
    float y;
    float radius;
};

// --- N-Dimensional Obstacle ---
struct ObstacleND {
    Eigen::VectorXf center;  
    float radius;            
    
    ObstacleND() : center(Eigen::VectorXf::Zero(2)), radius(0.0f) {}
    ObstacleND(const Eigen::VectorXf& c, float r) : center(c), radius(r) {}
    
    ObstacleND(float x, float y, float r) : center(2), radius(r) {
        center << x, y;
    }
    
    ObstacleND(float x, float y, float z, float r) : center(3), radius(r) {
        center << x, y, z;
    }
    
    float x() const { return center.size() > 0 ? center(0) : 0.0f; }
    float y() const { return center.size() > 1 ? center(1) : 0.0f; }
    float z() const { return center.size() > 2 ? center(2) : 0.0f; }
    size_t dimensions() const { return center.size(); }
};

class ObstacleMap {
public:
    explicit ObstacleMap(size_t dimensions = 2, unsigned int seed = std::random_device{}()) 
        : dimensions_(dimensions), map_width_(MAP_WIDTH), 
          map_height_(MAP_HEIGHT), map_depth_(MAP_DEPTH), gen_(seed) 
    {
        std::cout << "ObstacleMap initialized (" << dimensions_ << "D) with seed: " << seed << "\n";
    }

    void setSeed(unsigned int seed) {
        gen_.seed(seed);
    }
    
    void addObstacle(const ObstacleND& obstacle) {
        if (obstacle.dimensions() == dimensions_) {
            obstacles_.push_back(obstacle);
        } else {
            std::cerr << "Warning: Dimension mismatch. Expected " << dimensions_ << "D\n";
        }
    }

    // --- JSON Logic ---
    bool saveToJSON(const std::string& filename) const {
        try {
            json j;
            j["metadata"] = {
                {"dimensions", dimensions_}, {"map_width", map_width_},
                {"map_height", map_height_}, {"map_depth", map_depth_},
                {"num_obstacles", obstacles_.size()}
            };
            
            auto& obs_array = j["obstacles"] = json::array();
            for (const auto& obs : obstacles_) {
                std::vector<float> center_vec(obs.center.data(), obs.center.data() + obs.center.size());
                obs_array.push_back({{"radius", obs.radius}, {"center", center_vec}});
            }
            
            std::ofstream file(filename);
            if (!file) return false;
            file << j.dump(2);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Save error: " << e.what() << "\n";
            return false;
        }
    }
    
    bool loadFromJSON(const std::string& filename, bool clear_existing = true) {
        try {
            std::ifstream file(filename);
            if (!file) return false;
            json j; file >> j;
            
            if (clear_existing) obstacles_.clear();
            
            if (j.contains("metadata")) {
                dimensions_ = j["metadata"]["dimensions"];
                map_width_ = j["metadata"]["map_width"];
                map_height_ = j["metadata"]["map_height"];
                if (j["metadata"].contains("map_depth")) map_depth_ = j["metadata"]["map_depth"];
            }
            
            for (const auto& obs_j : j["obstacles"]) {
                std::vector<float> c_vec = obs_j["center"].get<std::vector<float>>();
                if (c_vec.size() != dimensions_) continue;
                
                Eigen::VectorXf center = Eigen::Map<Eigen::VectorXf>(c_vec.data(), c_vec.size());
                obstacles_.emplace_back(center, obs_j["radius"].get<float>());
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Load error: " << e.what() << "\n";
            return false;
        }
    }
    
    void addObstacle(float x, float y, float radius) {
        if (dimensions_ == 2) obstacles_.emplace_back(x, y, radius);
    }

    void addObstacle(float x, float y, float z, float radius) {
        if (dimensions_ == 3) obstacles_.emplace_back(x, y, z, radius);
    }
    
    void clearRegion(const Eigen::VectorXf& center, float clearance_radius) {
        if (center.size() != dimensions_) return;
        auto it = std::remove_if(obstacles_.begin(), obstacles_.end(), [&](const ObstacleND& obs) {
            return (obs.center - center).norm() <= clearance_radius;
        });
        obstacles_.erase(it, obstacles_.end());
    }
    
    void clearStartGoalRegions(const Eigen::VectorXf& start, const Eigen::VectorXf& goal, float clearance_radius) {
        auto it = std::remove_if(obstacles_.begin(), obstacles_.end(), [&](const ObstacleND& obs) {
            return (obs.center - start).norm() <= clearance_radius || (obs.center - goal).norm() <= clearance_radius;
        });
        obstacles_.erase(it, obstacles_.end());
    }
    
    void clear() { obstacles_.clear(); }
    
    void generateRandom(size_t count, float radius, float min_spacing_factor = 2.0f) {
        Eigen::VectorXf min_b = Eigen::VectorXf::Constant(dimensions_, OBSTACLE_RADIUS);
        Eigen::VectorXf max_b = Eigen::VectorXf::Constant(dimensions_, 100.0f - OBSTACLE_RADIUS);
        
        if (dimensions_ >= 1) max_b(0) = map_width_ - OBSTACLE_RADIUS;
        if (dimensions_ >= 2) max_b(1) = map_height_ - OBSTACLE_RADIUS;
        if (dimensions_ >= 3) max_b(2) = map_depth_ - OBSTACLE_RADIUS;
        
        generateRandomND(count, radius, min_b, max_b, min_spacing_factor);
    }

    void generateRandomND(size_t count, float radius, const Eigen::VectorXf& min_bounds, 
                        const Eigen::VectorXf& max_bounds, float min_spacing_factor = 2.0f) {
        if (min_bounds.size() != dimensions_) return;
        
        const float min_spacing_sq = std::pow(radius * min_spacing_factor, 2);
        std::vector<std::uniform_real_distribution<float>> dists;
        for (size_t d = 0; d < dimensions_; ++d) {
            dists.emplace_back(min_bounds(d) + radius, max_bounds(d) - radius);
        }
        
        size_t attempts = 0;
        const size_t MAX_ATTEMPTS = count * 200;
        
        while (obstacles_.size() < count && attempts++ < MAX_ATTEMPTS) {
            Eigen::VectorXf new_center(dimensions_);
            for (size_t d = 0; d < dimensions_; ++d) new_center(d) = dists[d](gen_);
            
            bool overlap = std::any_of(obstacles_.begin(), obstacles_.end(), [&](const ObstacleND& obs) {
                return (new_center - obs.center).squaredNorm() < min_spacing_sq;
            });
            
            if (!overlap) obstacles_.emplace_back(new_center, radius);
        }
    }
    
    float computeSDF(const Eigen::VectorXf& query_point) const {
        if (query_point.size() != dimensions_ || obstacles_.empty()) return std::numeric_limits<float>::max();
        
        float min_sdf = std::numeric_limits<float>::max();
        for (const auto& obs : obstacles_) {
            float dist = (query_point - obs.center).norm() - obs.radius;
            if (std::abs(dist) < std::abs(min_sdf)) min_sdf = dist;
        }
        return min_sdf;
    }
    
    bool isInCollision(const Eigen::VectorXf& query, float safety = 0.0f) const {
        return computeSDF(query) < safety;
    }
    
    // Getters
    const std::vector<ObstacleND>& getObstacles() const { return obstacles_; }
    std::vector<ObstacleND>& getObstacles() { return obstacles_; }
    size_t size() const { return obstacles_.size(); }
    bool empty() const { return obstacles_.empty(); }
    size_t getDimensions() const { return dimensions_; }
    float getMapWidth() const { return map_width_; }
    float getMapHeight() const { return map_height_; }
    
    void setMapSize(float w, float h) { map_width_ = w; map_height_ = h; }
    void setMapSize(float w, float h, float d) { map_width_ = w; map_height_ = h; map_depth_ = d; }
    
    std::vector<Obstacle> toLegacy2D() const {
        std::vector<Obstacle> legacy;
        for (const auto& obs : obstacles_) {
            if (obs.dimensions() >= 2) legacy.push_back({obs.x(), obs.y(), obs.radius});
        }
        return legacy;
    }
    
    void loadFromLegacy2D(const std::vector<Obstacle>& legacy) {
        clear(); dimensions_ = 2;
        for (const auto& obs : legacy) obstacles_.emplace_back(obs.x, obs.y, obs.radius);
    }

private:
    std::vector<ObstacleND> obstacles_;
    size_t dimensions_;
    float map_width_, map_height_, map_depth_;
    std::mt19937 gen_;
};

// --- Global Helpers ---

inline float distanceSquared(float x1, float y1, float x2, float y2) {
    return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
}

inline float calculateSDF(float qx, float qy, const std::vector<Obstacle>& obstacles) {
    float min_dist_sq = std::numeric_limits<float>::max();
    for (const auto& obs : obstacles) {
        min_dist_sq = std::min(min_dist_sq, distanceSquared(qx, qy, obs.x, obs.y));
    }
    return std::sqrt(min_dist_sq) - OBSTACLE_RADIUS;
}

inline std::vector<Obstacle> generateObstacles(int count, float radius, int width, int height) {
    ObstacleMap map(2);
    map.setMapSize(static_cast<float>(width), static_cast<float>(height));
    map.generateRandom(count, radius);
    return map.toLegacy2D();
}