/**
 * @file CollisionAvoidanceTask.h
 * @brief Concrete PCE task for collision avoidance with obstacles
 * 
 * Optimized with:
 * - Structure of Arrays (SoA) layout for cache efficiency
 * - SIMD-friendly vectorized operations via Eigen
 * - OpenMP parallelization for batch evaluation
 */
#pragma once

#include "task.h"
#include "ObstacleMap.h"
#include "Trajectory.h"
#include "collision_utils.h"
#include "ForwardKinematics.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pce {

struct CollisionAvoidanceConfig {
    float map_width = 800.0f;
    float map_height = 600.0f;
    size_t num_obstacles = 20;
    float obstacle_radius = 20.0f;
    float clearance_distance = 100.0f;
    size_t num_dimensions = 2;
    float node_collision_radius = 15.0f;
    unsigned int random_seed = 999;
    float epsilon_sdf = 0.1f;
    float sigma_obs = 0.05f;
    float min_spacing_factor = 2.0f;
    
    // Parallelization settings
    int num_threads = -1;  // -1 = use all available
    size_t batch_chunk_size = 32;  // OpenMP chunk size for dynamic scheduling

    static CollisionAvoidanceConfig fromYAML(const YAML::Node& config) {
        CollisionAvoidanceConfig cfg;
        
        if (const auto& exp = config["experiment"]) {
            cfg.random_seed = exp["random_seed"].as<unsigned int>(cfg.random_seed);
        }
        
        if (const auto& env = config["environment"]) {
            cfg.map_width = env["map_width"].as<float>(cfg.map_width);
            cfg.map_height = env["map_height"].as<float>(cfg.map_height);
            cfg.num_obstacles = env["num_obstacles"].as<size_t>(cfg.num_obstacles);
            cfg.obstacle_radius = env["obstacle_radius"].as<float>(cfg.obstacle_radius);
            cfg.clearance_distance = env["clearance_distance"].as<float>(cfg.clearance_distance);
            cfg.min_spacing_factor = env["min_spacing_factor"].as<float>(cfg.min_spacing_factor);

            if (const auto& cost = env["cost"]) {
                cfg.epsilon_sdf = cost["epsilon_sdf"].as<float>(cfg.epsilon_sdf);
                cfg.sigma_obs = cost["sigma_obs"].as<float>(cfg.sigma_obs);
            }
        }
        
        if (const auto& mp = config["motion_planning"]) {
            cfg.num_dimensions = mp["num_dimensions"].as<size_t>(cfg.num_dimensions);
            cfg.node_collision_radius = mp["node_collision_radius"].as<float>(cfg.node_collision_radius);
        }
        
        if (const auto& perf = config["performance"]) {
            cfg.num_threads = perf["num_threads"].as<int>(cfg.num_threads);
            cfg.batch_chunk_size = perf["batch_chunk_size"].as<size_t>(cfg.batch_chunk_size);
        }
        
        return cfg;
    }
    
    void validate() const {
        if (map_width <= 0 || map_height <= 0) 
            throw std::invalid_argument("Map dimensions must be positive");
        if (num_dimensions < 2 || num_dimensions > 3) 
            throw std::invalid_argument("Only 2D/3D supported");
        if (epsilon_sdf <= 0 || sigma_obs <= 0) 
            throw std::invalid_argument("SDF params must be positive");
        if (obstacle_radius <= 0 || node_collision_radius <= 0) 
            throw std::invalid_argument("Radii must be positive");
        if (min_spacing_factor < 1.0f) 
            throw std::invalid_argument("Min spacing factor must be >= 1.0");
    }
};


/**
 * @brief Structure of Arrays layout for obstacle data
 * 
 * Optimized for SIMD operations:
 * - Contiguous memory access patterns
 * - Vectorized distance computations
 * - Cache-friendly data layout
 */
struct ObstacleDataSoA {
    Eigen::MatrixXf centers;        // D x K matrix (D dimensions, K obstacles)
    Eigen::VectorXf radii;          // K vector of obstacle radii
    Eigen::VectorXf combined_radii; // K vector of (obstacle_radius + node_radius)
    size_t num_obstacles = 0;
    size_t num_dimensions = 0;
    
    ObstacleDataSoA() = default;
    
    /**
     * @brief Build SoA from vector of obstacles
     */
    void buildFrom(const std::vector<ObstacleND>& obstacles, float node_collision_radius) {
        if (obstacles.empty()) {
            num_obstacles = 0;
            num_dimensions = 0;
            centers.resize(0, 0);
            radii.resize(0);
            combined_radii.resize(0);
            return;
        }
        
        num_obstacles = obstacles.size();
        num_dimensions = obstacles[0].dimensions();
        
        centers.resize(num_dimensions, num_obstacles);
        radii.resize(num_obstacles);
        combined_radii.resize(num_obstacles);
        
        for (size_t k = 0; k < num_obstacles; ++k) {
            centers.col(k) = obstacles[k].center;
            radii(k) = obstacles[k].radius;
            combined_radii(k) = obstacles[k].radius + node_collision_radius;
        }
    }
    
    void updateNodeRadius(float node_collision_radius) {
        combined_radii = radii.array() + node_collision_radius;
    }
    
    bool empty() const { return num_obstacles == 0; }
};


/**
 * @brief High-performance collision avoidance task with SIMD optimization
 */
class CollisionAvoidanceTask : public Task {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;

    explicit CollisionAvoidanceTask(const YAML::Node& config, 
                                    std::shared_ptr<ForwardKinematics> custom_fk = nullptr)
        : num_nodes_(0), total_time_(0.0f) 
    {
        std::cout << "\n=== Initializing CollisionAvoidanceTask (SIMD Optimized) ===\n";
        
        config_ = CollisionAvoidanceConfig::fromYAML(config);
        config_.validate();
        
        epsilon_sdf_ = config_.epsilon_sdf;
        sigma_obs_ = config_.sigma_obs;
        node_collision_radius_ = config_.node_collision_radius;
        batch_chunk_size_ = config_.batch_chunk_size;
        
        // Configure OpenMP threads
        #ifdef _OPENMP
        if (config_.num_threads > 0) {
            omp_set_num_threads(config_.num_threads);
        }
        num_threads_ = omp_get_max_threads();
        std::cout << "OpenMP enabled: " << num_threads_ << " threads\n";
        #else
        num_threads_ = 1;
        std::cout << "OpenMP not available: single-threaded mode\n";
        #endif
        
        // Setup Obstacle Map
        obstacle_map_ = std::make_shared<ObstacleMap>(config_.num_dimensions, config_.random_seed);
        obstacle_map_->setMapSize(config_.map_width, config_.map_height);
        obstacle_map_->generateRandom(config_.num_obstacles, config_.obstacle_radius, 
                                      config_.min_spacing_factor);
        
        // Clear regions near Start/Goal
        if (const auto& mp = config["motion_planning"]) {
            if (mp["start_position"] && mp["goal_position"]) {
                auto start_v = mp["start_position"].as<std::vector<float>>();
                auto goal_v = mp["goal_position"].as<std::vector<float>>();
                
                VectorXf start_vec = Eigen::Map<VectorXf>(start_v.data(), start_v.size());
                VectorXf goal_vec = Eigen::Map<VectorXf>(goal_v.data(), goal_v.size());
                
                obstacle_map_->clearRegion(start_vec, config_.clearance_distance);
                obstacle_map_->clearRegion(goal_vec, config_.clearance_distance);
            }
        }
        
        // Store obstacles and build SoA
        obstacles_ = obstacle_map_->getObstacles();
        rebuildObstacleSoA();
        
        // Initialize FK
        if (custom_fk) {
            fk_ = custom_fk;
        } else {
            fk_ = std::make_shared<IdentityFK>(config_.num_dimensions);
        }
        
        std::cout << "Task Ready: " << obstacles_.size() << " obstacles active (SoA optimized).\n";
    }

    const CollisionAvoidanceConfig& getConfig() const { return config_; }

    void initialize(size_t num_dimensions, const TrajectoryNode& start, const TrajectoryNode& goal,
                    size_t num_nodes, float total_time) override {
        num_dimensions_ = num_dimensions;
        start_node_ = start;
        goal_node_ = goal;
        num_nodes_ = num_nodes;
        total_time_ = total_time;
        obstacles_ = obstacle_map_->getObstacles();
        rebuildObstacleSoA();
    }

    // ==================== Single Trajectory Costs ====================
    
    float computeStateCost(const Trajectory& trajectory) const override {
        return computeStateCostVectorized(trajectory);
    }

    float computeStateCostSimple(const Trajectory& trajectory) const override {
        return computeStateCostVectorized(trajectory);
    }

    // ==================== Batch Trajectory Costs (Parallelized) ====================
    
    /**
     * @brief Batch evaluation with OpenMP parallelization
     */
    std::vector<float> computeStateCost(
        const std::vector<Trajectory>& trajectories) const override 
    {
        return computeBatchParallel(trajectories);
    }
    
    std::vector<float> computeStateCostSimple(
        const std::vector<Trajectory>& trajectories) const override 
    {
        return computeBatchParallel(trajectories);
    }

    // ==================== Accessors ====================
    
    const std::vector<ObstacleND>& getObstacles() const { return obstacles_; }
    std::shared_ptr<ObstacleMap> getObstacleMap() const { return obstacle_map_; }
    const ObstacleDataSoA& getObstacleSoA() const { return obs_soa_; }
    int getNumThreads() const { return num_threads_; }

    void setCollisionParameters(float epsilon_sdf, float sigma_obs) {
        epsilon_sdf_ = epsilon_sdf;
        sigma_obs_ = sigma_obs;
    }

    void setNodeCollisionRadius(float radius) {
        if (radius > 0) {
            node_collision_radius_ = radius;
            obs_soa_.updateNodeRadius(radius);
        }
    }
    
    void addObstacle(const ObstacleND& obstacle) {
        obstacles_.push_back(obstacle);
        rebuildObstacleSoA();
    }
    
    void clearObstacles() {
        obstacles_.clear();
        rebuildObstacleSoA();
    }

private:
    // ==================== Core Vectorized Computation ====================
    
    /**
     * @brief Parallel batch evaluation
     */
    std::vector<float> computeBatchParallel(
        const std::vector<Trajectory>& trajectories) const 
    {
        const size_t M = trajectories.size();
        std::vector<float> costs(M);
        
        if (M == 0) return costs;
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, batch_chunk_size_)
        #endif
        for (size_t m = 0; m < M; ++m) {
            costs[m] = computeStateCostVectorized(trajectories[m]);
        }
        
        return costs;
    }
    
    /**
     * @brief Compute collision cost for single position against all obstacles (SIMD)
     */
    float computePositionCostVectorized(const VectorXf& task_pos) const {
        if (obs_soa_.empty()) return 0.0f;
        
        // diff = pos (broadcasted) - centers  [D x K]
        MatrixXf diff = obs_soa_.centers.colwise() - task_pos;
        
        // distances = ||diff||_2 for each column  [K]
        VectorXf distances = diff.colwise().norm();
        
        // SDF = distance - combined_radius  [K]
        VectorXf sdfs = distances - obs_soa_.combined_radii;
        
        // Hinge loss: max(0, epsilon - sdf)  [K]
        VectorXf hinges = (epsilon_sdf_ - sdfs.array()).max(0.0f);
        
        // Total cost: sigma * sum(hinge^2)
        return sigma_obs_ * hinges.squaredNorm();
    }
    
    /**
     * @brief Compute collision cost for entire trajectory (vectorized)
     */
    float computeStateCostVectorized(const Trajectory& trajectory) const {
        if (trajectory.nodes.empty() || obs_soa_.empty()) return 0.0f;
        
        const size_t N = trajectory.nodes.size();
        float total_cost = 0.0f;
        
        // Skip boundary nodes (indices 0 and N-1)
        for (size_t i = 1; i < N - 1; ++i) {
            const VectorXf task_pos = fk_->compute(trajectory.nodes[i].position);
            total_cost += computePositionCostVectorized(task_pos);
        }
        
        return total_cost;
    }
    
    void rebuildObstacleSoA() {
        obs_soa_.buildFrom(obstacles_, node_collision_radius_);
    }

    // ==================== Member Variables ====================
    
    std::shared_ptr<ObstacleMap> obstacle_map_;
    std::vector<ObstacleND> obstacles_;
    ObstacleDataSoA obs_soa_;
    
    std::shared_ptr<ForwardKinematics> fk_;
    CollisionAvoidanceConfig config_;

    float epsilon_sdf_;
    float sigma_obs_;
    float node_collision_radius_;

    size_t num_dimensions_;
    size_t num_nodes_;
    float total_time_;
    TrajectoryNode start_node_;
    TrajectoryNode goal_node_;
    
    int num_threads_;
    size_t batch_chunk_size_;
};

} // namespace pce