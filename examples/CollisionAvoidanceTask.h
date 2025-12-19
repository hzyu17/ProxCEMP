/**
 * @file CollisionAvoidanceTask.h
 * @brief Concrete PCE task for collision avoidance with obstacles
 **/
#pragma once

#include "task.h"
#include "ObstacleMap.h"
#include "Trajectory.h"
#include "collision_utils.h"
#include "ForwardKinematics.h"
#include <Eigen/Sparse>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <iostream>

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
        
        return cfg;
    }
    
    void validate() const {
        if (map_width <= 0 || map_height <= 0) throw std::invalid_argument("Map dimensions must be positive");
        if (num_dimensions < 2 || num_dimensions > 3) throw std::invalid_argument("Only 2D/3D supported");
        if (epsilon_sdf <= 0 || sigma_obs <= 0) throw std::invalid_argument("SDF params must be positive");
        if (obstacle_radius <= 0 || node_collision_radius <= 0) throw std::invalid_argument("Radii must be positive");
        if (min_spacing_factor < 1.0f) throw std::invalid_argument("Min spacing factor must be >= 1.0");
    }
};

class CollisionAvoidanceTask : public Task {
public:
    using SparseMatrixXf = Eigen::SparseMatrix<float>;

    explicit CollisionAvoidanceTask(const YAML::Node& config, 
                                    std::shared_ptr<ForwardKinematics> custom_fk = nullptr)
        : num_nodes_(0), total_time_(0.0f) 
    {
        std::cout << "\n=== Initializing CollisionAvoidanceTask ===\n";
        
        config_ = CollisionAvoidanceConfig::fromYAML(config);
        config_.validate();
        
        epsilon_sdf_ = config_.epsilon_sdf;
        sigma_obs_ = config_.sigma_obs;
        node_collision_radius_ = config_.node_collision_radius;
        
        // Setup Obstacle Map
        obstacle_map_ = std::make_shared<ObstacleMap>(config_.num_dimensions, config_.random_seed);
        obstacle_map_->setMapSize(config_.map_width, config_.map_height);
        obstacle_map_->generateRandom(config_.num_obstacles, config_.obstacle_radius, config_.min_spacing_factor);
        
        // Clear regions near Start/Goal
        if (const auto& mp = config["motion_planning"]) {
            if (mp["start_position"] && mp["goal_position"]) {
                auto start_v = mp["start_position"].as<std::vector<float>>();
                auto goal_v = mp["goal_position"].as<std::vector<float>>();
                
                Eigen::VectorXf start_vec = Eigen::Map<Eigen::VectorXf>(start_v.data(), start_v.size());
                Eigen::VectorXf goal_vec = Eigen::Map<Eigen::VectorXf>(goal_v.data(), goal_v.size());
                
                obstacle_map_->clearRegion(start_vec, config_.clearance_distance);
                obstacle_map_->clearRegion(goal_vec, config_.clearance_distance);
            }
        }
        
        // Initialize FK
        if (custom_fk) {
            fk_ = custom_fk;
        } else {
            fk_ = std::make_shared<IdentityFK>(config_.num_dimensions);
        }
        
        std::cout << "Task Ready: " << obstacle_map_->size() << " obstacles active.\n";
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
    }

    float computeStateCost(const Trajectory& trajectory) const override {
        if (trajectory.nodes.empty()) return 0.0f;

        float total_cost = 0.0f;
        const size_t N = trajectory.nodes.size();

        // Skip boundary nodes
        for (size_t i = 1; i < N - 1; ++i) {
            const Eigen::VectorXf task_pos = fk_->compute(trajectory.nodes[i].position);
            
            for (const auto& obs : obstacles_) {
                if (obs.dimensions() != static_cast<size_t>(task_pos.size())) continue;
                
                float dist = (task_pos - obs.center).norm();
                float sdf = dist - (obs.radius + node_collision_radius_);
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - sdf);
                
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
            }
        }
        return total_cost;
    }

    float computeStateCostSimple(const Trajectory& trajectory) const override {
        return computeStateCost(trajectory);
    }

    const std::vector<ObstacleND>& getObstacles() const { return obstacles_; }
    std::shared_ptr<ObstacleMap> getObstacleMap() const { return obstacle_map_; }

    void setCollisionParameters(float epsilon_sdf, float sigma_obs) {
        epsilon_sdf_ = epsilon_sdf;
        sigma_obs_ = sigma_obs;
    }

    void setNodeCollisionRadius(float radius) {
        if (radius > 0) node_collision_radius_ = radius;
    }

private:
    std::shared_ptr<ObstacleMap> obstacle_map_;
    std::vector<ObstacleND> obstacles_;
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
};

} // namespace pce