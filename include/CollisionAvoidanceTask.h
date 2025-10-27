/**
 * @file CollisionAvoidanceTask.h
 * @brief Concrete PCE task for collision avoidance with obstacles
 * 
 * This task implementation handles:
 * - Obstacle management via ObstacleMap
 * - SDF-based collision cost computation
 * - Smoothness cost via R-matrix (acceleration minimization)
 * 
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

/**
 * @brief Configuration parameters for collision avoidance
 * Maps directly to the YAML configuration structure
 * (Does NOT include start/goal - those are managed by MotionPlanner)
 */
struct CollisionAvoidanceConfig {
    // Environment parameters (from environment section)
    float map_width = 800.0f;
    float map_height = 600.0f;
    size_t num_obstacles = 20;
    float obstacle_radius = 20.0f;
    float clearance_distance = 100.0f;
    
    // Motion planning parameters (from motion_planning section)
    size_t num_dimensions = 2;
    float node_collision_radius = 15.0f;  // Radius for collision checking
    
    // Experiment parameters
    unsigned int random_seed = 999;
    
    // Collision detection parameters (SDF-based collision cost)
    // These can be added to YAML under environment section, or use defaults
    float epsilon_sdf = 0.1f;   // SDF threshold for collision detection
    float sigma_obs = 0.05f;    // Smoothing parameter for collision cost
    
    // Obstacle generation parameters
    float min_spacing_factor = 2.0f;  // Minimum spacing = radius * factor
    
    /**
     * @brief Load configuration from YAML node
     * @param config YAML configuration node
     * @return CollisionAvoidanceConfig instance
     */
    static CollisionAvoidanceConfig fromYAML(const YAML::Node& config) {
        CollisionAvoidanceConfig cfg;
        
        // Read experiment parameters
        if (config["experiment"]) {
            const auto& exp = config["experiment"];
            if (exp["random_seed"]) {
                cfg.random_seed = exp["random_seed"].as<unsigned int>();
            }
        }
        
        // Read environment parameters
        if (config["environment"]) {
            const auto& env = config["environment"];
            cfg.map_width = env["map_width"].as<float>(cfg.map_width);
            cfg.map_height = env["map_height"].as<float>(cfg.map_height);
            cfg.num_obstacles = env["num_obstacles"].as<size_t>(cfg.num_obstacles);
            cfg.obstacle_radius = env["obstacle_radius"].as<float>(cfg.obstacle_radius);
            cfg.clearance_distance = env["clearance_distance"].as<float>(cfg.clearance_distance);
            
            if (env["min_spacing_factor"]) {
                cfg.min_spacing_factor = env["min_spacing_factor"].as<float>();
            }

            // Load cost parameters
            if (env["cost"]) {
                const YAML::Node& cost = env["cost"];
                
                if (cost["epsilon_sdf"]) {
                    cfg.epsilon_sdf = cost["epsilon_sdf"].as<float>();
                }
                if (cost["sigma_obs"]) {
                    cfg.sigma_obs = cost["sigma_obs"].as<float>();
                }
            }
        }
        
        // Read motion planning parameters
        if (config["motion_planning"]) {
            const auto& mp = config["motion_planning"];
            cfg.num_dimensions = mp["num_dimensions"].as<size_t>(cfg.num_dimensions);
            cfg.node_collision_radius = mp["node_collision_radius"].as<float>(cfg.node_collision_radius);
        }
        
        return cfg;
    }
    
    /**
     * @brief Validate configuration parameters
     */
    void validate() const {
        if (map_width <= 0 || map_height <= 0) {
            throw std::invalid_argument("Map dimensions must be positive");
        }
        
        if (num_dimensions < 2 || num_dimensions > 3) {
            throw std::invalid_argument("Only 2D and 3D environments are currently supported");
        }
        
        if (epsilon_sdf <= 0 || sigma_obs <= 0) {
            throw std::invalid_argument("SDF parameters must be positive");
        }
        
        if (obstacle_radius <= 0) {
            throw std::invalid_argument("Obstacle radius must be positive");
        }
        
        if (node_collision_radius <= 0) {
            throw std::invalid_argument("Node collision radius must be positive");
        }
        
        if (clearance_distance < 0) {
            throw std::invalid_argument("Clearance distance must be non-negative");
        }
        
        if (min_spacing_factor < 1.0f) {
            throw std::invalid_argument("Minimum spacing factor must be >= 1.0");
        }
    }
};


/**
 * @brief PCE Task for collision-free motion planning in obstacle environments.
 * 
 * This task encapsulates all obstacle-related logic and cost computations,
 * keeping the PCE optimizer agnostic to the specific problem.
 * 
 * The task is fully self-contained - it creates its own obstacle map, FK, and
 * manages all environment setup from the configuration.
 */
class CollisionAvoidanceTask : public Task {
public:
    using SparseMatrixXf = Eigen::SparseMatrix<float>;

    /**
     * @brief Constructor - creates fully configured task from YAML config
     * @param config YAML configuration node containing all parameters
     * @param custom_fk Optional custom forward kinematics. If nullptr, creates IdentityFK.
     * 
     * This constructor:
     * 1. Loads CollisionAvoidanceConfig from YAML
     * 2. Creates and configures ObstacleMap
     * 3. Generates obstacles
     * 4. Clears start/goal regions
     * 5. Creates forward kinematics (custom or Identity)
     */
    explicit CollisionAvoidanceTask(
        const YAML::Node& config,
        std::shared_ptr<ForwardKinematics> custom_fk = nullptr)
        : num_nodes_(0)
        , total_time_(0.0f)
    {
        std::cout << "\n=== Initializing CollisionAvoidanceTask ===\n";
        
        // 1. Load configuration from YAML
        config_ = CollisionAvoidanceConfig::fromYAML(config);
        config_.validate();
        
        epsilon_sdf_ = config_.epsilon_sdf;
        sigma_obs_ = config_.sigma_obs;
        node_collision_radius_ = config_.node_collision_radius;
        
        std::cout << "Configuration loaded:\n";
        std::cout << "  Map size: " << config_.map_width << " x " 
                  << config_.map_height << "\n";
        std::cout << "  Dimensions: " << config_.num_dimensions << "D\n";
        std::cout << "  Obstacles: " << config_.num_obstacles << "\n";
        std::cout << "  Obstacle radius: " << config_.obstacle_radius << "\n";
        std::cout << "  Node collision radius: " << config_.node_collision_radius << "\n";
        std::cout << "  Random seed: " << config_.random_seed << "\n\n";

        std::cout << "Cost:\n";
        std::cout << "  Epsilon SDF:          " << config_.epsilon_sdf << "\n";
        std::cout << "  Sigma obs:            " << config_.sigma_obs << "\n";
        
        // 2. Create obstacle map
        std::cout << "Creating obstacle map...\n";
        obstacle_map_ = std::make_shared<ObstacleMap>(
            config_.num_dimensions,
            config_.random_seed
        );
        
        obstacle_map_->setMapSize(
            config_.map_width, 
            config_.map_height
        );
        
        // 3. Generate random obstacles
        std::cout << "Generating obstacles...\n";
        obstacle_map_->generateRandom(
            config_.num_obstacles,
            config_.obstacle_radius,
            config_.min_spacing_factor
        );
        
        std::cout << "Generated " << obstacle_map_->size() << " obstacles\n";
        
        // 4. Read start and goal positions to clear regions
        if (config["motion_planning"]) {
            const auto& mp = config["motion_planning"];
            
            if (mp["start_position"] && mp["goal_position"]) {
                std::vector<float> start_pos = 
                    mp["start_position"].as<std::vector<float>>();
                std::vector<float> goal_pos = 
                    mp["goal_position"].as<std::vector<float>>();
                
                // Convert to Eigen vectors
                Eigen::VectorXf start_vec(start_pos.size());
                Eigen::VectorXf goal_vec(goal_pos.size());
                for (size_t i = 0; i < start_pos.size(); ++i) {
                    start_vec(i) = start_pos[i];
                    goal_vec(i) = goal_pos[i];
                }
                
                std::cout << "\nClearing start/goal regions...\n";
                std::cout << "  Start: [" << start_vec.transpose() << "]\n";
                std::cout << "  Goal:  [" << goal_vec.transpose() << "]\n";
                
                size_t initial_count = obstacle_map_->size();
                obstacle_map_->clearRegion(start_vec, config_.clearance_distance);
                obstacle_map_->clearRegion(goal_vec, config_.clearance_distance);
                size_t final_count = obstacle_map_->size();
                
                std::cout << "Removed " << (initial_count - final_count) 
                          << " obstacles near start/goal\n";
                std::cout << "Final obstacle count: " << final_count << "\n";
            }
        }
        
        // 5. Create forward kinematics (custom or Identity)
        if (custom_fk) {
            fk_ = custom_fk;
            std::cout << "\nUsing custom forward kinematics\n";
        } else {
            std::cout << "\nCreating Identity forward kinematics...\n";
            try {
                fk_ = std::make_shared<IdentityFK>(config_.num_dimensions);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    std::string("Failed to create IdentityFK: ") + e.what() +
                    "\nIf IdentityFK doesn't match your ForwardKinematics interface, "
                    "please provide a custom FK implementation as the second parameter."
                );
            }
        }
        
        std::cout << "\n=== CollisionAvoidanceTask Ready ===\n\n";
    }

    /**
     * @brief Get configuration
     */
    const CollisionAvoidanceConfig& getConfig() const { return config_; }

    /**
     * @brief Initialize task with trajectory parameters
     * Called by the motion planner to set up trajectory-specific parameters
     */
    void initialize(size_t num_dimensions,
                   const PathNode& start,
                   const PathNode& goal,
                   size_t num_nodes,
                   float total_time) override {
        
        num_dimensions_ = num_dimensions;
        start_node_ = start;
        goal_node_ = goal;
        num_nodes_ = num_nodes;
        total_time_ = total_time;

        // Verify dimensions match configuration
        if (num_dimensions_ != config_.num_dimensions) {
            std::cerr << "Warning: Task dimensions (" << num_dimensions_ 
                      << ") differ from config (" << config_.num_dimensions << ")\n";
        }

        // Cache obstacles for fast access during optimization
        obstacles_ = obstacle_map_->getObstacles();
        
        std::cout << "Task initialized for optimization:\n"
                  << "  Dimensions: " << num_dimensions_ << "\n"
                  << "  Waypoints: " << num_nodes_ << "\n"
                  << "  Active obstacles: " << obstacles_.size() << "\n"
                  << "  Collision radius: " << node_collision_radius_ << "\n";

        // Precompute R matrix for smoothness cost
        // computeRMatrix(num_nodes, total_time);
    }

    /**
     * @brief Compute collision cost using signed distance field
     */
    float computeCollisionCost(const Trajectory& trajectory) const override {
        if (trajectory.nodes.empty()) {
            return 0.0f;
        }

        float total_cost = 0.0f;
        const size_t N = trajectory.nodes.size();

        // Maximum exponent to prevent overflow (exp(20) ≈ 5e8)
        const float MAX_EXPONENT = 20.0f;
        const float MAX_COST_PER_NODE = std::exp(MAX_EXPONENT);

        // Skip first and last nodes (fixed boundary conditions)
        for (size_t i = 1; i < N - 1; ++i) {
            const PathNode& node = trajectory.nodes[i];
            
            // Transform to task space if needed
            Eigen::VectorXf task_pos = fk_->compute(node.position);
            
            // Compute minimum signed distance to all obstacles
            float min_sdf = std::numeric_limits<float>::infinity();
            
            for (const auto& obs : obstacles_) {
                // Ensure dimensionality matches
                if (obs.dimensions() != task_pos.size()) {
                    continue;
                }
                
                // Compute signed distance
                float dist = (task_pos - obs.center).norm();
                float sdf = dist - (obs.radius + node_collision_radius_);

                // Squared Hinge loss function: max(0, epsilon - signed_distance)^2
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - sdf);
                
                // Weighted squared hinge loss
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
                
                // if (sdf < min_sdf) {
                //     min_sdf = sdf;
                // }
            }
            
            
        }

        return total_cost;
    }

    // /**
    //  * @brief Compute smoothness cost using R matrix (acceleration minimization)
    //  */
    // float computeSmoothnessCost(const Trajectory& trajectory) const override {
    //     if (trajectory.nodes.empty() || R_matrix_.rows() == 0) {
    //         return 0.0f;
    //     }

    //     const size_t N = trajectory.nodes.size();
    //     const size_t D = num_dimensions_;
        
    //     if (N != static_cast<size_t>(R_matrix_.rows())) {
    //         std::cerr << "Warning: Trajectory size mismatch with R matrix\n";
    //         return 0.0f;
    //     }

    //     float total_cost = 0.0f;

    //     // Compute smoothness cost for each dimension
    //     for (size_t d = 0; d < D; ++d) {
    //         // Extract positions for dimension d
    //         Eigen::VectorXf positions(N);
    //         for (size_t i = 0; i < N; ++i) {
    //             positions(i) = trajectory.nodes[i].position(d);
    //         }

    //         // Compute quadratic form: positions^T * R * positions
    //         Eigen::VectorXf R_pos = R_matrix_ * positions;
    //         total_cost += positions.dot(R_pos);
    //     }

    //     return total_cost;
    // }

    /**
     * @brief Get reference to obstacles (for visualization/debugging)
     */
    const std::vector<ObstacleND>& getObstacles() const {
        return obstacles_;
    }

    /**
     * @brief Get reference to obstacle map
     */
    std::shared_ptr<ObstacleMap> getObstacleMap() const {
        return obstacle_map_;
    }

    // /**
    //  * @brief Get the R matrix used for smoothness cost
    //  */
    // const SparseMatrixXf& getRMatrix() const {
    //     return R_matrix_;
    // }

    /**
     * @brief Set collision cost parameters (can override config values)
     */
    void setCollisionParameters(float epsilon_sdf, float sigma_obs) {
        epsilon_sdf_ = epsilon_sdf;
        sigma_obs_ = sigma_obs;
        std::cout << "Updated collision parameters: epsilon_sdf=" << epsilon_sdf 
                  << ", sigma_obs=" << sigma_obs << "\n";
    }

    /**
     * @brief Set node collision radius (can override config value)
     */
    void setNodeCollisionRadius(float radius) {
        if (radius > 0) {
            node_collision_radius_ = radius;
            std::cout << "Updated node collision radius: " << radius << "\n";
        }
    }

protected:
    // /**
    //  * @brief Precompute the R = A^T * A matrix for smoothness cost
    //  */
    // void computeRMatrix(size_t num_nodes, float total_time) {
    //     float dt = total_time / static_cast<float>(num_nodes - 1);
    //     float dt_sq = dt * dt;
    //     float scale = 1.0f / (dt_sq * dt_sq);  // 1/dt^4

    //     R_matrix_.resize(num_nodes, num_nodes);
    //     R_matrix_.setZero();
        
    //     if (num_nodes < 3) {
    //         return;  // R is zero for N < 3
    //     }

    //     R_matrix_.reserve(Eigen::VectorXi::Constant(num_nodes, 5));
    //     std::vector<Eigen::Triplet<float>> triplets;

    //     // Build pentadiagonal matrix R = A^T * A
    //     // Main diagonal: R[i,i]
    //     triplets.emplace_back(0, 0, scale * 1.0f);
    //     triplets.emplace_back(1, 1, scale * 5.0f);
    //     for (size_t i = 2; i < num_nodes - 2; ++i) {
    //         triplets.emplace_back(i, i, scale * 6.0f);
    //     }
    //     triplets.emplace_back(num_nodes - 2, num_nodes - 2, scale * 5.0f);
    //     triplets.emplace_back(num_nodes - 1, num_nodes - 1, scale * 1.0f);

    //     // First off-diagonal: R[i,i+1] and R[i+1,i]
    //     triplets.emplace_back(0, 1, scale * (-2.0f));
    //     triplets.emplace_back(1, 0, scale * (-2.0f));
        
    //     for (size_t i = 1; i < num_nodes - 2; ++i) {
    //         triplets.emplace_back(i, i + 1, scale * (-4.0f));
    //         triplets.emplace_back(i + 1, i, scale * (-4.0f));
    //     }
        
    //     triplets.emplace_back(num_nodes - 2, num_nodes - 1, scale * (-2.0f));
    //     triplets.emplace_back(num_nodes - 1, num_nodes - 2, scale * (-2.0f));

    //     // Second off-diagonal: R[i,i+2] and R[i+2,i]
    //     triplets.emplace_back(0, 2, scale * 1.0f);
    //     triplets.emplace_back(2, 0, scale * 1.0f);
        
    //     for (size_t i = 1; i < num_nodes - 2; ++i) {
    //         triplets.emplace_back(i, i + 2, scale * 1.0f);
    //         triplets.emplace_back(i + 2, i, scale * 1.0f);
    //     }

    //     R_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    //     R_matrix_.makeCompressed();
    // }

private:
    // Obstacle management
    std::shared_ptr<ObstacleMap> obstacle_map_;
    std::vector<ObstacleND> obstacles_;  // Cached for performance

    // Forward kinematics
    std::shared_ptr<ForwardKinematics> fk_;

    // Configuration
    CollisionAvoidanceConfig config_;

    // Cost parameters (from config, can be overridden)
    float epsilon_sdf_;           // SDF threshold for collision detection
    float sigma_obs_;             // Smoothing parameter for collision cost
    float node_collision_radius_; // Radius for collision checking

    // Trajectory parameters
    size_t num_dimensions_;
    size_t num_nodes_;
    float total_time_;
    PathNode start_node_;
    PathNode goal_node_;

    // // Smoothness cost matrix
    // SparseMatrixXf R_matrix_;
};

}  // namespace pce