/**
 * @file MotionPlanner.h (Refactored Version - Config in Initialize)
 * @brief Abstract base class for motion planning algorithms
 * 
 * Key principles:
 * - Planner manages trajectory structure and optimization flow
 * - Task manages problem-specific data (obstacles, costs, constraints)
 * - Clean separation of concerns
 * - Configuration loaded during initialize(), solve() is parameter-free
 */
#pragma once

#include <pce/Trajectory.h>
#include "ForwardKinematics.h"
#include <vector> 
#include <random> 
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>      
#include <sstream>      
#include <iomanip>      
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <filesystem>


struct MotionPlannerConfig {
    // Motion planning parameters
    size_t num_dimensions = 2;
    size_t num_discretization = 100;
    float total_time = 12.0f;
    float node_collision_radius = 15.0f;
    
    // Start and goal positions
    std::vector<float> start_position;
    std::vector<float> goal_position;
    
    // Experiment parameters
    unsigned int random_seed = 42;
    bool visualize_initial_state = false;
    
    // Environment parameters
    float clearance_distance = 2.0f;
    
    /**
     * @brief Load configuration from YAML file
     * @param config_file_path Path to YAML configuration file
     * @return true if loading successful
     */
    virtual bool loadFromFile(const std::string& config_file_path) {
        try {
            YAML::Node config = YAML::LoadFile(config_file_path);
            return loadFromYAML(config);
        } catch (const std::exception& e) {
            std::cerr << "Error loading config from file: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * @brief Load configuration from YAML node
     * @param config YAML node containing configuration
     * @return true if loading successful
     */
    virtual bool loadFromYAML(const YAML::Node& config) {
        try {
            // Load motion planning parameters
            if (config["motion_planning"]) {
                const YAML::Node& mp = config["motion_planning"];
                
                if (mp["num_dimensions"]) {
                    num_dimensions = mp["num_dimensions"].as<size_t>();
                }
                if (mp["num_discretization"]) {
                    num_discretization = mp["num_discretization"].as<size_t>();
                }
                if (mp["total_time"]) {
                    total_time = mp["total_time"].as<float>();
                }
                if (mp["node_collision_radius"]) {
                    node_collision_radius = mp["node_collision_radius"].as<float>();
                }
                if (mp["start_position"]) {
                    start_position = mp["start_position"].as<std::vector<float>>();
                }
                if (mp["goal_position"]) {
                    goal_position = mp["goal_position"].as<std::vector<float>>();
                }
            }
            
            // Load experiment parameters
            if (config["experiment"]) {
                const YAML::Node& exp = config["experiment"];
                
                if (exp["random_seed"]) {
                    random_seed = exp["random_seed"].as<unsigned int>();
                }
                if (exp["visualize_initial_state"]) {
                    visualize_initial_state = exp["visualize_initial_state"].as<bool>();
                }
            }
            
            // Load environment parameters
            if (config["environment"]) {
                const YAML::Node& env = config["environment"];
                
                if (env["clearance_distance"]) {
                    clearance_distance = env["clearance_distance"].as<float>();
                }
            }
            
            return validate();
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading config from YAML: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * @brief Validate configuration parameters
     * @return true if configuration is valid
     */
    virtual bool validate() const {
        
        if (num_discretization < 2) {
            std::cerr << "Error: num_discretization must be >= 2\n";
            return false;
        }
        
        if (total_time <= 0.0f) {
            std::cerr << "Error: total_time must be positive\n";
            return false;
        }
        
        if (node_collision_radius < 0.0f) {
            std::cerr << "Error: node_collision_radius must be non-negative\n";
            return false;
        }
        
        if (start_position.size() != num_dimensions) {
            std::cerr << "Error: start_position size doesn't match num_dimensions\n";
            return false;
        }
        
        if (goal_position.size() != num_dimensions) {
            std::cerr << "Error: goal_position size doesn't match num_dimensions\n";
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Print configuration to console
     */
    virtual void print() const {
        std::cout << "=== Motion Planner Configuration ===\n";
        std::cout << "Motion Planning:\n";
        std::cout << "  Dimensions:           " << num_dimensions << "\n";
        std::cout << "  Discretization:       " << num_discretization << "\n";
        std::cout << "  Total time:           " << total_time << "\n";
        std::cout << "  Node radius:          " << node_collision_radius << "\n";
        std::cout << "  Start: [";
        for (size_t i = 0; i < start_position.size(); ++i) {
            std::cout << start_position[i];
            if (i < start_position.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Goal:  [";
        for (size_t i = 0; i < goal_position.size(); ++i) {
            std::cout << goal_position[i];
            if (i < goal_position.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        std::cout << "Experiment:\n";
        std::cout << "  Random seed:          " << random_seed << "\n";
        std::cout << "  Visualize initial:    " << (visualize_initial_state ? "true" : "false") << "\n";
        
        std::cout << "Environment:\n";
        std::cout << "  Clearance distance:   " << clearance_distance << "\n";
        std::cout << "\n";
    }
    
    virtual ~MotionPlannerConfig() = default;
};


/**
 * @brief Enumerates the available interpolation methods for initial path generation.
 */
enum class InterpolationMethod {
    LINEAR,
    BEZIER
};

// Forward declarations
struct ObstacleND;  // Only for getObstacles() return type

class MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;

    MotionPlanner() {
        std::random_device rd;
        random_engine_.seed(rd());
    }
    
    virtual ~MotionPlanner() {
        closeLog();
    }

    /**
     * @brief Get planner name (must be implemented by derived classes)
     */
    virtual std::string getPlannerName() const = 0;

    /**
     * @brief Initialize planner with configuration object
     * @param config Configuration object containing all parameters
     * @return true if initialization successful
     * 
     * This method:
     * 1. Stores configuration
     * 2. Validates configuration
     * 3. Sets up environment (if needed)
     * 4. Initializes trajectory structure
     * 
     * After calling this, the planner is ready for optimize()
     */
    virtual bool initialize(const MotionPlannerConfig& config) {
        // Store configuration
        config_ = std::make_shared<MotionPlannerConfig>(config);
        
        // Validate configuration
        if (!config_->validate()) {
            std::cerr << "Failed to validate configuration\n";
            return false;
        }
        
        // Extract parameters from config
        num_dimensions_ = config_->num_dimensions;
        num_nodes_ = config_->num_discretization;
        total_time_ = config_->total_time;
        node_radius_ = config_->node_collision_radius;
        random_seed_ = config_->random_seed;
        
        // Seed random engine
        random_engine_.seed(random_seed_);
        
        // Print configuration
        std::cout << "Initializing " << getPlannerName() << " planner...\n";
        
        // Open log file
        openLog();
        // log("=== Configuration ===");
        // logf("Planner: %s", getPlannerName().c_str());
        // logf("Dimensions: %zu", num_dimensions_);
        // logf("Nodes: %zu", num_nodes_);
        // logf("Total time: %.2f", total_time_);
        // logf("Node radius: %.2f", node_radius_);
        // logf("Random seed: %u", random_seed_);
        // log("");
        
        // Initialize trajectory
        if (!initializeTrajectory()) {
            std::cerr << "Failed to initialize trajectory\n";
            return false;
        }
        
        // Log planner-specific configuration
        // logPlannerSpecificConfig();
        
        is_initialized_ = true;
        
        std::cout << "Planner initialized successfully\n\n";
        return true;
    }

    /**
     * @brief Solve the motion planning problem (parameter-free)
     * @return true if optimization successful
     * 
     * Configuration must be loaded via initialize() first.
     * This method just runs the optimization.
     */
    virtual bool solve() {
        if (!is_initialized_) {
            std::cerr << "Error: Planner not initialized. Call initialize() first.\n";
            return false;
        }
        
        // Run optimization
        return optimize();
    }

    /**
     * @brief Visualize the initial trajectory state
     */
    void showInitialState() {
        if (current_trajectory_.nodes.empty()) {
            std::cerr << "Warning: Cannot visualize - trajectory not initialized\n";
            return;
        }
        
        std::cout << "Initial state ready for visualization\n";
        // Visualization can be added here if needed
    }

    /**
     * @brief Check if visualization should be shown
     */
    bool shouldVisualizeInitialState() const {
        return config_ ? config_->visualize_initial_state : false;
    }

    /**
     * @brief Initialize the trajectory structure (NO obstacles parameter)
     * 
     * This method ONLY handles trajectory interpolation and R-matrix setup.
     * Obstacle management is entirely in the Task.
     * 
     * @param num_dims Number of dimensions
     * @param start Starting configuration
     * @param goal Goal configuration
     * @param num_nodes Number of trajectory waypoints
     * @param total_time Total trajectory duration
     * @param method Interpolation method
     */
    virtual void initializeTrajectoryStructure(InterpolationMethod method = InterpolationMethod::LINEAR) {
        
        if (num_nodes_ < 2) {
            throw std::invalid_argument("num_nodes must be >= 2");
        }

        // Store trajectory parameters
        current_trajectory_.nodes.clear();
        current_trajectory_.total_time = total_time_;
        current_trajectory_.start_index = 0;
        current_trajectory_.goal_index = num_nodes_ - 1;

        // Update forward kinematics
        fk_ = std::make_shared<IdentityFK>(num_dimensions_);

        // Build interpolated trajectory
        for (size_t i = 0; i < num_nodes_; ++i) {
            float t = static_cast<float>(i) / (num_nodes_ - 1);
            Eigen::VectorXf pos = (1 - t) * start_node_.position + t * goal_node_.position;
            float radius = start_node_.radius;
            current_trajectory_.nodes.emplace_back(pos, radius);
        }

        // Compute R = A^T A for smoothness costs
        computeRMatrix(num_nodes_, total_time_);
    }

    /**
     * @brief Run the optimization algorithm (pure virtual)
     */
    virtual bool optimize() = 0;

    /**
     * @brief Collision cost computation (delegated to Task in derived classes)
     */
    virtual float computeCollisionCost(const Trajectory& trajectory) const {
        std::cerr << "Error: computeCollisionCost not implemented\n";
        return std::numeric_limits<float>::infinity();
    }

    // /**
    //  * @brief Computes the L2-norm squared smoothness cost for N-dimensional trajectories.
    //  * Equivalent to sum over all dimensions: Σ_d (X_d^T A^T A X_d)
    //  * @return The total smoothness cost (non-negative).
    //  */
    // float computeSmoothnessCost() const {
    //     return computeSmoothnessCost(current_trajectory_);
    // }

    float computeSmoothnessCost(const Trajectory& traj) const {
        const size_t N = traj.nodes.size();
        const size_t D = traj.dimensions();
        
        // Handle edge cases
        if (N < 3) {
            return 0.0f;  // No smoothness cost for < 3 nodes
        }
        
        // Extract all trajectory positions into a D x N matrix
        Eigen::MatrixXf Y(D, N);

        for (size_t i = 0; i < N; ++i) {
            Y.col(i) = traj.nodes[i].position;
        }
        
        // Compute total cost: sum over all dimensions
        float total_cost = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            Eigen::VectorXf Y_d = Y.row(d).transpose();  // Extract dimension d (N x 1 vector)
            
            // Debug: Check Y_d
            float y_min = Y_d.minCoeff();
            float y_max = Y_d.maxCoeff();
            float y_mean = Y_d.mean();
            
            // Compute quadratic form: Y_d^T * R * Y_d
            Eigen::VectorXf RY_d = R_matrix_ * Y_d;
            float cost_d = Y_d.dot(RY_d);
            
            // Debug output if negative
            if (cost_d < 0) {
                std::cerr << "\n=== NEGATIVE SMOOTHNESS COST ===" << "\n";
                std::cerr << "Dimension: " << d << "\n";
                std::cerr << "Cost: " << cost_d << "\n";
                std::cerr << "Y_d range: [" << y_min << ", " << y_max << "]\n";
                std::cerr << "Y_d mean: " << y_mean << "\n";
                std::cerr << "Y_d norm: " << Y_d.norm() << "\n";
                std::cerr << "RY_d norm: " << RY_d.norm() << "\n";
                
                // Check if R is symmetric
                std::cerr << "R_matrix size: " << R_matrix_.rows() << "x" << R_matrix_.cols() << "\n";
                std::cerr << "R_matrix nonzeros: " << R_matrix_.nonZeros() << "\n";
                
                // Print first few values
                std::cerr << "First 5 Y_d values: ";
                for (int i = 0; i < std::min(5, (int)Y_d.size()); ++i) {
                    std::cerr << Y_d(i) << " ";
                }
                std::cerr << "\n";
                
                // Check symmetry by sampling
                float r_01 = R_matrix_.coeff(0, 1);
                float r_10 = R_matrix_.coeff(1, 0);
                std::cerr << "R(0,1) = " << r_01 << ", R(1,0) = " << r_10 << "\n";
                
                std::cerr << "================================\n\n";
            }
            
            total_cost += cost_d;
        }
        
        return total_cost;
    }

    /**
     * @brief Get forward kinematics instance
     */
    std::shared_ptr<ForwardKinematics> getForwardKinematics() const {
        return fk_;
    }

    /**
     * @brief Get current trajectory
     */
    const Trajectory& getCurrentTrajectory() const {
        return current_trajectory_;
    }

    /**
     * @brief Get trajectory history
     */
    const std::vector<Trajectory>& getTrajectoryHistory() const {
        return trajectory_history_;
    }

    /**
     * @brief Get number of dimensions
     */
    size_t getNumDimensions() const {
        return num_dimensions_;
    }

    /**
     * @brief Get configuration
     */
    std::shared_ptr<const MotionPlannerConfig> getConfig() const {
        return config_;
    }

    /**
     * @brief Check if planner is initialized
     */
    bool isInitialized() const {
        return is_initialized_;
    }

public:

    /**
     * @brief Samples M noise matrices from N(0, R^{-1})
     * @param M Number of samples
     * @param N Number of trajectory nodes
     * @param D Number of dimensions
     * @return Vector of M noise matrices (D x N each)
     */
    std::vector<Eigen::MatrixXf> sampleNoiseMatrices(size_t M, size_t N, size_t D) {
        std::vector<Eigen::MatrixXf> epsilon_samples;
        epsilon_samples.reserve(M);
        
        for (size_t m = 0; m < M; ++m) {
            Eigen::MatrixXf epsilon_m(D, N);
            
            for (size_t d = 0; d < D; ++d) {
                std::vector<float> noise_1d = sampleSmoothnessNoise(N, random_engine_);
                for (size_t i = 0; i < N; ++i) {
                    epsilon_m(d, i) = noise_1d[i];
                }
            }
            
            epsilon_samples.push_back(epsilon_m);
        }
        
        return epsilon_samples;
    }

protected:

    /**
     * @brief Samples a 1D vector from the Gaussian distribution N(0, R^{-1}).
     * This vector represents the smoothness noise applied to one coordinate dimension.
     * The implementation relies on the efficient banded structure of R.
     * @param N The size of the vector to sample (number of trajectory nodes).
     * @param rng The random number generator engine.
     * @return A vector of size N containing the sampled noise.
     */
    virtual std::vector<float> sampleSmoothnessNoise(size_t num_nodes, std::mt19937& random_engine) const {
        if (num_nodes < 3) {
            return std::vector<float>(num_nodes, 0.0f);
        }
        
        Eigen::VectorXd z(num_nodes - 2);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < num_nodes - 2; ++i) {
            z(i) = dist(random_engine);
        }
        
        Eigen::VectorXd noise = L_solver_.solve(z);
        if (L_solver_.info() != Eigen::Success) {
            std::cerr << "Cholesky solve failed in sampleSmoothnessNoise\n";
            return std::vector<float>(num_nodes, 0.0f);
        }
        
        std::vector<float> result(num_nodes, 0.0f);
        for (size_t i = 0; i < num_nodes - 2; ++i) {
            result[i + 1] = static_cast<float>(noise(i));
        }
        return result;
    }

    /**
     * @brief Pre-computes and caches the Cholesky factorization of the smoothness 
     *        precision matrix R for the free nodes.
     * @param N Total number of nodes in the trajectory.
     * @param total_time Total time duration of the trajectory.
     */
    virtual void precomputeCholeskyFactorization(size_t num_nodes, float total_time) {
        if (num_nodes < 3) return;
        
        const size_t N_free = num_nodes - 2;
        
        // Compute scaling (use double for intermediate calculations)
        double dt = static_cast<double>(total_time) / static_cast<double>(num_nodes - 1);
        double dt_sq = dt * dt;
        double scale = 1.0 / (dt_sq * dt_sq);  // Use double precision
        
        // Build R_free with DOUBLE precision
        Eigen::SparseMatrix<double> R_free(N_free, N_free);
        R_free.reserve(Eigen::VectorXi::Constant(N_free, 5));
        
        std::vector<Eigen::Triplet<double>> triplets;  // Use double
        
        if (N_free == 1) {
            triplets.emplace_back(0, 0, 6.0 * scale);
        } else if (N_free == 2) {
            triplets.emplace_back(0, 0, 5.0 * scale);
            triplets.emplace_back(1, 1, 5.0 * scale);
            triplets.emplace_back(0, 1, -4.0 * scale);
            triplets.emplace_back(1, 0, -4.0 * scale);
        } else {
            // Main diagonal: [5, 6, 6, ..., 6, 5]
            triplets.emplace_back(0, 0, 5.0 * scale);
            for (size_t i = 1; i < N_free - 1; ++i) {
                triplets.emplace_back(i, i, 6.0 * scale);
            }
            triplets.emplace_back(N_free - 1, N_free - 1, 5.0 * scale);
            
            // Off-diagonal 1: all -4
            for (size_t i = 0; i < N_free - 1; ++i) {
                triplets.emplace_back(i, i + 1, -4.0 * scale);
                triplets.emplace_back(i + 1, i, -4.0 * scale);
            }
            
            // Off-diagonal 2: all 1
            for (size_t i = 0; i < N_free - 2; ++i) {
                triplets.emplace_back(i, i + 2, 1.0 * scale);
                triplets.emplace_back(i + 2, i, 1.0 * scale);
            }
        }
        
        R_free.setFromTriplets(triplets.begin(), triplets.end());
        R_free.makeCompressed();
        
        // Perform Cholesky decomposition
        L_solver_.compute(R_free);
        
        if (L_solver_.info() != Eigen::Success) {
            std::cerr << "ERROR: Cholesky decomposition failed!\n";
            std::cerr << "  N_free: " << N_free << ", dt: " << dt 
                    << ", scale: " << scale << "\n";
            throw std::runtime_error("Cholesky decomposition failed");
        }
        
        std::cout << "Cholesky factorization computed for " << N_free 
                << " free nodes (scale=" << scale << ")\n";
    }

    
    /**
     * @brief Initialize trajectory from config
     */
    bool initializeTrajectory() {
        try {
            if (!config_) {
                std::cerr << "Error: No configuration loaded\n";
                return false;
            }
            
            // Get start/goal from config
            const auto& start_vec = config_->start_position;
            const auto& goal_vec = config_->goal_position;
            
            // Construct start_node_ and goal_node_ for general size start_vec and goal_vec
            start_node_ = PathNode(Eigen::VectorXf::Map(start_vec.data(), start_vec.size()), node_radius_);
            goal_node_ = PathNode(Eigen::VectorXf::Map(goal_vec.data(), goal_vec.size()), node_radius_);

            // Initialize trajectory structure
            initializeTrajectoryStructure(InterpolationMethod::LINEAR);
            
            // Task-specific initialization (handled by derived class)
            initializeTask();
            
            std::cout << "Trajectory initialized: " << num_nodes_ << " nodes\n";
            std::cout << "  Start: [" << start_node_.position.transpose() << "]\n";
            std::cout << "  Goal:  [" << goal_node_.position.transpose() << "]\n";
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing trajectory: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * @brief Initialize task with trajectory parameters
     * Must be overridden by derived classes to call task->initialize()
     */
    virtual void initializeTask() = 0;

    // ============= R Matrix Computation =============
    
    void computeRMatrix(size_t num_nodes, float total_time) {
        float dt = total_time / static_cast<float>(num_nodes - 1);
        float dt_sq = dt * dt;
        float scale = 1.0f / (dt_sq * dt_sq);

        R_matrix_.resize(num_nodes, num_nodes);
        R_matrix_.setZero();
        
        if (num_nodes < 3) {
            return;
        }

        R_matrix_.reserve(Eigen::VectorXi::Constant(num_nodes, 5));
        std::vector<Eigen::Triplet<float>> triplets;

        // Main diagonal
        triplets.emplace_back(0, 0, scale * 1.0f);
        triplets.emplace_back(1, 1, scale * 5.0f);
        for (size_t i = 2; i < num_nodes - 2; ++i) {
            triplets.emplace_back(i, i, scale * 6.0f);
        }
        triplets.emplace_back(num_nodes - 2, num_nodes - 2, scale * 5.0f);
        triplets.emplace_back(num_nodes - 1, num_nodes - 1, scale * 1.0f);

        // First off-diagonal
        triplets.emplace_back(0, 1, scale * (-2.0f));
        triplets.emplace_back(1, 0, scale * (-2.0f));
        for (size_t i = 1; i < num_nodes - 2; ++i) {
            triplets.emplace_back(i, i + 1, scale * (-4.0f));
            triplets.emplace_back(i + 1, i, scale * (-4.0f));
        }
        triplets.emplace_back(num_nodes - 2, num_nodes - 1, scale * (-2.0f));
        triplets.emplace_back(num_nodes - 1, num_nodes - 2, scale * (-2.0f));

        // Second off-diagonal
        triplets.emplace_back(0, 2, scale * 1.0f);
        triplets.emplace_back(2, 0, scale * 1.0f);
        for (size_t i = 1; i < num_nodes - 2; ++i) {
            triplets.emplace_back(i, i + 2, scale * 1.0f);
            triplets.emplace_back(i + 2, i, scale * 1.0f);
        }

        R_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        R_matrix_.makeCompressed();

        // Compute Cholesky decomposition
        if (num_nodes >= 3) {
            precomputeCholeskyFactorization(num_nodes, total_time);
        }

        storeTrajectory();
        
    }

    // ============= Trajectory Utilities =============
    
    /**
     * @brief Store current trajectory in history
     */
    void storeTrajectory() {
        trajectory_history_.push_back(current_trajectory_);
    }

    /**
     * @brief Convert trajectory to matrix format [D x N]
     */
    Eigen::MatrixXf trajectoryToMatrix() const {
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        
        Eigen::MatrixXf Y(D, N);
        for (size_t i = 0; i < N; ++i) {
            Y.col(i) = current_trajectory_.nodes[i].position;
        }
        return Y;
    }

    /**
     * @brief Updates trajectory from matrix and enforces start/goal constraints
     * @param Y_new New trajectory matrix (D x N)
     */
    void updateTrajectoryFromMatrix(const Eigen::MatrixXf& Y_new) {
        const size_t N = Y_new.cols();
        
        for (size_t i = 0; i < N; ++i) {
            current_trajectory_.nodes[i].position = Eigen::VectorXf(Y_new.col(i));
        }
        
        // Enforce start/goal constraints
        current_trajectory_.nodes[0].position = start_node_.position;
        current_trajectory_.nodes[N - 1].position = goal_node_.position;
    }

    /**
     * @brief Create perturbed trajectory
     */
    Trajectory createPerturbedTrajectory(const Eigen::MatrixXf& Y_k, 
                                         const Eigen::MatrixXf& epsilon) const {
        Eigen::MatrixXf Y_perturbed = Y_k + epsilon;
        return matrixToTrajectory(Y_perturbed);
    }

    /**
     * @brief Converts matrix to trajectory format
     * @param Y Matrix (D x N) to convert
     * @return Trajectory with positions from Y
     */
    Trajectory matrixToTrajectory(const Eigen::MatrixXf& Y) const {
        Trajectory traj = current_trajectory_;  // Copy metadata
        
        const size_t N = Y.cols();
        const size_t D = Y.rows();
        
        for (size_t i = 0; i < N; ++i) {
            traj.nodes[i].position = Eigen::VectorXf(Y.col(i));
        }
        
        return traj;
    }

    Trajectory matrixToTrajectory(const MatrixXf& positions, const Trajectory& reference) const {
        Trajectory traj;
        traj.total_time = reference.total_time;
        traj.start_index = reference.start_index;
        traj.goal_index = reference.goal_index;
        
        const size_t N = positions.cols();
        traj.nodes.reserve(N);
        
        for (size_t i = 0; i < N; ++i) {
            Eigen::VectorXf position = Eigen::VectorXf(positions.col(i));
            
            float radius = (i < reference.nodes.size()) ? reference.nodes[i].radius : 0.5f;
            PathNode node(position, radius);
            traj.nodes.push_back(node);
        }
        
        return traj;
    }


    // ============= Logging Utilities =============
    
    void log(const std::string& message) {
        if (!log_file_.is_open()) {
            openLog();
        }
        log_file_ << message << "\n";
        std::cout << message << "\n";
    }

    template<typename... Args>
    void logf(const char* format, Args... args) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), format, args...);
        log(std::string(buffer));
    }

    void logSeparator(char c = '-', size_t length = 80) {
        log(std::string(length, c));
    }

    virtual void logPlannerSpecificConfig() {
        // Override in derived classes
    }

    std::string getLogFilename() const {
        return log_filename_;
    }

private:
    void openLog() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "logs/" << getPlannerName() << "_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
        log_filename_ = ss.str();
        
        std::filesystem::create_directories("logs");
        log_file_.open(log_filename_);
        
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Could not open log file\n";
        }
    }

    void closeLog() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

protected:
    // Configuration
    std::shared_ptr<MotionPlannerConfig> config_;
    
    // Forward kinematics (identity by default)
    std::shared_ptr<ForwardKinematics> fk_;
    
    // Trajectory state
    Trajectory current_trajectory_;
    std::vector<Trajectory> trajectory_history_;
    PathNode start_node_;
    PathNode goal_node_;
    
    // Parameters (extracted from config)
    size_t num_dimensions_;
    size_t num_nodes_;
    float total_time_;
    float node_radius_;
    unsigned int random_seed_ = 42;
    
    // Optimization matrices
    SparseMatrixXf R_matrix_;

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> L_solver_;
    
    // State
    bool is_initialized_ = false;
    
    // Random number generation
    mutable std::mt19937 random_engine_;
    
    // Logging
    std::ofstream log_file_;
    std::string log_filename_;
};
