#pragma once

#include "Trajectory.h"
#include "ObstacleMap.h"
#include "ForwardKinematics.h"
#include "collision_utils.h"
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

/**
 * @brief Enumerates the available interpolation methods for initial path generation.
 */
enum class InterpolationMethod {
    LINEAR,
    BEZIER
};

/**
 * @brief Structure to hold the non-zero diagonals of the symmetric, pentadiagonal matrix R = A^T * A.
 * R is defined by four non-zero bands (diagonals).
 */
struct RMatrixDiagonals {
    std::vector<float> main_diag; // R[i, i]
    std::vector<float> diag1;     // R[i, i+1] / R[i+1, i]
    std::vector<float> diag2;     // R[i, i+2] / R[i+2, i]
};


// Forward declare visualization function
void visualizeInitialState(const std::vector<ObstacleND>& obstacles,
                          const Trajectory& trajectory,
                          const std::string& window_title);

/**
 * @brief Abstract base class for motion planning algorithms (e.g., Trajectory Optimization).
 * Fully Eigen-based for N-dimensional support.
 */
class MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;

    // Constructor - initializes random engine and FK (default identity)
    MotionPlanner() {
        std::random_device rd;
        random_engine_.seed(rd());
    }
    
    virtual ~MotionPlanner() {
        closeLog();
    }

    /**
     * @brief Visualize the initial trajectory state before optimization
     * Shows obstacles, initial trajectory, and collision status
     * Blocks until user presses SPACE/ENTER to continue
     */
    void showInitialState() {
        if (current_trajectory_.nodes.empty()) {
            std::cerr << "Warning: Cannot visualize - trajectory not initialized\n";
            return;
        }
        
        std::string window_title = getPlannerName() + " - Initial State";
        ::visualizeInitialState(obstacles_, current_trajectory_, window_title);
    }
    
    /**
     * @brief Check if visualization should be shown (from config)
     */
    bool shouldVisualizeInitialState() const {
        if (config_["experiment"] && config_["experiment"]["visualize_initial_state"]) {
            return config_["experiment"]["visualize_initial_state"].as<bool>();
        }
        return false;  // Default: don't visualize
    }

    /**
     * @brief Initialize the planner without running optimization
     * Useful for visualization and testing
     * @param config_file_path Path to YAML config file
     * @return true if successful
     */
    bool initializeOnly(const std::string& config_file_path) {
        // 1. Load configuration
        if (!loadConfiguration(config_file_path)) {
            std::cerr << "Failed to load configuration\n";
            return false;
        }
        
        // 2. Setup environment
        if (!setupEnvironment()) {
            std::cerr << "Failed to setup environment\n";
            return false;
        }
        
        // 3. Initialize trajectory
        if (!initializeTrajectory()) {
            std::cerr << "Failed to initialize trajectory\n";
            return false;
        }
        
        is_initialized_ = true;
        return true;
    }
    
    /**
     * @brief Get forward kinematics instance
     */
    std::shared_ptr<ForwardKinematics> getForwardKinematics() const {
        return fk_;
    }

    /**
     * @brief Initializes the trajectory from start to goal using the specified interpolation method.
     * @param start The starting path node.
     * @param goal The goal path node.
     * @param num_steps The number of nodes in the initial trajectory.
     * @param total_time The time duration of the trajectory.
     * @param method The interpolation method to use (LINEAR or BEZIER).
     * @param obstacles Reference to the obstacles vector (obstacles near start/goal will be removed).
     * @param clearance_radius The radius around start and goal within which obstacles are removed.
     */

    virtual void initialize(const size_t num_dims,
                            const PathNode& start, 
                            const PathNode& goal, 
                            size_t num_nodes, 
                            float total_time, 
                            InterpolationMethod method, 
                            ObstacleMap& obstacle_map,
                            float clearance_dist) {

        std::cout << "\n========== DEBUG initialize() ==========\n";
        std::cout << "Input parameters:\n";
        std::cout << "  num_dims: " << num_dims << "\n";
        std::cout << "  start.position.size(): " << start.position.size() << "\n";
        std::cout << "  goal.position.size(): " << goal.position.size() << "\n";
        std::cout << "  start.position: [";
        for (int i = 0; i < start.position.size(); ++i) {
            std::cout << start.position(i);
            if (i < start.position.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  goal.position: [";
        for (int i = 0; i < goal.position.size(); ++i) {
            std::cout << goal.position(i);
            if (i < goal.position.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        if (num_nodes < 2) {
            throw std::invalid_argument("num_nodes must be >= 2");
        }

        // Clear any prior state
        current_trajectory_.nodes.clear();
        current_trajectory_.total_time = total_time;
        current_trajectory_.start_index = 0;
        current_trajectory_.goal_index = num_nodes - 1;
        total_time_ = total_time;
        num_nodes_ = num_nodes;
        start_node_ = start;
        goal_node_ = goal;
        num_dimensions_ = num_dims;

        // Update FK to match num_dimensions_
        fk_ = std::make_shared<IdentityFK>(num_dimensions_);

        // Clear obstacles near start/goal
        obstacle_map.clearStartGoalRegions(start.position, goal.position, clearance_dist);
        
        // Store obstacles from map
        obstacles_ = obstacle_map.getObstacles();

        // --- Build interpolated nodes (linear for simplicity; extend to Bezier if needed) ---
        for (size_t i = 0; i < num_nodes; ++i) {
            float t = static_cast<float>(i) / (num_nodes - 1);
            Eigen::VectorXf pos = (1 - t) * start.position + t * goal.position;
            float radius = start.radius;
            current_trajectory_.nodes.emplace_back(pos, radius);
        }

        // Compute R = A^T A
        float dt = total_time / static_cast<float>(num_nodes - 1);
        float dt_sq = dt * dt;
        float scale = 1.0f / (dt_sq * dt_sq); // Correctly compute 1/dt^4

        R_matrix_.resize(num_nodes, num_nodes);
        R_matrix_.setZero();
        if (num_nodes >= 3) {
            R_matrix_.reserve(Eigen::VectorXi::Constant(num_nodes, 5));
            std::vector<Eigen::Triplet<float>> triplets;
            
            // Main diagonal
            triplets.emplace_back(0, 0, 1.0f * scale);
            triplets.emplace_back(1, 1, 5.0f * scale);
            for (size_t i = 2; i < num_nodes - 2; ++i) {
                triplets.emplace_back(i, i, 6.0f * scale);
            }
            triplets.emplace_back(num_nodes - 2, num_nodes - 2, 5.0f * scale);
            triplets.emplace_back(num_nodes - 1, num_nodes - 1, 1.0f * scale);
            
            // Off-diagonal 1: pattern is -2, -4, -4, ..., -4, -2
            triplets.emplace_back(0, 1, -2.0f * scale);
            triplets.emplace_back(1, 0, -2.0f * scale);
            
            for (size_t i = 1; i < num_nodes - 2; ++i) {
                triplets.emplace_back(i, i + 1, -4.0f * scale);
                triplets.emplace_back(i + 1, i, -4.0f * scale);
            }
            
            triplets.emplace_back(num_nodes - 2, num_nodes - 1, -2.0f * scale);
            triplets.emplace_back(num_nodes - 1, num_nodes - 2, -2.0f * scale);
            
            // Off-diagonal 2: all are 1
            for (size_t i = 0; i < num_nodes - 2; ++i) {
                triplets.emplace_back(i, i + 2, 1.0f * scale);
                triplets.emplace_back(i + 2, i, 1.0f * scale);
            }
            
            R_matrix_.setFromTriplets(triplets.begin(), triplets.end());

            float epsilon = 1e-4f * scale;  // Small regularization relative to scale
            for (size_t i = 0; i < num_nodes; ++i) {
                R_matrix_.coeffRef(i, i) += epsilon;
            }

            std::cout << "\n=== Verifying R Matrix ===\n";
            std::cout << "R size: " << R_matrix_.rows() << "x" << R_matrix_.cols() << "\n";
            std::cout << "R nonzeros: " << R_matrix_.nonZeros() << "\n";
            std::cout << "scale = " << scale << "\n";

            // Print structure for small matrices
            if (num_nodes <= 10) {
                Eigen::MatrixXf R_dense = Eigen::MatrixXf(R_matrix_);
                std::cout << "R matrix:\n" << R_dense << "\n";
            }

            // Check eigenvalues (convert to dense for small matrices)
            if (num_nodes <= 200) {
                Eigen::MatrixXf R_dense = Eigen::MatrixXf(R_matrix_);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(R_dense);
                
                float min_eig = es.eigenvalues().minCoeff();
                float max_eig = es.eigenvalues().maxCoeff();
                
                std::cout << "Eigenvalues: min = " << min_eig << ", max = " << max_eig << "\n";
                
                if (min_eig < -1e-6) {
                    std::cerr << "ERROR: R matrix has negative eigenvalues! Not PSD!\n";
                    std::cerr << "Smallest eigenvalues: " << es.eigenvalues().head(5).transpose() << "\n";
                } else {
                    std::cout << "✓ R matrix is positive semi-definite\n";
                }
            }
            std::cout << "=========================\n\n";

            R_matrix_.makeCompressed();
        }


        // Compute Cholesky decomposition
        if (num_nodes >= 3) {
            precomputeCholeskyFactorization(num_nodes, total_time);
        }

        storeTrajectory();

        std::cout << "Planner initialized with " << num_nodes << " nodes in " 
                << start.dimensions() << "D space.\n";
    }

    /**
     * @brief Seeds the random number generator for reproducibility.
     * @param seed The seed value.
     */
    void seedRandomEngine(unsigned int seed) {
        random_engine_.seed(seed);
        std::cout << "MotionPlanner seed set to: " << seed << "\n";
    }

    /**
     * @brief Main interface: solve the motion planning problem
     * @param config_file_path Path to YAML config file
     * @return true if successful
     */
    virtual bool solve(const std::string& config_file_path) {
        if (!is_initialized_) {
            // 1. Load configuration
            if (!loadConfiguration(config_file_path)) {
                std::cerr << "Failed to load configuration from: " << config_file_path << "\n";
                return false;
            }
            
            // 2. Initialize logger
            std::string planner_name = getPlannerName();
            std::string log_dir = config_["experiment"]["log_directory"] 
                                ? config_["experiment"]["log_directory"].as<std::string>() 
                                : "./logs";
            initializeLogger(planner_name, log_dir);
            
            // 3. Log experiment setup
            logExperimentSetup();
            
            // 4. Setup environment
            if (!setupEnvironment()) {
                log("Failed to setup environment");
                return false;
            }
            
            // 5. Initialize trajectory
            if (!initializeTrajectory()) {
                log("Failed to initialize trajectory");
                return false;
            }

            is_initialized_ = true;
        }

        // Initialize logger (always do this, even if continuing)
        std::string planner_name = getPlannerName();
        std::string log_dir = config_["experiment"]["log_directory"] 
                             ? config_["experiment"]["log_directory"].as<std::string>() 
                             : "./logs";
        initializeLogger(planner_name, log_dir);
        
        // Log configuration
        logExperimentSetup();
        logConfiguration();
        
        // 8. Run optimization
        log("--- Starting Optimization ---");
        bool success = optimize();
        
        // 9. Log final state
        logFinalState();
        
        // 10. Close log
        closeLog();
        
        return success;
    }

    /**
     * @brief Pure virtual method for algorithm-specific optimization
     */
    virtual bool optimize() = 0;
    
    /**
     * @brief Get the planner name (e.g., "PCEM", "NGD")
     */
    virtual std::string getPlannerName() const = 0;

    /**
     * @brief N-dimensional collision cost using Squared Hinge Loss.
     */
    virtual float computeCollisionCost(const Trajectory& traj, const std::vector<ObstacleND>& obstacles) const {
        float total_cost = 0.0f;

        for (const auto& node : traj.nodes) {
            for (const auto& obs : obstacles) {
                if (node.dimensions() != obs.dimensions()) {
                    continue; 
                }
                
                VectorXf diff = node.position - obs.center;
                float dist_to_center = diff.norm();
                float signed_distance = dist_to_center - obs.radius - node.radius;
                
                // Squared Hinge loss function: max(0, epsilon - signed_distance)^2
                float hinge_loss = std::max(0.0f, epsilon_sdf_ - signed_distance);
                
                // Weighted squared hinge loss
                total_cost += sigma_obs_ * hinge_loss * hinge_loss;
            }
        }
        return total_cost;
    }

    /**
     * @brief Computes the collision cost of the current trajectory.
     */
    float computeCollisionCost(const std::vector<ObstacleND>& obstacles) const {
        return computeCollisionCost(current_trajectory_, obstacles);
    }

    /**
     * @brief Computes the L2-norm squared smoothness cost for N-dimensional trajectories.
     * Equivalent to sum over all dimensions: Σ_d (X_d^T A^T A X_d)
     * @return The total smoothness cost (non-negative).
     */
    float computeSmoothnessCost() const {
        return computeSmoothnessCost(current_trajectory_);
    }

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
            
            // ✅ Debug: Check Y_d
            float y_min = Y_d.minCoeff();
            float y_max = Y_d.maxCoeff();
            float y_mean = Y_d.mean();
            
            // Compute quadratic form: Y_d^T * R * Y_d
            Eigen::VectorXf RY_d = R_matrix_ * Y_d;
            float cost_d = Y_d.dot(RY_d);
            
            // ✅ Debug output if negative
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

    /**
     * @brief Returns the current optimized trajectory.
     */
    const Trajectory& getCurrentTrajectory() const {
        return current_trajectory_;
    }
    
    /**
     * @brief Sets the current trajectory (useful for warm-starting).
     */
    void setInitialTrajectory(const Trajectory& trajectory) {
        current_trajectory_ = trajectory;
    }

    /**
     * @brief Converts current trajectory to matrix format (D x N)
     * @return Matrix where rows are dimensions, columns are timesteps
     */
    Eigen::MatrixXf trajectoryToMatrix() const {
        const size_t N = current_trajectory_.nodes.size();
        const size_t D = num_dimensions_;
        
        Eigen::MatrixXf Y(D, N);
        for (size_t i = 0; i < N; ++i) {
            if (current_trajectory_.nodes[i].position.size() != D) {
                throw std::runtime_error(
                    "Node " + std::to_string(i) + " has dimension " + 
                    std::to_string(current_trajectory_.nodes[i].position.size()) +
                    " but expected " + std::to_string(D)
                );
            }
            Y.col(i) = Eigen::MatrixXf(current_trajectory_.nodes[i].position);
        }

        return Y;
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
    
    /**
     * @brief Virtual getter for obstacles (needed for visualization/cost from base class).
     */
    virtual const std::vector<ObstacleND>& getObstacles() const {
        return obstacles_;
    }

    /**
     * @brief Returns the history of trajectories captured during optimization.
     */
    const std::vector<Trajectory>& getTrajectoryHistory() const {
        return trajectory_history_;
    }

    /**
     * @brief Returns the number of dimensions in the configuration space.
     */
    size_t getNumDimensions() const {
        return num_dimensions_;
    }
    
    /**
     * @brief Sets the forward kinematics model.
     * @param fk Shared pointer to ForwardKinematics object
     */
    void setForwardKinematics(std::shared_ptr<ForwardKinematics> fk) {
        fk_ = fk;
    }


    /**
     * @brief Creates a perturbed trajectory: Ỹ = Y + ε
     * @param Y_k Current trajectory as matrix (D x N)
     * @param epsilon Noise matrix (D x N)
     * @return Perturbed trajectory
     */
    Trajectory createPerturbedTrajectory(const Eigen::MatrixXf& Y_k, 
                                         const Eigen::MatrixXf& epsilon) const {
        Eigen::MatrixXf Y_perturbed = Y_k + epsilon;
        return matrixToTrajectory(Y_perturbed);
    }
    
    /**
     * @brief Applies forward kinematics to convert config space to workspace.
     * @param config_traj Trajectory in configuration space
     * @return Trajectory in workspace
     */
    Trajectory applyForwardKinematics(const Trajectory& config_traj) const {
        if (!fk_) {
            throw std::runtime_error("Forward kinematics not set");
        }
        return fk_->apply(config_traj);
    }
    
    /**
     * @brief Computes collision cost in workspace coordinates.
     * Applies forward kinematics before collision checking.
     * @param config_traj Trajectory in configuration space
     * @param obstacles Obstacles in workspace
     * @return Collision cost
     */
    float computeCollisionCostFK(const Trajectory& config_traj, 
                                  const std::vector<ObstacleND>& obstacles) const {
        // Convert to workspace coordinates
        Trajectory workspace_traj = applyForwardKinematics(config_traj);
        
        // Compute collision cost in workspace
        return computeCollisionCost(workspace_traj, obstacles);
    }


    /**
     * @brief Initializes the log file with timestamp-based filename
     * @param prefix Optional prefix for the log filename (e.g., "PCEM", "NGD")
     * @param log_dir Optional directory path (default: current directory)
     */
    void initializeLogger(const std::string& prefix = "planner", 
                     const std::string& log_dir = "./logs") {
        // ✓ Check if logging is enabled in config
        if (config_["experiment"] && config_["experiment"]["enable_logging"]) {
            bool enable_logging_config = config_["experiment"]["enable_logging"].as<bool>();
            if (!enable_logging_config) {
                logging_enabled_ = false;
                return;  // Skip logger initialization
            }
        }
        
        // Create logs directory if needed
        std::filesystem::path log_path(log_dir);
        try {
            if (!std::filesystem::exists(log_path)) {
                std::filesystem::create_directories(log_path);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error creating log directory: " << e.what() << "\n";
            logging_enabled_ = false;
            return;
        }
        
        // Generate timestamp-based filename
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << log_dir << "/" << prefix << "_" 
        << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S")
        << "_" << std::setfill('0') << std::setw(3) << ms.count()
        << ".log";
        
        log_filename_ = ss.str();
        
        if (log_file_.is_open()) {
            log_file_.close();
        }
        
        log_file_.open(log_filename_, std::ios::out | std::ios::trunc);
        
        if (!log_file_.is_open()) {
            logging_enabled_ = false;
        } else {
            logging_enabled_ = true;
            log("=== Motion Planner Log Started ===");
            log("Log file: " + log_filename_);
            log("Timestamp: " + getCurrentTimestamp());
            log("");
        }
    }


    /**
     * @brief Logs a message to both console and file
     * @param message The message to log
     * @param console_only If true, only print to console (not file)
     */
    void log(const std::string& message, bool console_only = false) {
        // Always print to console
        std::cout << message << std::endl;
        
        // Also write to file if logging is enabled
        if (logging_enabled_ && !console_only && log_file_.is_open()) {
            log_file_ << message << std::endl;
            log_file_.flush();  // Ensure immediate write
        }
    }
    
    /**
     * @brief Logs formatted data (printf-style)
     */
    template<typename... Args>
    void logf(const char* format, Args... args) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), format, args...);
        log(std::string(buffer));
    }
    
    /**
     * @brief Logs a separator line
     */
    void logSeparator(char c = '=', size_t length = 60) {
        log(std::string(length, c));
    }
    
    /**
     * @brief Closes the log file
     */
    void closeLog() {
        if (log_file_.is_open()) {
            log("");
            log("=== Motion Planner Log Ended ===");
            log("Timestamp: " + getCurrentTimestamp());
            log_file_.close();
            logging_enabled_ = false;
        }
    }
    
    /**
     * @brief Gets the current log filename
     */
    std::string getLogFilename() const {
        return log_filename_;
    }
    
    /**
     * @brief Checks if logging is enabled
     */
    bool isLoggingEnabled() const {
        return logging_enabled_;
    }


protected:
    bool is_initialized_ = false;
    Trajectory current_trajectory_;
    PathNode start_node_;
    PathNode goal_node_;
    size_t num_dimensions_;  // Dimensionality of configuration space

    // configuration parameters
    YAML::Node config_;
    std::string config_file_path_;
    unsigned int random_seed_ = 42;
    size_t num_obstacles_initial_ = 0;
    std::unique_ptr<ObstacleMap> obstacle_map_;

    std::vector<ObstacleND> obstacles_;
    Eigen::SparseMatrix<float> R_matrix_;
    float total_time_ = 0.0f;
    size_t num_nodes_ = 0;
    float epsilon_sdf_ = 20.0f;
    float sigma_obs_ = 1.0f;
    
    // Forward kinematics model
    std::shared_ptr<ForwardKinematics> fk_;

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> L_solver_;
    
    std::vector<Trajectory> trajectory_history_; // Stores trajectory at each iteration
    
    // Random engine is mutable so it can be used in const member functions
    mutable std::mt19937 random_engine_;


    /**
     * @brief Load configuration from YAML file
     */
    bool loadConfiguration(const std::string& config_file_path) {
        try {
            config_file_path_ = config_file_path;
            config_ = YAML::LoadFile(config_file_path);
            
            std::cout << "Configuration loaded from: " << config_file_path << "\n";

            if (config_["motion_planning"] && config_["motion_planning"]["num_dimensions"]) {
                num_dimensions_ = config_["motion_planning"]["num_dimensions"].as<size_t>();
                std::cout << "State dimensions: " << num_dimensions_ << "D\n";
            } else {
                std::cerr << "Error: num_dimensions not found in config\n";
                return false;
            }
            
            // Read random seed
            if (config_["experiment"] && config_["experiment"]["random_seed"]) {
                random_seed_ = config_["experiment"]["random_seed"].as<unsigned int>();
                seedRandomEngine(random_seed_);
                std::cout << "Using random seed: " << random_seed_ << "\n";
            }
            
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading config: " << e.what() << "\n";
            return false;
        }
    }


    /**
     * @brief Initialize trajectory from config
     */
    bool initializeTrajectory() {
        try {
            const YAML::Node& mp = config_["motion_planning"];
            size_t num_nodes = mp["num_discretization"].as<size_t>();
            total_time_ = mp["total_time"].as<float>();
            
            // Initialize with linear interpolation
            initialize(num_dimensions_, start_node_, goal_node_, num_nodes, total_time_, 
                      InterpolationMethod::LINEAR, *obstacle_map_, 0.0f);
            
            std::cout << "Trajectory initialized (" << num_dimensions_ << "D): " 
                  << num_nodes << " nodes over " << total_time_ << "s\n";
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing trajectory: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * @brief Saves a snapshot of the current trajectory to the history.
     */
    void storeTrajectory() {
        trajectory_history_.push_back(current_trajectory_);
    }

    // Virtual function for optimization criteria (subclasses implement this)
    virtual bool checkConvergence() const {
        // Default implementation for basic example
        return false;
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

protected:
    // Logging members
    std::ofstream log_file_;
    std::string log_filename_;
    bool logging_enabled_ = false;
    
    /**
     * @brief Gets current timestamp as string
     */
    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }


    /**
     * @brief Logs the base configuration parameters
     */
    void logConfiguration() {
        logSeparator('=');
        log("CONFIGURATION");
        logSeparator('=');
        log("");
        
        log("--- Motion Planning Parameters ---");
        logf("  Dimensions:           %zu", num_dimensions_);
        logf("  Number of nodes:      %zu", current_trajectory_.nodes.size());
        logf("  Total time:           %.2f s", total_time_);
        logf("  Node collision radius: %.2f", current_trajectory_.nodes.empty() ? 0.0f : current_trajectory_.nodes[0].radius);
        log("");
        
        log("--- Start/Goal Configuration ---");
        logf("  Start position:       [%.2f, %.2f]", start_node_.position(0), start_node_.position(1));
        logf("  Goal position:        [%.2f, %.2f]", goal_node_.position(0), goal_node_.position(1));
        logf("  Start-goal distance:  %.2f", (goal_node_.position - start_node_.position).norm());
        log("");
        
        log("--- Obstacle Map ---");
        logf("  Number of obstacles:  %zu", obstacles_.size());
        if (!obstacles_.empty()) {
            float min_radius = obstacles_[0].radius;
            float max_radius = obstacles_[0].radius;
            for (const auto& obs : obstacles_) {
                min_radius = std::min(min_radius, obs.radius);
                max_radius = std::max(max_radius, obs.radius);
            }
            logf("  Obstacle radius range: [%.2f, %.2f]", min_radius, max_radius);
        }
        log("");
        
        log("--- Smoothness Matrix ---");
        logf("  R matrix size:        %ld x %ld", R_matrix_.rows(), R_matrix_.cols());
        logf("  R matrix non-zeros:   %ld", R_matrix_.nonZeros());
        log("");
        
        // Call derived class to log planner-specific config
        logPlannerSpecificConfig();
        
        logSeparator('=');
        log("");
    }

    /**
     * @brief Setup environment: create obstacle map and define start/goal
     */
    bool setupEnvironment() {
        try {
            // Read environment config
            const YAML::Node& env = config_["environment"];
            
            float map_width = env["map_width"].as<float>();
            float map_height = env["map_height"].as<float>();
            float map_depth = (num_dimensions_ >= 3 && env["map_depth"]) ? 
                          env["map_depth"].as<float>() : 0.0f;
            size_t num_obstacles = env["num_obstacles"].as<size_t>();
            float obstacle_radius = env["obstacle_radius"].as<float>();
            float clearance_dist = env["clearance_distance"].as<float>();
            
            // Create obstacle map
            obstacle_map_ = std::make_unique<ObstacleMap>(num_dimensions_, random_seed_);
            if (num_dimensions_ == 2) {
                obstacle_map_->setMapSize(map_width, map_height);
            } else if (num_dimensions_ == 3) {
                obstacle_map_->setMapSize(map_width, map_height, map_depth);
            }
            
            std::cout << "Creating ObstacleMap with dimensions: " << num_dimensions_ << "\n";
            obstacle_map_->generateRandom(num_obstacles, obstacle_radius);
            
            num_obstacles_initial_ = obstacle_map_->size();
            std::cout << "Generated " << num_obstacles_initial_ << " obstacles\n";

            // Read start/goal from config
            auto start_vec = env["start_position"].as<std::vector<float>>();
            auto goal_vec = env["goal_position"].as<std::vector<float>>();

            std::cout << "start_vec.size(): " << start_vec.size() << "\n";
            std::cout << "goal_vec.size(): " << goal_vec.size() << "\n";
            std::cout << "num_dimensions_: " << num_dimensions_ << "\n";
            
            float node_radius = config_["motion_planning"]["node_collision_radius"].as<float>();
            
            if (num_dimensions_ == 2){
                start_node_ = PathNode(start_vec[0], start_vec[1], node_radius);
                goal_node_ = PathNode(goal_vec[0], goal_vec[1], node_radius);
            } else if (num_dimensions_ == 3) {
                start_node_ = PathNode(start_vec[0], start_vec[1], start_vec[2], node_radius);
                goal_node_ = PathNode(goal_vec[0], goal_vec[1], goal_vec[2], node_radius);
            } else {
                throw std::runtime_error("Invalid number of dimensions");
            }

            std::cout << "After PathNode creation:\n";
            std::cout << "  start_node_.position.size(): " << start_node_.position.size() << "\n";
            std::cout << "  goal_node_.position.size(): " << goal_node_.position.size() << "\n";
            
            // Update FK
            fk_ = std::make_shared<IdentityFK>(num_dimensions_);

            // Validate
            if (start_vec.size() != num_dimensions_) {
                std::cerr << "ERROR: start_vec.size() != num_dimensions_\n";
                return false;
            }
            
            if (goal_vec.size() != num_dimensions_) {
                std::cerr << "ERROR: goal_vec.size() != num_dimensions_\n";
                return false;
            }

            std::cout << "About to call clearStartGoalRegions...\n";
            std::cout << "  Obstacle map dimensions: " << obstacle_map_->getDimensions() << "\n";
            std::cout << "  First obstacle dimension: " << (obstacle_map_->getObstacles().empty() ? 0 : obstacle_map_->getObstacles()[0].dimensions()) << "\n";
            
            // Clear obstacles near start/goal
            obstacle_map_->clearStartGoalRegions(start_node_.position, goal_node_.position, clearance_dist);
            
            std::cout << "clearStartGoalRegions completed successfully\n";

            // Store cleared obstacles
            obstacles_ = obstacle_map_->getObstacles();
            
            std::cout << "Environment setup: " << num_obstacles_initial_ 
                      << " obstacles → " << obstacles_.size() 
                      << " after clearance\n";
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error setting up environment: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * @brief Log experiment setup information
     */
    void logExperimentSetup() {
        logSeparator('=');
        log("EXPERIMENT SETUP");
        logSeparator('=');
        log("");
        
        log("--- Configuration File ---");
        log("  Path: " + config_file_path_);
        log("");
        
        log("--- Random Seed ---");
        logf("  Random seed: %u", random_seed_);
        log("  Note: Fixed seed ensures reproducible results");
        log("");
        
        const YAML::Node& env = config_["environment"];
        
        log("--- Map Configuration ---");
        logf("  Map dimensions: %.0f x %.0f", 
             env["map_width"].as<float>(), 
             env["map_height"].as<float>());
        logf("  Initial obstacles: %zu", num_obstacles_initial_);
        logf("  Obstacle radius: %.2f", env["obstacle_radius"].as<float>());
        logf("  Clearance distance: %.2f", env["clearance_distance"].as<float>());
        logf("  Obstacles after clearance: %zu", obstacles_.size());
        logf("  Removed obstacles: %zu", num_obstacles_initial_ - obstacles_.size());
        log("");
        
        logSeparator('-', 60);
        log("");
    }

    /**
     * @brief Log final optimization state
     */
    void logFinalState() {
        log("");
        logSeparator('=');
        log("FINAL RESULTS");
        logSeparator('=');
        log("");
        
        float final_collision = computeCollisionCost(current_trajectory_, obstacles_);
        float final_smoothness = computeSmoothnessCost(current_trajectory_);
        float final_total = final_collision + final_smoothness;
        
        logf("Final Collision Cost:  %.6f", final_collision);
        logf("Final Smoothness Cost: %.6f", final_smoothness);
        logf("Final Total Cost:      %.6f", final_total);
        log("");
        
        // Check if collision-free
        if (final_collision < 1e-6) {
            log("✓ Trajectory is COLLISION-FREE!");
        } else {
            log("⚠ Trajectory has collisions");
        }
        
        log("");
        log("Log saved to: " + getLogFilename());
        logSeparator('=');
    }
    
    /**
     * @brief Virtual method for derived classes to log their specific configuration
     * Override this in PCEM, NGD, STOMP, etc.
     */
    virtual void logPlannerSpecificConfig() {
        // Default: no planner-specific config
    }
    
    /**
     * @brief Logs the full YAML configuration (if available)
     */
    void logYAMLConfig(const YAML::Node& config) {
        log("--- YAML Configuration ---");
        
        if (config["motion_planning"]) {
            log("motion_planning:");
            const auto& mp = config["motion_planning"];
            if (mp["num_dimensions"]) 
                logf("  num_dimensions: %d", mp["num_dimensions"].as<int>());
            if (mp["num_discretization"]) 
                logf("  num_discretization: %d", mp["num_discretization"].as<int>());
            if (mp["total_time"]) 
                logf("  total_time: %.2f", mp["total_time"].as<float>());
            if (mp["node_collision_radius"]) 
                logf("  node_collision_radius: %.2f", mp["node_collision_radius"].as<float>());
            if (mp["random_seed"]) 
                logf("  random_seed: %u", mp["random_seed"].as<unsigned int>());
            log("");
        }
    }


private:
    /**
     * @brief Removes obstacles that are within clearance_radius of the start or goal positions.
     * Works for N-dimensional obstacles using Eigen vector operations.
     * @param obstacles Reference to the obstacles vector (modified in-place).
     * @param start The starting path node.
     * @param goal The goal path node.
     * @param clearance_radius The radius within which obstacles are removed.
     */
    void removeObstaclesNearPoints(std::vector<ObstacleND>& obstacles,
                                   const PathNode& start,
                                   const PathNode& goal,
                                   float clearance_radius) {
        size_t initial_count = obstacles.size();
        
        // Use erase-remove idiom to efficiently remove obstacles
        obstacles.erase(
            std::remove_if(obstacles.begin(), obstacles.end(),
                [&](const ObstacleND& obs) {
                    // Ensure dimensionality matches
                    if (obs.dimensions() != start.dimensions()) {
                        return false;  // Keep obstacles with mismatched dimensions
                    }
                    
                    // Calculate N-dimensional distance from obstacle center to start
                    float dist_to_start = (obs.center - start.position).norm();
                    
                    // Calculate N-dimensional distance from obstacle center to goal
                    float dist_to_goal = (obs.center - goal.position).norm();
                    
                    // Remove if obstacle is within clearance radius of either start or goal
                    return (dist_to_start <= clearance_radius) || 
                           (dist_to_goal <= clearance_radius);
                }),
            obstacles.end()
        );
        
        size_t removed_count = initial_count - obstacles.size();
        if (removed_count > 0) {
            std::cout << "Removed " << removed_count 
                      << " obstacle(s) near start/goal positions.\n";
        }
    }
};
