#pragma once

#include "Trajectory.h"
#include "ObstacleMap.h"
#include <vector> 
#include <random> 

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

/**
 * @brief Abstract base class for motion planning algorithms (e.g., Trajectory Optimization).
 */
class MotionPlanner {
public:
    // Virtual destructor for proper inheritance
    virtual ~MotionPlanner() = default;

    /**
     * @brief Initializes the trajectory from start to goal using the specified interpolation method.
     * @param start The starting path node.
     * @param goal The goal path node.
     * @param num_steps The number of nodes in the initial trajectory.
     * @param total_time The time duration of the trajectory.
     * @param method The interpolation method to use (LINEAR or BEZIER).
     */
    void initialize(const PathNode& start, 
                    const PathNode& goal, 
                    size_t num_steps, 
                    float total_time, 
                    InterpolationMethod method);

    /**
     * @brief Pure virtual function for the optimization loop (algorithm specific).
     * @return True if optimization was successful, false otherwise.
     */
    virtual bool optimize() = 0;

    /**
     * @brief Computes the collision cost of the current trajectory against the environment.
     * @param obstacles The list of static obstacles.
     * @return The total collision cost (non-negative).
     */
    float computeCollisionCost(const std::vector<Obstacle>& obstacles) const;

    /**
     * @brief Computes the L2-norm squared smoothness cost, equivalent to $X^T A^T A X$.
     * @return The total smoothness cost (non-negative).
     */
    float computeSmoothnessCost() const;

    float computeSmoothnessCost(const Trajectory& trajectory) const;

    /**
     * @brief Computes the non-zero diagonals of the smoothness precision matrix R = A^T * A.
     * R is an N x N pentadiagonal symmetric matrix.
     * NOTE: This assumes the start/goal nodes are fixed (constrained).
     * @param N The length of the coordinate vector (number of nodes).
     * @return The RMatrixDiagonals structure.
     */
    RMatrixDiagonals getSmoothnessMatrixRDiagonals(size_t N) const;

    /**
     * @brief Samples a 1D vector from the Gaussian distribution N(0, R^{-1}).
     * This vector represents the smoothness noise applied to one coordinate (X or Y).
     * The implementation relies on the efficient banded structure of R.
     * @param N The size of the vector to sample (number of trajectory nodes).
     * @param rng The random number generator engine.
     * @return A vector of size N containing the sampled noise.
     */
    std::vector<float> sampleSmoothnessNoise(size_t N, std::mt19937& rng) const;

    /**
     * @brief Returns the current optimized trajectory.
     */
    const Trajectory& getCurrentTrajectory() const {
        return current_trajectory_;
    }
    
    /**
     * @brief Virtual getter for obstacles (needed for visualization/cost from base class).
     */
    virtual const std::vector<Obstacle>& getObstacles() const = 0;

    /**
     * @brief Returns the history of trajectories captured during optimization.
     */
    const std::vector<Trajectory>& getTrajectoryHistory() const {
        return trajectory_history_;
    }

protected:
    Trajectory current_trajectory_;
    PathNode start_node_;
    PathNode goal_node_;
    
    std::vector<Trajectory> trajectory_history_; // Stores trajectory at each iteration

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
};

/**
 * @brief Example concrete implementation of a Motion Planner (for basic use).
 */
class BasicPlanner : public MotionPlanner {
public:
    BasicPlanner(const std::vector<Obstacle>& obs) : obstacles_(obs) {}

    // Simple implementation: just initialize and return (no real optimization yet)
    bool optimize() override;

    const std::vector<Obstacle>& getObstacles() const override {
        return obstacles_;
    }

private:
    const std::vector<Obstacle>& obstacles_;
};
