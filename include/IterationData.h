#pragma once

#include <vector>
#include <Eigen/Core>
#include "Trajectory.h"

/**
 * @brief Data collected at each optimization iteration
 */
struct IterationData {
    int iteration;
    Trajectory mean_trajectory;
    std::vector<Trajectory> samples;  // Can be empty if not storing samples
    float total_cost;
    float collision_cost;
    float smoothness_cost;
    float best_sample_cost;
    float worst_sample_cost;
    float mean_sample_cost;
    
    // Timing (optional)
    double iteration_time_ms;
};

/**
 * @brief Container for full optimization history
 */
struct OptimizationHistory {
    std::vector<IterationData> iterations;
    
    // Final results
    Trajectory final_trajectory;
    float final_cost;
    bool converged;
    int total_iterations;
    double total_time_ms;
    
    // Methods
    void clear() {
        iterations.clear();
        converged = false;
        total_iterations = 0;
        total_time_ms = 0.0;
    }
    
    void addIteration(const IterationData& data) {
        iterations.push_back(data);
    }
    
    // Get cost history as vector (for plotting)
    std::vector<float> getCostHistory() const {
        std::vector<float> costs;
        costs.reserve(iterations.size());
        for (const auto& it : iterations) {
            costs.push_back(it.total_cost);
        }
        return costs;
    }
    
    std::vector<float> getCollisionCostHistory() const {
        std::vector<float> costs;
        costs.reserve(iterations.size());
        for (const auto& it : iterations) {
            costs.push_back(it.collision_cost);
        }
        return costs;
    }
    
    std::vector<float> getSmoothnessCostHistory() const {
        std::vector<float> costs;
        costs.reserve(iterations.size());
        for (const auto& it : iterations) {
            costs.push_back(it.smoothness_cost);
        }
        return costs;
    }
};
