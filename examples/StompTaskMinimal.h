/**
 * @file StompTaskMinimal.h
 * @brief Ultra-minimal STOMP Task adapter for debugging
 * 
 * This version has NO member variables to isolate the crash.
 */
#pragma once

#include <stomp/task.h>
#include <stomp/utils.h>
#include <Eigen/Dense>
#include <iostream>

namespace pce {

/**
 * @brief Ultra-minimal STOMP Task for debugging
 * 
 * This class has NO member variables to match MinimalTask that works.
 */
class StompTaskMinimal : public stomp::Task {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    
    StompTaskMinimal() {
        std::cout << "StompTaskMinimal constructor\n";
    }
    
    ~StompTaskMinimal() {
        std::cout << "StompTaskMinimal destructor\n";
    }
    
    bool generateNoisyParameters(const MatrixXd& parameters,
                                 std::size_t start_timestep,
                                 std::size_t num_timesteps,
                                 int iteration_number,
                                 int rollout_number,
                                 MatrixXd& parameters_noise,
                                 MatrixXd& noise) override 
    {
        noise = MatrixXd::Random(parameters.rows(), parameters.cols()) * 10.0;
        // Zero out start and end
        noise.col(0).setZero();
        noise.col(noise.cols()-1).setZero();
        parameters_noise = parameters + noise;
        return true;
    }
    
    bool computeNoisyCosts(const MatrixXd& parameters,
                           std::size_t start_timestep,
                           std::size_t num_timesteps,
                           int iteration_number,
                           int rollout_number,
                           VectorXd& costs,
                           bool& validity) override 
    {
        costs = VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    
    bool computeCosts(const MatrixXd& parameters,
                      std::size_t start_timestep,
                      std::size_t num_timesteps,
                      int iteration_number,
                      VectorXd& costs,
                      bool& validity) override 
    {
        costs = VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    
    bool filterNoisyParameters(std::size_t start_timestep,
                               std::size_t num_timesteps,
                               int iteration_number,
                               int rollout_number,
                               MatrixXd& parameters,
                               bool& filtered) override 
    {
        filtered = false;
        return true;
    }
    
    bool filterParameterUpdates(std::size_t start_timestep,
                                std::size_t num_timesteps,
                                int iteration_number,
                                const MatrixXd& parameters,
                                MatrixXd& updates) override 
    {
        return true;
    }
    
    void postIteration(std::size_t start_timestep,
                       std::size_t num_timesteps,
                       int iteration_number,
                       double cost,
                       const MatrixXd& parameters) override 
    {
        if (iteration_number % 10 == 0) {
            std::cout << "  Iteration " << iteration_number << " cost: " << cost << "\n";
        }
    }
    
    void done(bool success, int total_iterations, double final_cost,
              const MatrixXd& parameters) override 
    {
        std::cout << "Optimization " << (success ? "succeeded" : "failed") 
                  << " after " << total_iterations << " iterations\n";
    }
};

} // namespace pce