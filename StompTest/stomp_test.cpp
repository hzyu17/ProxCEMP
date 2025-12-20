/**
 * @file stomp_test.cpp
 * @brief Minimal test to verify STOMP library installation
 * 
 * Compile with:
 * g++ -std=c++17 stomp_test.cpp -o stomp_test -lstomp -I/path/to/eigen3
 */

#include <stomp/stomp.h>
#include <stomp/task.h>
#include <iostream>
#include <memory>
#include "../examples/StompTaskMinimal.h"   


/**
 * @brief Minimal Task implementation for testing
 */
class MinimalTask : public stomp::Task {
public:
    MinimalTask() {
        std::cout << "MinimalTask created\n";
    }
    
    ~MinimalTask() {
        std::cout << "MinimalTask destroyed\n";
    }
    
    bool generateNoisyParameters(const Eigen::MatrixXd& parameters,
                                 std::size_t start_timestep,
                                 std::size_t num_timesteps,
                                 int iteration_number,
                                 int rollout_number,
                                 Eigen::MatrixXd& parameters_noise,
                                 Eigen::MatrixXd& noise) override 
    {
        noise = Eigen::MatrixXd::Random(parameters.rows(), parameters.cols()) * 0.1;
        parameters_noise = parameters + noise;
        return true;
    }
    
    bool computeNoisyCosts(const Eigen::MatrixXd& parameters,
                           std::size_t start_timestep,
                           std::size_t num_timesteps,
                           int iteration_number,
                           int rollout_number,
                           Eigen::VectorXd& costs,
                           bool& validity) override 
    {
        costs = Eigen::VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    
    bool computeCosts(const Eigen::MatrixXd& parameters,
                      std::size_t start_timestep,
                      std::size_t num_timesteps,
                      int iteration_number,
                      Eigen::VectorXd& costs,
                      bool& validity) override 
    {
        costs = Eigen::VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    
    bool filterNoisyParameters(std::size_t start_timestep,
                               std::size_t num_timesteps,
                               int iteration_number,
                               int rollout_number,
                               Eigen::MatrixXd& parameters,
                               bool& filtered) override 
    {
        filtered = false;
        return true;
    }
    
    bool filterParameterUpdates(std::size_t start_timestep,
                                std::size_t num_timesteps,
                                int iteration_number,
                                const Eigen::MatrixXd& parameters,
                                Eigen::MatrixXd& updates) override 
    {
        return true;
    }
    
    void postIteration(std::size_t start_timestep,
                       std::size_t num_timesteps,
                       int iteration_number,
                       double cost,
                       const Eigen::MatrixXd& parameters) override 
    {
        std::cout << "  Iteration " << iteration_number << " cost: " << cost << "\n";
    }
    
    void done(bool success, int total_iterations, double final_cost,
              const Eigen::MatrixXd& parameters) override 
    {
        std::cout << "Optimization " << (success ? "succeeded" : "failed") 
                  << " after " << total_iterations << " iterations\n";
    }
};


int main() {

    {
        std::cout << "\n=== STOMP Minimal Test ===\n";
        
        // Print size checks
        std::cout << "Size checks:\n";
        std::cout << "  sizeof(stomp::StompConfiguration): " << sizeof(stomp::StompConfiguration) << "\n";
        std::cout << "  sizeof(std::shared_ptr<stomp::Task>): " << sizeof(std::shared_ptr<stomp::Task>) << "\n";
        std::cout << "  sizeof(pce::StompTaskMinimal): " << sizeof(pce::StompTaskMinimal) << "\n";
        
        // Create minimal config
        stomp::StompConfiguration stomp_config;
        stomp_config.num_timesteps = 50;
        stomp_config.num_iterations = 10;
        stomp_config.num_dimensions = 2;
        stomp_config.num_rollouts = 5;
        stomp_config.max_rollouts = 10;
        stomp_config.num_iterations_after_valid = 0;
        stomp_config.delta_t = 0.1;
        stomp_config.control_cost_weight = 0.0;
        stomp_config.exponentiated_cost_sensitivity = 10.0;
        stomp_config.initialization_method = stomp::TrajectoryInitializations::LINEAR_INTERPOLATION;
        
        std::cout << "Config created\n";
        
        // Create minimal task
        auto minimal_task = std::make_shared<pce::StompTaskMinimal>();
        std::cout << "Task created, use_count: " << minimal_task.use_count() << "\n";
        
        // Create STOMP
        std::cout << "Creating STOMP...\n";
        stomp::Stomp* stomp = nullptr;
        try {
            stomp = new stomp::Stomp(stomp_config, minimal_task);
            std::cout << "STOMP created successfully!\n";
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << "\n";
        }
        
        if (stomp) {
            // Run a quick solve
            Eigen::VectorXd start(2), goal(2);
            start << 50.0, 50.0;
            goal << 750.0, 550.0;
            
            Eigen::MatrixXd result;
            bool success = stomp->solve(start, goal, result);
            std::cout << "Minimal test result: " << (success ? "SUCCESS" : "FAILED") << "\n";
            
            delete stomp;
        }
        
        std::cout << "=== Minimal Test Complete ===\n\n";
    }



    std::cout << "=== STOMP Library Test ===\n\n";
    
    // Print sizes for ABI debugging
    std::cout << "Size checks:\n";
    std::cout << "  sizeof(stomp::StompConfiguration): " << sizeof(stomp::StompConfiguration) << "\n";
    std::cout << "  sizeof(std::shared_ptr<stomp::Task>): " << sizeof(std::shared_ptr<stomp::Task>) << "\n";
    std::cout << "  sizeof(Eigen::MatrixXd): " << sizeof(Eigen::MatrixXd) << "\n";
    std::cout << "\n";
    
    // Create configuration
    std::cout << "Creating StompConfiguration...\n";
    stomp::StompConfiguration config;
    config.num_timesteps = 20;
    config.num_iterations = 10;
    config.num_dimensions = 2;
    config.num_rollouts = 5;
    config.max_rollouts = 10;
    config.num_iterations_after_valid = 0;
    config.delta_t = 0.1;
    config.control_cost_weight = 0.0;
    config.exponentiated_cost_sensitivity = 10.0;
    config.initialization_method = stomp::TrajectoryInitializations::LINEAR_INTERPOLATION;
    
    std::cout << "Configuration created:\n";
    std::cout << "  num_timesteps: " << config.num_timesteps << "\n";
    std::cout << "  num_dimensions: " << config.num_dimensions << "\n";
    std::cout << "  num_rollouts: " << config.num_rollouts << "\n";
    std::cout << "  max_rollouts: " << config.max_rollouts << "\n";
    std::cout << "\n";
    
    // Create task
    std::cout << "Creating task...\n";
    auto task = std::make_shared<MinimalTask>();
    std::cout << "Task created, use_count: " << task.use_count() << "\n\n";
    
    // Create STOMP
    std::cout << "Creating STOMP optimizer...\n";
    stomp::Stomp* stomp = nullptr;
    try {
        stomp = new stomp::Stomp(config, task);
        std::cout << "STOMP created successfully!\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    
    // Run optimization
    std::cout << "Running solve()...\n";
    Eigen::VectorXd start(2), goal(2);
    start << 0.0, 0.0;
    goal << 100.0, 100.0;
    
    Eigen::MatrixXd result;
    bool success = stomp->solve(start, goal, result);
    
    std::cout << "\nResult: " << (success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "Result matrix size: " << result.rows() << " x " << result.cols() << "\n";
    
    // Cleanup
    delete stomp;
    
    std::cout << "\n=== Test Complete ===\n";
    return 0;
}