/**
 * @file stomp_standalone_test.cpp
 * @brief Completely standalone STOMP test - NO project dependencies
 * 
 * Compile:
 *   g++ -std=c++17 -fsanitize=address -fno-omit-frame-pointer -g \
 *       stomp_standalone_test.cpp -o stomp_standalone_test \
 *       -lstomp $(pkg-config --cflags --libs eigen3)
 * 
 * Run:
 *   ./stomp_standalone_test
 */

#include <iostream>
#include <memory>
#include <stomp/stomp.h>
#include <stomp/task.h>
#include <Eigen/Dense>

class TestTask : public stomp::Task {
public:
    TestTask() { std::cout << "TestTask created (size=" << sizeof(*this) << ")\n"; }
    ~TestTask() { std::cout << "TestTask destroyed\n"; }
    
    bool generateNoisyParameters(const Eigen::MatrixXd& parameters, std::size_t, std::size_t,
                                 int, int, Eigen::MatrixXd& parameters_noise, Eigen::MatrixXd& noise) override {
        noise = Eigen::MatrixXd::Random(parameters.rows(), parameters.cols()) * 10.0;
        noise.col(0).setZero();
        noise.col(noise.cols()-1).setZero();
        parameters_noise = parameters + noise;
        return true;
    }
    
    bool computeNoisyCosts(const Eigen::MatrixXd&, std::size_t, std::size_t num_timesteps,
                           int, int, Eigen::VectorXd& costs, bool& validity) override {
        costs = Eigen::VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    
    bool computeCosts(const Eigen::MatrixXd&, std::size_t, std::size_t num_timesteps,
                      int, Eigen::VectorXd& costs, bool& validity) override {
        costs = Eigen::VectorXd::Zero(num_timesteps);
        validity = true;
        return true;
    }
    
    bool filterNoisyParameters(std::size_t, std::size_t, int, int, Eigen::MatrixXd&, bool& filtered) override {
        filtered = false;
        return true;
    }
    
    bool filterParameterUpdates(std::size_t, std::size_t, int, const Eigen::MatrixXd&, Eigen::MatrixXd&) override {
        return true;
    }
    
    void postIteration(std::size_t, std::size_t, int iter, double cost, const Eigen::MatrixXd&) override {
        if (iter % 10 == 0) std::cout << "  Iter " << iter << ": cost=" << cost << "\n";
    }
    
    void done(bool success, int total_iterations, double, const Eigen::MatrixXd&) override {
        std::cout << (success ? "SUCCESS" : "FAILED") << " after " << total_iterations << " iterations\n";
    }
};

int main() {
    std::cout << "=== Standalone STOMP Test ===\n";
    std::cout << "sizeof(stomp::StompConfiguration): " << sizeof(stomp::StompConfiguration) << "\n";
    std::cout << "sizeof(TestTask): " << sizeof(TestTask) << "\n\n";
    
    stomp::StompConfiguration cfg;
    cfg.num_timesteps = 50;
    cfg.num_iterations = 100;
    cfg.num_dimensions = 2;
    cfg.num_rollouts = 10;
    cfg.max_rollouts = 20;
    cfg.num_iterations_after_valid = 0;
    cfg.delta_t = 0.1;
    cfg.control_cost_weight = 0.0;
    cfg.exponentiated_cost_sensitivity = 10.0;
    cfg.initialization_method = stomp::TrajectoryInitializations::LINEAR_INTERPOLATION;
    
    std::cout << "Creating task...\n";
    auto task = std::make_shared<TestTask>();
    
    std::cout << "Creating STOMP...\n";
    stomp::Stomp* stomp = new stomp::Stomp(cfg, task);
    std::cout << "STOMP created!\n\n";
    
    Eigen::VectorXd start(2), goal(2);
    start << 50.0, 50.0;
    goal << 750.0, 550.0;
    
    std::cout << "Running solve()...\n";
    Eigen::MatrixXd result;
    stomp->solve(start, goal, result);
    
    std::cout << "\nResult: " << result.rows() << " x " << result.cols() << "\n";
    
    delete stomp;
    std::cout << "=== Test Complete ===\n";
    return 0;
}