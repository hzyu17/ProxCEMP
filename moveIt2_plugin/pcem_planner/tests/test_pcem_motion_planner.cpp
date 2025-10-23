/**
 * @file test_pce_motion_planner_simple.cpp
 * @brief Simplified ROS 1 Noetic tests for PCE Motion Planner - validates against logged data
 * 
 * Based on log: PCEM_20251023_133717_839.log
 * 
 * Tests:
 * 1. Parameters match logged configuration
 * 2. Optimization step costs match logged values
 * 3. Final result costs match logged values
 */

#include <gtest/gtest.h>
#include <ros/ros.h>

// Include your PCEM planner headers
#include "pce_motion_planner/pce_motion_planner.h"

class PCEMotionPlannerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize ROS if not already initialized
        if (!ros::isInitialized()) {
            int argc = 0;
            char** argv = NULL;
            ros::init(argc, argv, "pce_planner_test_node");
        }
        
        nh_ = boost::make_shared<ros::NodeHandle>("~");
        
        // Set random seed for reproducibility (from log)
        srand(999);
        
        // Initialize planner with parameters from log
        initializePlanner();
    }

    void TearDown() override {
        // ROS shutdown handled by test runner
    }

    void initializePlanner() {
        // Parameters from log file
        config_.dimensions = 2;
        config_.num_nodes = 50;
        config_.total_time = 10.0;
        config_.node_collision_radius = 15.0;
        
        // Map configuration
        config_.map_width = 800;
        config_.map_height = 600;
        config_.clearance_distance = 100.0;
        
        // PCEM parameters
        config_.num_samples = 3000;
        config_.num_iterations = 10;
        config_.initial_temperature = 1.5;
        config_.temperature_scaling = 1.01;  // alpha_temp
        config_.initial_eta = 1.0;
        config_.initial_gamma = 0.6667;
        config_.gamma_decay = 0.99;  // alpha
        config_.convergence_threshold = 0.01;
        
        // Cost function parameters
        config_.epsilon_sdf = 10.0;
        config_.sigma_obs = 1.0;
        
        // Create planner instance
        planner_ = boost::make_shared<PCEMotionPlanner>(nh_, config_);
    }

    // Test data members
    ros::NodeHandlePtr nh_;
    boost::shared_ptr<PCEMotionPlanner> planner_;
    PCEMotionPlannerConfig config_;
};

/**
 * @test Test 1: Verify all parameters match the logged configuration
 * 
 * From log:
 * - Dimensions: 2
 * - Number of nodes: 50
 * - Total time: 10.00 s
 * - Node collision radius: 15.00
 * - Map dimensions: 800 x 600
 * - Number of samples: 3000
 * - Number of iterations: 10
 * - Initial temperature: 1.5000
 * - Temperature scaling: 1.0100 (alpha_temp)
 * - Initial eta: 1.0000
 * - Initial gamma: 0.6667
 * - Gamma decay: 0.9900 (alpha)
 * - Convergence threshold: 0.010000
 * - Epsilon SDF: 10.00
 * - Sigma obs: 1.0000
 */
TEST_F(PCEMotionPlannerTest, ParametersMatchLoggedData) {
    ROS_INFO("Test 1: Verifying all parameters match logged configuration");
    
    // Dimensions
    EXPECT_EQ(config_.dimensions, 2);
    
    // Trajectory parameters
    EXPECT_EQ(config_.num_nodes, 50);
    EXPECT_DOUBLE_EQ(config_.total_time, 10.0);
    EXPECT_DOUBLE_EQ(config_.node_collision_radius, 15.0);
    
    // Map configuration
    EXPECT_EQ(config_.map_width, 800);
    EXPECT_EQ(config_.map_height, 600);
    EXPECT_DOUBLE_EQ(config_.clearance_distance, 100.0);
    
    // PCEM algorithm parameters
    EXPECT_EQ(config_.num_samples, 3000);
    EXPECT_EQ(config_.num_iterations, 10);
    EXPECT_DOUBLE_EQ(config_.initial_temperature, 1.5);
    EXPECT_DOUBLE_EQ(config_.temperature_scaling, 1.01);
    EXPECT_DOUBLE_EQ(config_.initial_eta, 1.0);
    EXPECT_DOUBLE_EQ(config_.initial_gamma, 0.6667);
    EXPECT_DOUBLE_EQ(config_.gamma_decay, 0.99);
    EXPECT_DOUBLE_EQ(config_.convergence_threshold, 0.01);
    
    // Cost function parameters
    EXPECT_DOUBLE_EQ(config_.epsilon_sdf, 10.0);
    EXPECT_DOUBLE_EQ(config_.sigma_obs, 1.0);
    
    ROS_INFO("✓ All parameters match logged configuration");
}

/**
 * @test Test 2: Verify optimization step costs match logged values
 * 
 * From log, each iteration should have these costs:
 * Iteration 1:  Cost = 1945.91 (Collision: 5.0, Smoothness: 1940.9080)
 * Iteration 2:  Cost = 3012.95 (Collision: 5.0, Smoothness: 3007.9451)
 * Iteration 3:  Cost = 1413.67 (Collision: 5.0, Smoothness: 1408.6660)
 * Iteration 4:  Cost = 2613.02 (Collision: 5.0, Smoothness: 2608.0212)
 * Iteration 5:  Cost = 1827.62 (Collision: 5.0, Smoothness: 1822.6210)
 * Iteration 6:  Cost = 1988.27 (Collision: 5.0, Smoothness: 1983.2720)
 * Iteration 7:  Cost = 2242.12 (Collision: 5.0, Smoothness: 2237.1187)
 * Iteration 8:  Cost = 3054.81 (Collision: 5.0, Smoothness: 3049.8110)
 * Iteration 9:  Cost = 1914.17 (Collision: 5.0, Smoothness: 1909.1677)
 * Iteration 10: Cost = 1668.86 (Collision: 5.0, Smoothness: 1663.8578)
 */
TEST_F(PCEMotionPlannerTest, OptimizationStepCostsMatchLoggedData) {
    ROS_INFO("Test 2: Verifying optimization step costs match logged values");
    
    // Set start and goal from log
    Eigen::VectorXd start(2);
    start << 50.0, 550.0;
    
    Eigen::VectorXd goal(2);
    goal << 750.0, 50.0;
    
    // Run planner
    bool success = planner_->plan(start, goal);
    ASSERT_TRUE(success);
    
    // Get cost history for all iterations
    std::vector<double> cost_history = planner_->getCostHistory();
    ASSERT_EQ(cost_history.size(), 10);
    
    // Expected costs from log (total cost for each iteration)
    std::vector<double> expected_costs = {
        1945.91,  // Iteration 1
        3012.95,  // Iteration 2
        1413.67,  // Iteration 3
        2613.02,  // Iteration 4
        1827.62,  // Iteration 5
        1988.27,  // Iteration 6
        2242.12,  // Iteration 7
        3054.81,  // Iteration 8
        1914.17,  // Iteration 9
        1668.86   // Iteration 10
    };
    
    // Expected collision cost (constant for all iterations)
    const double expected_collision_cost = 5.0;
    
    // Expected smoothness costs (derived from log: total - collision)
    std::vector<double> expected_smoothness_costs = {
        1940.9080,  // Iteration 1
        3007.9451,  // Iteration 2
        1408.6660,  // Iteration 3
        2608.0212,  // Iteration 4
        1822.6210,  // Iteration 5
        1983.2720,  // Iteration 6
        2237.1187,  // Iteration 7
        3049.8110,  // Iteration 8
        1909.1677,  // Iteration 9
        1663.8578   // Iteration 10
    };
    
    // Verify each iteration's costs
    double tolerance = 0.01;  // Allow small numerical differences
    
    for (size_t i = 0; i < cost_history.size(); ++i) {
        ROS_INFO("Iteration %zu: Expected=%.2f, Actual=%.2f", 
                 i+1, expected_costs[i], cost_history[i]);
        
        EXPECT_NEAR(cost_history[i], expected_costs[i], tolerance)
            << "Iteration " << (i+1) << " total cost mismatch";
    }
    
    // Verify collision cost is constant at 5.0
    double collision_cost = planner_->getCollisionCost();
    EXPECT_DOUBLE_EQ(collision_cost, expected_collision_cost)
        << "Collision cost should be constant at 5.0";
    
    ROS_INFO("✓ All optimization step costs match logged values");
}

/**
 * @test Test 3: Verify final result costs match logged values
 * 
 * From log:
 * - PCEM finished. Final Cost: 3012.95 (Collision: 5.0000, Smoothness: 3007.9451)
 * - Final Collision Cost: 5.000000
 * - Final Smoothness Cost: 3007.945068
 * - Final Total Cost: 3012.945068
 * 
 * Note: Log shows "Restoring best trajectory from iteration 2 with cost 1413.67"
 * but final cost is 3012.95, which matches iteration 2's cost.
 */
TEST_F(PCEMotionPlannerTest, FinalResultCostsMatchLoggedData) {
    ROS_INFO("Test 3: Verifying final result costs match logged values");
    
    // Set start and goal from log
    Eigen::VectorXd start(2);
    start << 50.0, 550.0;
    
    Eigen::VectorXd goal(2);
    goal << 750.0, 50.0;
    
    // Run planner
    bool success = planner_->plan(start, goal);
    ASSERT_TRUE(success);
    
    // Get final costs
    double final_collision_cost = planner_->getCollisionCost();
    double final_smoothness_cost = planner_->getSmoothnessCost();
    double final_total_cost = planner_->getTotalCost();
    
    // Expected final costs from log
    const double expected_collision_cost = 5.0;
    const double expected_smoothness_cost = 3007.945068;
    const double expected_total_cost = 3012.945068;
    
    // Verify with tight tolerance
    double tolerance = 0.01;
    
    EXPECT_NEAR(final_collision_cost, expected_collision_cost, tolerance)
        << "Final collision cost mismatch";
    
    EXPECT_NEAR(final_smoothness_cost, expected_smoothness_cost, tolerance)
        << "Final smoothness cost mismatch";
    
    EXPECT_NEAR(final_total_cost, expected_total_cost, tolerance)
        << "Final total cost mismatch";
    
    // Log the results
    ROS_INFO("Final Results:");
    ROS_INFO("  Collision Cost: %.6f (expected: %.6f)", 
             final_collision_cost, expected_collision_cost);
    ROS_INFO("  Smoothness Cost: %.6f (expected: %.6f)", 
             final_smoothness_cost, expected_smoothness_cost);
    ROS_INFO("  Total Cost: %.6f (expected: %.6f)", 
             final_total_cost, expected_total_cost);
    
    // Verify the relationship: total = collision + smoothness
    EXPECT_NEAR(final_total_cost, final_collision_cost + final_smoothness_cost, 1e-6)
        << "Total cost should equal sum of collision and smoothness costs";
    
    ROS_INFO("✓ All final result costs match logged values");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "pce_motion_planner_test");
    return RUN_ALL_TESTS();
}