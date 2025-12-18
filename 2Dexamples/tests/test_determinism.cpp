#include <gtest/gtest.h>
#include <filesystem>
#include <cmath>
#include <sstream>
#include "../include/PCEMotionPlanner.h"
#include "../include/NGDMotionPlanner.h"

constexpr float COST_TOLERANCE = 1e-2f;

// Helper to suppress stdout
class SuppressOutput {
public:
    SuppressOutput() : old_cout(std::cout.rdbuf()) {
        std::cout.rdbuf(null_stream.rdbuf());
    }
    
    ~SuppressOutput() {
        std::cout.rdbuf(old_cout);
    }
    
private:
    std::stringstream null_stream;
    std::streambuf* old_cout;
};

std::string getTestConfigPath() {
    std::filesystem::path test_path(__FILE__);
    std::filesystem::path test_dir = test_path.parent_path();
    std::filesystem::path config_path = test_dir / "config.yaml";
    return std::filesystem::canonical(config_path).string();
}

TEST(Determinism, PCEMReproducibleCosts) {
    SuppressOutput suppress;
    
    std::string config_file = getTestConfigPath();
    
    ProximalCrossEntropyMotionPlanner planner;
    bool success = planner.solve(config_file);
    
    ASSERT_TRUE(success);
    
    const auto& history = planner.getTrajectoryHistory();
    
    // ✓ FIRST: Print what's actually stored
    std::cout << "\n=== ACTUAL HISTORY ===\n";
    for (size_t i = 0; i < history.size() && i < 15; ++i) {
        float collision = planner.computeStateCost(history[i], planner.getObstacles());
        float smoothness = planner.computeSmoothnessCost(history[i]);
        float total = collision + smoothness;
        
        std::cout << "history[" << i << "]: "
                  << "total=" << total 
                  << ", collision=" << collision 
                  << ", smoothness=" << smoothness << "\n";
    }
    std::cout << "======================\n\n";
}

TEST(Determinism, PCEMIdenticalRuns) {
    SuppressOutput suppress;  // ✓ Suppress all stdout
    
    std::string config_file = getTestConfigPath();
    
    ProximalCrossEntropyMotionPlanner planner1;
    bool success1 = planner1.solve(config_file);
    ASSERT_TRUE(success1);
    
    const auto& history1 = planner1.getTrajectoryHistory();
    
    ProximalCrossEntropyMotionPlanner planner2;
    bool success2 = planner2.solve(config_file);
    ASSERT_TRUE(success2);
    
    const auto& history2 = planner2.getTrajectoryHistory();
    
    ASSERT_EQ(history1.size(), history2.size());
    
    for (size_t i = 0; i < history1.size(); ++i) {
        float cost1_collision = planner1.computeStateCost(history1[i], planner1.getObstacles());
        float cost1_smoothness = planner1.computeSmoothnessCost(history1[i]);
        
        float cost2_collision = planner2.computeStateCost(history2[i], planner2.getObstacles());
        float cost2_smoothness = planner2.computeSmoothnessCost(history2[i]);
        
        EXPECT_FLOAT_EQ(cost1_collision, cost2_collision);
        EXPECT_FLOAT_EQ(cost1_smoothness, cost2_smoothness);
    }
}

TEST(Determinism, ObstacleGeneration) {
    SuppressOutput suppress;
    
    std::string config_file = getTestConfigPath();
    
    ProximalCrossEntropyMotionPlanner planner1;
    planner1.initializeOnly(config_file);
    
    ProximalCrossEntropyMotionPlanner planner2;
    planner2.initializeOnly(config_file);
    
    const auto& obs1 = planner1.getObstacles();
    const auto& obs2 = planner2.getObstacles();
    
    ASSERT_EQ(obs1.size(), obs2.size());
    
    for (size_t i = 0; i < obs1.size(); ++i) {
        EXPECT_FLOAT_EQ(obs1[i].radius, obs2[i].radius);
        for (size_t d = 0; d < obs1[i].center.size(); ++d) {
            EXPECT_FLOAT_EQ(obs1[i].center(d), obs2[i].center(d));
        }
    }
}

TEST(Determinism, NoiseSampling) {
    SuppressOutput suppress;
    
    std::string config_file = getTestConfigPath();
    
    ProximalCrossEntropyMotionPlanner planner1;
    planner1.initializeOnly(config_file);
    
    ProximalCrossEntropyMotionPlanner planner2;
    planner2.initializeOnly(config_file);
    
    const size_t N = 50;
    const size_t D = 2;
    const size_t M = 100;
    
    auto samples1 = planner1.sampleNoiseMatrices(M, N, D);
    auto samples2 = planner2.sampleNoiseMatrices(M, N, D);
    
    ASSERT_EQ(samples1.size(), samples2.size());
    
    for (size_t m = 0; m < M; ++m) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t n = 0; n < N; ++n) {
                EXPECT_FLOAT_EQ(samples1[m](d, n), samples2[m](d, n));
            }
        }
    }
}

TEST(Determinism, NGDReproducible) {
    SuppressOutput suppress;
    
    std::string config_file = getTestConfigPath();
    
    NGDMotionPlanner planner1;
    bool success1 = planner1.solve(config_file);
    ASSERT_TRUE(success1);
    
    NGDMotionPlanner planner2;
    bool success2 = planner2.solve(config_file);
    ASSERT_TRUE(success2);
    
    const auto& history1 = planner1.getTrajectoryHistory();
    const auto& history2 = planner2.getTrajectoryHistory();
    
    ASSERT_EQ(history1.size(), history2.size());
    
    const Trajectory& final1 = history1.back();
    const Trajectory& final2 = history2.back();
    
    float cost1_collision = planner1.computeStateCost(final1, planner1.getObstacles());
    float cost1_smoothness = planner1.computeSmoothnessCost(final1);
    
    float cost2_collision = planner2.computeStateCost(final2, planner2.getObstacles());
    float cost2_smoothness = planner2.computeSmoothnessCost(final2);
    
    EXPECT_FLOAT_EQ(cost1_collision, cost2_collision);
    EXPECT_FLOAT_EQ(cost1_smoothness, cost2_smoothness);
}

TEST(Determinism, DifferentSeedsProduceDifferentResults) {
    SuppressOutput suppress;
    
    std::string config_file = getTestConfigPath();
    
    ProximalCrossEntropyMotionPlanner planner1;
    planner1.initializeOnly(config_file);
    
    ProximalCrossEntropyMotionPlanner planner2;
    planner2.seedRandomEngine(123);
    
    const size_t N = 50;
    const size_t D = 2;
    const size_t M = 10;
    
    auto samples1 = planner1.sampleNoiseMatrices(M, N, D);
    auto samples2 = planner2.sampleNoiseMatrices(M, N, D);
    
    bool found_difference = false;
    for (size_t m = 0; m < M && !found_difference; ++m) {
        for (size_t d = 0; d < D && !found_difference; ++d) {
            for (size_t n = 0; n < N && !found_difference; ++n) {
                if (std::abs(samples1[m](d, n) - samples2[m](d, n)) > 1e-6) {
                    found_difference = true;
                }
            }
        }
    }
    
    EXPECT_TRUE(found_difference);
}