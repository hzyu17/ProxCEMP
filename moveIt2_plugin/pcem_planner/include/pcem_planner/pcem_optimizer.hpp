#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <random>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/distance_field/propagation_distance_field.hpp>

namespace pcem_planner
{

/**
 * @brief PCEM Optimizer using MoveIt's built-in distance fields
 */
class PCEMOptimizer
{
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    struct Parameters
    {
        size_t num_samples;
        size_t num_iterations;
        float temperature;
        float eta;
        float convergence_threshold;
        float alpha;
        float alpha_temp;
        float smoothness_weight;
        float collision_weight;
        unsigned int random_seed;
        
        // Distance field parameters
        double df_resolution;
        double df_size_x;
        double df_size_y;
        double df_size_z;
        double df_origin_x;
        double df_origin_y;
        double df_origin_z;
        
        // Default constructor (defined in .cpp)
        Parameters();
    };
    
    // Default constructor using default parameters
    PCEMOptimizer();
    
    // Constructor with parameters
    explicit PCEMOptimizer(const Parameters& params);
    
    /**
     * @brief Optimize trajectory using PCEM
     */
    bool optimize(
        const planning_scene::PlanningSceneConstPtr& planning_scene,
        const moveit::core::JointModelGroup* group,
        const Eigen::VectorXd& start_config,
        const Eigen::VectorXd& goal_config,
        size_t num_waypoints,
        std::vector<Eigen::VectorXd>& output_trajectory);
    
    // Getters
    const std::vector<std::vector<Eigen::VectorXd>>& getTrajectoryHistory() const 
    { 
        return trajectory_history_; 
    }
    
    float getBestCost() const { return best_cost_; }
    size_t getBestIteration() const { return best_iteration_; }
    
    // Exposed for testing
    SparseMatrixXf buildRegularizationMatrix(size_t num_waypoints) const;
    MatrixXf trajectoryToMatrix(const std::vector<Eigen::VectorXd>& trajectory) const;
    
private:
    Parameters params_;
    mutable std::mt19937 random_engine_;
    float gamma_;
    
    // Distance field
    std::shared_ptr<distance_field::PropagationDistanceField> distance_field_;
    
    // Collision geometry
    std::vector<std::string> important_links_;
    
    // History
    std::vector<std::vector<Eigen::VectorXd>> trajectory_history_;
    float best_cost_;
    size_t best_iteration_;
    
    // Helper functions
    void buildDistanceField(const planning_scene::PlanningSceneConstPtr& planning_scene);
    
    std::vector<Eigen::VectorXd> matrixToTrajectory(const MatrixXf& matrix) const;
    
    std::vector<MatrixXf> sampleNoiseMatrices(
        size_t num_samples,
        size_t num_waypoints,
        size_t num_joints) const;
    
    // --- TEMPORARY FUNCTION FOR TESTING CORE LOGIC ---
    float computeCollisionCost() const;
    // ---------------------------------------------------
    
    float computeSmoothnessCost(
        const std::vector<Eigen::VectorXd>& trajectory) const;
};

} // namespace pcem_planner
