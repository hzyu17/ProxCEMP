// ========================================================================
// prox_ce_planner.h - Complete ROS 1 Header
// ProxCE: Proximal Cross-Entropy Motion Planner
// ========================================================================

#ifndef PROX_CE_PLANNER_H
#define PROX_CE_PLANNER_H

#include <ros/ros.h>  // ROS 1
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <class_loader/class_loader.hpp>
#include <Eigen/Core>
#include <vector>
#include <string>

namespace prox_ce_planner {

/**
 * @brief ProxCE motion planner plugin for MoveIt (ROS 1)
 * 
 * Implements a stochastic trajectory optimization approach using
 * the Cross-Entropy method with noisy rollouts for exploration.
 */
class ProxCEPlanner : public planning_interface::PlannerManager {
public:
    ProxCEPlanner();
    virtual ~ProxCEPlanner();
    
    /**
     * @brief Configuration parameters for the planner
     * 
     * This struct is public so ProxCEPlanningContext can access it
     */
    struct Parameters {
        int num_timesteps;              ///< Number of waypoints in trajectory
        int num_iterations;             ///< Maximum optimization iterations
        int num_iterations_after_valid; ///< Continue optimizing after valid path found
        int num_rollouts;               ///< Number of noisy trajectories per iteration
        double noise_stddev;            ///< Standard deviation for Gaussian noise
        double obstacle_cost_weight;    ///< Weight for collision cost
        double smoothness_cost_weight;  ///< Weight for smoothness cost
        double delta_t;                 ///< Time step between waypoints (seconds)
        
        /// Constructor with default values
        Parameters() :
            num_timesteps(60),
            num_iterations(40),
            num_iterations_after_valid(0),
            num_rollouts(30),
            noise_stddev(0.05),
            obstacle_cost_weight(1.0),
            smoothness_cost_weight(0.1),
            delta_t(0.1) {}
    };
    
    // ====================================================================
    // PlannerManager Interface (Required)
    // ====================================================================
    
    /**
     * @brief Initialize the planner
     * @param model Robot model
     * @param ns Parameter namespace
     * @return true if initialization successful
     */
    bool initialize(const robot_model::RobotModelConstPtr& model,
                   const std::string& ns) override;
    
    /**
     * @brief Get description of this planner
     */
    std::string getDescription() const override;
    
    /**
     * @brief Get list of planning algorithms provided
     */
    void getPlanningAlgorithms(std::vector<std::string>& algs) const override;
    
    /**
     * @brief Get a planning context for a specific planning request
     */
    planning_interface::PlanningContextPtr getPlanningContext(
        const planning_scene::PlanningSceneConstPtr& planning_scene,
        const planning_interface::MotionPlanRequest& req,
        moveit_msgs::MoveItErrorCodes& error_code) const override;
    
    /**
     * @brief Check if this planner can service the given request
     */
    bool canServiceRequest(const planning_interface::MotionPlanRequest& req) const override;
    
protected:
    robot_model::RobotModelConstPtr robot_model_;  ///< Robot kinematic model
    ros::NodeHandle nh_;                          ///< ROS node handle
    std::string namespace_;                       ///< Parameter namespace
    Parameters params_;                           ///< Planner parameters
};


/**
 * @brief Planning context that executes the ProxCE algorithm
 * 
 * This class implements the actual planning logic:
 * 1. Generate initial trajectory (linear interpolation)
 * 2. Iteratively optimize using Cross-Entropy method
 * 3. Generate noisy rollouts
 * 4. Compute costs (collision + smoothness)
 * 5. Update trajectory using weighted average
 */
class ProxCEPlanningContext : public planning_interface::PlanningContext {
public:
    /**
     * @brief Constructor
     * @param name Planner name
     * @param group_name Planning group name
     * @param model Robot model
     * @param params Planner parameters
     */
    ProxCEPlanningContext(const std::string& name,
                         const std::string& group_name,
                         const robot_model::RobotModelConstPtr& model,
                         const ProxCEPlanner::Parameters& params);
    
    virtual ~ProxCEPlanningContext();
    
    // ====================================================================
    // PlanningContext Interface (Required)
    // ====================================================================
    
    /**
     * @brief Solve the planning problem
     * @param res Motion plan response
     * @return true if planning succeeded
     */
    bool solve(planning_interface::MotionPlanResponse& res) override;
    
    /**
     * @brief Solve and return detailed response
     */
    bool solve(planning_interface::MotionPlanDetailedResponse& res) override;
    
    /**
     * @brief Request termination of planning
     */
    bool terminate() override;
    
    /**
     * @brief Clear all internal state
     */
    void clear() override;
    
private:
    // ====================================================================
    // Core Planning Algorithm
    // ====================================================================
    
    /**
     * @brief Main planning function
     */
    bool planTrajectory(planning_interface::MotionPlanResponse& res);
    
    /**
     * @brief Generate initial trajectory via linear interpolation
     * @param trajectory Output trajectory matrix (timesteps x joints)
     */
    void generateInitialTrajectory(Eigen::MatrixXd& trajectory);
    
    /**
     * @brief Optimize trajectory using Cross-Entropy method
     * @param trajectory Input/output trajectory to optimize
     * @return true if valid collision-free trajectory found
     */
    bool optimizeTrajectory(Eigen::MatrixXd& trajectory);
    
    /**
     * @brief Generate noisy trajectory rollouts for exploration
     * @param trajectory Current trajectory
     * @param rollouts Output vector of noisy trajectories
     */
    void generateNoisyRollouts(const Eigen::MatrixXd& trajectory,
                              std::vector<Eigen::MatrixXd>& rollouts);
    
    // ====================================================================
    // Cost Functions
    // ====================================================================
    
    /**
     * @brief Compute total cost of trajectory
     */
    double computeTrajectoryCost(const Eigen::MatrixXd& trajectory);
    
    /**
     * @brief Compute collision cost (number of colliding waypoints)
     */
    double computeCollisionCost(const Eigen::MatrixXd& trajectory);
    
    /**
     * @brief Compute smoothness cost (acceleration squared)
     */
    double computeSmoothnessCost(const Eigen::MatrixXd& trajectory);
    
    // ====================================================================
    // Update and Validation
    // ====================================================================
    
    /**
     * @brief Update trajectory using weighted average of rollouts
     * @param trajectory Current trajectory to update
     * @param rollouts Vector of rollout trajectories
     * @param costs Cost for each rollout
     */
    void updateTrajectory(Eigen::MatrixXd& trajectory,
                         const std::vector<Eigen::MatrixXd>& rollouts,
                         const std::vector<double>& costs);
    
    /**
     * @brief Check if trajectory is collision-free
     */
    bool isTrajectoryValid(const Eigen::MatrixXd& trajectory);
    
    /**
     * @brief Convert Eigen matrix to RobotTrajectory
     */
    void matrixToRobotTrajectory(const Eigen::MatrixXd& matrix,
                                robot_trajectory::RobotTrajectory& traj);
    
    // ====================================================================
    // Member Variables
    // ====================================================================
    
    robot_model::RobotModelConstPtr robot_model_;            ///< Robot model
    planning_scene::PlanningSceneConstPtr planning_scene_;   ///< Planning scene
    ProxCEPlanner::Parameters params_;                       ///< Planner parameters
    
    std::vector<std::string> joint_names_;  ///< Names of joints in planning group
    int num_joints_;                        ///< Number of joints
    bool terminated_;                       ///< Flag for early termination
};

} // namespace prox_ce_planner

#endif // PROX_CE_PLANNER_H