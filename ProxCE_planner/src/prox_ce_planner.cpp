// ========================================================================
// prox_ce_planner.cpp - Complete ROS 1 Implementation
// ProxCE: Proximal Cross-Entropy Motion Planner
// ========================================================================

#include "prox_ce_planner.h"
#include <ros/ros.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <random>
#include <algorithm>
#include <limits>

namespace prox_ce_planner {

// ========================================================================
// ProxCEPlanner Implementation
// ========================================================================

ProxCEPlanner::ProxCEPlanner() : PlannerManager() {
    ROS_DEBUG("ProxCEPlanner constructor called");
}

ProxCEPlanner::~ProxCEPlanner() {
    ROS_DEBUG("ProxCEPlanner destructor called");
}

bool ProxCEPlanner::initialize(const robot_model::RobotModelConstPtr& model,
                               const std::string& ns) {
    robot_model_ = model;
    namespace_ = ns;
    
    // ROS 1: Create NodeHandle with namespace
    nh_ = ros::NodeHandle(ns);
    
    ROS_INFO("Initializing ProxCE Planner in namespace: %s", ns.c_str());
    
    // ROS 1: Load parameters using NodeHandle::param
    // Syntax: nh_.param(param_name, variable, default_value)
    nh_.param("num_timesteps", params_.num_timesteps, 60);
    nh_.param("num_iterations", params_.num_iterations, 40);
    nh_.param("num_iterations_after_valid", params_.num_iterations_after_valid, 0);
    nh_.param("num_rollouts", params_.num_rollouts, 30);
    nh_.param("noise_stddev", params_.noise_stddev, 0.05);
    nh_.param("obstacle_cost_weight", params_.obstacle_cost_weight, 1.0);
    nh_.param("smoothness_cost_weight", params_.smoothness_cost_weight, 0.1);
    nh_.param("delta_t", params_.delta_t, 0.1);
    
    // Log loaded parameters
    ROS_INFO("ProxCE Planner initialized with parameters:");
    ROS_INFO("  num_timesteps: %d", params_.num_timesteps);
    ROS_INFO("  num_iterations: %d", params_.num_iterations);
    ROS_INFO("  num_iterations_after_valid: %d", params_.num_iterations_after_valid);
    ROS_INFO("  num_rollouts: %d", params_.num_rollouts);
    ROS_INFO("  noise_stddev: %.4f", params_.noise_stddev);
    ROS_INFO("  obstacle_cost_weight: %.2f", params_.obstacle_cost_weight);
    ROS_INFO("  smoothness_cost_weight: %.2f", params_.smoothness_cost_weight);
    ROS_INFO("  delta_t: %.2f", params_.delta_t);
    
    return true;
}

std::string ProxCEPlanner::getDescription() const {
    return "ProxCE: Proximal Cross-Entropy Motion Planner";
}

void ProxCEPlanner::getPlanningAlgorithms(std::vector<std::string>& algs) const {
    algs.clear();
    algs.push_back("ProxCE");
}

planning_interface::PlanningContextPtr ProxCEPlanner::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const planning_interface::MotionPlanRequest& req,
    moveit_msgs::MoveItErrorCodes& error_code) const {
    
    error_code.val = moveit_msgs::MoveItErrorCodes::SUCCESS;
    
    if (req.group_name.empty()) {
        ROS_ERROR("ProxCE: No group specified in planning request");
        error_code.val = moveit_msgs::MoveItErrorCodes::INVALID_GROUP_NAME;
        return planning_interface::PlanningContextPtr();
    }
    
    if (!robot_model_->hasJointModelGroup(req.group_name)) {
        ROS_ERROR("ProxCE: Unknown planning group: %s", req.group_name.c_str());
        error_code.val = moveit_msgs::MoveItErrorCodes::INVALID_GROUP_NAME;
        return planning_interface::PlanningContextPtr();
    }
    
    ROS_DEBUG("ProxCE: Creating planning context for group '%s'", req.group_name.c_str());
    
    // Create planning context
    planning_interface::PlanningContextPtr context(
        new ProxCEPlanningContext("prox_ce_planner", req.group_name, 
                                 robot_model_, params_));
    
    context->setPlanningScene(planning_scene);
    context->setMotionPlanRequest(req);
    
    return context;
}

bool ProxCEPlanner::canServiceRequest(const planning_interface::MotionPlanRequest& req) const {
    if (req.goal_constraints.empty()) {
        ROS_ERROR("ProxCE: No goal constraints specified");
        return false;
    }
    
    // Currently only handle joint space goals
    if (req.goal_constraints[0].joint_constraints.empty()) {
        ROS_WARN("ProxCE: Only joint space goals are currently supported");
        return false;
    }
    
    return true;
}

// ========================================================================
// ProxCEPlanningContext Implementation
// ========================================================================

ProxCEPlanningContext::ProxCEPlanningContext(
    const std::string& name,
    const std::string& group_name,
    const robot_model::RobotModelConstPtr& model,
    const ProxCEPlanner::Parameters& params)
    : planning_interface::PlanningContext(name, group_name),
      robot_model_(model),
      params_(params),
      terminated_(false) {
    
    // Get joint model group
    const robot_model::JointModelGroup* jmg = 
        robot_model_->getJointModelGroup(group_name);
    
    if (jmg) {
        joint_names_ = jmg->getActiveJointModelNames();
        num_joints_ = joint_names_.size();
        ROS_DEBUG("ProxCE Context: Planning for %d joints in group '%s'", 
                 num_joints_, group_name.c_str());
    } else {
        ROS_ERROR("ProxCE Context: Failed to get joint model group '%s'", 
                 group_name.c_str());
        num_joints_ = 0;
    }
}

ProxCEPlanningContext::~ProxCEPlanningContext() {
    ROS_DEBUG("ProxCE Context destroyed");
}

bool ProxCEPlanningContext::solve(planning_interface::MotionPlanResponse& res) {
    terminated_ = false;
    ROS_INFO("ProxCE: Starting planning...");
    return planTrajectory(res);
}

bool ProxCEPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res) {
    planning_interface::MotionPlanResponse simple_res;
    bool success = solve(simple_res);
    
    res.trajectory_.push_back(simple_res.trajectory_);
    res.processing_time_.push_back(simple_res.planning_time_);
    res.description_.push_back("ProxCE");
    res.error_code_ = simple_res.error_code_;
    
    return success;
}

bool ProxCEPlanningContext::terminate() {
    ROS_INFO("ProxCE: Planning termination requested");
    terminated_ = true;
    return true;
}

void ProxCEPlanningContext::clear() {
    ROS_DEBUG("ProxCE: Clearing context");
    terminated_ = false;
}

// ========================================================================
// Core Planning Algorithm
// ========================================================================

bool ProxCEPlanningContext::planTrajectory(planning_interface::MotionPlanResponse& res) {
    ros::WallTime start_time = ros::WallTime::now();
    
    if (num_joints_ == 0) {
        ROS_ERROR("ProxCE: No joints to plan for");
        res.error_code_.val = moveit_msgs::MoveItErrorCodes::INVALID_GROUP_NAME;
        return false;
    }
    
    // Initialize trajectory matrix: rows = timesteps, cols = joints
    Eigen::MatrixXd trajectory(params_.num_timesteps, num_joints_);
    
    ROS_INFO("ProxCE: Generating initial trajectory...");
    generateInitialTrajectory(trajectory);
    
    ROS_INFO("ProxCE: Optimizing trajectory (max %d iterations)...", params_.num_iterations);
    bool success = optimizeTrajectory(trajectory);
    
    if (success) {
        ROS_INFO("ProxCE: Planning succeeded!");
        
        // Create robot trajectory
        res.trajectory_.reset(new robot_trajectory::RobotTrajectory(
            robot_model_, getGroupName()));
        
        matrixToRobotTrajectory(trajectory, *res.trajectory_);
        
        // Add time parameterization
        trajectory_processing::IterativeParabolicTimeParameterization time_param;
        if (!time_param.computeTimeStamps(*res.trajectory_, 1.0)) {
            ROS_WARN("ProxCE: Failed to compute time stamps");
        }
        
        res.error_code_.val = moveit_msgs::MoveItErrorCodes::SUCCESS;
        
        ROS_INFO("ProxCE: Generated trajectory with %zu waypoints", 
                res.trajectory_->getWayPointCount());
    } else {
        ROS_WARN("ProxCE: Planning failed to find collision-free solution");
        res.error_code_.val = moveit_msgs::MoveItErrorCodes::PLANNING_FAILED;
    }
    
    res.planning_time_ = (ros::WallTime::now() - start_time).toSec();
    ROS_INFO("ProxCE: Planning time: %.3f seconds", res.planning_time_);
    
    return success;
}

void ProxCEPlanningContext::generateInitialTrajectory(Eigen::MatrixXd& trajectory) {
    const moveit_msgs::RobotState& start_state = request_.start_state;
    const moveit_msgs::Constraints& goal_constraints = request_.goal_constraints[0];
    
    Eigen::VectorXd start_joints(num_joints_);
    Eigen::VectorXd goal_joints(num_joints_);
    
    // Extract start positions
    for (size_t i = 0; i < num_joints_; ++i) {
        const std::string& joint_name = joint_names_[i];
        bool found = false;
        
        for (size_t j = 0; j < start_state.joint_state.name.size(); ++j) {
            if (start_state.joint_state.name[j] == joint_name) {
                start_joints(i) = start_state.joint_state.position[j];
                found = true;
                break;
            }
        }
        
        if (!found) {
            ROS_WARN("ProxCE: Start position not found for joint '%s', using 0.0", 
                    joint_name.c_str());
            start_joints(i) = 0.0;
        }
    }
    
    // Extract goal positions
    for (size_t i = 0; i < num_joints_; ++i) {
        const std::string& joint_name = joint_names_[i];
        bool found = false;
        
        for (const auto& jc : goal_constraints.joint_constraints) {
            if (jc.joint_name == joint_name) {
                goal_joints(i) = jc.position;
                found = true;
                break;
            }
        }
        
        if (!found) {
            ROS_ERROR("ProxCE: Goal position not found for joint '%s'", joint_name.c_str());
            goal_joints(i) = start_joints(i);
        }
    }
    
    // Linear interpolation between start and goal
    for (int t = 0; t < params_.num_timesteps; ++t) {
        double alpha = static_cast<double>(t) / (params_.num_timesteps - 1);
        trajectory.row(t) = (1.0 - alpha) * start_joints + alpha * goal_joints;
    }
    
    ROS_DEBUG("ProxCE: Initial trajectory generated via linear interpolation");
}

bool ProxCEPlanningContext::optimizeTrajectory(Eigen::MatrixXd& trajectory) {
    bool found_valid = false;
    int iterations_after_valid = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    
    for (int iter = 0; iter < params_.num_iterations && !terminated_; ++iter) {
        // Generate noisy rollouts
        std::vector<Eigen::MatrixXd> rollouts;
        generateNoisyRollouts(trajectory, rollouts);
        
        // Compute costs for all rollouts
        std::vector<double> costs(rollouts.size());
        for (size_t i = 0; i < rollouts.size(); ++i) {
            costs[i] = computeTrajectoryCost(rollouts[i]);
        }
        
        // Find best cost
        double iter_best_cost = *std::min_element(costs.begin(), costs.end());
        if (iter_best_cost < best_cost) {
            best_cost = iter_best_cost;
        }
        
        // Update trajectory based on rollout costs
        updateTrajectory(trajectory, rollouts, costs);
        
        // Check if we have a valid trajectory
        if (isTrajectoryValid(trajectory)) {
            if (!found_valid) {
                found_valid = true;
                ROS_INFO("ProxCE: Found valid collision-free trajectory at iteration %d", iter);
            }
            
            iterations_after_valid++;
            if (iterations_after_valid >= params_.num_iterations_after_valid) {
                ROS_INFO("ProxCE: Completed %d iterations after finding valid solution", 
                        iterations_after_valid);
                break;
            }
        }
        
        // Progress logging
        if ((iter + 1) % 10 == 0) {
            ROS_DEBUG("ProxCE: Iteration %d/%d, Best cost: %.4f, Valid: %s", 
                     iter + 1, params_.num_iterations, best_cost, 
                     found_valid ? "Yes" : "No");
        }
    }
    
    if (terminated_) {
        ROS_WARN("ProxCE: Planning was terminated early");
    }
    
    return found_valid;
}

void ProxCEPlanningContext::generateNoisyRollouts(
    const Eigen::MatrixXd& trajectory,
    std::vector<Eigen::MatrixXd>& rollouts) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist(0.0, params_.noise_stddev);
    
    rollouts.clear();
    rollouts.reserve(params_.num_rollouts);
    
    for (int r = 0; r < params_.num_rollouts; ++r) {
        Eigen::MatrixXd noisy_traj = trajectory;
        
        // Add Gaussian noise to all waypoints except start and goal
        for (int t = 1; t < params_.num_timesteps - 1; ++t) {
            for (int j = 0; j < num_joints_; ++j) {
                noisy_traj(t, j) += noise_dist(gen);
            }
        }
        
        rollouts.push_back(noisy_traj);
    }
}

double ProxCEPlanningContext::computeTrajectoryCost(const Eigen::MatrixXd& trajectory) {
    double collision_cost = computeCollisionCost(trajectory);
    double smoothness_cost = computeSmoothnessCost(trajectory);
    
    return params_.obstacle_cost_weight * collision_cost +
           params_.smoothness_cost_weight * smoothness_cost;
}

double ProxCEPlanningContext::computeCollisionCost(const Eigen::MatrixXd& trajectory) {
    double total_cost = 0.0;
    
    robot_state::RobotState robot_state(robot_model_);
    collision_detection::CollisionRequest collision_request;
    collision_detection::CollisionResult collision_result;
    
    collision_request.contacts = false;  // We only need binary collision info
    collision_request.cost = false;
    collision_request.distance = false;
    
    for (int t = 0; t < params_.num_timesteps; ++t) {
        // Set robot state to waypoint
        for (int j = 0; j < num_joints_; ++j) {
            robot_state.setJointPositions(joint_names_[j], &trajectory(t, j));
        }
        robot_state.update();
        
        // Check collision
        collision_result.clear();
        planning_scene_->checkCollision(collision_request, collision_result, robot_state);
        
        if (collision_result.collision) {
            total_cost += 1.0;  // Unit penalty for collision
        }
    }
    
    return total_cost;
}

double ProxCEPlanningContext::computeSmoothnessCost(const Eigen::MatrixXd& trajectory) {
    double total_cost = 0.0;
    
    // Compute acceleration cost (second derivative approximation)
    for (int t = 1; t < params_.num_timesteps - 1; ++t) {
        Eigen::VectorXd acceleration = 
            trajectory.row(t - 1) - 2.0 * trajectory.row(t) + trajectory.row(t + 1);
        total_cost += acceleration.squaredNorm();
    }
    
    return total_cost / (params_.num_timesteps - 2);  // Normalize
}

void ProxCEPlanningContext::updateTrajectory(
    Eigen::MatrixXd& trajectory,
    const std::vector<Eigen::MatrixXd>& rollouts,
    const std::vector<double>& costs) {
    
    // Cross-Entropy Method: Compute probability weights using softmax
    double min_cost = *std::min_element(costs.begin(), costs.end());
    double max_cost = *std::max_element(costs.begin(), costs.end());
    
    // Avoid numerical issues if all costs are the same
    if (max_cost - min_cost < 1e-10) {
        return;  // Keep current trajectory
    }
    
    std::vector<double> exp_costs(costs.size());
    double sum_exp = 0.0;
    
    // Compute exponential weights (lower cost = higher weight)
    for (size_t i = 0; i < costs.size(); ++i) {
        exp_costs[i] = std::exp(-(costs[i] - min_cost) / (max_cost - min_cost + 1e-10));
        sum_exp += exp_costs[i];
    }
    
    // Normalize to probabilities
    std::vector<double> probabilities(costs.size());
    for (size_t i = 0; i < costs.size(); ++i) {
        probabilities[i] = exp_costs[i] / (sum_exp + 1e-10);
    }
    
    // Weighted average of rollouts
    Eigen::MatrixXd new_trajectory = Eigen::MatrixXd::Zero(params_.num_timesteps, num_joints_);
    for (size_t i = 0; i < rollouts.size(); ++i) {
        new_trajectory += probabilities[i] * rollouts[i];
    }
    
    // Update trajectory (keep start and goal fixed)
    for (int t = 1; t < params_.num_timesteps - 1; ++t) {
        trajectory.row(t) = new_trajectory.row(t);
    }
}

bool ProxCEPlanningContext::isTrajectoryValid(const Eigen::MatrixXd& trajectory) {
    // Trajectory is valid if it has no collisions
    return computeCollisionCost(trajectory) < 0.5;
}

void ProxCEPlanningContext::matrixToRobotTrajectory(
    const Eigen::MatrixXd& matrix,
    robot_trajectory::RobotTrajectory& traj) {
    
    for (int t = 0; t < params_.num_timesteps; ++t) {
        robot_state::RobotState state(robot_model_);
        
        // Set joint positions
        for (int j = 0; j < num_joints_; ++j) {
            state.setJointPositions(joint_names_[j], &matrix(t, j));
        }
        
        state.update();
        traj.addSuffixWayPoint(state, params_.delta_t);
    }
}

} // namespace prox_ce_planner

// ========================================================================
// Plugin Registration (MUST be at end of file)
// ========================================================================

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(prox_ce_planner::ProxCEPlanner,
                       planning_interface::PlannerManager)