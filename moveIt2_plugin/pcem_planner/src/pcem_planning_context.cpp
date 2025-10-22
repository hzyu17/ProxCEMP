#include <pcem_planner/pcem_planning_context.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/constraint_samplers/constraint_sampler_manager.hpp>
#include <moveit/utils/logger.hpp>
#include <chrono>

namespace pcem_planner
{

namespace
{
rclcpp::Logger getLogger()
{
    return moveit::getLogger("pcem_planning_context");
}
}

PCEMPlanningContext::PCEMPlanningContext(
    const std::string& name,
    const std::string& group,
    const PCEMOptimizer::Parameters& params)
    : planning_interface::PlanningContext(name, group)
    , params_(params)
    , terminate_flag_(false)
{
}

void PCEMPlanningContext::solve(planning_interface::MotionPlanResponse& res)
{
    auto start_time = std::chrono::steady_clock::now();
    
    res.planner_id = "PCEM";
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    
    RCLCPP_INFO(getLogger(), "Starting PCEM planning");
    
    // Get start and goal states
    const auto& req = getMotionPlanRequest();
    moveit::core::RobotState start_state(
        *getPlanningScene()->getCurrentStateUpdated(req.start_state));
    
    // Sample goal state
    moveit::core::RobotState goal_state(start_state);
    constraint_samplers::ConstraintSamplerManager sampler_manager;
    auto goal_sampler = sampler_manager.selectSampler(
        getPlanningScene(), getGroupName(), req.goal_constraints.at(0));
    
    if (!goal_sampler || !goal_sampler->sample(goal_state))
    {
        RCLCPP_ERROR(getLogger(), "Failed to sample goal state");
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
        return;
    }
    
    // Get joint configurations
    const auto* group = getPlanningScene()->getRobotModel()->getJointModelGroup(getGroupName());
    Eigen::VectorXd start_config, goal_config;
    start_state.copyJointGroupPositions(group, start_config);
    goal_state.copyJointGroupPositions(group, goal_config);
    
    RCLCPP_INFO(getLogger(), "Start config: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
                start_config[0], start_config[1], start_config[2], start_config[3],
                start_config[4], start_config[5], start_config[6]);
    RCLCPP_INFO(getLogger(), "Goal config:  [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
                goal_config[0], goal_config[1], goal_config[2], goal_config[3],
                goal_config[4], goal_config[5], goal_config[6]);
    
    // Create optimizer
    PCEMOptimizer optimizer(params_);
    
    // Optimize
    size_t num_waypoints = 30;
    std::vector<Eigen::VectorXd> optimized_trajectory;
    
    bool success = optimizer.optimize(
        getPlanningScene(),
        group,
        start_config,
        goal_config,
        num_waypoints,
        optimized_trajectory);
    
    if (success)
    {
        // Convert to RobotTrajectory
        res.trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(
            start_state.getRobotModel(), group);
        
        for (const auto& config : optimized_trajectory)
        {
            moveit::core::RobotState state(start_state);
            state.setJointGroupPositions(group, config);
            res.trajectory->addSuffixWayPoint(state, 0.1);
        }
        
        RCLCPP_INFO(getLogger(), 
                    "✓ PCEM succeeded: %zu waypoints, best cost: %.4f (iteration %zu)",
                    res.trajectory->getWayPointCount(),
                    optimizer.getBestCost(),
                    optimizer.getBestIteration());
    }
    else
    {
        RCLCPP_ERROR(getLogger(), "✗ PCEM optimization failed");
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    }
    
    // Planning time
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - start_time);
    res.planning_time = elapsed.count();
    
    RCLCPP_INFO(getLogger(), "Total planning time: %.3f s", res.planning_time);
}

void PCEMPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res)
{
    RCLCPP_ERROR(getLogger(), "Detailed response not implemented");
}

bool PCEMPlanningContext::terminate()
{
    terminate_flag_ = true;
    return true;
}

void PCEMPlanningContext::clear()
{
    terminate_flag_ = false;
}

} // namespace pcem_planner
