#include <class_loader/class_loader.hpp>
#include <moveit/planning_interface/planning_interface.hpp>  // Changed .h to .hpp
#include <pcem_planner/pcem_planning_context.hpp>
#include <moveit/utils/logger.hpp>

namespace pcem_planner
{

namespace
{
rclcpp::Logger getLogger()
{
    return moveit::getLogger("pcem_planner_manager");
}
}

class PCEMPlannerManager : public planning_interface::PlannerManager
{
public:
    bool initialize(const moveit::core::RobotModelConstPtr& model,
                    const rclcpp::Node::SharedPtr& node,
                    const std::string& parameter_namespace) override
    {
        robot_model_ = model;
        node_ = node;
        
        // Declare and read parameters
        std::string ns = parameter_namespace.empty() ? "pcem" : parameter_namespace;
        
        node_->declare_parameter(ns + ".num_samples", 3000);
        node_->declare_parameter(ns + ".num_iterations", 50);
        node_->declare_parameter(ns + ".temperature", 1.5);
        node_->declare_parameter(ns + ".eta", 1.0);
        node_->declare_parameter(ns + ".convergence_threshold", 0.01);
        node_->declare_parameter(ns + ".df_resolution", 0.02);
        
        params_.num_samples = node_->get_parameter(ns + ".num_samples").as_int();
        params_.num_iterations = node_->get_parameter(ns + ".num_iterations").as_int();
        params_.temperature = node_->get_parameter(ns + ".temperature").as_double();
        params_.eta = node_->get_parameter(ns + ".eta").as_double();
        params_.convergence_threshold = node_->get_parameter(ns + ".convergence_threshold").as_double();
        params_.df_resolution = node_->get_parameter(ns + ".df_resolution").as_double();
        
        RCLCPP_INFO(getLogger(), "PCEM Planner initialized with:");
        RCLCPP_INFO(getLogger(), "  num_samples: %zu", params_.num_samples);
        RCLCPP_INFO(getLogger(), "  num_iterations: %zu", params_.num_iterations);
        RCLCPP_INFO(getLogger(), "  temperature: %.2f", params_.temperature);
        RCLCPP_INFO(getLogger(), "  eta: %.2f", params_.eta);
        RCLCPP_INFO(getLogger(), "  df_resolution: %.3f m", params_.df_resolution);
        
        return true;
    }
    
    std::string getDescription() const override
    {
        return "Proximal Cross-Entropy Method (PCEM) with MoveIt Distance Fields";
    }
    
    void getPlanningAlgorithms(std::vector<std::string>& algs) const override
    {
        algs = {"PCEM"};
    }
    
    planning_interface::PlanningContextPtr getPlanningContext(
        const planning_scene::PlanningSceneConstPtr& planning_scene,
        const planning_interface::MotionPlanRequest& req,
        moveit_msgs::msg::MoveItErrorCodes& error_code) const override
    {
        auto context = std::make_shared<PCEMPlanningContext>(
            "PCEM", req.group_name, params_);
        
        context->setPlanningScene(planning_scene);
        context->setMotionPlanRequest(req);
        
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
        return context;
    }
    
    bool canServiceRequest(const planning_interface::MotionPlanRequest& req) const override
    {
        return !req.goal_constraints.empty() && !req.group_name.empty();
    }
    
    void setPlannerConfigurations(const planning_interface::PlannerConfigurationMap&) override {}
    
private:
    moveit::core::RobotModelConstPtr robot_model_;
    rclcpp::Node::SharedPtr node_;
    mutable PCEMOptimizer::Parameters params_;
};

} // namespace pcem_planner

CLASS_LOADER_REGISTER_CLASS(pcem_planner::PCEMPlannerManager,
                            planning_interface::PlannerManager)
