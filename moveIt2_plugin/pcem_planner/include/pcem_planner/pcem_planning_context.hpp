#pragma once

#include <moveit/planning_interface/planning_interface.hpp>  // Changed .h to .hpp
#include <pcem_planner/pcem_optimizer.hpp>

namespace pcem_planner
{

class PCEMPlanningContext : public planning_interface::PlanningContext
{
public:
    PCEMPlanningContext(
        const std::string& name,
        const std::string& group,
        const PCEMOptimizer::Parameters& params);
    
    void solve(planning_interface::MotionPlanResponse& res) override;
    void solve(planning_interface::MotionPlanDetailedResponse& res) override;
    bool terminate() override;
    void clear() override;
    
private:
    PCEMOptimizer::Parameters params_;
    std::atomic<bool> terminate_flag_;
};

} // namespace pcem_planner
