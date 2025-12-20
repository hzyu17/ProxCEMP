#include "PCEMotionPlanner.h"
#include "../examples/CartPoleTask.h"

int main() {
    // 1. Create cart-pole task
    pce::CartPoleConfig cart_config;
    cart_config.N = 50;
    cart_config.T = 5.0f;
    cart_config.mu_terminal = 1000.0f;
    
    auto task = std::make_shared<pce::CartPoleTask>(cart_config);
    
    // 2. Create PCE planner
    ProximalCrossEntropyMotionPlanner planner(task);
    
    // 3. Configure PCE
    PCEConfig pce_config;
    pce_config.num_dimensions = 1;           // Control is 1D
    pce_config.num_discretization = cart_config.N;    // Time steps
    pce_config.total_time = cart_config.T;
    pce_config.num_samples = 500;
    pce_config.num_iterations = 200;
    pce_config.temperature = 2.0f;
    pce_config.temperature_final = 0.1f;
    pce_config.cov_scale_initial = 10.0f;    // Large initial exploration for controls
    pce_config.cov_scale_final = 0.1f;
    
    // Start/goal in control space (initial guess: zero control)
    pce_config.start_position = std::vector<float>(1, 0.0f);
    pce_config.goal_position = std::vector<float>(1, 0.0f);
    
    planner.initialize(pce_config);
    
    // 4. Solve
    planner.solve();
    
    // 5. Extract optimal control sequence
    Trajectory optimal_controls = planner.getCurrentTrajectory();
    
    // 6. Reconstruct full state trajectory
    auto states = task->forwardSimulate(optimal_controls);
    
    return 0;
}