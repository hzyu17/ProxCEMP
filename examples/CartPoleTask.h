#pragma once

#include "task.h"
#include <cmath>

namespace pce {

struct CartPoleConfig {
    // System parameters
    float m1 = 0.5f;      // cart mass
    float m2 = 0.5f;      // pole mass
    float l = 0.5f;       // pole length
    float g = 9.81f;      // gravity
    
    // Problem setup
    float T = 5.0f;       // total time
    size_t N = 50;        // number of nodes
    
    // Goal state: swing up to [d, pi, 0, 0]
    float goal_position = 10.0f;
    float goal_angle = M_PI;
    
    // Penalty weights
    float mu_terminal = 1000.0f;    // terminal state penalty
    float mu_control = 1.0f;        // control effort weight
    float mu_bounds = 100.0f;       // bound violation penalty
    
    // State/control bounds
    float p_max = 15.0f;
    float theta_max = 2.0f * M_PI;
    float v_max = 20.0f;
    float omega_max = 20.0f;
    float u_max = 200.0f;
    
    float dt() const { return T / (N - 1); }
};


class CartPoleTask : public Task {
public:
    explicit CartPoleTask(const CartPoleConfig& config = CartPoleConfig())
        : config_(config) {}
    
    void initialize(size_t num_dimensions,
                   const TrajectoryNode& start,
                   const TrajectoryNode& goal,
                   size_t num_nodes,
                   float total_time) override {
        // For single shooting: num_dimensions = 1 (control only)
        // num_nodes = N time steps
        config_.N = num_nodes;
        config_.T = total_time;
        
        // Initial state from start node (if provided)
        x_init_ = Eigen::Vector4f::Zero();
        
        // Goal state
        x_goal_ << config_.goal_position, config_.goal_angle, 0.0f, 0.0f;
    }
    
    /**
     * @brief Cart-pole dynamics: dx/dt = f(x, u)
     */
    Eigen::Vector4f dynamics(const Eigen::Vector4f& x, float u) const {
        float p = x(0), theta = x(1), p_dot = x(2), theta_dot = x(3);
        
        float sin_t = std::sin(theta);
        float cos_t = std::cos(theta);
        float sin2_t = sin_t * sin_t;
        
        float m1 = config_.m1, m2 = config_.m2, l = config_.l, g = config_.g;
        
        // Equations of motion
        float denom = m1 + m2 * sin2_t;
        
        float p_ddot = (l * m2 * sin_t * theta_dot * theta_dot 
                       + u 
                       + m2 * g * cos_t * sin_t) / denom;
        
        float theta_ddot = -(l * m2 * cos_t * sin_t * theta_dot * theta_dot 
                            + u * cos_t 
                            + (m1 + m2) * g * sin_t) / (l * denom);
        
        return Eigen::Vector4f(p_dot, theta_dot, p_ddot, theta_ddot);
    }
    
    /**
     * @brief RK4 integration step
     */
    Eigen::Vector4f rk4Step(const Eigen::Vector4f& x, float u, float dt) const {
        Eigen::Vector4f k1 = dynamics(x, u);
        Eigen::Vector4f k2 = dynamics(x + 0.5f * dt * k1, u);
        Eigen::Vector4f k3 = dynamics(x + 0.5f * dt * k2, u);
        Eigen::Vector4f k4 = dynamics(x + dt * k3, u);
        return x + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    }
    
    /**
     * @brief Forward simulate trajectory given control sequence
     * @param controls Control inputs at each node (N x 1 trajectory)
     * @return Full state trajectory (N x 4)
     */
    std::vector<Eigen::Vector4f> forwardSimulate(const Trajectory& controls) const {
        const size_t N = controls.nodes.size();
        const float dt = config_.dt();
        
        std::vector<Eigen::Vector4f> states(N);
        states[0] = x_init_;
        
        for (size_t k = 0; k < N - 1; ++k) {
            float u = controls.nodes[k].position(0);  // Control is 1D
            states[k + 1] = rk4Step(states[k], u, dt);
        }
        
        return states;
    }
    
    /**
     * @brief Compute total cost for unconstrained formulation
     * 
     * J = J_control + μ_term * J_terminal + μ_bound * J_bounds
     */
    float computeStateCost(const Trajectory& trajectory) const override {
        // Forward simulate to get states
        std::vector<Eigen::Vector4f> states = forwardSimulate(trajectory);
        
        const size_t N = trajectory.nodes.size();
        const float dt = config_.dt();
        
        float cost = 0.0f;
        
        // === 1. Control effort ===
        for (size_t k = 0; k < N - 1; ++k) {
            float u_k = trajectory.nodes[k].position(0);
            float u_k1 = trajectory.nodes[k + 1].position(0);
            cost += config_.mu_control * (dt / 2.0f) * (u_k * u_k + u_k1 * u_k1);
        }
        
        // === 2. Terminal state penalty ===
        Eigen::Vector4f terminal_error = states[N - 1] - x_goal_;
        cost += config_.mu_terminal * terminal_error.squaredNorm();
        
        // === 3. State bound violations ===
        for (size_t k = 0; k < N; ++k) {
            cost += config_.mu_bounds * computeBoundPenalty(states[k], 
                                                            trajectory.nodes[k].position(0));
        }
        
        return cost;
    }
    
    /**
     * @brief Simplified cost for batch evaluation
     */
    std::vector<float> computeStateCostSimple(
        const std::vector<Trajectory>& trajectories) const override {
        
        std::vector<float> costs(trajectories.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < trajectories.size(); ++i) {
            costs[i] = computeStateCost(trajectories[i]);
        }
        
        return costs;
    }
    
    float computeStateCostSimple(const Trajectory& trajectory) const override {
        return computeStateCost(trajectory);
    }
    
private:
    /**
     * @brief Quadratic penalty for bound violations
     */
    float computeBoundPenalty(const Eigen::Vector4f& x, float u) const {
        float penalty = 0.0f;
        
        // Position bounds
        penalty += std::pow(std::max(0.0f, std::abs(x(0)) - config_.p_max), 2);
        
        // Velocity bounds  
        penalty += std::pow(std::max(0.0f, std::abs(x(2)) - config_.v_max), 2);
        penalty += std::pow(std::max(0.0f, std::abs(x(3)) - config_.omega_max), 2);
        
        // Control bounds
        penalty += std::pow(std::max(0.0f, std::abs(u) - config_.u_max), 2);
        
        return penalty;
    }
    
    CartPoleConfig config_;
    Eigen::Vector4f x_init_;
    Eigen::Vector4f x_goal_;
};

} // namespace pce