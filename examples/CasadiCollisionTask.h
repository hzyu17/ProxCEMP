/**
 * @file CasadiCollisionTask.h
 * @brief CasADi symbolic collision cost functions for trajectory optimization
 * 
 * Provides smooth, differentiable collision costs that enable:
 * - Analytical gradients via CasADi automatic differentiation
 * - Direct use with IPOPT/SQP without finite differences
 * - Smooth approximations of hinge loss for better convergence
 */
#pragma once

#include "CollisionAvoidanceTask.h"
#include <casadi/casadi.hpp>

namespace pce {

/**
 * @brief Utility class for CasADi symbolic collision cost construction
 * 
 * This class provides static methods and a wrapper around CollisionAvoidanceTask
 * to build smooth, differentiable collision costs for use with CasADi-based optimizers.
 */
class CasadiCollisionCost {
public:
    using VectorXf = Eigen::VectorXf;
    using MatrixXf = Eigen::MatrixXf;
    
    // ==================== Smooth Approximation Functions ====================
    
    /**
     * @brief Smooth approximation of max(0, x) using softplus
     * 
     * softplus(x, k) = log(1 + exp(k*x)) / k
     * 
     * Properties:
     * - Smooth everywhere (infinitely differentiable)
     * - Approaches max(0, x) as k -> infinity
     * - k=20 is a good balance between smoothness and accuracy
     * 
     * @param x Input symbolic expression
     * @param k Sharpness parameter (higher = closer to max, but less smooth)
     * @return Smooth approximation of max(0, x)
     */
    static casadi::SX softplus(const casadi::SX& x, double k = 20.0) {
        using namespace casadi;
        // Numerically stable: for large x, softplus(x) ≈ x
        return if_else(x > 10.0/k, x, log(1.0 + exp(k * x)) / k);
    }
    
    /**
     * @brief Smooth max using polynomial transition (very smooth)
     * 
     * For x < 0: returns 0
     * For x > d: returns x - d/2  
     * For 0 <= x <= d: returns x^2/(2d) (smooth quadratic transition)
     * 
     * @param x Input symbolic expression
     * @param d Transition width (smaller = sharper)
     */
    static casadi::SX smoothReluPoly(const casadi::SX& x, double d = 0.5) {
        using namespace casadi;
        return if_else(x < 0, SX::zeros(x.size1(), x.size2()),
               if_else(x > d, x - d/2, x*x/(2*d)));
    }
    
    /**
     * @brief Smooth max using sqrt-based formula
     * 
     * smooth_relu(x) = (sqrt(x^2 + eps) + x) / 2
     * 
     * @param x Input symbolic expression
     * @param eps Smoothing parameter
     */
    static casadi::SX smoothReluSqrt(const casadi::SX& x, double eps = 0.1) {
        using namespace casadi;
        return (sqrt(x*x + eps) + x) / 2.0;
    }
    
    // ==================== Symbolic Cost Builders ====================
    
    /**
     * @brief Build symbolic collision cost for a single 2D/3D position
     * 
     * @param pos_x X coordinate (symbolic)
     * @param pos_y Y coordinate (symbolic)
     * @param pos_z Z coordinate (symbolic, ignored if 2D)
     * @param obs_soa Obstacle data in SoA format
     * @param epsilon_sdf SDF margin
     * @param sigma_obs Cost weight
     * @param scale Optional scaling factor
     * @return Symbolic collision cost for this position
     */
    static casadi::SX buildPositionCost(
        const casadi::SX& pos_x,
        const casadi::SX& pos_y,
        const casadi::SX& pos_z,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        double scale = 1.0) 
    {
        using namespace casadi;
        
        if (obs_soa.empty()) return SX::zeros(1);
        
        const size_t K = obs_soa.num_obstacles;
        const size_t D = obs_soa.num_dimensions;
        
        SX cost = 0;
        
        for (size_t k = 0; k < K; ++k) {
            SX dx = pos_x - static_cast<double>(obs_soa.centers(0, k));
            SX dy = pos_y - static_cast<double>(obs_soa.centers(1, k));
            SX dist_sq = dx*dx + dy*dy;
            
            if (D == 3) {
                SX dz = pos_z - static_cast<double>(obs_soa.centers(2, k));
                dist_sq += dz*dz;
            }
            
            // Add small epsilon to avoid gradient singularity at zero distance
            SX dist = sqrt(dist_sq + 1e-6);
            
            double combined_r = static_cast<double>(obs_soa.combined_radii(k));
            SX sdf = dist - combined_r;
            
            // Smooth hinge: softplus(epsilon - sdf)^2
            SX hinge = softplus(epsilon_sdf - sdf);
            cost += hinge * hinge;
        }
        
        return sigma_obs * scale * cost;
    }
    
    /**
     * @brief Build symbolic collision cost for entire trajectory
     * 
     * @param Y Decision variables: flattened trajectory [(N-2) * D]
     * @param N Total number of trajectory nodes
     * @param D Number of dimensions (2 or 3)
     * @param obs_soa Obstacle data
     * @param epsilon_sdf SDF margin
     * @param sigma_obs Cost weight
     * @param scale Optional scaling factor
     * @return Symbolic collision cost expression
     */
    static casadi::SX buildTrajectoryCost(
        const casadi::SX& Y,
        size_t N, size_t D,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        double scale = 1.0) 
    {
        using namespace casadi;
        
        if (obs_soa.empty()) return SX::zeros(1);
        
        SX total_cost = 0;
        
        // Process interior nodes (skip fixed endpoints)
        for (size_t i = 1; i < N - 1; ++i) {
            size_t idx = (i - 1) * D;
            
            SX pos_x = Y(idx);
            SX pos_y = Y(idx + 1);
            SX pos_z = (D == 3) ? Y(idx + 2) : SX::zeros(1);
            
            total_cost += buildPositionCost(pos_x, pos_y, pos_z, obs_soa, 
                                            epsilon_sdf, sigma_obs, 1.0);
        }
        
        return scale * total_cost;
    }
    
    /**
     * @brief Build symbolic smoothness cost (acceleration squared)
     * 
     * @param Y Decision variables
     * @param N Number of nodes
     * @param D Number of dimensions
     * @param start_pos Start position (fixed)
     * @param goal_pos Goal position (fixed)
     * @param total_time Total trajectory time
     * @param scale Optional scaling factor
     * @return Symbolic smoothness cost
     */
    static casadi::SX buildSmoothnessCost(
        const casadi::SX& Y,
        size_t N, size_t D,
        const VectorXf& start_pos,
        const VectorXf& goal_pos,
        float total_time,
        double scale = 1.0) 
    {
        using namespace casadi;
        
        // Build full trajectory
        std::vector<SX> traj(N * D);
        
        for (size_t d = 0; d < D; ++d)
            traj[d] = static_cast<double>(start_pos(d));
        
        for (size_t i = 0; i < N - 2; ++i)
            for (size_t d = 0; d < D; ++d)
                traj[(i + 1) * D + d] = Y(i * D + d);
        
        for (size_t d = 0; d < D; ++d)
            traj[(N - 1) * D + d] = static_cast<double>(goal_pos(d));
        
        float dt = total_time / (N - 1);
        SX cost = 0;
        
        for (size_t i = 1; i < N - 1; ++i) {
            for (size_t d = 0; d < D; ++d) {
                SX y_prev = traj[(i - 1) * D + d];
                SX y_curr = traj[i * D + d];
                SX y_next = traj[(i + 1) * D + d];
                SX accel = (y_prev - 2*y_curr + y_next) / (dt * dt);
                cost += accel * accel;
            }
        }
        
        return scale * cost;
    }
    
    /**
     * @brief Build combined cost (smoothness + collision)
     */
    static casadi::SX buildTotalCost(
        const casadi::SX& Y,
        size_t N, size_t D,
        const VectorXf& start_pos,
        const VectorXf& goal_pos,
        float total_time,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        float collision_weight,
        double scale = 1.0) 
    {
        casadi::SX smooth = buildSmoothnessCost(Y, N, D, start_pos, goal_pos, total_time, scale);
        casadi::SX coll = buildTrajectoryCost(Y, N, D, obs_soa, epsilon_sdf, sigma_obs, scale);
        return smooth + collision_weight * coll;
    }
    
    // ==================== Pre-compiled Function Factories ====================
    
    /**
     * @brief Create CasADi Function for collision cost + gradient
     */
    static casadi::Function createCollisionCostFunction(
        size_t N, size_t D,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        double scale = 1.0) 
    {
        using namespace casadi;
        
        size_t n_vars = (N - 2) * D;
        SX Y = SX::sym("Y", n_vars);
        
        SX cost = buildTrajectoryCost(Y, N, D, obs_soa, epsilon_sdf, sigma_obs, scale);
        SX grad = gradient(cost, Y);
        
        return Function("coll_cost", {Y}, {cost, grad}, {"Y"}, {"cost", "grad"});
    }
    
    /**
     * @brief Create CasADi Function for total cost + gradient
     */
    static casadi::Function createTotalCostFunction(
        size_t N, size_t D,
        const VectorXf& start_pos,
        const VectorXf& goal_pos,
        float total_time,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        float collision_weight,
        double scale = 1.0) 
    {
        using namespace casadi;
        
        size_t n_vars = (N - 2) * D;
        SX Y = SX::sym("Y", n_vars);
        
        SX cost = buildTotalCost(Y, N, D, start_pos, goal_pos, total_time,
                                 obs_soa, epsilon_sdf, sigma_obs, collision_weight, scale);
        SX grad = gradient(cost, Y);
        
        return Function("total_cost", {Y}, {cost, grad}, {"Y"}, {"cost", "grad"});
    }
    
    /**
     * @brief Create CasADi NLP dictionary for direct use with nlpsol
     */
    static casadi::SXDict createNLP(
        size_t N, size_t D,
        const VectorXf& start_pos,
        const VectorXf& goal_pos,
        float total_time,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        float collision_weight,
        double scale = 1.0) 
    {
        using namespace casadi;
        
        size_t n_vars = (N - 2) * D;
        SX Y = SX::sym("Y", n_vars);
        
        SX cost = buildTotalCost(Y, N, D, start_pos, goal_pos, total_time,
                                 obs_soa, epsilon_sdf, sigma_obs, collision_weight, scale);
        
        return {{"x", Y}, {"f", cost}};
    }
};


/**
 * @brief Wrapper class that extends CollisionAvoidanceTask with CasADi support
 * 
 * Provides convenient methods to create CasADi cost functions from task parameters.
 */
class CasadiCollisionTask : public CollisionAvoidanceTask {
public:
    using CollisionAvoidanceTask::CollisionAvoidanceTask;
    
    /**
     * @brief Create collision cost function from task parameters
     */
    casadi::Function createCollisionCostFunction(size_t N, double scale = 1.0) const {
        return CasadiCollisionCost::createCollisionCostFunction(
            N, num_dimensions_, obs_soa_, epsilon_sdf_, sigma_obs_, scale);
    }
    
    /**
     * @brief Create total cost function from task parameters
     */
    casadi::Function createTotalCostFunction(size_t N, float collision_weight, double scale = 1.0) const {
        return CasadiCollisionCost::createTotalCostFunction(
            N, num_dimensions_, start_node_.position, goal_node_.position, total_time_,
            obs_soa_, epsilon_sdf_, sigma_obs_, collision_weight, scale);
    }
    
    /**
     * @brief Create NLP for direct use with IPOPT/SQP
     */
    casadi::SXDict createNLP(size_t N, float collision_weight, double scale = 1.0) const {
        return CasadiCollisionCost::createNLP(
            N, num_dimensions_, start_node_.position, goal_node_.position, total_time_,
            obs_soa_, epsilon_sdf_, sigma_obs_, collision_weight, scale);
    }
    
    /**
     * @brief Build symbolic collision cost for this task
     */
    casadi::SX buildCollisionCostSymbolic(const casadi::SX& Y, size_t N, double scale = 1.0) const {
        return CasadiCollisionCost::buildTrajectoryCost(
            Y, N, num_dimensions_, obs_soa_, epsilon_sdf_, sigma_obs_, scale);
    }
    
    /**
     * @brief Build symbolic smoothness cost for this task
     */
    casadi::SX buildSmoothnessCostSymbolic(const casadi::SX& Y, size_t N, double scale = 1.0) const {
        return CasadiCollisionCost::buildSmoothnessCost(
            Y, N, num_dimensions_, start_node_.position, goal_node_.position, total_time_, scale);
    }
    
    /**
     * @brief Build symbolic total cost for this task
     */
    casadi::SX buildTotalCostSymbolic(const casadi::SX& Y, size_t N, 
                                       float collision_weight, double scale = 1.0) const {
        return CasadiCollisionCost::buildTotalCost(
            Y, N, num_dimensions_, start_node_.position, goal_node_.position, total_time_,
            obs_soa_, epsilon_sdf_, sigma_obs_, collision_weight, scale);
    }
};

} // namespace pce