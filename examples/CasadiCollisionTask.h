/**
 * @file CasadiCollisionTask.h
 * @brief CasADi symbolic collision cost functions for trajectory optimization
 * 
 * Provides smooth, differentiable collision costs. 
 * 
 * NOTE: Smoothness cost should be computed using MotionPlanner's R_matrix_ 
 * and computeSmoothnessCost() to ensure consistency. This file focuses on
 * collision cost only.
 */
#pragma once

#include "CollisionAvoidanceTask.h"
#include <casadi/casadi.hpp>

namespace pce {

/**
 * @brief Utility class for CasADi symbolic collision cost construction
 */
class CasadiCollisionCost {
public:
    using VectorXf = Eigen::VectorXf;
    using MatrixXf = Eigen::MatrixXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;
    
    // ==================== Smooth Approximation Functions ====================
    
    /**
     * @brief Smooth approximation of max(0, x) using softplus
     * 
     * softplus(x, k) = log(1 + exp(k*x)) / k
     * 
     * @param x Input symbolic expression
     * @param k Sharpness parameter (higher = closer to max, but less smooth)
     * @return Smooth approximation of max(0, x)
     */
    static casadi::SX softplus(const casadi::SX& x, double k = 20.0) {
        using namespace casadi;
        return if_else(x > 10.0/k, x, log(1.0 + exp(k * x)) / k);
    }
    
    /**
     * @brief Smooth max using polynomial transition
     */
    static casadi::SX smoothReluPoly(const casadi::SX& x, double d = 0.5) {
        using namespace casadi;
        return if_else(x < 0, SX::zeros(x.size1(), x.size2()),
               if_else(x > d, x - d/2, x*x/(2*d)));
    }
    
    // ==================== R-Matrix Conversion ====================
    
    /**
     * @brief Convert Eigen sparse R-matrix to CasADi DM
     * 
     * Use this to convert MotionPlanner's R_matrix_ for use in CasADi.
     * This ensures the symbolic smoothness cost matches computeSmoothnessCost().
     */
    static casadi::DM sparseRMatrixToDM(const SparseMatrixXf& R_sparse) {
        size_t n = R_sparse.rows();
        casadi::DM R(n, n);
        
        // Fill with zeros first
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                R(i, j) = 0.0;
            }
        }
        
        // Copy non-zero entries
        for (int k = 0; k < R_sparse.outerSize(); ++k) {
            for (SparseMatrixXf::InnerIterator it(R_sparse, k); it; ++it) {
                R(it.row(), it.col()) = static_cast<double>(it.value());
            }
        }
        
        return R;
    }
    
    // ==================== Symbolic Cost Builders ====================
    
    /**
     * @brief Build symbolic collision cost for a single 2D/3D position
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
            
            SX dist = sqrt(dist_sq + 1e-6);
            double combined_r = static_cast<double>(obs_soa.combined_radii(k));
            SX sdf = dist - combined_r;
            SX hinge = softplus(epsilon_sdf - sdf);
            cost += hinge * hinge;
        }
        
        return sigma_obs * scale * cost;
    }
    
    /**
     * @brief Build symbolic collision cost for entire trajectory
     * 
     * @param Y Decision variables (interior nodes only, flattened as [x0,y0,x1,y1,...])
     * @param N Total number of nodes (including fixed endpoints)
     * @param D Number of dimensions
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
        
        // Interior nodes only (indices 1 to N-2)
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
     * @brief Build symbolic smoothness cost using provided R-matrix
     * 
     * Computes: sum_d(Y_d^T * R * Y_d) / scale_sq
     * 
     * To match MotionPlanner::computeSmoothnessCost():
     * 1. Pass R_matrix from MotionPlanner (converted via sparseRMatrixToDM)
     * 2. Pass normalize_scale = max absolute position value from trajectory
     * 
     * @param Y Decision variables (interior nodes, flattened)
     * @param N Total number of nodes  
     * @param D Number of dimensions
     * @param start_pos Fixed start position
     * @param goal_pos Fixed goal position
     * @param R_matrix R matrix from MotionPlanner (use sparseRMatrixToDM to convert)
     * @param normalize_scale Max absolute position for normalization (0 = no normalization)
     * @param obj_scale Additional objective scaling
     */
    static casadi::SX buildSmoothnessCost(
        const casadi::SX& Y,
        size_t N, size_t D,
        const VectorXf& start_pos,
        const VectorXf& goal_pos,
        const casadi::DM& R_matrix,
        double normalize_scale = 0.0,
        double obj_scale = 1.0) 
    {
        using namespace casadi;
        
        SX cost = 0;
        
        // Process each dimension: Y_d^T * R * Y_d
        for (size_t d = 0; d < D; ++d) {
            // Build full trajectory vector for this dimension
            SX Y_d = SX::zeros(N, 1);
            
            // Start (fixed)
            Y_d(0) = static_cast<double>(start_pos(d));
            
            // Interior nodes (decision variables)
            for (size_t i = 0; i < N - 2; ++i) {
                Y_d(i + 1) = Y(i * D + d);
            }
            
            // Goal (fixed)
            Y_d(N - 1) = static_cast<double>(goal_pos(d));
            
            // Y_d^T * R * Y_d
            SX R_sx = SX(R_matrix);
            SX RY = mtimes(R_sx, Y_d);
            
            cost += mtimes(Y_d.T(), RY);
        }
        
        // Apply normalization (matches MotionPlanner::computeSmoothnessCost)
        if (normalize_scale > 0) {
            double scale_sq = std::max(normalize_scale * normalize_scale, 1e-6);
            cost = cost / scale_sq;
        }
        
        return obj_scale * cost;
    }
    
    /**
     * @brief Build total cost (smoothness + collision)
     */
    static casadi::SX buildTotalCost(
        const casadi::SX& Y,
        size_t N, size_t D,
        const VectorXf& start_pos,
        const VectorXf& goal_pos,
        const casadi::DM& R_matrix,
        const ObstacleDataSoA& obs_soa,
        float epsilon_sdf,
        float sigma_obs,
        float collision_weight,
        double normalize_scale = 0.0,
        double obj_scale = 1.0) 
    {
        casadi::SX smooth = buildSmoothnessCost(Y, N, D, start_pos, goal_pos, 
                                                 R_matrix, normalize_scale, obj_scale);
        casadi::SX coll = buildTrajectoryCost(Y, N, D, obs_soa, epsilon_sdf, sigma_obs, obj_scale);
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
};


/**
 * @brief Wrapper class that extends CollisionAvoidanceTask with CasADi support
 */
class CasadiCollisionTask : public CollisionAvoidanceTask {
public:
    using CollisionAvoidanceTask::CollisionAvoidanceTask;
    
    /**
     * @brief Build symbolic collision cost
     */
    casadi::SX buildCollisionCostSymbolic(
        const casadi::SX& Y, 
        size_t N, 
        double scale = 1.0) const 
    {
        return CasadiCollisionCost::buildTrajectoryCost(
            Y, N, num_dimensions_, obs_soa_, epsilon_sdf_, sigma_obs_, scale);
    }
    
    /**
     * @brief Build symbolic smoothness cost using provided R-matrix
     * 
     * @param Y Decision variables
     * @param N Number of nodes
     * @param R_matrix R matrix from MotionPlanner (use getRMatrixDM() to convert)
     * @param normalize_scale Max position scale for normalization
     * @param obj_scale Objective scaling
     */
    casadi::SX buildSmoothnessCostSymbolic(
        const casadi::SX& Y,
        size_t N,
        const casadi::DM& R_matrix,
        double normalize_scale = 0.0,
        double obj_scale = 1.0) const 
    {
        return CasadiCollisionCost::buildSmoothnessCost(
            Y, N, num_dimensions_, start_node_.position, goal_node_.position,
            R_matrix, normalize_scale, obj_scale);
    }
    
    /**
     * @brief Build symbolic total cost
     */
    casadi::SX buildTotalCostSymbolic(
        const casadi::SX& Y,
        size_t N,
        const casadi::DM& R_matrix,
        float collision_weight,
        double normalize_scale = 0.0,
        double obj_scale = 1.0) const 
    {
        return CasadiCollisionCost::buildTotalCost(
            Y, N, num_dimensions_, start_node_.position, goal_node_.position,
            R_matrix, obs_soa_, epsilon_sdf_, sigma_obs_, collision_weight,
            normalize_scale, obj_scale);
    }
    
    /**
     * @brief Create collision cost function
     */
    casadi::Function createCollisionCostFunction(size_t N, double scale = 1.0) const {
        return CasadiCollisionCost::createCollisionCostFunction(
            N, num_dimensions_, obs_soa_, epsilon_sdf_, sigma_obs_, scale);
    }
    
    /**
     * @brief Convert sparse R-matrix to CasADi DM
     */
    static casadi::DM convertRMatrixToDM(const Eigen::SparseMatrix<float>& R_sparse) {
        return CasadiCollisionCost::sparseRMatrixToDM(R_sparse);
    }
};

} // namespace pce