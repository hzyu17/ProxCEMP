#include "../include/MotionPlanner.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric> // For std::iota (used in noise sampling)

// --- MotionPlanner Base Class Implementations ---

void MotionPlanner::initialize(const PathNode& start,
                               const PathNode& goal,
                               size_t num_steps,
                               float total_time,
                               InterpolationMethod method)
{
    start_node_ = start;
    goal_node_ = goal;

    // Clear history on initialization
    trajectory_history_.clear();

    switch (method) {
        case InterpolationMethod::LINEAR:
            current_trajectory_ = generateInterpolatedTrajectoryLinear(start, goal, num_steps, total_time);
            break;
        case InterpolationMethod::BEZIER:
            current_trajectory_ = generateInterpolatedTrajectoryBezier(start, goal, num_steps, total_time);
            break;
        default:
            std::cerr << "Error: Unknown interpolation method selected. Using LINEAR.\n";
            current_trajectory_ = generateInterpolatedTrajectoryLinear(start, goal, num_steps, total_time);
            break;
    }

    std::cout << "Planner initialized with " << current_trajectory_.nodes.size() << " nodes.\n";
}

float MotionPlanner::computeCollisionCost(const std::vector<Obstacle>& obstacles) const {
    float total_cost = 0.0f;

    // Collision cost is defined as the sum of how deep each node penetrates an obstacle.
    for (const auto& node : current_trajectory_.nodes) {
        // SDF is calculated based on the obstacle map
        float sdf_value = calculateSDF(node.x, node.y, obstacles);

        // Signed Distance (SDF) relative to the node's collision radius:
        // A collision occurs if SDF - node.radius < 0 
        float effective_sdf = sdf_value - node.radius;

        // If the effective_sdf is negative, there is penetration (collision).
        if (effective_sdf < 0.0f) {
            // Penetration depth is -effective_sdf
            total_cost += (-effective_sdf);
        }
    }
    return total_cost;
}

float MotionPlanner::computeSmoothnessCost() const {
    float smoothness_cost = 0.0f;
    const auto& nodes = current_trajectory_.nodes;
    size_t N = nodes.size();

    if (N < 3) return 0.0f;

    // Sum of squared second differences for x and y coordinates (acceleration cost)
    for (size_t i = 1; i < N - 1; ++i) {
        // Second difference in x: x_{i-1} - 2x_i + x_{i+1}
        float d2x = nodes[i - 1].x - 2.0f * nodes[i].x + nodes[i + 1].x;
        smoothness_cost += std::pow(d2x, 2);
    }

    for (size_t i = 1; i < N - 1; ++i) {
        // Second difference in y: y_{i-1} - 2y_i + y_{i+1}
        float d2y = nodes[i - 1].y - 2.0f * nodes[i].y + nodes[i + 1].y;
        smoothness_cost += std::pow(d2y, 2);
    }

    // Total smoothness cost (L2 norm of acceleration squared)
    return smoothness_cost;
}

float MotionPlanner::computeSmoothnessCost(const Trajectory& trajectory) const {
    float smoothness_cost = 0.0f;
    const auto& nodes = trajectory.nodes;
    size_t N = nodes.size();

    if (N < 3) return 0.0f;

    // Sum of squared second differences for x and y coordinates (acceleration cost)
    for (size_t i = 1; i < N - 1; ++i) {
        // Second difference in x: x_{i-1} - 2x_i + x_{i+1}
        float d2x = nodes[i - 1].x - 2.0f * nodes[i].x + nodes[i + 1].x;
        smoothness_cost += std::pow(d2x, 2);
    }

    for (size_t i = 1; i < N - 1; ++i) {
        // Second difference in y: y_{i-1} - 2y_i + y_{i+1}
        float d2y = nodes[i - 1].y - 2.0f * nodes[i].y + nodes[i + 1].y;
        smoothness_cost += std::pow(d2y, 2);
    }

    // Total smoothness cost (L2 norm of acceleration squared)
    return smoothness_cost;
}

RMatrixDiagonals MotionPlanner::getSmoothnessMatrixRDiagonals(size_t N) const {
    // R = A^T A, where A is the finite difference matrix.
    // For physical trajectories, we scale by 1/(Δt)⁴ to account for acceleration units
    
    RMatrixDiagonals R;
    if (N < 3) return R;
    
    R.main_diag.resize(N);  // R[i, i]
    R.diag1.resize(N - 1);  // R[i, i+1]
    R.diag2.resize(N - 2);  // R[i, i+2]
    
    // --- Compute Time Scaling Factor ---
    // Get the trajectory (adjust method name based on your class)
    const Trajectory& traj = getCurrentTrajectory();
    
    // Calculate uniform time step: nodes are evenly distributed over total_time
    float dt = traj.total_time / (N - 1);
    
    // For R = A^T A where A has 1/(Δt)² scaling, we need 1/(Δt)⁴
    float dt_sq = dt * dt;
    float dt_4 = dt_sq * dt_sq;
    float scale = 1.0f / dt_4;
    
    // Handle edge case: if total_time is zero or not set
    if (traj.total_time < 1e-6f) {
        scale = 1.0f;  // No scaling - fall back to geometric smoothness
    }
    
    // --- R[i, i] (Main Diagonal) ---
    R.main_diag[0] = 1.0f * scale;
    R.main_diag[1] = 5.0f * scale;
    for (size_t i = 2; i < N - 2; ++i) {
        R.main_diag[i] = 6.0f * scale;  // 1 + 4 + 1
    }
    R.main_diag[N - 2] = 5.0f * scale;
    R.main_diag[N - 1] = 1.0f * scale;
    
    // --- R[i, i+1] (Off-Diagonal 1) ---
    R.diag1[0] = -2.0f * scale;
    R.diag1[1] = -4.0f * scale;
    for (size_t i = 2; i < N - 2; ++i) {
        R.diag1[i] = -4.0f * scale;
    }
    R.diag1[N - 2] = -2.0f * scale;
    
    // --- R[i, i+2] (Off-Diagonal 2) ---
    R.diag2[0] = 1.0f * scale;
    for (size_t i = 1; i < N - 2; ++i) {
        R.diag2[i] = 1.0f * scale;
    }
    
    // R is symmetric, so R[i+k, i] = R[i, i+k].
    return R;
}


std::vector<float> MotionPlanner::sampleSmoothnessNoise(size_t N, std::mt19937& rng) const {
    if (N < 3) return std::vector<float>(N, 0.0f);
    
    // The number of FREE nodes (excluding fixed start and goal)
    size_t N_free = N - 2;
    
    // --- 1. Get R matrix for FREE nodes only (indices 1 to N-2) ---
    // We need the smoothness matrix for the unconstrained nodes
    RMatrixDiagonals R_bands = getSmoothnessMatrixRDiagonals(N);
    
    // Extract the submatrix for free nodes (skip first and last rows/columns)
    std::vector<float> R_main(N_free);
    std::vector<float> R_diag1(N_free - 1);
    std::vector<float> R_diag2(N_free - 2);
    
    // Copy the free node portion of R
    for (size_t i = 0; i < N_free; ++i) {
        R_main[i] = R_bands.main_diag[i + 1];  // Skip first fixed node
    }
    for (size_t i = 0; i < N_free - 1; ++i) {
        R_diag1[i] = R_bands.diag1[i + 1];  // Skip first fixed node
    }
    for (size_t i = 0; i < N_free - 2; ++i) {
        R_diag2[i] = R_bands.diag2[i + 1];  // Skip first fixed node
    }
    
    // --- 2. Cholesky Factorization of R_free (R_free = L * L^T) ---
    std::vector<float> L0(N_free);
    std::vector<float> L1(N_free - 1);
    std::vector<float> L2(N_free - 2);
    
    // Initial boundary conditions
    L0[0] = std::sqrt(R_main[0]);
    if (N_free > 1) {
        L1[0] = R_diag1[0] / L0[0];
    }
    if (N_free > 2) {
        L2[0] = R_diag2[0] / L0[0];
    }
    
    // General case: i = 1
    if (N_free > 1) {
        L0[1] = std::sqrt(R_main[1] - L1[0] * L1[0]);
        if (N_free > 2) {
            L1[1] = (R_diag1[1] - L2[0] * L1[0]) / L0[1];
        }
        if (N_free > 3) {
            L2[1] = R_diag2[1] / L0[1];
        }
    }
    
    // General case: i = 2 to N_free-3
    for (size_t i = 2; i < N_free - 2; ++i) {
        L0[i] = std::sqrt(R_main[i] - L1[i - 1] * L1[i - 1] - L2[i - 2] * L2[i - 2]);
        L1[i] = (R_diag1[i] - L2[i - 1] * L1[i - 1]) / L0[i];
        L2[i] = R_diag2[i] / L0[i];
    }
    
    // Boundary case: i = N_free-2
    if (N_free > 2) {
        L0[N_free - 2] = std::sqrt(R_main[N_free - 2] - L1[N_free - 3] * L1[N_free - 3] - L2[N_free - 4] * L2[N_free - 4]);
        if (N_free > 3) {
            L1[N_free - 2] = (R_diag1[N_free - 2] - L2[N_free - 3] * L1[N_free - 3]) / L0[N_free - 2];
        }
    }
    
    // Boundary case: i = N_free-1
    if (N_free > 2) {
        L0[N_free - 1] = std::sqrt(R_main[N_free - 1] - L1[N_free - 2] * L1[N_free - 2] - L2[N_free - 3] * L2[N_free - 3]);
    }
    
    // --- 3. Generate White Noise Z for free nodes only ---
    std::vector<float> Z(N_free);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    for (size_t i = 0; i < N_free; ++i) {
        Z[i] = normal_dist(rng);
    }
    
    // --- 4. Solve L^T * S_free = Z for S_free (Back-Substitution) ---
    std::vector<float> S_free(N_free);
    
    for (int i = N_free - 1; i >= 0; --i) {
        float sum = Z[i];
        
        // Off-diagonal 1
        if (i + 1 < static_cast<int>(N_free)) {
            sum -= L1[i] * S_free[i + 1];
        }
        
        // Off-diagonal 2
        if (i + 2 < static_cast<int>(N_free)) {
            sum -= L2[i] * S_free[i + 2];
        }
        
        // Diagonal
        S_free[i] = sum / L0[i];
    }
    
    // --- 5. Construct full noise vector with zeros at boundaries ---
    std::vector<float> S(N, 0.0f);
    S[0] = 0.0f;           // Fixed start node
    S[N - 1] = 0.0f;       // Fixed goal node
    
    // Copy free node noise
    for (size_t i = 0; i < N_free; ++i) {
        S[i + 1] = S_free[i];
    }
    
    return S;
}


// --- BasicPlanner Concrete Implementation ---

bool BasicPlanner::optimize() {
    std::cout << "Running basic optimization check...\n";

    // 1. Store the initial trajectory (Iteration 0)
    storeTrajectory();

    // Example of using the new functions:
    size_t N = current_trajectory_.nodes.size();
    RMatrixDiagonals R_bands = getSmoothnessMatrixRDiagonals(N);
    std::cout << "R Matrix Main Diagonal Size: " << R_bands.main_diag.size() << " (R[0,0] = " << R_bands.main_diag[0] << ", R[N-1,N-1] = " << R_bands.main_diag[N-1] << ")\n";

    std::mt19937 rng;
    std::vector<float> noise_x = sampleSmoothnessNoise(N, rng);
    std::cout << "Sampled smoothness noise vector size: " << noise_x.size() << "\n";


    float collision_cost = computeCollisionCost(obstacles_);
    float smoothness_cost = computeSmoothnessCost();
    std::cout << "Initial Collision Cost (Iteration 0): " << collision_cost << "\n";
    std::cout << "Initial Smoothness Cost (Iteration 0): " << smoothness_cost << "\n";

    // Check if the initial path is already collision-free
    if (collision_cost < 0.01f) {
        std::cout << "Initial trajectory is nearly collision-free.\n";
        return true;
    }

    // --- Simulate a single optimization step (Iteration 1) ---
    // Example: current_trajectory_.nodes[10].x += 5.0f;

    // 2. Store the result of the first optimization step
    storeTrajectory();

    float new_collision_cost = computeCollisionCost(obstacles_);
    float new_smoothness_cost = computeSmoothnessCost();
    std::cout << "Optimized Collision Cost (Iteration 1): " << new_collision_cost << "\n";
    std::cout << "Optimized Smoothness Cost (Iteration 1): " << new_smoothness_cost << "\n";

    // For now, we simulate success after the checks.
    return true;
}
