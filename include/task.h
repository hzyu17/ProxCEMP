/**
 * @file PCETask.h
 * @brief Abstract task interface for Proximal Cross-Entropy motion planning
 * 
 * This interface follows the STOMP architecture pattern, separating the
 * optimization algorithm from the problem-specific cost computations.
 * 
 * @author Motion Planning Team
 * @date 2025
 */
#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

// Forward declarations
struct Trajectory;
struct PathNode;

namespace pce {

class Task;
typedef std::shared_ptr<Task> TaskPtr;

/**
 * @brief Abstract base class defining the interface for PCE optimization tasks.
 * 
 * Similar to STOMP's Task interface, this separates problem definition (costs,
 * constraints) from the optimization algorithm (PCE). Implementations of this
 * interface define:
 * - How to compute collision costs
 * - How to compute smoothness/control costs  
 * - Any problem-specific filtering or constraints
 */
class Task {
public:
    virtual ~Task() = default;

    /**
     * @brief Compute the collision cost for a given trajectory.
     * 
     * This is the state-dependent cost term that measures how well the
     * trajectory avoids obstacles or satisfies other geometric constraints.
     * 
     * @param trajectory The trajectory to evaluate
     * @return The collision cost (0 = collision-free)
     */
    virtual float computeCollisionCost(const Trajectory& trajectory) const = 0;

    /**
     * @brief Compute collision costs for multiple trajectories (batch processing).
     * 
     * This overload enables efficient batch evaluation of trajectory samples,
     * which is critical for PCE optimization. Implementations may use GPU
     * acceleration or parallel CPU processing.
     * 
     * Default implementation: calls single-trajectory version in a loop.
     * Derived classes should override for better performance.
     * 
     * @param trajectories Vector of trajectories to evaluate (typically many samples)
     * @return Vector of collision costs (one per trajectory)
     */
    virtual std::vector<float> computeCollisionCost(
        const std::vector<Trajectory>& trajectories) const 
    {
        // Default implementation: sequential evaluation
        // Derived classes should override this for batch processing
        std::vector<float> costs;
        costs.reserve(trajectories.size());
        
        for (const auto& traj : trajectories) {
            costs.push_back(computeCollisionCost(traj));
        }
        
        return costs;
    }/**
     * @brief Compute the smoothness cost for a given trajectory.
     * 
     * This is the control cost term that penalizes jerky or high-acceleration
     * motions, encouraging smooth trajectories.
     * 
     * @param trajectory The trajectory to evaluate
     * @return The smoothness cost (lower = smoother)
     */
    // virtual float computeSmoothnessCost(const Trajectory& trajectory) const = 0;

    // /**
    //  * @brief Compute the total cost (collision + smoothness).
    //  * 
    //  * @param trajectory The trajectory to evaluate
    //  * @return The total cost
    //  */
    // virtual float computeTotalCost(const Trajectory& trajectory) const {
    //     return computeCollisionCost(trajectory) + computeSmoothnessCost(trajectory);
    // }

    /**
     * @brief Filter a trajectory to satisfy constraints (e.g., joint limits).
     * 
     * This method can be used to project trajectories onto feasible space,
     * similar to STOMP's filterNoisyParameters.
     * 
     * @param trajectory The trajectory to filter (modified in-place)
     * @param iteration_number Current optimization iteration
     * @return True if filtering was applied, false otherwise
     */
    virtual bool filterTrajectory(Trajectory& trajectory, int iteration_number) {
        return false;  // Default: no filtering
    }

    /**
     * @brief Called at the end of each optimization iteration.
     * 
     * Allows the task to log, visualize, or adapt based on optimization progress.
     * 
     * @param iteration_number Current iteration count
     * @param cost Current trajectory cost
     * @param trajectory Current optimized trajectory
     */
    virtual void postIteration(int iteration_number, 
                               float cost,
                               const Trajectory& trajectory) {
        // Default: do nothing
    }

    /**
     * @brief Called when optimization completes.
     * 
     * @param success Whether optimization succeeded
     * @param total_iterations Number of iterations performed
     * @param final_cost Final trajectory cost
     * @param trajectory Final optimized trajectory
     */
    virtual void done(bool success,
                      int total_iterations,
                      float final_cost,
                      const Trajectory& trajectory) {
        // Default: do nothing
    }

    /**
     * @brief Initialize the task with problem parameters.
     * 
     * Called before optimization begins. Implementations can set up
     * problem-specific data structures here.
     * 
     * @param num_dimensions Dimensionality of the configuration space
     * @param start Starting configuration
     * @param goal Goal configuration
     * @param num_nodes Number of trajectory waypoints
     * @param total_time Total trajectory duration
     */
    virtual void initialize(size_t num_dimensions,
                           const PathNode& start,
                           const PathNode& goal,
                           size_t num_nodes,
                           float total_time) {
        // Default: do nothing
    }
};

}  // namespace pce