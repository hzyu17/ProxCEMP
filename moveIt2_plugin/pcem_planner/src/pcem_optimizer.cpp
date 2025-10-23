#include <pcem_planner/pcem_optimizer.hpp>
#include <moveit/utils/logger.hpp>
#include <chrono>
#include <limits>
#include <cmath>

namespace pcem_planner
{

namespace
{
rclcpp::Logger getLogger()
{
    return moveit::getLogger("pcem_optimizer");
}
}

// Parameters default constructor
PCEMOptimizer::Parameters::Parameters()
    : num_samples(3000)
    , num_iterations(50)
    , temperature(1.5f)
    , eta(1.0f)
    , convergence_threshold(0.01f)
    , alpha(0.99f)
    , alpha_temp(1.01f)
    , smoothness_weight(1.0f)
    , collision_weight(1.0f)
    , random_seed(42)
    , df_resolution(0.02)
    , df_size_x(3.0)
    , df_size_y(3.0)
    , df_size_z(2.0)
    , df_origin_x(-1.5)
    , df_origin_y(-1.5)
    , df_origin_z(0.0)
{
}

// Default constructor
PCEMOptimizer::PCEMOptimizer()
    : PCEMOptimizer(Parameters())
{
}

// Constructor with parameters
PCEMOptimizer::PCEMOptimizer(const Parameters& params)
    : params_(params)
    , gamma_(params.eta / params.temperature)
    , best_cost_(std::numeric_limits<float>::infinity())
    , best_iteration_(0)
{
    random_engine_.seed(params_.random_seed);
}

void PCEMOptimizer::buildDistanceField(
    const planning_scene::PlanningSceneConstPtr& planning_scene)
{
    RCLCPP_INFO(getLogger(), "Building distance field...");
    
    // Create distance field
    distance_field_ = std::make_shared<distance_field::PropagationDistanceField>(
        params_.df_size_x,
        params_.df_size_y,
        params_.df_size_z,
        params_.df_resolution,
        params_.df_origin_x,
        params_.df_origin_y,
        params_.df_origin_z,
        0.5  // max_distance
    );
    
    // Get collision objects from world
    const collision_detection::WorldConstPtr& world = planning_scene->getWorld();
    std::vector<std::string> object_ids = world->getObjectIds();
    
    RCLCPP_INFO(getLogger(), "Found %zu collision objects", object_ids.size());
    
    size_t total_shapes = 0;
    for (const auto& obj_id : object_ids)
    {
        collision_detection::World::ObjectConstPtr obj = world->getObject(obj_id);
        if (!obj)
            continue;
        
        // Add each shape individually
        for (size_t i = 0; i < obj->shapes_.size(); ++i)
        {
            distance_field_->addShapeToField(obj->shapes_[i].get(), obj->shape_poses_[i]);
            total_shapes++;
        }
    }
    
    if (total_shapes > 0)
    {
        RCLCPP_INFO(getLogger(), "Added %zu shapes to distance field", total_shapes);
    }
    else
    {
        RCLCPP_WARN(getLogger(), "No collision objects found - distance field is empty");
    }
}

bool PCEMOptimizer::optimize(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const moveit::core::JointModelGroup* group,
    const Eigen::VectorXd& start_config,
    const Eigen::VectorXd& goal_config,
    size_t num_waypoints,
    std::vector<Eigen::VectorXd>& output_trajectory)
{
    RCLCPP_INFO(getLogger(), "Starting PCEM optimization");
    RCLCPP_INFO(getLogger(), "  Samples: %zu, Iterations: %zu, Waypoints: %zu",
                params_.num_samples, params_.num_iterations, num_waypoints);
    
    const size_t num_joints = start_config.size();
    
    if (num_waypoints < 3)
    {
        RCLCPP_ERROR(getLogger(), "Need at least 3 waypoints");
        return false;
    }
    
    // Build distance field from planning scene
    auto df_start = std::chrono::steady_clock::now();
    buildDistanceField(planning_scene);
    auto df_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - df_start).count();
    RCLCPP_INFO(getLogger(), "Distance field built in %ld ms", df_time);
    
    // Get important links for collision checking
    important_links_ = group->getLinkModelNames();
    RCLCPP_INFO(getLogger(), "Checking collision for %zu links", important_links_.size());
    
    // Initialize trajectory with linear interpolation
    std::vector<Eigen::VectorXd> trajectory;
    trajectory.reserve(num_waypoints);
    
    for (size_t i = 0; i < num_waypoints; ++i)
    {
        double alpha = static_cast<double>(i) / (num_waypoints - 1);
        Eigen::VectorXd config = (1.0 - alpha) * start_config + alpha * goal_config;
        trajectory.push_back(config);
    }
    
    // Build regularization matrix
    SparseMatrixXf R_matrix = buildRegularizationMatrix(num_waypoints);
    
    // Initial cost
    // float collision_cost = computeCollisionCost(planning_scene, group, trajectory);
    float collision_cost = computeCollisionCost();
    float smoothness_cost = computeSmoothnessCost(trajectory);
    float cost = params_.collision_weight * collision_cost + 
                 params_.smoothness_weight * smoothness_cost;
    
    best_cost_ = cost;
    best_iteration_ = 0;
    trajectory_history_.clear();
    trajectory_history_.push_back(trajectory);
    
    RCLCPP_INFO(getLogger(), "Initial cost: %.4f (collision: %.4f, smoothness: %.4f)",
                cost, collision_cost, smoothness_cost);
    
    // PCEM optimization loop
    float temperature = params_.temperature;
    float gamma = gamma_;
    
    for (size_t iteration = 1; iteration <= params_.num_iterations; ++iteration)
    {
        // Update parameters
        gamma *= std::pow(params_.alpha, static_cast<float>(iteration - 1));
        temperature *= params_.alpha_temp;
        
        // Convert trajectory to matrix (joints × waypoints)
        MatrixXf Y_k = trajectoryToMatrix(trajectory);
        
        // Sample noise matrices
        std::vector<MatrixXf> epsilon_samples = 
            sampleNoiseMatrices(params_.num_samples, num_waypoints, num_joints);
        
        // Compute weights for each sample
        Eigen::VectorXf weights(params_.num_samples);
        float max_exponent = -std::numeric_limits<float>::infinity();
        
        for (size_t m = 0; m < params_.num_samples; ++m)
        {
            // Perturbed trajectory
            MatrixXf Y_perturbed = Y_k + epsilon_samples[m];
            std::vector<Eigen::VectorXd> sample_traj = matrixToTrajectory(Y_perturbed);
            
            // Fix endpoints
            sample_traj[0] = start_config;
            sample_traj[num_waypoints - 1] = goal_config;
            
            // Compute collision cost using distance field
            // float sample_collision = computeCollisionCost(planning_scene, group, sample_traj);
            float sample_collision = computeCollisionCost();
            
            // Regularization term (sum over all joints)
            float reg_term = 0.0f;
            for (size_t j = 0; j < num_joints; ++j)
            {
                Eigen::VectorXf epsilon_j = epsilon_samples[m].row(j).transpose();
                Eigen::VectorXf Y_k_j = Y_k.row(j).transpose();
                
                // ε^T * R * Y for this joint
                Eigen::VectorXf R_Y_j = R_matrix * Y_k_j;
                reg_term += epsilon_j.dot(R_Y_j);
            }
            
            // Compute weight exponent
            float exponent = -gamma * (sample_collision + reg_term) / temperature;
            
            if (exponent > max_exponent)
            {
                max_exponent = exponent;
            }
            weights(m) = exponent;
        }
        
        // Normalize weights (log-sum-exp trick)
        weights = (weights.array() - max_exponent).exp();
        float weight_sum = weights.sum();
        
        if (weight_sum < 1e-10f)
        {
            RCLCPP_WARN(getLogger(), "Weight sum too small, stopping");
            break;
        }
        
        weights /= weight_sum;
        
        // Weighted update
        MatrixXf Y_kplus1 = MatrixXf::Zero(num_joints, num_waypoints);
        for (size_t m = 0; m < params_.num_samples; ++m)
        {
            Y_kplus1 += weights(m) * (Y_k + epsilon_samples[m]);
        }
        
        // Convert back to trajectory
        trajectory = matrixToTrajectory(Y_kplus1);
        
        // Fix endpoints
        trajectory[0] = start_config;
        trajectory[num_waypoints - 1] = goal_config;
        
        // Store in history
        trajectory_history_.push_back(trajectory);
        
        // Compute new cost
        collision_cost = computeCollisionCost();
        // collision_cost = computeCollisionCost(planning_scene, group, trajectory);
        smoothness_cost = computeSmoothnessCost(trajectory);
        float new_cost = params_.collision_weight * collision_cost + 
                        params_.smoothness_weight * smoothness_cost;
        
        // Track best
        if (new_cost < best_cost_)
        {
            best_cost_ = new_cost;
            best_iteration_ = iteration;
        }
        
        RCLCPP_INFO(getLogger(), 
                    "Iter %zu: cost=%.4f (coll=%.4f, smooth=%.4f), gamma=%.4f, temp=%.4f",
                    iteration, new_cost, collision_cost, smoothness_cost, gamma, temperature);
        
        // Check convergence
        if (iteration > 1 && std::abs(cost - new_cost) < params_.convergence_threshold)
        {
            RCLCPP_INFO(getLogger(), "Converged at iteration %zu", iteration);
            break;
        }
        
        cost = new_cost;
    }
    
    // Restore best trajectory
    if (best_iteration_ < trajectory_history_.size())
    {
        trajectory = trajectory_history_[best_iteration_];
        RCLCPP_INFO(getLogger(), "Restoring best trajectory from iteration %zu (cost: %.4f)",
                    best_iteration_, best_cost_);
    }
    
    output_trajectory = trajectory;
    
    RCLCPP_INFO(getLogger(), "PCEM optimization complete");
    return true;
}


// --- TEMPORARY FUNCTION FOR TESTING CORE LOGIC ---
float PCEMOptimizer::computeCollisionCost() const
{
    // ROS_INFO_STREAM("Using MOCKED collision cost: 5.0f");
    return 5.0f; // Arbitrary constant value
}
// ---------------------------------------------------


float PCEMOptimizer::computeCollisionCost(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const moveit::core::JointModelGroup* group,
    const std::vector<Eigen::VectorXd>& trajectory) const
{
    if (!distance_field_)
    {
        RCLCPP_ERROR(getLogger(), "Distance field not initialized!");
        return std::numeric_limits<float>::infinity();
    }
    
    float total_cost = 0.0f;
    const float safety_margin = 0.05f;  // 5cm
    const float collision_penalty = 100.0f;
    
    moveit::core::RobotState state(planning_scene->getRobotModel());
    
    for (const auto& config : trajectory)
    {
        // Set robot state
        state.setJointGroupPositions(group, config);
        state.update();
        
        // Check each important link
        for (const auto& link_name : important_links_)
        {
            const Eigen::Isometry3d& link_transform = state.getGlobalLinkTransform(link_name);
            Eigen::Vector3d link_position = link_transform.translation();
            
            // Get distance from distance field
            double distance = distance_field_->getDistance(
                link_position.x(),
                link_position.y(),
                link_position.z()
            );
            
            // Compute cost
            if (distance < 0.0)
            {
                // In collision
                total_cost += collision_penalty * std::abs(distance);
            }
            else if (distance < safety_margin)
            {
                // Close to collision - soft penalty
                double normalized = (safety_margin - distance) / safety_margin;
                total_cost += normalized * normalized;
            }
        }
    }
    
    return total_cost;
}

float PCEMOptimizer::computeSmoothnessCost(
    const std::vector<Eigen::VectorXd>& trajectory) const
{
    if (trajectory.size() < 3)
    {
        return 0.0f;
    }
    
    float cost = 0.0f;
    
    // Compute finite differences (acceleration)
    for (size_t i = 1; i < trajectory.size() - 1; ++i)
    {
        Eigen::VectorXd accel = trajectory[i+1] - 2.0 * trajectory[i] + trajectory[i-1];
        cost += static_cast<float>(accel.squaredNorm());
    }
    
    return cost;
}

PCEMOptimizer::MatrixXf PCEMOptimizer::trajectoryToMatrix(
    const std::vector<Eigen::VectorXd>& trajectory) const
{
    const size_t num_waypoints = trajectory.size();
    const size_t num_joints = trajectory[0].size();
    
    MatrixXf matrix(num_joints, num_waypoints);
    
    for (size_t i = 0; i < num_waypoints; ++i)
    {
        matrix.col(i) = trajectory[i].cast<float>();
    }
    
    return matrix;
}

std::vector<Eigen::VectorXd> PCEMOptimizer::matrixToTrajectory(
    const MatrixXf& matrix) const
{
    const size_t num_waypoints = matrix.cols();
    
    std::vector<Eigen::VectorXd> trajectory;
    trajectory.reserve(num_waypoints);
    
    for (size_t i = 0; i < num_waypoints; ++i)
    {
        Eigen::VectorXd config = matrix.col(i).cast<double>();
        trajectory.push_back(config);
    }
    
    return trajectory;
}

std::vector<PCEMOptimizer::MatrixXf> PCEMOptimizer::sampleNoiseMatrices(
    size_t num_samples,
    size_t num_waypoints,
    size_t num_joints) const
{
    std::vector<MatrixXf> samples;
    samples.reserve(num_samples);
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (size_t m = 0; m < num_samples; ++m)
    {
        MatrixXf epsilon(num_joints, num_waypoints);
        
        for (size_t i = 0; i < num_joints; ++i)
        {
            for (size_t j = 0; j < num_waypoints; ++j)
            {
                epsilon(i, j) = dist(random_engine_);
            }
        }
        
        // Zero out endpoints
        epsilon.col(0).setZero();
        epsilon.col(num_waypoints - 1).setZero();
        
        samples.push_back(epsilon);
    }
    
    return samples;
}

PCEMOptimizer::SparseMatrixXf PCEMOptimizer::buildRegularizationMatrix(
    size_t num_waypoints) const
{
    // Build finite difference matrix for acceleration
    SparseMatrixXf D(num_waypoints - 2, num_waypoints);
    
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve((num_waypoints - 2) * 3);
    
    for (size_t i = 0; i < num_waypoints - 2; ++i)
    {
        triplets.emplace_back(i, i, 1.0f);
        triplets.emplace_back(i, i+1, -2.0f);
        triplets.emplace_back(i, i+2, 1.0f);
    }
    
    D.setFromTriplets(triplets.begin(), triplets.end());
    
    // R = D^T * D
    SparseMatrixXf R = D.transpose() * D;
    
    return R;
}

} // namespace pcem_planner
