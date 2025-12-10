#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "Trajectory.h" 

/**
 * @brief Abstract base class for forward kinematics transformations.
 * Maps configuration space (e.g., joint angles) to workspace (e.g., Cartesian positions).
 */
class ForwardKinematics {
public:
    virtual ~ForwardKinematics() = default;
    
    /**
     * @brief Computes workspace position from configuration space position.
     * @param config_position Configuration space vector (e.g., joint angles)
     * @return Workspace position vector (e.g., end-effector position)
     */
    virtual Eigen::VectorXf compute(const Eigen::VectorXf& config_position) const = 0;
    
    /**
     * @brief Applies FK to a single PathNode.
     * @param config_node Node in configuration space
     * @return Node in workspace (position transformed, radius preserved)
     */
    PathNode apply(const PathNode& config_node) const {
        Eigen::VectorXf workspace_pos = compute(config_node.position);
        return PathNode(workspace_pos, config_node.radius);
    }
    
    /**
     * @brief Applies FK to an entire trajectory.
     * @param config_traj Trajectory in configuration space
     * @return Trajectory in workspace
     */
    Trajectory apply(const Trajectory& config_traj) const {
        Trajectory workspace_traj;
        workspace_traj.total_time = config_traj.total_time;
        workspace_traj.start_index = config_traj.start_index;
        workspace_traj.goal_index = config_traj.goal_index;
        workspace_traj.nodes.reserve(config_traj.nodes.size());
        
        for (const auto& config_node : config_traj.nodes) {
            workspace_traj.nodes.push_back(apply(config_node));
        }
        
        return workspace_traj;
    }
    
    /**
     * @brief Gets the expected configuration space dimensionality.
     */
    virtual size_t getConfigDimension() const = 0;
    
    /**
     * @brief Gets the workspace dimensionality.
     */
    virtual size_t getWorkspaceDimension() const = 0;
};

/**
 * @brief Identity forward kinematics (for Cartesian planning).
 * Configuration space = Workspace.
 */
class IdentityFK : public ForwardKinematics {
public:
    explicit IdentityFK(size_t dimensions) : dimensions_(dimensions) {}
    
    Eigen::VectorXf compute(const Eigen::VectorXf& config_position) const override {
        return config_position;  // Identity mapping
    }
    
    size_t getConfigDimension() const override { return dimensions_; }
    size_t getWorkspaceDimension() const override { return dimensions_; }
    
private:
    size_t dimensions_;
};

/**
 * @brief 2-Link Planar Arm Forward Kinematics.
 * Configuration: [theta1, theta2] (2 joint angles)
 * Workspace: [x, y] (end-effector position in 2D)
 */
class PlanarArm2LinkFK : public ForwardKinematics {
public:
    PlanarArm2LinkFK(float link1_length, float link2_length)
        : L1_(link1_length), L2_(link2_length) {}
    
    Eigen::VectorXf compute(const Eigen::VectorXf& config_position) const override {
        if (config_position.size() != 2) {
            throw std::runtime_error("PlanarArm2Link expects 2D configuration [theta1, theta2]");
        }
        
        float theta1 = config_position(0);
        float theta2 = config_position(1);
        
        // End-effector position
        float x = L1_ * std::cos(theta1) + L2_ * std::cos(theta1 + theta2);
        float y = L1_ * std::sin(theta1) + L2_ * std::sin(theta1 + theta2);
        
        Eigen::VectorXf workspace_pos(2);
        workspace_pos << x, y;
        return workspace_pos;
    }
    
    /**
     * @brief Get elbow position (for visualization)
     */
    Eigen::VectorXf getElbowPosition(const Eigen::VectorXf& config_position) const {
        if (config_position.size() != 2) {
            throw std::runtime_error("PlanarArm2Link expects 2D configuration");
        }
        
        float theta1 = config_position(0);
        
        Eigen::VectorXf elbow_pos(2);
        elbow_pos << L1_ * std::cos(theta1), L1_ * std::sin(theta1);
        return elbow_pos;
    }
    
    size_t getConfigDimension() const override { return 2; }
    size_t getWorkspaceDimension() const override { return 2; }
    
    float getLink1Length() const { return L1_; }
    float getLink2Length() const { return L2_; }
    
private:
    float L1_;  // Link 1 length
    float L2_;  // Link 2 length
};

/**
 * @brief 3-Link Planar Arm Forward Kinematics.
 * Configuration: [theta1, theta2, theta3] (3 joint angles)
 * Workspace: [x, y] (end-effector position in 2D)
 */
class PlanarArm3LinkFK : public ForwardKinematics {
public:
    PlanarArm3LinkFK(float link1_length, float link2_length, float link3_length)
        : L1_(link1_length), L2_(link2_length), L3_(link3_length) {}
    
    Eigen::VectorXf compute(const Eigen::VectorXf& config_position) const override {
        if (config_position.size() != 3) {
            throw std::runtime_error("PlanarArm3Link expects 3D configuration [theta1, theta2, theta3]");
        }
        
        float theta1 = config_position(0);
        float theta2 = config_position(1);
        float theta3 = config_position(2);
        
        // End-effector position
        float x = L1_ * std::cos(theta1) 
                + L2_ * std::cos(theta1 + theta2) 
                + L3_ * std::cos(theta1 + theta2 + theta3);
        float y = L1_ * std::sin(theta1) 
                + L2_ * std::sin(theta1 + theta2) 
                + L3_ * std::sin(theta1 + theta2 + theta3);
        
        Eigen::VectorXf workspace_pos(2);
        workspace_pos << x, y;
        return workspace_pos;
    }
    
    size_t getConfigDimension() const override { return 3; }
    size_t getWorkspaceDimension() const override { return 2; }
    
private:
    float L1_, L2_, L3_;
};

/**
 * @brief 3-Link Spatial Arm Forward Kinematics.
 * Configuration: [theta1, theta2, theta3] (3 revolute joints)
 * Workspace: [x, y, z] (end-effector position in 3D)
 * Simple model: base rotation + 2 pitch joints
 */
class SpatialArm3LinkFK : public ForwardKinematics {
public:
    SpatialArm3LinkFK(float link1_length, float link2_length, float link3_length)
        : L1_(link1_length), L2_(link2_length), L3_(link3_length) {}
    
    Eigen::VectorXf compute(const Eigen::VectorXf& config_position) const override {
        if (config_position.size() != 3) {
            throw std::runtime_error("SpatialArm3Link expects 3D configuration");
        }
        
        float theta1 = config_position(0);  // Base rotation (around Z)
        float theta2 = config_position(1);  // Shoulder pitch (around Y)
        float theta3 = config_position(2);  // Elbow pitch (around Y)
        
        // Simplified FK
        float r = L1_ + L2_ * std::cos(theta2) + L3_ * std::cos(theta2 + theta3);
        float z = L2_ * std::sin(theta2) + L3_ * std::sin(theta2 + theta3);
        
        float x = r * std::cos(theta1);
        float y = r * std::sin(theta1);
        
        Eigen::VectorXf workspace_pos(3);
        workspace_pos << x, y, z;
        return workspace_pos;
    }
    
    size_t getConfigDimension() const override { return 3; }
    size_t getWorkspaceDimension() const override { return 3; }
    
private:
    float L1_, L2_, L3_;
};

/**
 * @brief Denavit-Hartenberg (DH) based Forward Kinematics.
 * Generic implementation for serial manipulators using DH parameters.
 */
class DHForwardKinematics : public ForwardKinematics {
public:
    struct DHParameters {
        float a;      // Link length
        float alpha;  // Link twist
        float d;      // Link offset
        float theta;  // Joint angle (variable for revolute)
        bool is_revolute;  // True for revolute, false for prismatic
    };
    
    DHForwardKinematics(const std::vector<DHParameters>& dh_params)
        : dh_params_(dh_params) {}
    
    Eigen::VectorXf compute(const Eigen::VectorXf& config_position) const override {
        if (config_position.size() != dh_params_.size()) {
            throw std::runtime_error("Configuration dimension doesn't match DH parameters");
        }
        
        // Build transformation matrix
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        
        for (size_t i = 0; i < dh_params_.size(); ++i) {
            float theta = dh_params_[i].is_revolute ? 
                         config_position(i) : dh_params_[i].theta;
            float d = dh_params_[i].is_revolute ? 
                     dh_params_[i].d : config_position(i);
            
            Eigen::Matrix4f Ti = dhTransform(
                dh_params_[i].a, 
                dh_params_[i].alpha, 
                d, 
                theta
            );
            
            T = T * Ti;
        }
        
        // Extract position from transformation matrix
        Eigen::VectorXf workspace_pos(3);
        workspace_pos << T(0, 3), T(1, 3), T(2, 3);
        return workspace_pos;
    }
    
    size_t getConfigDimension() const override { return dh_params_.size(); }
    size_t getWorkspaceDimension() const override { return 3; }
    
private:
    std::vector<DHParameters> dh_params_;
    
    Eigen::Matrix4f dhTransform(float a, float alpha, float d, float theta) const {
        Eigen::Matrix4f T;
        float ct = std::cos(theta);
        float st = std::sin(theta);
        float ca = std::cos(alpha);
        float sa = std::sin(alpha);
        
        T << ct,    -st*ca,  st*sa,   a*ct,
             st,     ct*ca, -ct*sa,   a*st,
             0,      sa,     ca,      d,
             0,      0,      0,       1;
        
        return T;
    }
};

/**
 * @brief Factory function to create FK objects
 */
inline std::shared_ptr<ForwardKinematics> createFK(const std::string& type, 
                                                   const std::vector<float>& params = {}) {
    if (type == "identity") {
        size_t dims = params.empty() ? 2 : static_cast<size_t>(params[0]);
        return std::make_shared<IdentityFK>(dims);
    }
    else if (type == "planar_2link") {
        if (params.size() < 2) throw std::runtime_error("PlanarArm2Link requires 2 link lengths");
        return std::make_shared<PlanarArm2LinkFK>(params[0], params[1]);
    }
    else if (type == "planar_3link") {
        if (params.size() < 3) throw std::runtime_error("PlanarArm3Link requires 3 link lengths");
        return std::make_shared<PlanarArm3LinkFK>(params[0], params[1], params[2]);
    }
    else if (type == "spatial_3link") {
        if (params.size() < 3) throw std::runtime_error("SpatialArm3Link requires 3 link lengths");
        return std::make_shared<SpatialArm3LinkFK>(params[0], params[1], params[2]);
    }
    else {
        throw std::runtime_error("Unknown FK type: " + type);
    }
}
