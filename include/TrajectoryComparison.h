#pragma once

#include "MotionPlanner.h"
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

/**
 * @brief Utility class for comparing trajectories from different planners
 */
class TrajectoryComparison {
public:
    /**
     * @brief Save a trajectory to a text file
     */
    static bool saveTrajectory(const Trajectory& traj, 
                               const std::string& filename,
                               const std::string& planner_name) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return false;
        }
        
        file << std::fixed << std::setprecision(8);
        
        // Header
        file << "# Planner: " << planner_name << "\n";
        file << "# Total time: " << traj.total_time << "\n";
        file << "# Num nodes: " << traj.nodes.size() << "\n";
        file << "# Dimensions: " << traj.dimensions() << "\n";
        file << "# Format: node_index, x, y, z, ..., radius\n";
        
        // Data
        for (size_t i = 0; i < traj.nodes.size(); ++i) {
            file << i;
            for (size_t d = 0; d < traj.nodes[i].position.size(); ++d) {
                file << ", " << traj.nodes[i].position[d];
            }
            file << ", " << traj.nodes[i].radius << "\n";
        }
        
        file.close();
        std::cout << "Trajectory saved to: " << filename << "\n";
        return true;
    }
    
    /**
     * @brief Load a trajectory from a text file
     */
    static bool loadTrajectory(Trajectory& traj, const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return false;
        }
        
        std::string line;
        size_t num_nodes = 0;
        size_t dimensions = 0;
        
        // Parse header
        while (std::getline(file, line)) {
            if (line[0] != '#') break;
            
            if (line.find("# Num nodes:") != std::string::npos) {
                sscanf(line.c_str(), "# Num nodes: %zu", &num_nodes);
            }
            if (line.find("# Dimensions:") != std::string::npos) {
                sscanf(line.c_str(), "# Dimensions: %zu", &dimensions);
            }
            if (line.find("# Total time:") != std::string::npos) {
                sscanf(line.c_str(), "# Total time: %f", &traj.total_time);
            }
        }
        
        traj.nodes.clear();
        traj.nodes.reserve(num_nodes);
        
        // Parse data (first line already in 'line')
        do {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string token;
            std::vector<float> values;
            
            while (std::getline(iss, token, ',')) {
                values.push_back(std::stof(token));
            }
            
            if (values.size() >= dimensions + 2) {  // index + position + radius
                Eigen::VectorXf pos(dimensions);
                for (size_t d = 0; d < dimensions; ++d) {
                    pos[d] = values[d + 1];  // Skip index
                }
                float radius = values.back();
                traj.nodes.emplace_back(pos, radius);
            }
        } while (std::getline(file, line));
        
        file.close();
        return true;
    }
    
    /**
     * @brief Compute detailed comparison between two trajectories
     */
    static void compareTrajectories(const Trajectory& traj1, 
                                    const Trajectory& traj2,
                                    const std::string& name1,
                                    const std::string& name2,
                                    const Eigen::SparseMatrix<float>& R_matrix) {
        
        std::cout << "\n========================================\n";
        std::cout << "  Trajectory Comparison\n";
        std::cout << "========================================\n\n";
        
        std::cout << "Comparing: " << name1 << " vs " << name2 << "\n\n";
        
        // Basic checks
        if (traj1.nodes.size() != traj2.nodes.size()) {
            std::cout << "WARNING: Different number of nodes!\n";
            std::cout << "  " << name1 << ": " << traj1.nodes.size() << " nodes\n";
            std::cout << "  " << name2 << ": " << traj2.nodes.size() << " nodes\n";
            return;
        }
        
        const size_t N = traj1.nodes.size();
        const size_t D = traj1.dimensions();
        
        std::cout << "Number of nodes: " << N << "\n";
        std::cout << "Dimensions: " << D << "\n\n";
        
        // Convert to matrices
        Eigen::MatrixXf Y1(D, N);
        Eigen::MatrixXf Y2(D, N);
        
        for (size_t i = 0; i < N; ++i) {
            Y1.col(i) = traj1.nodes[i].position;
            Y2.col(i) = traj2.nodes[i].position;
        }
        
        // Overall difference
        Eigen::MatrixXf diff = Y1 - Y2;
        float frobenius_norm = diff.norm();
        float max_diff = diff.cwiseAbs().maxCoeff();
        float mean_diff = diff.cwiseAbs().mean();
        
        std::cout << "--- Overall Trajectory Difference ---\n";
        std::cout << "  Frobenius norm:  " << frobenius_norm << "\n";
        std::cout << "  Max absolute:    " << max_diff << "\n";
        std::cout << "  Mean absolute:   " << mean_diff << "\n\n";
        
        // Start/Goal comparison
        std::cout << "--- Start Node Comparison ---\n";
        Eigen::VectorXf start_diff = Y1.col(0) - Y2.col(0);
        std::cout << "  " << name1 << " start: " << Y1.col(0).transpose() << "\n";
        std::cout << "  " << name2 << " start: " << Y2.col(0).transpose() << "\n";
        std::cout << "  Difference:        " << start_diff.transpose() << "\n";
        std::cout << "  Distance:          " << start_diff.norm() << "\n\n";
        
        std::cout << "--- Goal Node Comparison ---\n";
        Eigen::VectorXf goal_diff = Y1.col(N-1) - Y2.col(N-1);
        std::cout << "  " << name1 << " goal: " << Y1.col(N-1).transpose() << "\n";
        std::cout << "  " << name2 << " goal: " << Y2.col(N-1).transpose() << "\n";
        std::cout << "  Difference:       " << goal_diff.transpose() << "\n";
        std::cout << "  Distance:         " << goal_diff.norm() << "\n\n";
        
        // Interior waypoints comparison
        float interior_max = 0.0f;
        size_t interior_max_idx = 0;
        for (size_t i = 1; i < N-1; ++i) {
            float dist = (Y1.col(i) - Y2.col(i)).norm();
            if (dist > interior_max) {
                interior_max = dist;
                interior_max_idx = i;
            }
        }
        
        std::cout << "--- Interior Waypoints (nodes 1 to " << N-2 << ") ---\n";
        std::cout << "  Max difference: " << interior_max << " at node " << interior_max_idx << "\n";
        std::cout << "  Node " << interior_max_idx << " " << name1 << ": " 
                  << Y1.col(interior_max_idx).transpose() << "\n";
        std::cout << "  Node " << interior_max_idx << " " << name2 << ": " 
                  << Y2.col(interior_max_idx).transpose() << "\n\n";
        
        // Smoothness cost comparison
        std::cout << "--- Smoothness Cost: Y^T * R * Y ---\n";
        float smooth1 = 0.0f;
        float smooth2 = 0.0f;
        
        for (size_t d = 0; d < D; ++d) {
            Eigen::VectorXf Y1_d = Y1.row(d).transpose();
            Eigen::VectorXf Y2_d = Y2.row(d).transpose();
            
            Eigen::VectorXf RY1_d = R_matrix * Y1_d;
            Eigen::VectorXf RY2_d = R_matrix * Y2_d;
            
            smooth1 += Y1_d.dot(RY1_d);
            smooth2 += Y2_d.dot(RY2_d);
        }
        
        std::cout << "  " << name1 << " smoothness: " << smooth1 << "\n";
        std::cout << "  " << name2 << " smoothness: " << smooth2 << "\n";
        std::cout << "  Difference:         " << std::abs(smooth1 - smooth2) << "\n";
        std::cout << "  Relative diff:      " << std::abs(smooth1 - smooth2) / std::max(smooth1, smooth2) * 100 << "%\n\n";
        
        // Per-dimension analysis
        std::cout << "--- Per-Dimension Analysis ---\n";
        for (size_t d = 0; d < D; ++d) {
            Eigen::VectorXf Y1_d = Y1.row(d).transpose();
            Eigen::VectorXf Y2_d = Y2.row(d).transpose();
            
            float dim_diff = (Y1_d - Y2_d).norm();
            
            Eigen::VectorXf RY1_d = R_matrix * Y1_d;
            Eigen::VectorXf RY2_d = R_matrix * Y2_d;
            
            float smooth1_d = Y1_d.dot(RY1_d);
            float smooth2_d = Y2_d.dot(RY2_d);
            
            std::cout << "  Dimension " << d << ":\n";
            std::cout << "    Position difference: " << dim_diff << "\n";
            std::cout << "    " << name1 << " smoothness: " << smooth1_d << "\n";
            std::cout << "    " << name2 << " smoothness: " << smooth2_d << "\n";
            std::cout << "    Smoothness diff:     " << std::abs(smooth1_d - smooth2_d) << "\n";
        }
        std::cout << "\n";
        
        // R matrix diagnostics
        std::cout << "--- R Matrix Diagnostics ---\n";
        std::cout << "  R matrix size: " << R_matrix.rows() << " x " << R_matrix.cols() << "\n";
        std::cout << "  R matrix nnz:  " << R_matrix.nonZeros() << "\n";
        
        // Compute contribution from first/last nodes
        Eigen::VectorXf first_col = Eigen::VectorXf(R_matrix.col(0));
        Eigen::VectorXf last_col = Eigen::VectorXf(R_matrix.col(N-1));
        
        std::cout << "  R first column norm: " << first_col.norm() << "\n";
        std::cout << "  R last column norm:  " << last_col.norm() << "\n";
        
        // Check endpoint contribution to smoothness
        float endpoint_contrib1 = 0.0f;
        float endpoint_contrib2 = 0.0f;
        
        for (size_t d = 0; d < D; ++d) {
            float start1 = Y1(d, 0);
            float goal1 = Y1(d, N-1);
            float start2 = Y2(d, 0);
            float goal2 = Y2(d, N-1);
            
            endpoint_contrib1 += start1 * (R_matrix.coeff(0, 0) * start1) + 
                                goal1 * (R_matrix.coeff(N-1, N-1) * goal1);
            endpoint_contrib2 += start2 * (R_matrix.coeff(0, 0) * start2) + 
                                goal2 * (R_matrix.coeff(N-1, N-1) * goal2);
        }
        
        std::cout << "\n--- Endpoint Contribution to Smoothness ---\n";
        std::cout << "  " << name1 << " endpoints: " << endpoint_contrib1 << "\n";
        std::cout << "  " << name2 << " endpoints: " << endpoint_contrib2 << "\n";
        std::cout << "  Difference:        " << std::abs(endpoint_contrib1 - endpoint_contrib2) << "\n";
        
        std::cout << "\n========================================\n\n";
    }
    
    /**
     * @brief Save comparison results to a file
     */
    static void saveComparisonToFile(const Trajectory& traj1, 
                                     const Trajectory& traj2,
                                     const std::string& name1,
                                     const std::string& name2,
                                     const Eigen::SparseMatrix<float>& R_matrix,
                                     const std::string& output_file) {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::ofstream file(output_file);
        std::cout.rdbuf(file.rdbuf());
        
        compareTrajectories(traj1, traj2, name1, name2, R_matrix);
        
        std::cout.rdbuf(old_cout);
        file.close();
        
        std::cout << "Comparison saved to: " << output_file << "\n";
    }
};
