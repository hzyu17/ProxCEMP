/**
 * @file MotionPlanner.h
 * @brief Cleaned Abstract base class for motion planning algorithms
 */
#pragma once

#include "Trajectory.h"
#include "ForwardKinematics.h"
#include <vector> 
#include <random> 
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>      
#include <sstream>      
#include <iomanip>      
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <filesystem>

/**
 * @brief Covariance scheduling strategies
 */
enum class CovarianceSchedule {
    CONSTANT, LINEAR, EXPONENTIAL, COSINE, STEP, ADAPTIVE
};

struct MotionPlannerConfig {
    size_t num_dimensions = 2;
    size_t num_discretization = 100;
    float total_time = 12.0f;
    float node_collision_radius = 15.0f;
    
    std::vector<float> start_position;
    std::vector<float> goal_position;
    
    unsigned int random_seed = 42;
    bool visualize_initial_state = false;
    float clearance_distance = 2.0f;

    CovarianceSchedule cov_schedule = CovarianceSchedule::COSINE;
    float cov_scale_initial = 1.0f;
    float cov_scale_final = 0.01f;
    float cov_decay_rate = 0.9f;
    size_t cov_step_interval = 3;
    float cov_step_factor = 0.5f;
    float cov_adaptive_threshold = 0.05f;
    
    virtual bool loadFromFile(const std::string& config_file_path) {
        try {
            return loadFromYAML(YAML::LoadFile(config_file_path));
        } catch (const std::exception& e) {
            std::cerr << "Error loading config from file: " << e.what() << "\n";
            return false;
        }
    }
    
    virtual bool loadFromYAML(const YAML::Node& config) {
        try {
            if (auto mp = config["motion_planning"]) {
                if (mp["num_dimensions"])      num_dimensions = mp["num_dimensions"].as<size_t>();
                if (mp["num_discretization"])  num_discretization = mp["num_discretization"].as<size_t>();
                if (mp["total_time"])          total_time = mp["total_time"].as<float>();
                if (mp["node_collision_radius"]) node_collision_radius = mp["node_collision_radius"].as<float>();
                if (mp["start_position"])      start_position = mp["start_position"].as<std::vector<float>>();
                if (mp["goal_position"])       goal_position = mp["goal_position"].as<std::vector<float>>();
            }
            
            if (auto exp = config["experiment"]) {
                if (exp["random_seed"])             random_seed = exp["random_seed"].as<unsigned int>();
                if (exp["visualize_initial_state"]) visualize_initial_state = exp["visualize_initial_state"].as<bool>();
            }
            
            if (auto env = config["environment"]) {
                if (env["clearance_distance"]) clearance_distance = env["clearance_distance"].as<float>();
            }

            if (auto planner = config["pce_planner"] ? config["pce_planner"] : config["ngd_planner"]) {
                if (planner["covariance_schedule"]) {
                    std::string s = planner["covariance_schedule"].as<std::string>();
                    static const std::unordered_map<std::string, CovarianceSchedule> schedule_map = {
                        {"constant", CovarianceSchedule::CONSTANT}, {"linear", CovarianceSchedule::LINEAR},
                        {"exponential", CovarianceSchedule::EXPONENTIAL}, {"cosine", CovarianceSchedule::COSINE},
                        {"step", CovarianceSchedule::STEP}, {"adaptive", CovarianceSchedule::ADAPTIVE}
                    };
                    if (schedule_map.count(s)) cov_schedule = schedule_map.at(s);
                }
                if (planner["cov_scale_initial"])      cov_scale_initial = planner["cov_scale_initial"].as<float>();
                if (planner["cov_scale_final"])        cov_scale_final = planner["cov_scale_final"].as<float>();
                if (planner["cov_decay_rate"])         cov_decay_rate = planner["cov_decay_rate"].as<float>();
                if (planner["cov_step_interval"])      cov_step_interval = planner["cov_step_interval"].as<size_t>();
                if (planner["cov_step_factor"])        cov_step_factor = planner["cov_step_factor"].as<float>();
                if (planner["cov_adaptive_threshold"]) cov_adaptive_threshold = planner["cov_adaptive_threshold"].as<float>();
            }
            return validate();
        } catch (const std::exception& e) {
            std::cerr << "Error loading config from YAML: " << e.what() << "\n";
            return false;
        }
    }
    
    virtual bool validate() const {
        if (num_discretization < 2 || total_time <= 0.0f || node_collision_radius < 0.0f) return false;
        return (start_position.size() == num_dimensions && goal_position.size() == num_dimensions);
    }
    
    virtual void print() const {
        std::cout << "=== Motion Planner Configuration ===\n"
                  << "  Dimensions: " << num_dimensions << " | Nodes: " << num_discretization << "\n"
                  << "  Time: " << total_time << " | Seed: " << random_seed << "\n";
    }
    virtual ~MotionPlannerConfig() = default;
};

enum class InterpolationMethod { LINEAR, BEZIER };

class MotionPlanner {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    using SparseMatrixXf = Eigen::SparseMatrix<float>;

    MotionPlanner() { random_engine_.seed(std::random_device{}()); }
    virtual ~MotionPlanner() { closeLog(); }

    virtual std::string getPlannerName() const = 0;

    float computeCovarianceScale(size_t iteration, float prev_cost = 0.0f, float curr_cost = 0.0f) {
        float t = static_cast<float>(iteration - 1);
        float T = static_cast<float>(num_nodes_);

        switch (cov_schedule_) {
            case CovarianceSchedule::CONSTANT: return cov_scale_initial_;
            case CovarianceSchedule::LINEAR:   return cov_scale_initial_ + (cov_scale_final_ - cov_scale_initial_) * (t / T);
            case CovarianceSchedule::EXPONENTIAL: return std::max(cov_scale_final_, cov_scale_initial_ * std::pow(cov_decay_rate_, t));
            case CovarianceSchedule::COSINE:   return cov_scale_final_ + 0.5f * (cov_scale_initial_ - cov_scale_final_) * (1.0f + std::cos(M_PI * t / T));
            case CovarianceSchedule::STEP:     return std::max(cov_scale_final_, cov_scale_initial_ * std::pow(cov_step_factor_, std::floor(t / cov_step_interval_)));
            case CovarianceSchedule::ADAPTIVE: {
                if (iteration <= 1) return cov_scale_current_;
                float improvement = (prev_cost - curr_cost) / (std::abs(prev_cost) + 1e-6f);
                if (improvement > cov_adaptive_threshold_) cov_scale_current_ *= cov_decay_rate_;
                else if (improvement < 0) cov_scale_current_ *= (1.0f + 0.1f * (1.0f - cov_decay_rate_));
                return cov_scale_current_ = std::clamp(cov_scale_current_, cov_scale_final_, cov_scale_initial_);
            }
            default: return cov_scale_initial_;
        }
    }

    virtual bool initialize(const MotionPlannerConfig& config) {
        config_ = std::make_shared<MotionPlannerConfig>(config);
        if (!config_->validate()) return false;
        
        num_dimensions_ = config_->num_dimensions;
        num_nodes_ = config_->num_discretization;
        total_time_ = config_->total_time;
        node_radius_ = config_->node_collision_radius;
        random_seed_ = config_->random_seed;
        random_engine_.seed(random_seed_);
        
        config_->print();
        openLog();
        if (!initializeTrajectory()) return false;
        
        cov_schedule_ = config.cov_schedule;
        cov_scale_initial_ = config.cov_scale_initial;
        cov_scale_final_ = config.cov_scale_final;
        cov_decay_rate_ = config.cov_decay_rate;
        cov_step_interval_ = config.cov_step_interval;
        cov_step_factor_ = config.cov_step_factor;
        cov_adaptive_threshold_ = config.cov_adaptive_threshold;
        cov_scale_current_ = cov_scale_initial_;

        return is_initialized_ = true;
    }

    virtual bool solve() {
        if (!is_initialized_) {
            std::cerr << "Error: Planner not initialized.\n";
            return false;
        }
        return optimize();
    }

    void showInitialState() {
        if (current_trajectory_.nodes.empty()) return;
        std::cout << "Initial state ready for visualization\n";
    }

    bool shouldVisualizeInitialState() const {
        return config_ && config_->visualize_initial_state;
    }

    virtual void initializeTrajectoryStructure(InterpolationMethod method = InterpolationMethod::LINEAR) {
        current_trajectory_.nodes.clear();
        current_trajectory_.total_time = total_time_;
        current_trajectory_.start_index = 0;
        current_trajectory_.goal_index = num_nodes_ - 1;
        fk_ = std::make_shared<IdentityFK>(num_dimensions_);
        
        for (size_t i = 0; i < num_nodes_; ++i) {
            float t = static_cast<float>(i) / (num_nodes_ - 1);
            VectorXf pos = (1 - t) * start_node_.position + t * goal_node_.position;
            current_trajectory_.nodes.emplace_back(pos, start_node_.radius);
        }
        computeRMatrix(num_nodes_, total_time_);
    }

    virtual bool optimize() = 0;

    virtual float computeStateCost(const Trajectory& trajectory) const {
        std::cerr << "Error: computeStateCost not implemented\n";
        return std::numeric_limits<float>::infinity();
    }

    float computeSmoothnessCost(const Trajectory& traj) const {
        if (traj.nodes.size() < 3) return 0.0f;
        float total_cost = 0.0f;
        for (size_t d = 0; d < num_dimensions_; ++d) {
            VectorXf Y_d(traj.nodes.size());
            for (size_t i = 0; i < traj.nodes.size(); ++i) Y_d(i) = traj.nodes[i].position(d);
            total_cost += Y_d.dot(R_matrix_ * Y_d);
        }
        return total_cost;
    }

    std::shared_ptr<ForwardKinematics> getForwardKinematics() const { return fk_; }
    const Trajectory& getCurrentTrajectory() const { return current_trajectory_; }
    const std::vector<Trajectory>& getTrajectoryHistory() const { return trajectory_history_; }
    size_t getNumDimensions() const { return num_dimensions_; }
    std::shared_ptr<const MotionPlannerConfig> getConfig() const { return config_; }
    bool isInitialized() const { return is_initialized_; }

    std::vector<Eigen::MatrixXf> sampleNoiseMatrices(size_t M, size_t N, size_t D) {
        std::vector<Eigen::MatrixXf> samples(M, MatrixXf(D, N));
        for (auto& eps : samples) {
            for (size_t d = 0; d < D; ++d) {
                auto noise = sampleSmoothnessNoise(N, random_engine_);
                for (size_t i = 0; i < N; ++i) eps(d, i) = noise[i];
            }
        }
        return samples;
    }

protected:
    CovarianceSchedule cov_schedule_;
    float cov_scale_initial_, cov_scale_final_, cov_decay_rate_, cov_step_factor_, cov_adaptive_threshold_, cov_scale_current_;
    size_t cov_step_interval_;

    virtual std::vector<float> sampleSmoothnessNoise(size_t n, std::mt19937& rng) const {
        if (n < 3) return std::vector<float>(n, 0.0f);
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd z(n - 2);
        for (int i = 0; i < z.size(); ++i) z(i) = dist(rng);
        Eigen::VectorXd noise = L_solver_.solve(z);
        
        std::vector<float> res(n, 0.0f);
        for (int i = 0; i < n - 2; ++i) res[i + 1] = static_cast<float>(noise(i));
        return res;
    }

    virtual void precomputeCholeskyFactorization(size_t n, float time) {
        if (n < 3) return;
        size_t nf = n - 2;
        double dt = static_cast<double>(time) / (n - 1);
        double scale = 1.0 / std::pow(dt, 4);
        
        Eigen::SparseMatrix<double> Rf(nf, nf);
        std::vector<Eigen::Triplet<double>> triplets;
        auto add = [&](int i, int j, double v) { 
            triplets.emplace_back(i, j, v * scale); 
            if(i != j) triplets.emplace_back(j, i, v * scale);
        };

        for (size_t i = 0; i < nf; ++i) {
            add(i, i, (i == 0 || i == nf - 1) ? 5.0 : 6.0);
            if (i < nf - 1) add(i, i + 1, -4.0);
            if (i < nf - 2) add(i, i + 2, 1.0);
        }
        Rf.setFromTriplets(triplets.begin(), triplets.end());
        L_solver_.compute(Rf);
    }

    bool initializeTrajectory() {
        try {
            if (!config_) return false;
            start_node_ = TrajectoryNode(VectorXf::Map(config_->start_position.data(), num_dimensions_), node_radius_);
            goal_node_  = TrajectoryNode(VectorXf::Map(config_->goal_position.data(), num_dimensions_), node_radius_);
            initializeTrajectoryStructure();
            initializeTask();
            return true;
        } catch (...) { return false; }
    }

    virtual void initializeTask() = 0;

    void computeRMatrix(size_t n, float time) {
        float dt = time / (n - 1);
        float scale = 1.0f / std::pow(dt, 4);
        R_matrix_ = SparseMatrixXf(n, n);
        if (n < 3) { R_matrix_.setIdentity(); return; }

        std::vector<Eigen::Triplet<float>> triplets;
        auto add = [&](int i, int j, float v) { 
            triplets.emplace_back(i, j, v * scale); 
            if(i != j) triplets.emplace_back(j, i, v * scale);
        };

        for (size_t i = 0; i < n; ++i) {
            float diag = (i == 0 || i == n - 1) ? 1.0f : (i == 1 || i == n - 2) ? 5.0f : 6.0f;
            add(i, i, diag);
            if (i < n - 1) add(i, i + 1, (i == 0 || i == n - 2) ? -2.0f : -4.0f);
            if (i < n - 2) add(i, i + 2, 1.0f);
        }
        R_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        precomputeCholeskyFactorization(n, time);
        storeTrajectory();
    }

    void storeTrajectory() { trajectory_history_.push_back(current_trajectory_); }

    Eigen::MatrixXf trajectoryToMatrix() const {
        MatrixXf Y(num_dimensions_, current_trajectory_.nodes.size());
        for (size_t i = 0; i < current_trajectory_.nodes.size(); ++i) Y.col(i) = current_trajectory_.nodes[i].position;
        return Y;
    }

    void updateTrajectoryFromMatrix(const Eigen::MatrixXf& Y_new) {
        for (size_t i = 0; i < Y_new.cols(); ++i) current_trajectory_.nodes[i].position = Y_new.col(i);
        current_trajectory_.nodes.front().position = start_node_.position;
        current_trajectory_.nodes.back().position = goal_node_.position;
    }

    Trajectory createPerturbedTrajectory(const Eigen::MatrixXf& Y_k, const Eigen::MatrixXf& epsilon) const {
        return matrixToTrajectory(Y_k + epsilon);
    }

    Trajectory matrixToTrajectory(const Eigen::MatrixXf& Y) const {
        Trajectory traj = current_trajectory_;
        for (size_t i = 0; i < Y.cols(); ++i) traj.nodes[i].position = Y.col(i);
        return traj;
    }

    Trajectory matrixToTrajectory(const MatrixXf& positions, const Trajectory& reference) const {
        Trajectory traj = reference;
        for (size_t i = 0; i < positions.cols(); ++i) {
            if (i < traj.nodes.size()) traj.nodes[i].position = positions.col(i);
            else traj.nodes.emplace_back(positions.col(i), 0.5f);
        }
        return traj;
    }

    void log(const std::string& msg) { if (log_file_.is_open()) log_file_ << msg << "\n"; std::cout << msg << "\n"; }
    template<typename... Args> void logf(const char* fmt, Args... args) {
        char buf[1024]; snprintf(buf, sizeof(buf), fmt, args...); log(std::string(buf));
    }
    void logSeparator(char c = '-', size_t len = 80) { log(std::string(len, c)); }
    virtual void logPlannerSpecificConfig() {}
    std::string getLogFilename() const { return log_filename_; }

private:
    void openLog() {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream ss; ss << "logs/" << getPlannerName() << "_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S") << ".log";
        log_filename_ = ss.str();
        std::filesystem::create_directories("logs");
        log_file_.open(log_filename_);
    }
    void closeLog() { if (log_file_.is_open()) log_file_.close(); }

protected:
    std::shared_ptr<MotionPlannerConfig> config_;
    std::shared_ptr<ForwardKinematics> fk_;
    Trajectory current_trajectory_;
    std::vector<Trajectory> trajectory_history_;
    TrajectoryNode start_node_, goal_node_;
    size_t num_dimensions_, num_nodes_;
    float total_time_, node_radius_;
    unsigned int random_seed_ = 42;
    SparseMatrixXf R_matrix_;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> L_solver_;
    bool is_initialized_ = false;
    mutable std::mt19937 random_engine_;
    std::ofstream log_file_;
    std::string log_filename_;
};