# ProxCEMP: Proximal Cross-Entropy Motion Planning

A modern C++ motion planning library implementing sampling-based trajectory optimization algorithms with emphasis on collision avoidance and smooth motion generation.

## Overview

ProxCEMP provides efficient, gradient-free optimization methods for trajectory planning in configuration spaces with obstacles. The library features a clean task-based architecture that separates optimization algorithms from problem-specific cost computations.

### Key Features

- **Multiple Optimization Algorithms**
  - **PCE (Proximal Cross-Entropy Method)**: Weighted sample-based trajectory optimization
  - **NGD (Natural Gradient Descent)**: Gradient-based optimization using natural gradients
  
- **Flexible Architecture**
  - Task-based design separating algorithm from problem definition
  - Header-only core library for easy integration
  - Support for arbitrary N-dimensional configuration spaces
  - Pluggable forward kinematics for robot-specific transformations

- **Robust Collision Avoidance**
  - Signed distance field (SDF) based collision detection
  - Configurable safety margins and collision thresholds
  - N-dimensional obstacle representation

- **Trajectory Quality**
  - Smoothness optimization via acceleration minimization
  - Configurable trajectory discretization
  - Start/goal constraint enforcement

- **Visualization & Analysis**
  - Real-time SFML-based visualization (2D/3D)
  - Trajectory history visualization
  - Collision status indicators
  - Export to PNG images

## Installation

### Prerequisites

**Required:**
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+
- [Eigen3](https://eigen.tuxfamily.org/) (3.3+)

**Optional:**
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) (for configuration files)
- [SFML 3](https://www.sfml-dev.org/) (for visualization)
- [GoogleTest](https://github.com/google/googletest) (for testing)
- [nlohmann/json](https://github.com/nlohmann/json) (for obstacle map serialization)

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libyaml-cpp-dev \
    libsfml-dev \
    libgtest-dev \
    nlohmann-json3-dev

# Clone repository
git clone --recurse-submodules https://github.com/hzyu17/ProxCEMP.git
cd ProxCEMP

# Build core library
mkdir build && cd build
cmake ..
make
sudo make install
```

### macOS

```bash
# Install dependencies via Homebrew
brew install cmake eigen yaml-cpp sfml googletest nlohmann-json

# Clone and build
git clone --recurse-submodules https://github.com/hzyu17/ProxCEMP.git
cd ProxCEMP
mkdir build && cd build
cmake ..
make
sudo make install
```

### Building 2D Examples

```bash
cd 2Dexamples
mkdir build && cd build
cmake ..
make

# Run examples
./main                              # Compare PCE vs NGD planners
./visualize_collision_checking      # Interactive collision visualization
./visualize_noisy_trj              # Visualize smoothness noise distribution
```

## Quick Start

### 1. Basic 2D Planning Example

```cpp
#include <PCEMotionPlanner.h>
#include <CollisionAvoidanceTask.h>
#include <yaml-cpp/yaml.h>

int main() {
    // Load configuration
    YAML::Node config = YAML::LoadFile("config.yaml");
    
    // Create collision avoidance task
    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);
    
    // Create PCE planner
    auto planner = std::make_shared<ProximalCrossEntropyMotionPlanner>(task);
    
    // Load planner configuration
    PCEConfig pce_config;
    pce_config.loadFromFile("config.yaml");
    
    // Initialize and solve
    planner->initialize(pce_config);
    bool success = planner->solve();
    
    // Get optimized trajectory
    const Trajectory& result = planner->getCurrentTrajectory();
    
    return success ? 0 : 1;
}
```

### 2. Configuration File Example

```yaml
# config.yaml
experiment:
  random_seed: 999
  visualize_initial_state: true

environment:
  map_width: 800
  map_height: 600
  num_obstacles: 20
  obstacle_radius: 20.0
  clearance_distance: 100.0
  
  cost:
    epsilon_sdf: 10.0
    sigma_obs: 10.0

motion_planning:
  num_dimensions: 2
  num_discretization: 50
  total_time: 10.0
  node_collision_radius: 15.0
  start_position: [50.0, 550.0]
  goal_position: [750.0, 50.0]

pce_planner:
  num_samples: 3000
  num_iterations: 20
  eta: 1.0
  temperature: 1.5
  convergence_threshold: 0.01
```

### 3. Custom Task Implementation

```cpp
#include <task.h>

class MyCustomTask : public pce::Task {
public:
    float computeCollisionCost(const Trajectory& trajectory) const override {
        // Your custom collision cost computation
        float cost = 0.0f;
        // ... implementation ...
        return cost;
    }
    
    bool filterTrajectory(Trajectory& trajectory, int iteration) override {
        // Optional: apply constraints (e.g., joint limits)
        return false;  // Return true if modified
    }
};
```

## Project Structure

```
ProxCEMP/
├── include/                      # Header-only library
│   ├── PCEMotionPlanner.h       # PCE algorithm implementation
│   ├── NGDMotionPlanner.h       # NGD algorithm implementation
│   ├── MotionPlanner.h          # Base planner interface
│   ├── task.h                   # Task interface (STOMP-style)
│   ├── CollisionAvoidanceTask.h # Collision avoidance task
│   ├── Trajectory.h             # Trajectory representation
│   ├── ObstacleMap.h            # N-D obstacle management
│   ├── ForwardKinematics.h      # FK transformations
│   └── visualization.h          # SFML visualization utilities
│
├── 2Dexamples/                  # 2D planning examples
│   ├── src/
│   │   ├── main.cpp            # PCE vs NGD comparison
│   │   ├── visualize_collision_checking.cpp
│   │   └── visualize_noisy_trj.cpp
│   ├── configs/
│   │   └── config.yaml         # Example configuration
│   └── tests/
│       └── test_determinism.cpp
│
├── config_editor_tkinter.py     # MuJoCo mobile manipulator tools
├── motion_planner.py
├── view_trajectory.py
│
├── Dockers/                     # ROS MoveIt benchmarking
│   └── ROS1MoveIt/
│
├── CMakeLists.txt              # Core library build
└── README.md
```

## Examples

### 2D Point Navigation

The `2Dexamples/` directory contains three visualization tools:

1. **`main`**: Compare PCE and NGD planners side-by-side
   ```bash
   cd 2Dexamples/build
   ./main
   ```

2. **`visualize_collision_checking`**: Interactive trajectory editor
   - Press `SPACE` to run optimization
   - Press `R` to reset trajectory
   - Press `C` to toggle collision spheres
   - Press `ESC` to quit

3. **`visualize_noisy_trj`**: Visualize smoothness noise distribution N(0, R⁻¹)
   - Shows how sampling noise spreads in workspace
   - Useful for understanding algorithm behavior

### MuJoCo Mobile Manipulator

Plan collision-free trajectories for a 12-DOF mobile manipulator:

```bash
# Setup environment
bash setup.sh
source activate.sh

# 1. Configure start/goal positions
python config_editor_tkinter.py

# 2. Plan trajectory
python motion_planner.py

# 3. Visualize result
python view_trajectory.py
```

## Algorithm Details

### Proximal Cross-Entropy Method (PCE)

PCE is a weighted sampling-based optimization method that:
1. Samples trajectories from a smoothness prior N(Y_k, R⁻¹)
2. Evaluates collision costs for each sample
3. Computes importance weights based on costs
4. Updates trajectory via weighted mean: Y_{k+1} = Σ w_m (Y_k + ε_m)

**Key parameters:**
- `num_samples`: Number of trajectory samples per iteration
- `temperature`: Controls weight concentration (higher = more uniform)
- `eta/gamma`: Learning rate parameters

### Natural Gradient Descent (NGD)

NGD uses the natural gradient of expected cost:
1. Samples trajectories from smoothness prior
2. Computes collision costs
3. Estimates gradient: ∇ = E[S(Ỹ)ε]
4. Updates: Y_{k+1} = (1-η)Y_k - η∇

**Key parameters:**
- `learning_rate`: Step size for gradient updates
- `temperature`: Scales gradient estimates

## Configuration Options

### Environment

| Parameter | Description | Default |
|-----------|-------------|---------|
| `map_width` | Environment width | 800 |
| `map_height` | Environment height | 600 |
| `num_obstacles` | Number of obstacles | 20 |
| `obstacle_radius` | Obstacle size | 20.0 |
| `clearance_distance` | Start/goal clearance | 100.0 |

### Cost Function

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epsilon_sdf` | SDF collision threshold | 10.0 |
| `sigma_obs` | Collision cost weight | 10.0 |

### Motion Planning

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_dimensions` | Config space dimensions | 2 |
| `num_discretization` | Trajectory waypoints | 50 |
| `total_time` | Trajectory duration | 10.0 |
| `node_collision_radius` | Safety margin | 15.0 |

## Testing

Run unit tests to verify deterministic behavior:

```bash
cd 2Dexamples/build
make test

# Or run directly
./test_pcem
```

Tests verify:
- Reproducible results with fixed random seeds
- Consistent obstacle generation
- Deterministic noise sampling
- Algorithm convergence

## ROS MoveIt Benchmarking

Compare ProxCEMP against ROS MoveIt planners (OMPL, STOMP, CHOMP):

```bash
cd Dockers/ROS1MoveIt
docker build -t moveit-stomp:noetic .
docker run -it --name moveit_stomp moveit-stomp:noetic

# Inside container
source /root/catkin_ws/devel/setup.bash
roslaunch moveit_resources_panda_moveit_config demo.launch pipeline:=stomp
```

## Performance Tips

1. **Tune sampling rate**: Start with 1000-3000 samples, increase for complex environments
2. **Adjust temperature**: Higher temperature (1.5-2.0) for exploration, lower (0.5-1.0) for refinement
3. **Discretization tradeoff**: More nodes = smoother but slower
4. **Early stopping**: Use `convergence_threshold` to stop when improvement is negligible

## Citation

If you use ProxCEMP in your research, please cite:

```bibtex
@software{proxcemp2025,
  title={ProxCEMP: Proximal Cross-Entropy Motion Planning},
  author={Yu, Hongzhe and others},
  year={2025},
  url={https://github.com/hzyu17/ProxCEMP}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Related Work

- **STOMP**: Stochastic Trajectory Optimization for Motion Planning
- **CHOMP**: Covariant Hamiltonian Optimization for Motion Planning  
- **TrajOpt**: Sequential Convex Optimization for motion planning
- **GPMP**: Gaussian Process Motion Planning

## Support

- **Issues**: [GitHub Issues](https://github.com/hzyu17/ProxCEMP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hzyu17/ProxCEMP/discussions)

## Acknowledgments

This project builds on research in sampling-based motion planning and trajectory optimization. Special thanks to the robotics community for open-source tools and algorithms.
