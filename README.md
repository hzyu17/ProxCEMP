# Motion Planning Package For Mobile Manipulator 

This package provides a set of Python scripts to simulate and plan collision-free motion for a mobile manipulator robot in a MuJoCo environment. The robot has a mobile base with 3 degrees of freedom (X, Y, yaw) and a 9-DOF arm, operating in a scene with a target object. The package includes tools to configure start and goal positions, plan a path using an RRT algorithm, and visualize the resulting trajectory.

## Features
- **Configuration Editor**: Interactively set start and goal configurations using a Tkinter GUI (`config_editor_tkinter.py`).
- **Motion Planning**: Plan a collision-free path using Rapidly-exploring Random Trees (RRT) (`motion_planner.py`).
- **Trajectory Visualization**: Visualize the planned path in a MuJoCo viewer (`view_trajectory.py`).
- **MuJoCo Model**: Custom robot model with a mobile base and arm (`robot_movable.xml`).

## Prerequisites
- **Python 3.11**
- **Required Python packages**:
  - `mujoco` (MuJoCo Python bindings)
  - `numpy`
  - `glfw` (for visualization)
  - `tkinter` (included with standard Python)
- **MuJoCo**: Install the MuJoCo physics engine (version 2.3 or later).
- **Assets**: Ensure STL and OBJ mesh files and textures are in the `robot_files/assets/` directory.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hzyu17/ProxCEMP.git
   cd ProxCEMP
   ```

2. **Set Up Environment**:
   Run the provided setup script to install dependencies and create a virtual environment:
     ```bash
     bash setup.sh
     ```

### Directory Structure: Ensure the following structure:
```
ProxCEMP/
├── robot_files/
│   ├── manipulation_scene.xml
│   ├── robot_movable.xml
│   ├── assets/
│   │   ├── link_base_v.obj
│   │   ├── link_torso_v.obj
│   │   ├── ... (other STL/OBJ files)
│   │   ├── robot_texture.png
│   │   ├── finger_base_texture.png
│   │   ├── finger_tip_texture.png
├── config_editor_tkinter.py
├── motion_planner.py
├── view_trajectory.py
├── robot_configs.pkl (generated)
├── planned_path.pkl (generated)
```


# Usage
The package operates in three steps: configure, plan, and visualize.

## 1. Configure Start and Goal Positions
Run the configuration editor to set the robot's start and goal configurations:
python config_editor_tkinter.py


### GUI Controls:
Select START or GOAL mode using radio buttons.
Adjust sliders for Base X, Base Y, Base Yaw, and 9 arm joints (torso, shoulder, bicep, elbow, forearm, wrist, gripper, finger_right, finger_left).


### Output: 
Saves robot_configs.pkl with start/goal configurations (12 DOFs: 3 base + 9 arm).

## 2. Plan a Collision-Free Path
Run the motion planner to generate a path from start to goal:
```
python motion_planner.py
```

### Process:

Loads robot_configs.pkl.
Uses RRT to plan a collision-free path, avoiding obstacles (e.g., table, walls).
Interpolates the path for smooth motion.


### Output: 
Saves planned_path.pkl with the trajectory (list of 12-DOF configurations).



## 3. Visualize the Trajectory
Run the viewer to visualize the planned path
```
python view_trajectory.py
```

Process:
Loads planned_path.pkl.
Animates the robot following the trajectory in the MuJoCo viewer.


Controls:
Use mouse buttons to rotate (left), pan (right), or zoom (middle/scroll).
Close the window to exit.

