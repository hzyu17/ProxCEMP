#!/usr/bin/env python3
"""
Motion Planning for Mobile Manipulation (Robot Only)
"""

import mujoco
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import time
import pickle

@dataclass
class Task:
    name: str
    target_pos: np.ndarray
    task_type: str = "reach"

class RRTPlanner:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_nq: int):
        self.model = model
        self.data = data
        self.robot_nq = robot_nq  # Robot DOFs (12: 3 base + 9 arm)
        
        # Store default qpos for non-robot DOFs
        self.default_qpos = self.data.qpos.copy()
        
        # Check for base joints (2 translations + 1 yaw = 3 DOFs)
        self.has_base_joints = self.robot_nq >= 3
        if self.has_base_joints:
            self.base_qpos_dim = 3  # X, Y, yaw
            self.arm_indices = list(range(3, self.robot_nq))
            self.num_arm_joints = self.robot_nq - self.base_qpos_dim  # 12 - 3 = 9
            self.q_min = self.model.jnt_range[3:3+self.num_arm_joints, 0]  # Arm joints
            self.q_max = self.model.jnt_range[3:3+self.num_arm_joints, 1]
        else:
            self.base_qpos_dim = 0
            self.arm_indices = list(range(self.robot_nq))
            self.num_arm_joints = self.robot_nq
            self.q_min = self.model.jnt_range[:, 0]
            self.q_max = self.model.jnt_range[:, 1]
        
        self.step_size = 0.15
        self.max_iterations = 3000
        self.goal_tolerance = 0.15
        self.goal_bias = 0.15
        
        print(f"Planner: {self.model.nq} DOFs (total), {self.robot_nq} DOFs (robot), Base joints: {self.has_base_joints}")
        print(f"Arm indices: {self.arm_indices}, q_min length: {len(self.q_min)}")
    
    def check_collisions_detailed(self, q: np.ndarray) -> tuple[bool, list]:
        if len(q) != self.robot_nq:
            raise ValueError(f"Input q has {len(q)} DOFs, expected {self.robot_nq}")
        self.data.qpos[:] = self.default_qpos
        self.data.qpos[:self.robot_nq] = q
        mujoco.mj_forward(self.model, self.data)
        
        collisions = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            if geom1_id == 0 or geom2_id == 0:  # Skip floor
                continue
            
            if geom1_name and geom2_name:
                if ('table' in geom1_name or 'table' in geom2_name) and \
                   ('object' in geom1_name or 'object' in geom2_name):
                    continue
            
            if contact.dist < -0.005:
                collisions.append({
                    'geom1': geom1_name or f"geom_{geom1_id}",
                    'geom2': geom2_name or f"geom_{geom2_id}",
                    'dist': contact.dist
                })
        
        return len(collisions) == 0, collisions
    
    def is_collision_free(self, q: np.ndarray) -> bool:
        collision_free, _ = self.check_collisions_detailed(q)
        return collision_free
    
    def sample_config(self) -> np.ndarray:
        q = np.zeros(self.robot_nq)
        if self.has_base_joints:
            q[0] = np.random.uniform(-5, 5)  # Base X
            q[1] = np.random.uniform(-5, 5)  # Base Y
            q[2] = np.random.uniform(-np.pi, np.pi)  # Yaw
        for i in range(self.num_arm_joints):
            q[i + self.base_qpos_dim] = np.random.uniform(self.q_min[i], self.q_max[i])
        return q
    
    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        if self.has_base_joints:
            # Weight translation and yaw differently
            trans_dist = np.linalg.norm(q1[:2] - q2[:2])  # X, Y
            yaw_dist = min(abs(q1[2] - q2[2]), 2 * np.pi - abs(q1[2] - q2[2]))  # Angular distance
            arm_dist = np.linalg.norm(q1[3:] - q2[3:])  # Arm joints
            return 0.5 * trans_dist + 0.5 * yaw_dist + arm_dist
        return np.linalg.norm(q1 - q2)
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        direction = q_to - q_from
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return q_to.copy()
        
        q_new = q_from + (direction / dist) * self.step_size
        if self.has_base_joints:
            q_new[0] = np.clip(q_new[0], -5, 5)  # Clip X
            q_new[1] = np.clip(q_new[1], -5, 5)  # Clip Y
            q_new[2] = np.clip(q_new[2], -np.pi, np.pi)  # Clip yaw
            for i in range(self.num_arm_joints):
                q_new[i + self.base_qpos_dim] = np.clip(q_new[i + self.base_qpos_dim], self.q_min[i], self.q_max[i])
        return q_new
    
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, task: Task) -> Optional[List[np.ndarray]]:
        print(f"\nPlanning to goal configuration (distance: {self.distance(q_start, q_goal):.3f})...")
        
        collision_free, collisions = self.check_collisions_detailed(q_start)
        if not collision_free:
            print("  ERROR: Start in collision!")
            for col in collisions[:5]:
                print(f"    {col['geom1']} <-> {col['geom2']} (dist: {col['dist']:.4f})")
            return None
        
        collision_free, collisions = self.check_collisions_detailed(q_goal)
        if not collision_free:
            print("  ERROR: Goal in collision!")
            for col in collisions[:5]:
                print(f"    {col['geom1']} <-> {col['geom2']} (dist: {col['dist']:.4f})")
            return None
        
        tree = {0: q_start.copy()}
        parent = {0: None}
        node_id = 1
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            q_rand = q_goal.copy() if np.random.random() < self.goal_bias else self.sample_config()
            nearest_id = min(tree.keys(), key=lambda i: self.distance(tree[i], q_rand))
            q_new = self.steer(tree[nearest_id], q_rand)
            
            if self.is_collision_free(q_new):
                tree[node_id] = q_new
                parent[node_id] = nearest_id
                
                if self.distance(q_new, q_goal) < self.goal_tolerance:
                    print(f"  SUCCESS! {iteration} iters, {time.time()-start_time:.2f}s")
                    
                    path = [q_goal]
                    current = node_id
                    while parent[current] is not None:
                        path.append(tree[current])
                        current = parent[current]
                    path.append(q_start)
                    path.reverse()
                    
                    print(f"  Path: {len(path)} waypoints")
                    return path
                
                node_id += 1
            
            if iteration % 500 == 0 and iteration > 0:
                print(f"  Iter {iteration}, tree size: {len(tree)}")
        
        print("  FAILED!")
        return None
    
    def interpolate_path(self, path: List[np.ndarray], max_step: float = 0.05) -> List[np.ndarray]:
        if len(path) <= 1:
            return path
        
        print(f"Interpolating ({len(path)} waypoints)...")
        interpolated = [path[0].copy()]
        
        for i in range(len(path) - 1):
            q1, q2 = path[i], path[i + 1]
            dist = self.distance(q1, q2)
            num_steps = max(2, int(np.ceil(dist / max_step)))
            
            for j in range(1, num_steps):
                q_interp = q1 + (j / num_steps) * (q2 - q1)
                if self.has_base_joints:
                    q_interp[0] = np.clip(q_interp[0], -5, 5)  # Clip X
                    q_interp[1] = np.clip(q_interp[1], -5, 5)  # Clip Y
                    q_interp[2] = np.clip(q_interp[2], -np.pi, np.pi)  # Clip yaw
                interpolated.append(q_interp)
            
            interpolated.append(q2.copy())
        
        print(f"  Interpolated to {len(interpolated)} waypoints")
        return interpolated

def load_robot_configs(filename='robot_configs.pkl'):
    try:
        with open(filename, 'rb') as f:
            configs = pickle.load(f)
            print(f"Loaded configs: {list(configs.keys())}")
            return configs
    except FileNotFoundError:
        print(f"Configuration file {filename} not found!")
        print("Run: python config_editor_tkinter.py")
        return None

def find_collision_free_start(model, data, planner, robot_nq):
    print("\nSearching for collision-free start configuration...")
    
    if model.nkey > 0:
        for key_id in range(model.nkey):
            key_qpos = model.key_qpos[key_id][:robot_nq]
            collision_free, _ = planner.check_collisions_detailed(key_qpos)
            if collision_free:
                print(f"  Found keyframe {key_id} is collision-free")
                return key_qpos.copy()
    
    test_configs = [
        [0.0, 0.0, 0.0] + [0] * (robot_nq - 3),  # All zeros
        [0.5, 0.0, 0.0] + [0] * (robot_nq - 3),  # Moved forward
        [0.0, 0.5, 0.0] + [0] * (robot_nq - 3),  # Moved sideways
        [0.5, 0.0, 0.0, 0.3, -0.3, 0.2] + [0] * (robot_nq - 6),  # With arm pose
    ]
    
    for i, config in enumerate(test_configs):
        if len(config) == robot_nq:
            collision_free, collisions = planner.check_collisions_detailed(config)
            if collision_free:
                print(f"  Found test config {i} is collision-free")
                return np.array(config)
            else:
                print(f"  Test config {i} has {len(collisions)} collisions")
    
    print("  Trying random sampling...")
    for attempt in range(100):
        q = planner.sample_config()
        if planner.is_collision_free(q):
            print(f"  Found collision-free config after {attempt+1} samples")
            return q
    
    print("  WARNING: Could not find collision-free start!")
    return None

def main():
    model_path = "robot_files/manipulation_scene.xml"
    
    print("Loading configurations...")
    configs = load_robot_configs()
    
    if configs is None:
        print("\nPlease create configurations first:")
        print("  python config_editor_tkinter.py")
        return
    
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    robot_nq = configs['robot_nq']
    q_start = np.array(configs['start'])
    q_goal = np.array(configs['goal'])
    
    task = configs.get('task', Task(name="reach_target", target_pos=np.array([1.5, 0.3, 0.53])))
    
    print(f"\nLoaded configurations:")
    print(f"  Robot DOFs: {robot_nq}")
    print(f"  Total DOFs: {model.nq}")
    print(f"  Task: {task.name} (type: {task.task_type}, target: {task.target_pos})")
    print(f"\nStart config:")
    print(f"  Base: [{q_start[0]:.3f}, {q_start[1]:.3f}, yaw: {q_start[2]:.3f}]")
    if len(q_start) > 3:
        print(f"  Joints: {q_start[3:]}")
    print(f"\nGoal config:")
    print(f"  Base: [{q_goal[0]:.3f}, {q_goal[1]:.3f}, yaw: {q_goal[2]:.3f}]")
    if len(q_goal) > 3:
        print(f"  Joints: {q_goal[3:]}")
    
    planner = RRTPlanner(model, data, robot_nq)
    
    print("\nVerifying configurations...")
    collision_free, collisions = planner.check_collisions_detailed(q_start)
    if not collision_free:
        print(f"WARNING: Start has {len(collisions)} collisions!")
        for col in collisions[:3]:
            print(f"  {col['geom1']} <-> {col['geom2']}")
        q_start = find_collision_free_start(model, data, planner, robot_nq)
        if q_start is None:
            print("ERROR: Cannot find valid start config")
            return
        print("Using new collision-free start config")
    
    path = planner.plan(q_start, q_goal, task)
    
    if path:
        path = planner.interpolate_path(path, max_step=0.05)
        
        robot_path = [q[:robot_nq] for q in path]
        
        path_data = {
            'path': robot_path,
            'robot_nq': robot_nq,
            'model_path': model_path,
            'task': task
        }
        
        with open('planned_path.pkl', 'wb') as f:
            pickle.dump(path_data, f)
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Path saved to: planned_path.pkl")
        print(f"  Waypoints: {len(robot_path)}")
        print(f"  Goal Config: {q_goal}")
        print(f"{'='*60}")
        print("\nNext: python view_trajectory.py")
    else:
        print("\nFailed to plan!")

if __name__ == "__main__":
    main()