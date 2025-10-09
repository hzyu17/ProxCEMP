#!/usr/bin/env python3
"""
View trajectory interactively in MuJoCo
Compatible with macOS
"""

import mujoco
import numpy as np
import pickle
import glfw
import time
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    target_pos: np.ndarray
    task_type: str = "reach"

def load_path(filename='planned_path.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found! Run motion_planner.py first.")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

class TrajectoryViewer:
    def __init__(self, model_path, path, robot_nq):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.path = path
        self.robot_nq = robot_nq  # Number of robot DOFs
        self.path_idx = 0
        self.playing = True
        self.speed = 1.0
        
        # Save initial object configuration
        self.initial_qpos = self.data.qpos.copy()

        # Initialize GLFW
        if not glfw.init():
            raise Exception("Could not initialize GLFW")
        
        # Create window
        self.window = glfw.create_window(1200, 900, "Trajectory Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Could not create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Setup MuJoCo visualization
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Camera setup
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 4.0
        self.cam.lookat[:] = [1.0, 0, 0.5]
        
        # Setup callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)
        
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
    
    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.playing = not self.playing
                print("Playing" if self.playing else "Paused")
            elif key == glfw.KEY_R:
                self.path_idx = 0
                print("Reset to start")
            elif key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_UP:
                self.speed = min(3.0, self.speed + 0.2)
                print(f"Speed: {self.speed:.1f}x")
            elif key == glfw.KEY_DOWN:
                self.speed = max(0.2, self.speed - 0.2)
                print(f"Speed: {self.speed:.1f}x")
            elif key == glfw.KEY_LEFT:
                self.path_idx = max(0, self.path_idx - 10)
            elif key == glfw.KEY_RIGHT:
                self.path_idx = min(len(self.path) - 1, self.path_idx + 10)
    
    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        x, y = glfw.get_cursor_pos(window)
        self.lastx = x
        self.lasty = y
    
    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos
        
        width, height = glfw.get_window_size(window)
        
        if self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)
        elif self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)
        elif self.button_middle:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
            mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)
    
    def scroll(self, window, xoffset, yoffset):
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)
    
    def render_overlay(self, viewport):
        """Render text overlay with info"""
        # Prepare overlay text
        status = "PLAYING" if self.playing else "PAUSED"
        overlay = [
            f"Waypoint: {self.path_idx + 1}/{len(self.path)}",
            f"Status: {status}",
            f"Speed: {self.speed:.1f}x",
            "",
            "Controls:",
            "SPACE - Play/Pause",
            "R - Reset",
            "↑↓ - Speed",
            "←→ - Skip",
            "Q/ESC - Quit",
            "",
            "Mouse:",
            "Left - Rotate",
            "Right - Pan",
            "Scroll - Zoom"
        ]
        
        for i, line in enumerate(overlay):
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                line,
                "",
                self.context
            )
    
    def run(self):
        print("\n" + "="*60)
        print("TRAJECTORY VIEWER")
        print("="*60)
        print(f"Trajectory: {len(self.path)} waypoints")
        print("\nControls:")
        print("  SPACE    - Play/Pause")
        print("  R        - Reset to start")
        print("  ↑/↓      - Increase/Decrease speed")
        print("  ←/→      - Step backward/forward")
        print("  Q or ESC - Quit")
        print("\nMouse:")
        print("  Left drag   - Rotate camera")
        print("  Right drag  - Pan camera")
        print("  Scroll      - Zoom")
        print("="*60 + "\n")
        
        last_update = time.time()
        frame_time = 1.0 / 30.0  # 30 fps
        
        while not glfw.window_should_close(self.window):
            # Update trajectory
            current_time = time.time()
            if self.playing and (current_time - last_update) > (frame_time / self.speed):
                self.path_idx = (self.path_idx + 1) % len(self.path)
                last_update = current_time
            
            # IMPORTANT: Only update robot configuration, keep object fixed
            self.data.qpos[:] = self.initial_qpos  # Reset all
            self.data.qpos[:self.robot_nq] = self.path[self.path_idx]  # Update only robot
            mujoco.mj_forward(self.model, self.data)
            
            # Render
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            # Update scene
            mujoco.mjv_updateScene(
                self.model, self.data, self.opt, None, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL, self.scene
            )
            
            # Render scene
            mujoco.mjr_render(viewport, self.scene, self.context)
            
            # Render overlay
            self.render_overlay(viewport=viewport)
            
            # Swap buffers
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        # Cleanup
        glfw.terminate()

def main():
    print("Loading trajectory...")
    path_data = load_path('planned_path.pkl')
    
    if path_data is None:
        return
    
    print(f"Loaded: {len(path_data['path'])} waypoints")
    print(f"Robot DOFs: {path_data['robot_nq']}")
    print(f"Task: {path_data['task'].name}")
    print(f"Target (informational): {path_data['task'].target_pos}")
    print(f"Goal Config: {path_data['path'][-1]}")
    
    viewer = TrajectoryViewer(
        path_data['model_path'], 
        path_data['path'],
        path_data['robot_nq']  # Pass robot DOF count
    )
    viewer.run()

if __name__ == "__main__":
    main()