#!/usr/bin/env python3
"""
Configuration Editor with Tkinter GUI (no extra dependencies)
"""

import mujoco
import numpy as np
import glfw
import pickle
import tkinter as tk
from tkinter import ttk
import threading
from dataclasses import dataclass


@dataclass
class Task:
    name: str
    target_pos: np.ndarray
    task_type: str = "reach"

class ConfigEditorTkinter:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Find robot DOFs (excluding target_object)
        self.robot_nq = self.model.nq
        try:
            obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == obj_body_id:
                    self.robot_nq = self.model.jnt_qposadr[i]
                    break
        except:
            pass
        
        print(f"Robot DOFs: {self.robot_nq}")
        
        # Configurations
        self.start_config = np.zeros(self.robot_nq)
        self.goal_config = np.zeros(self.robot_nq)
        
        # Build joint info
        self.joint_info = []
        if self.robot_nq >= 3:
            self.joint_info.append({'name': 'Base X', 'qpos_idx': 0, 'min': -5.0, 'max': 5.0})
            self.joint_info.append({'name': 'Base Y', 'qpos_idx': 1, 'min': -5.0, 'max': 5.0})
            self.joint_info.append({'name': 'Base Yaw', 'qpos_idx': 2, 'min': -np.pi, 'max': np.pi})
        
        # Arm joints
        for i in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jnt_type = self.model.jnt_type[i]
            qpos_adr = self.model.jnt_qposadr[i]
            
            if jnt_type == 0 or qpos_adr >= self.robot_nq:  # Skip non-hinge or out-of-range
                continue
            if qpos_adr < 3:  # Skip base joints (X, Y, yaw)
                continue
            
            q_min, q_max = self.model.jnt_range[i]
            self.joint_info.append({
                'name': jnt_name or f"joint_{i}",
                'qpos_idx': qpos_adr,
                'min': float(q_min),
                'max': float(q_max)
            })
        
        print("Joint Info:", self.joint_info)
        
        # Create Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Configuration Editor")
        self.root.geometry("450x800")
        
        self.current_mode = tk.StringVar(value="start")
        self.sliders = []
        self.value_labels = []
        self.setup_gui()
        
        # GLFW window for visualization
        self.setup_viewer()
        
        # Update loop
        self.running = True
        self.load_current_config()

    def setup_gui(self):
        """Setup Tkinter GUI"""
        default_font = ('TkDefaultFont', 12)
        style = ttk.Style()
        style.configure("TLabel", font=default_font)
        style.configure("TRadiobutton", font=default_font)
        style.configure("TButton", font=default_font, padding=8)
        
        mode_frame = ttk.Frame(self.root, padding="10")
        mode_frame.pack(fill=tk.X)
        
        ttk.Label(mode_frame, text="Mode:", style="TLabel").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="START", variable=self.current_mode, 
                        value="start", command=self.on_mode_change, style="TRadiobutton").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="GOAL", variable=self.current_mode, 
                        value="goal", command=self.on_mode_change, style="TRadiobutton").pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        canvas = tk.Canvas(self.root, height=600)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for info in self.joint_info:
            frame = ttk.Frame(scrollable_frame, padding="5")
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            label = ttk.Label(frame, text=info['name'], width=20, style="TLabel", anchor="w")
            label.pack(side=tk.LEFT, padx=5)
            
            slider = tk.Scale(frame, from_=info['min'], to=info['max'], 
                            resolution=0.01, orient=tk.HORIZONTAL, 
                            command=lambda val, idx=info['qpos_idx']: self.on_slider_change(idx, val),
                            font=default_font)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            slider.set(self.data.qpos[info['qpos_idx']])
            
            value_label = ttk.Label(frame, text=f"{self.data.qpos[info['qpos_idx']]:.3f}", width=8, style="TLabel")
            value_label.pack(side=tk.LEFT, padx=5)
            
            self.sliders.append((info['qpos_idx'], slider))
            self.value_labels.append((info['qpos_idx'], value_label))
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(button_frame, text="Copy START to GOAL", 
                command=self.copy_start_to_goal, style="TButton").pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(button_frame, text="Copy GOAL to START", 
                command=self.copy_goal_to_start, style="TButton").pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(button_frame, text="Reset to Zero", 
                command=self.reset_config, style="TButton").pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(button_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save Configurations", 
                command=self.save_to_file, style="TButton").pack(fill=tk.X, padx=5, pady=5, ipady=10)

    def setup_viewer(self):
        """Setup GLFW viewer"""
        if not glfw.init():
            raise Exception("Could not initialize GLFW")
        
        self.window = glfw.create_window(1200, 900, "Robot View", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Could not create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 4.0
        self.cam.lookat[:] = [1.0, 0, 0.5]
        
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_scroll_callback(self.window, self.scroll)
        
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.lastx = 0
        self.lasty = 0
    
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
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 
                                 dx/height, dy/height, self.scene, self.cam)
        elif self.button_right:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 
                                 dx/height, dy/height, self.scene, self.cam)
        elif self.button_middle:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 
                                 dx/height, dy/height, self.scene, self.cam)
    
    def scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 
                             0.0, -0.05 * yoffset, self.scene, self.cam)
    
    def on_mode_change(self):
        self.load_current_config()
        print(f"Switched to {self.current_mode.get().upper()} mode")
    
    def on_slider_change(self, qpos_idx, value):
        print(f"Slider changed: qpos_idx={qpos_idx}, value={value}, qpos before={self.data.qpos[:self.robot_nq].copy()}")
        self.data.qpos[qpos_idx] = float(value)
        self.save_current_config()
        self.update_value_labels()
        mujoco.mj_forward(self.model, self.data)
    
    def update_value_labels(self):
        for qpos_idx, label in self.value_labels:
            label.config(text=f"{self.data.qpos[qpos_idx]:.3f}")
    
    def save_current_config(self):
        config = self.data.qpos[:self.robot_nq].copy()
        if self.current_mode.get() == "start":
            self.start_config = config
        else:
            self.goal_config = config
    
    def load_current_config(self):
        config = self.start_config if self.current_mode.get() == "start" else self.goal_config
        self.data.qpos[:self.robot_nq] = config
        for qpos_idx, slider in self.sliders:
            slider.set(self.data.qpos[qpos_idx])
        self.update_value_labels()
        mujoco.mj_forward(self.model, self.data)
    
    def copy_start_to_goal(self):
        self.goal_config = self.start_config.copy()
        if self.current_mode.get() == "goal":
            self.load_current_config()
        print("Copied START → GOAL")
    
    def copy_goal_to_start(self):
        self.start_config = self.goal_config.copy()
        if self.current_mode.get() == "start":
            self.load_current_config()
        print("Copied GOAL → START")
    
    def reset_config(self):
        config = np.zeros(self.robot_nq)
        if self.current_mode.get() == "start":
            self.start_config = config
        else:
            self.goal_config = config
        self.load_current_config()
        print(f"Reset {self.current_mode.get().upper()}")
    
    def save_to_file(self):
        data = {
            'start': self.start_config,
            'goal': self.goal_config,
            'robot_nq': self.robot_nq,
            'task': Task(name="reach_target", target_pos=np.array([1.5, 0.3, 0.53]))
        }
        with open('robot_configs.pkl', 'wb') as f:
            pickle.dump(data, f)
        print("Configurations saved to robot_configs.pkl")
    
    def update_viewer(self):
        mujoco.mj_forward(self.model, self.data)
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
    
    def run(self):
        print("\n" + "="*60)
        print("CONFIGURATION EDITOR")
        print("="*60)
        print("Use the GUI to set START and GOAL configurations")
        print("="*60 + "\n")
        
        def update_viewer_periodic():
            if self.running and not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.update_viewer()
                self.root.after(16, update_viewer_periodic)
            else:
                self.on_closing()
        
        self.root.after(16, update_viewer_periodic)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.running = False
        glfw.terminate()
        self.root.destroy()

def main():
    model_path = "robot_files/manipulation_scene.xml"
    print("Loading model...")
    editor = ConfigEditorTkinter(model_path)
    editor.run()

if __name__ == "__main__":
    main()