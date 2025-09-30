import mujoco
import mujoco.viewer
import time
import numpy as np

# Path to the model XML file
model_path = 'planning_envs/point3D_scene.xml'

try:
    # Load the model and data
    print(f"Loading model from: {model_path}")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    print("Model loaded successfully!")
    
    # Print some model info to verify it's loading correctly
    print(f"Number of bodies: {m.nbody}")
    print(f"Number of geoms: {m.ngeom}")
    
    # List all geom names to verify your coordinate axes are there
    print("Geom names:")
    for i in range(m.ngeom):
        geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name:
            print(f"  {i}: {geom_name}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Launch the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    print("Starting simulation with coordinate frames...")
    
    # Enable frame visualization
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    # Controller gains for the PD controller
    Kp = 50.0
    Kd = 10.0

    # Generate a simple desired trajectory (a figure-eight)
    def desired_trajectory(t):
        p_des = np.array([
            2,
            2,
            1.5
        ])
        v_des = np.array([
            0,0,0
        ])
        return p_des, v_des

    duration = 20.0
    start_time = time.time()
    
    while True:
        sim_time = d.time

        # Get the desired position and velocity from the trajectory function
        p_desired, v_desired = desired_trajectory(sim_time)

        # Get current position and velocity from the simulation
        p_current = d.qpos[0:3]
        v_current = d.qvel[0:3]
        
        # Compute control forces based on the PD controller
        error_p = p_desired - p_current
        error_v = v_desired - v_current
        
        # Calculate the desired acceleration
        a_desired = Kp * error_p + Kd * error_v
        
        # Add a gravity compensation term for the z-axis
        a_desired[2] += -m.opt.gravity[2]

        # Apply forces as control inputs
        d.ctrl[0] = m.body('robot_body').mass[0] * a_desired[0]
        d.ctrl[1] = m.body('robot_body').mass[0] * a_desired[1]
        d.ctrl[2] = m.body('robot_body').mass[0] * a_desired[2]
        
        # Step the simulation
        mujoco.mj_step(m, d)
        viewer.sync()
    
    print("Simulation finished.")