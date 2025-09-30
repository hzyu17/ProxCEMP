import mujoco
import mujoco.viewer
import numpy as np

# Path to the model XML file
model_path = 'planning_envs/point3D_scene.xml'

# Load the model
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

print("=== GEOMETRY INFORMATION ===")
print(f"Total number of geoms: {m.ngeom}")
print()

# Print details for each geometry
for i in range(m.ngeom):
    geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    if geom_name:
        geom_type = m.geom_type[i]
        geom_size = m.geom_size[i]
        geom_rgba = m.geom_rgba[i]
        
        # Convert geom type number to name
        type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 
                     5: 'cylinder', 6: 'box', 7: 'mesh'}
        type_name = type_names.get(geom_type, f'unknown({geom_type})')
        
        print(f"Geom {i}: '{geom_name}'")
        print(f"  Type: {type_name}")
        print(f"  Size: {geom_size}")
        print(f"  RGBA: {geom_rgba}")
        print()

# Specifically check robot geometry
robot_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'robot_geom')
if robot_geom_id >= 0:
    robot_size = m.geom_size[robot_geom_id]
    robot_rgba = m.geom_rgba[robot_geom_id]
    print(f"ROBOT GEOMETRY:")
    print(f"  Expected size: 1.0 (if you changed it)")
    print(f"  Actual size: {robot_size[0]} (radius)")
    print(f"  Expected color: [0.0, 0.5, 0.9, 1.0] (blue)")
    print(f"  Actual color: {robot_rgba}")
    print()

# Check coordinate frame axes
for axis_name in ['x_axis', 'y_axis', 'z_axis']:
    axis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, axis_name)
    if axis_id >= 0:
        axis_size = m.geom_size[axis_id]
        axis_rgba = m.geom_rgba[axis_id]
        print(f"{axis_name.upper()}:")
        print(f"  Size: radius={axis_size[0]}, half_length={axis_size[1]}")
        print(f"  Color: {axis_rgba}")
        print()

print("=== FILE CONTENT CHECK ===")
# Read the actual file content to see what's in there
with open(model_path, 'r') as f:
    content = f.read()
    
# Look for the robot_geom line
import re
robot_pattern = r'<geom name="robot_geom"[^>]*>'
robot_match = re.search(robot_pattern, content)
if robot_match:
    print("Found robot_geom line in XML:")
    print(f"  {robot_match.group()}")
else:
    print("Could not find robot_geom in XML file!")
    
# Look for x_axis line
x_axis_pattern = r'<geom name="x_axis"[^>]*>'
x_axis_match = re.search(x_axis_pattern, content)
if x_axis_match:
    print("Found x_axis line in XML:")
    print(f"  {x_axis_match.group()}")
else:
    print("Could not find x_axis in XML file!")