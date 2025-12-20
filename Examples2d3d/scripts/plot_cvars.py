import pandas as pd
import subprocess
import yaml
import os
import shutil
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PICKLE_PATH = os.path.join(ROOT_DIR, "stats_results.pkl")
CONFIG_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
BINARY = "./main_mute" # Ensure this is the compiled version of the code above
FIGURES_DIR = os.path.join(ROOT_DIR, "figures/extremes")

os.makedirs(FIGURES_DIR, exist_ok=True)
df = pd.read_pickle(PICKLE_PATH)

def generate_extreme_plots(planner_name, cost_col):
    sorted_df = df.sort_values(by=cost_col)
    # Get Absolute Best and Absolute Worst
    targets = [("best", sorted_df.iloc[0]['seed']), 
               ("worst", sorted_df.iloc[-1]['seed'])]
    
    for label, seed in targets:
        seed = int(seed)
        print(f"Generating {label} plot for {planner_name} (Seed: {seed})")
        
        # 1. Update config
        with open(CONFIG_PATH, 'r') as f:
            conf = yaml.safe_load(f)
        conf['experiment']['random_seed'] = seed
        conf['experiment']['visualize_initial_state'] = True # Triggers image save in C++
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(conf, f)

        # 2. Run C++
        subprocess.run([BINARY], cwd=BUILD_DIR, capture_output=True)

        # 3. Move the file
        # C++ saves as: traj_result_{Planner}_seed_{seed}.png
        gen_file = f"traj_result_{planner_name}_seed_{seed}.png"
        old_path = os.path.join(BUILD_DIR, gen_file)
        new_path = os.path.join(FIGURES_DIR, f"{planner_name}_{label}_seed_{seed}.png")

        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
        else:
            print(f"  Warning: {gen_file} not found in build directory.")

# Run
for p in ["PCE", "NGD", "CasADi"]:
    generate_extreme_plots(p, f"{p.lower() if p != 'CasADi' else 'casadi'}_cost")