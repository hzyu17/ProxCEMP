import yaml
import subprocess
import pandas as pd
import numpy as np
import os
import pickle
import time
from tqdm import tqdm

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Use absolute paths to prevent "YAML::BadFile" errors
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
BINARY_NAME = "./main_mute" 
PICKLE_PATH = os.path.join(ROOT_DIR, "stats_results.pkl")

NUM_RUNS = 10
START_SEED = 1000

results = []

def get_stats(series):
    """Calculates Mean, Std, and Best/Worst 10% averages."""
    if series.empty:
        return {"Error": 0}
    n_10 = max(1, int(len(series) * 0.1))
    sorted_s = series.sort_values()
    return {
        "Mean": series.mean(),
        "Std": series.std(),
        "Best 10% Avg": sorted_s.head(n_10).mean(),
        "Worst 10% Avg": sorted_s.tail(n_10).mean()
    }

# ==========================================
# EXPERIMENT LOOP
# ==========================================
print(f"Starting statistical study: {NUM_RUNS} runs starting from seed {START_SEED}...")

for i in tqdm(range(NUM_RUNS)):
    current_seed = START_SEED + i
    
    # 1. Update YAML configuration with the new seed
    try:
        with open(CONFIG_PATH, 'r') as f:
            conf = yaml.safe_load(f)
        
        conf['experiment']['random_seed'] = current_seed
        conf['experiment']['visualize_initial_state'] = False 
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(conf, f)
    except Exception as e:
        print(f"\n[FATAL] Failed to write config: {e}")
        break

    # 2. Run the binary inside the build directory
    try:
        # We set cwd=BUILD_DIR so the C++ binary can find ../configs/config.yaml
        process = subprocess.run(
            [BINARY_NAME], 
            cwd=BUILD_DIR, 
            capture_output=True, 
            text=True, 
            timeout=120 # Planners may take longer on complex seeds
        )
        
        # 3. Parse the DATA_RESULT line from stdout
        found_data = False
        for line in process.stdout.split('\n'):
            if line.startswith("DATA_RESULT"):
                parts = line.split(',')
                # Format: DATA_RESULT, seed, pce_cost, pce_time, ngd_cost, ngd_time, cas_cost, cas_time
                results.append({
                    "seed": int(parts[1]),
                    "pce_cost": float(parts[2]),
                    "pce_time": float(parts[3]),
                    "ngd_cost": float(parts[4]),
                    "ngd_time": float(parts[5]),
                    "casadi_cost": float(parts[6]),
                    "casadi_time": float(parts[7])
                })
                found_data = True
        
        if not found_data:
            print(f"\n[ERROR] Seed {current_seed} failed to produce output. Check build/logs/.")
            if i == 0: # Print debug info for the first failure
                print(f"STDOUT: {process.stdout}")
                print(f"STDERR: {process.stderr}")

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Seed {current_seed} exceeded time limit.")
    except Exception as e:
        print(f"\n[ERROR] Seed {current_seed}: {e}")

# ==========================================
# POST-PROCESSING & ANALYSIS
# ==========================================
if not results:
    print("\n[CRITICAL] No data results were captured. Script exiting.")
    exit(1)

# Convert to DataFrame and save to Pickle
df = pd.DataFrame(results)
df.to_pickle(PICKLE_PATH)
print(f"\nResults saved to: {PICKLE_PATH}")

print("\n" + "="*60)
print(f"   STATISTICAL RESULTS (N={len(df)})")
print("="*60)

# Evaluate each planner
planners = [
    ("PCE", "pce_cost", "pce_time"),
    ("NGD", "ngd_cost", "ngd_time"),
    ("CasADi", "casadi_cost", "casadi_time")
]

for name, cost_col, time_col in planners:
    if cost_col in df.columns:
        stats = get_stats(df[cost_col])
        avg_time = df[time_col].mean()
        
        print(f"\n[{name}]")
        print(f"  Avg Time        : {avg_time:.4f}s")
        for key, val in stats.items():
            print(f"  {key:15}: {val:.4f}")

# Optional: Success rate (assuming cost < 10000 is a success/no-collision)
THRESHOLD = 10000.0
print("\n" + "="*60)
print(f"   SUCCESS RATES (Cost < {THRESHOLD})")
print("="*60)
for name, cost_col, _ in planners:
    success_rate = (df[cost_col] < THRESHOLD).mean() * 100
    print(f"  {name:15}: {success_rate:.2f}%")
