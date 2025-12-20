#!/usr/bin/env python3
"""
Multi-Solver Benchmark Script
=============================
Benchmarks PCE, NGD, and multiple CasADi solver variants (IPOPT, ADAM, LBFGS, SCP, etc.)
across multiple random seeds for statistical analysis.

Usage:
    python run_full_benchmark.py [--config experiment_config.yaml]
"""

import yaml
import subprocess
import pandas as pd
import numpy as np
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PLANNER_CONFIG_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
BINARY_NAME = "./main_mute"


class BenchmarkRunner:
    def __init__(self, experiment_config_path: str):
        """Initialize benchmark runner with experiment configuration."""
        with open(experiment_config_path, 'r') as f:
            self.exp_config = yaml.safe_load(f)
        
        self.num_runs = self.exp_config['benchmark']['num_runs']
        self.start_seed = self.exp_config['benchmark']['start_seed']
        self.timeout = self.exp_config['benchmark']['timeout_seconds']
        self.success_threshold = self.exp_config['benchmark']['success_threshold']
        
        # Store original planner config for restoration
        self.original_config = self._load_planner_config()
        
        # Results storage: {solver_name: [list of run results]}
        self.all_results: Dict[str, List[Dict]] = {}
        
    def _load_planner_config(self) -> Dict:
        """Load the planner configuration YAML."""
        with open(PLANNER_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_planner_config(self, config: Dict):
        """Save the planner configuration YAML."""
        with open(PLANNER_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _update_config_for_run(self, seed: int, casadi_solver: Optional[str] = None):
        """Update planner config with seed and optional CasADi solver type."""
        config = self._load_planner_config()
        config['experiment']['random_seed'] = seed
        config['experiment']['visualize_initial_state'] = False
        
        if casadi_solver is not None:
            config['casadi_planner']['solver'] = casadi_solver
        
        self._save_planner_config(config)
    
    def _parse_output(self, stdout: str, casadi_variant: str) -> Optional[Dict]:
        """
        Parse the DATA_RESULT line from binary output.
        
        Expected format:
        DATA_RESULT, seed, pce_cost, pce_time, ngd_cost, ngd_time, cas_cost, cas_time
        """
        for line in stdout.split('\n'):
            if line.startswith("DATA_RESULT"):
                parts = line.split(',')
                if len(parts) >= 8:
                    return {
                        "seed": int(parts[1]),
                        "pce_cost": float(parts[2]),
                        "pce_time": float(parts[3]),
                        "ngd_cost": float(parts[4]),
                        "ngd_time": float(parts[5]),
                        "casadi_cost": float(parts[6]),
                        "casadi_time": float(parts[7]),
                        "casadi_variant": casadi_variant
                    }
        return None
    
    def run_single_experiment(self, seed: int, casadi_solver: str, 
                               casadi_variant_name: str) -> Optional[Dict]:
        """Run a single experiment with given seed and CasADi solver."""
        self._update_config_for_run(seed, casadi_solver)
        
        try:
            process = subprocess.run(
                [BINARY_NAME],
                cwd=BUILD_DIR,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            result = self._parse_output(process.stdout, casadi_variant_name)
            if result is None:
                print(f"\n[WARN] Seed {seed}, {casadi_variant_name}: No DATA_RESULT in output")
                # Debug: print first failure's output
                if seed == self.start_seed:
                    print(f"STDOUT: {process.stdout[:500]}")
                    print(f"STDERR: {process.stderr[:500]}")
            return result
            
        except subprocess.TimeoutExpired:
            print(f"\n[TIMEOUT] Seed {seed}, {casadi_variant_name}")
            return None
        except Exception as e:
            print(f"\n[ERROR] Seed {seed}, {casadi_variant_name}: {e}")
            return None
    
    def run_casadi_variant_benchmark(self, variant_name: str, solver_key: str) -> List[Dict]:
        """Run benchmark for a specific CasADi solver variant."""
        results = []
        
        print(f"\n{'='*60}")
        print(f"  Benchmarking CasADi Variant: {variant_name} (solver={solver_key})")
        print(f"{'='*60}")
        
        for i in tqdm(range(self.num_runs), desc=f"{variant_name}"):
            seed = self.start_seed + i
            result = self.run_single_experiment(seed, solver_key, variant_name)
            if result:
                results.append(result)
        
        return results
    
    def run_full_benchmark(self):
        """Run the complete benchmark across all enabled solvers."""
        print(f"\n{'#'*60}")
        print(f"  FULL BENCHMARK SUITE")
        print(f"  Runs per config: {self.num_runs}")
        print(f"  Seeds: {self.start_seed} to {self.start_seed + self.num_runs - 1}")
        print(f"{'#'*60}")
        
        casadi_config = self.exp_config['solvers']['casadi']
        
        if not casadi_config['enabled']:
            print("[INFO] CasADi benchmarking disabled in config.")
            return
        
        # Run benchmark for each CasADi variant
        for variant in casadi_config['variants']:
            variant_name = variant['name']
            solver_key = variant['solver_key']
            
            results = self.run_casadi_variant_benchmark(variant_name, solver_key)
            
            # Store results keyed by CasADi variant name
            self.all_results[variant_name] = results
            
            # Brief summary after each variant
            if results:
                df = pd.DataFrame(results)
                avg_cas_cost = df['casadi_cost'].mean()
                avg_cas_time = df['casadi_time'].mean()
                print(f"  {variant_name}: Avg Cost={avg_cas_cost:.2f}, Avg Time={avg_cas_time:.4f}s, N={len(results)}")
        
        # Restore original config
        self._save_planner_config(self.original_config)
        print("\n[INFO] Original config restored.")
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """Combine all results into a single DataFrame."""
        all_rows = []
        for variant_name, results in self.all_results.items():
            for r in results:
                row = r.copy()
                row['casadi_variant'] = variant_name
                all_rows.append(row)
        return pd.DataFrame(all_rows)
    
    def compute_statistics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate Mean, Std, Best/Worst 10% averages."""
        if series.empty or len(series) == 0:
            return {"Mean": np.nan, "Std": np.nan, "Best10%": np.nan, "Worst10%": np.nan}
        
        n_10 = max(1, int(len(series) * 0.1))
        sorted_s = series.sort_values()
        
        return {
            "Mean": series.mean(),
            "Std": series.std(),
            "Best10%": sorted_s.head(n_10).mean(),
            "Worst10%": sorted_s.tail(n_10).mean()
        }
    
    def print_summary(self):
        """Print comprehensive summary statistics."""
        df = self.get_combined_dataframe()
        
        if df.empty:
            print("\n[CRITICAL] No results to summarize!")
            return
        
        print("\n" + "="*70)
        print("   COMPREHENSIVE BENCHMARK RESULTS")
        print("="*70)
        
        # PCE and NGD stats (should be consistent across CasADi variants)
        print("\n--- PCE Planner ---")
        pce_stats = self.compute_statistics(df['pce_cost'])
        pce_time = df['pce_time'].mean()
        print(f"  Avg Time     : {pce_time:.4f}s")
        for k, v in pce_stats.items():
            print(f"  {k:12}: {v:.4f}")
        
        print("\n--- NGD Planner ---")
        ngd_stats = self.compute_statistics(df['ngd_cost'])
        ngd_time = df['ngd_time'].mean()
        print(f"  Avg Time     : {ngd_time:.4f}s")
        for k, v in ngd_stats.items():
            print(f"  {k:12}: {v:.4f}")
        
        # CasADi variants comparison
        print("\n--- CasADi Variants Comparison ---")
        print(f"{'Variant':<12} {'Mean Cost':>12} {'Std':>10} {'Best10%':>10} {'Worst10%':>10} {'Avg Time':>10} {'Success%':>10}")
        print("-"*76)
        
        for variant_name in self.all_results.keys():
            variant_df = df[df['casadi_variant'] == variant_name]
            if variant_df.empty:
                continue
            
            stats = self.compute_statistics(variant_df['casadi_cost'])
            avg_time = variant_df['casadi_time'].mean()
            success_rate = (variant_df['casadi_cost'] < self.success_threshold).mean() * 100
            
            print(f"{variant_name:<12} {stats['Mean']:>12.2f} {stats['Std']:>10.2f} "
                  f"{stats['Best10%']:>10.2f} {stats['Worst10%']:>10.2f} "
                  f"{avg_time:>10.4f} {success_rate:>9.1f}%")
        
        # Success rates for all planners
        print("\n" + "="*70)
        print(f"   SUCCESS RATES (Cost < {self.success_threshold})")
        print("="*70)
        
        pce_success = (df['pce_cost'] < self.success_threshold).mean() * 100
        ngd_success = (df['ngd_cost'] < self.success_threshold).mean() * 100
        print(f"  PCE          : {pce_success:.1f}%")
        print(f"  NGD          : {ngd_success:.1f}%")
        
        for variant_name in self.all_results.keys():
            variant_df = df[df['casadi_variant'] == variant_name]
            if not variant_df.empty:
                success = (variant_df['casadi_cost'] < self.success_threshold).mean() * 100
                print(f"  {variant_name:<12}: {success:.1f}%")
    
    def save_results(self):
        """Save results to pickle and CSV files."""
        df = self.get_combined_dataframe()
        
        if df.empty:
            print("[WARN] No results to save.")
            return
        
        output_config = self.exp_config['output']
        
        # Save pickle
        pickle_path = os.path.join(ROOT_DIR, output_config['pickle_path'])
        df.to_pickle(pickle_path)
        print(f"\n[SAVED] Pickle: {pickle_path}")
        
        # Save CSV
        csv_path = os.path.join(ROOT_DIR, output_config['csv_path'])
        df.to_csv(csv_path, index=False)
        print(f"[SAVED] CSV: {csv_path}")
        
        # Save summary to text file
        summary_path = os.path.join(ROOT_DIR, output_config['summary_path'])
        self._save_summary_to_file(summary_path, df)
        print(f"[SAVED] Summary: {summary_path}")
    
    def _save_summary_to_file(self, path: str, df: pd.DataFrame):
        """Write summary statistics to a text file."""
        with open(path, 'w') as f:
            f.write(f"Benchmark Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Num Runs: {self.num_runs}\n")
            f.write(f"  Seed Range: {self.start_seed} - {self.start_seed + self.num_runs - 1}\n")
            f.write(f"  Success Threshold: {self.success_threshold}\n\n")
            
            # PCE stats
            pce_stats = self.compute_statistics(df['pce_cost'])
            f.write(f"PCE Planner:\n")
            f.write(f"  Mean Cost: {pce_stats['Mean']:.4f}\n")
            f.write(f"  Std: {pce_stats['Std']:.4f}\n")
            f.write(f"  Avg Time: {df['pce_time'].mean():.4f}s\n\n")
            
            # NGD stats
            ngd_stats = self.compute_statistics(df['ngd_cost'])
            f.write(f"NGD Planner:\n")
            f.write(f"  Mean Cost: {ngd_stats['Mean']:.4f}\n")
            f.write(f"  Std: {ngd_stats['Std']:.4f}\n")
            f.write(f"  Avg Time: {df['ngd_time'].mean():.4f}s\n\n")
            
            # CasADi variants
            f.write(f"CasADi Variants:\n")
            f.write(f"{'-'*70}\n")
            for variant_name in self.all_results.keys():
                variant_df = df[df['casadi_variant'] == variant_name]
                if variant_df.empty:
                    continue
                stats = self.compute_statistics(variant_df['casadi_cost'])
                avg_time = variant_df['casadi_time'].mean()
                success = (variant_df['casadi_cost'] < self.success_threshold).mean() * 100
                
                f.write(f"\n  [{variant_name}]\n")
                f.write(f"    Mean Cost: {stats['Mean']:.4f}\n")
                f.write(f"    Std: {stats['Std']:.4f}\n")
                f.write(f"    Best 10%: {stats['Best10%']:.4f}\n")
                f.write(f"    Worst 10%: {stats['Worst10%']:.4f}\n")
                f.write(f"    Avg Time: {avg_time:.4f}s\n")
                f.write(f"    Success Rate: {success:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Solver Benchmark Suite")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/solverbench_config.yaml',
        help='Path to experiment configuration YAML'
    )
    args = parser.parse_args()
    
    # Resolve config path
    if not os.path.isabs(args.config):
        config_path = os.path.join(ROOT_DIR, args.config)
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Experiment config not found: {config_path}")
        return 1
    
    print(f"[INFO] Using experiment config: {config_path}")
    
    runner = BenchmarkRunner(config_path)
    runner.run_full_benchmark()
    runner.print_summary()
    runner.save_results()
    
    print("\n[DONE] Benchmark complete!")
    return 0


if __name__ == "__main__":
    exit(main())