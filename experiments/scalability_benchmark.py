"""
Scalability Benchmark for Formal NAS.

Measures synthesis time vs. architecture complexity (max_nodes).
Essential for demonstrating the "Practical Limits" of SMT-based synthesis.
"""

import os
import sys
import time
import csv
import json
import z3
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.logic.temporal import Eventually, IsOp

# Load Secrets
load_dotenv()

NODES_SWEEP = [5, 8, 10, 12, 15, 20, 25, 30]
TRIALS_PER_POINT = 5 # Reduce randomness. Z3 is deterministic but system load varies.
TIMEOUT_SEC = 600 # 10 Minutes timeout per solve
PROJECT_NAME = "formal-nas-scalability"

def run_scalability_study():
    print(f"=== Formal NAS: Scalability Study ({PROJECT_NAME}) ===")
    
    # Init WandB
    wandb.init(project=PROJECT_NAME, name="synthesis_latency_sweep")
    
    results = []
    
    # Init CSV
    csv_file = "scalability_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Max_Nodes', 'Trial', 'Time_Sec', 'Result', 'Model_Nodes'])

    for max_nodes in NODES_SWEEP:
        print(f"\n--- Testing Max Nodes: {max_nodes} ---")
        
        for trial in range(TRIALS_PER_POINT):
            solver = z3.Solver()
            solver.set("timeout", TIMEOUT_SEC * 1000)
            
            # Setup Encoding
            # We use a simple constraint: "Must have Conv"
            # As max_nodes grows, the search space grows factorially.
            resource_limits = {"luts": 100000} # Loose limits to stress search space, not conflict
            
            encoding = DAGEncoding(solver, 
                                   max_nodes=max_nodes, 
                                   input_channels=3,
                                   resource_limits=resource_limits)
            
            solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
            
            # TIMING START
            start = time.time()
            status = solver.check()
            duration = time.time() - start
            # TIMING END
            
            result_str = str(status)
            found_nodes = 0
            
            if status == z3.sat:
                model = solver.model()
                arch = encoding.decode_architecture(model)
                found_nodes = len(arch)
                print(f"  Trial {trial}: {duration:.4f}s (SAT, {found_nodes} nodes)")
            else:
                print(f"  Trial {trial}: {duration:.4f}s ({result_str})")
            
            # Log Data
            row = {
                'max_nodes': max_nodes,
                'trial': trial,
                'time': duration,
                'result': result_str,
                'found_nodes': found_nodes
            }
            results.append(row)
            
            # WandB
            wandb.log(row)
            
            # CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([max_nodes, trial, duration, result_str, found_nodes])

    # Plotting
    print("\nGenerating Plot...")
    plot_scalability(results)
    print("Done. Results in 'scalability_results.csv' and 'scalability_plot.png'")
    wandb.finish()

def plot_scalability(results):
    # Aggregation
    data = {} # nodes -> [times]
    for r in results:
        if r['result'] == 'sat':
            if r['max_nodes'] not in data: data[r['max_nodes']] = []
            data[r['max_nodes']].append(r['time'])
            
    x = sorted(data.keys())
    y_mean = [sum(data[k])/len(data[k]) for k in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_mean, 'bo-', label='Mean Synthesis Time')
    
    # Scatter all points
    for k in x:
        plt.scatter([k]*len(data[k]), data[k], color='blue', alpha=0.3)
        
    plt.xlabel('Search Space Size (Max Nodes)')
    plt.ylabel('Time to Solution (s)')
    plt.title('Formal NAS Scalability: Synthesis Latency')
    plt.grid(True)
    plt.yscale('log') # Log scale is crucial for NP-hard problems
    plt.legend()
    plt.savefig('scalability_plot.png')
    
    # Log plot to WandB
    wandb.log({"scalability_plot": wandb.Image('scalability_plot.png')})

if __name__ == "__main__":
    run_scalability_study()
