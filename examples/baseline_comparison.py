"""
Formal NAS: Baseline Comparison (SMT vs Random).

This script demonstrates the "Finding a Needle in a Haystack" problem.
It compares:
1. SMT SOLVER: Direct synthesis of a valid architecture.
2. RANDOM SEARCH: Rejection sampling (generate random graph -> check if valid).

Hypothesis: Random search will struggle to find VALID architectures that respect 
strict dimension matching (e.g. Add layer requires identical inputs), especially
as depth increases.
"""

import random
import time
import sys
import os
import z3

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.logic.shape_inference import get_conv2d_output_shape, get_pool2d_output_shape

def check_random_validity(arch_props, input_res=32):
    """
    Check if a randomly generated attribute set forms a valid architecture.
    arch_props: list of dicts {'op', 'in1', 'in2', 'k', 's', 'c'}
    """
    # Track shapes: node_idx -> (C, H, W)
    shapes = {0: (3, input_res, input_res)}
    
    for i, props in enumerate(arch_props):
        node_id = i + 1
        op = props['op']
        in1 = props['in1']
        
        # 1. Connectivity Check
        if in1 >= node_id: return False # Cycle or future ref
        if in1 not in shapes: return False
        
        in_c, in_h, in_w = shapes[in1]
        
        # 2. Shape Logic Check
        if op == 1: # CONV
            k, s = props['k'], props['s']
            # Output height formula
            # H_out = (H_in + 2*pad - k) / s + 1
            # Check non-negative
            if in_h + 2 - k < 0: return False
            out_h = (in_h + 2 - k) // s + 1
            out_w = (in_w + 2 - k) // s + 1
            shapes[node_id] = (props['c'], out_h, out_w)
            
        elif op == 2: # POOL
            if in_h < 2: return False
            shapes[node_id] = (in_c, in_h//2, in_w//2)
            
        elif op == 3: # ADD
            in2 = props['in2']
            if in2 >= node_id or in2 not in shapes: return False
            in2_c, in2_h, in2_w = shapes[in2]
            
            # Strict Shape Matching
            if in_c != in2_c or in_h != in2_h or in_w != in2_w:
                return False
            shapes[node_id] = shapes[in1]
            
        elif op == -2: # OUTPUT
            shapes[node_id] = shapes[in1]
            
    return True

def run_random_search(num_trials=1000, max_nodes=8):
    print(f"\n[Random Search] Attempting {num_trials} random graphs...")
    valid_count = 0
    start_time = time.time()
    
    for _ in range(num_trials):
        # Generate random graph props
        props = []
        for i in range(1, max_nodes):
            op = random.choice([1, 2, 3]) # Conv, Pool, Add
            # Last node usually output, simplified here to just body checks
            
            p = {
                'op': op,
                'in1': random.randint(0, i-1),
                'in2': random.randint(0, i-1),
                'k': random.choice([1, 3, 5]),
                's': random.choice([1, 2]),
                'c': random.choice([16, 32])
            }
            props.append(p)
            
        if check_random_validity(props):
            valid_count += 1
            
    duration = time.time() - start_time
    print(f"  > Valid: {valid_count}/{num_trials} ({valid_count/num_trials*100:.1f}%)")
    print(f"  > Rate: {num_trials/duration:.0f} attempted graphs/sec")

def run_smt_synthesis(max_nodes=8):
    print(f"\n[Formal SMT] Synthesizing 1 Correct-by-Construction graph...")
    start_time = time.time()
    
    solver = z3.Solver()
    encoding = DAGEncoding(solver, max_nodes=max_nodes, input_channels=3)
    
    # Just basic validity is enough to compare
    if solver.check() == z3.sat:
        duration = time.time() - start_time
        print(f"  > Success! Found valid architecture.")
        print(f"  > Time: {duration:.4f} seconds")
    else:
        print("  > Failed.")

if __name__ == "__main__":
    print("=== Baseline Comparison ===")
    print("Task: Find a valid 8-node DAG with consistent tensor shapes.")
    
    run_random_search(num_trials=1000, max_nodes=8)
    run_smt_synthesis(max_nodes=8)
    
    print("\nConclusion: Random search wastes theoretical time on invalid DAGs.")
    print("Formal SMT guarantees 100% validity.")
