"""
Unified Loop Demo.

Demonstrates the "Self-Driving" NAS capability:
Spatiotemporal Integration of SMT (Spatial), P-STL (Logic), and CEGIS (Refinement).
"""

import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.search.unified_controller import UnifiedController

def run_demo():
    print("=== Spatiotemporal Formal NAS Demo ===")
    print("Goal: Find verified architecture with Accuracy >= 92% automatically.\n")
    
    # Init Controller (No WandB for local demo to avoid login prompt/errors)
    controller = UnifiedController(use_wandb=False)
    
    # Run Search
    # 10 Iterations should be enough for the Mock TransNAS to output a 'good' random arch.
    # In the mock, acc is random(70, 95). So ~20% chance of success per iter.
    best_arch = controller.run_search(max_iterations=10, target_acc=92.0)
    
    if best_arch:
        print("\n✅ Demo Successful: Found compliant architecture.")
        print(f"Topology: {[n['op'] for n in best_arch]}")
    else:
        print("\n⚠️ Demo Finished: No compliant architecture found in 10 steps.")

if __name__ == "__main__":
    run_demo()
