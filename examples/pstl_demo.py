"""
Parametric STL (P-STL) Demo.

Demonstrates "Automatic Parameter Synthesis" using P-STL and TransNAS traces.
Instead of checking manual bounds, we LEARN the bounds from the training dynamics.
"""

import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.stl.parametric import PSTLContext, ParametricPredicate
from formal_nas.benchmarks.transnas import get_benchmark

def run_pstl_demo():
    print("=== P-STL Parameter Synthesis Demo ===")
    
    # 1. Load Benchmark (Mock or Real)
    bench = get_benchmark() # Auto-detects
    
    # 2. Get a Training Trace for an "Unknown" Architecture
    # We pretend SMT generated "arch_001"
    trace = bench.get_training_trace(task="cifar10", arch_id="arch_001")
    print(f"\n[Trace] Loaded {len(trace)} epochs of training data.")
    print(f"  Final Epoch Status: {trace[-1]}")
    
    # 3. Define Parametric Specification
    # We want to characterize this architecture by finding its limits.
    
    ctx = PSTLContext()
    
    # Prop 1: What is the Maximum Accuracy it achieves? (Upper Bound)
    # G(Accuracy < theta_max) -> Minimize theta_max such that it holds? 
    # Actually, G(Acc < theta) is satisfied if theta >= Peak. Optimal = Peak.
    ctx.register_upper_bound("peak_acc", "accuracy")
    
    # Prop 2: What is the Minimum Loss it achieves? (Lower Bound)
    # G(Loss > theta_min) -> Optimize theta_min.
    ctx.register_lower_bound("min_loss", "loss")
    
    # Prop 3: Eventual Convergence?
    # F(Accuracy > theta_conv) -> Maximize theta_conv.
    ctx.register_eventual_goal("final_acc", "accuracy")
    
    # 4. Synthesize Parameters (The "Learning" Step)
    print("\n[P-STL] Synthesizing Performance Guarantees...")
    params = ctx.synthesize(trace)
    
    # 5. Report Certificates
    print("\n--- Certified Performance Bounds ---")
    print(f"  Guaranteed Peak Accuracy   : <= {params['peak_acc']:.2f}%")
    print(f"  Guaranteed Minimum Loss    : >= {params['min_loss']:.4f}")
    print(f"  Eventual Converged Accuracy: >= {params['final_acc']:.2f}%")
    
    print("\n[Analysis]")
    if params['final_acc'] > 90.0:
        print("  ✅ Architecture meets high-performance criteria (>90%).")
    else:
        print("  ⚠️ Architecture performance is mediocre.")
        
    # Novelty: We can now use these 'params' as fitness values in the outer loop!
    # "Maximize theta_acc subject to Structural Constraints"

if __name__ == "__main__":
    run_pstl_demo()
