"""
Formal NAS: Graph Synthesis Demo.

This script demonstrates the "Novelty Engine" capabilities:
1. SMT-Based DAG Synthesis (Arbitrary Graphs)
2. Symbolic Shape Logic (Correct-by-Construction)
3. Hardware Constraints (Symbolic FPGA Model)
4. Temporal Logic (TWTL)

It synthesizes a neural architecture that satisfies:
- Graph validity (connected, no dead ends)
- Resource limits (fits in small FPGA budget)
- Temporal rule: "Always(Conv -> Eventually(Pool))"
"""

import sys
import os
import z3

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.logic.temporal import Always, Eventually, Implies, IsOp, Atom, Next

def run_demo():
    print("=== Formal NAS: Novelty Engine Demo ===")
    
    # 1. Setup Solver
    solver = z3.Solver()
    
    # 2. Define Limits (Tight limits to force optimization)
    # 4000 LUTs is very small - forces the solver to be efficient!
    resource_limits = {"luts": 4000, "dsp": 100, "bram": 50000}
    print(f"Goal: Synthesize architecture within {resource_limits}...")
    
    # 3. Create Encoding (Max 6 nodes for demo speed)
    encoding = DAGEncoding(
        solver, 
        max_nodes=6, 
        input_channels=3, 
        resource_limits=resource_limits
    )
    
    # 4. Add Temporal Logic Constraints
    # Rule: "If we have a Conv layer, it must eventually be followed by Pooling"
    # This prevents the network from exploding feature map size with too many Convs
    # OP_CONV=1, OP_POOL=2
    print("Constraint: Always(Conv -> Eventually(Pool))")
    
    rule = Always(Implies(
        IsOp(1),                # If Op is Conv
        Next(Eventually(IsOp(2))) # Then Next node or later must be Pool
    ))
    solver.add(rule.encode(solver, encoding, 0))
    
    # Also force at least one Conv to make it interesting
    solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))

    # 5. Solve
    print("Solving... (Z3 is searching for topologies)")
    result = solver.check()
    
    if result == z3.sat:
        print("\n✅ Solution Found!")
        model = solver.model()
        
        # 6. Decode
        arch = encoding.decode_architecture(model)
        
        print("\nSynthesized Graph:")
        print("------------------")
        for node in arch:
            op_name = {
                -1: "INPUT", -2: "OUTPUT", 
                1: "CONV", 2: "POOL", 
                3: "ADD", 4: "CONCAT"
            }.get(node['op'], f"OP_{node['op']}")
            
            # Formatting
            shape_str = f"Shape: {node['shape']}"
            in_str = f"In: {node['in1']}"
            print(f"Node {node['id']}: {op_name:<8} | {in_str:<8} | {shape_str}")
        print("------------------")
        
    else:
        print("\n❌ Unsatisfiable! (Constraints too tight?)")

if __name__ == "__main__":
    run_demo()
