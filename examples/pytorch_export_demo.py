"""
PyTorch Export Verification Demo.

1. Synthesize a DAG with Residual Connections (Add).
2. Convert to PyTorch Model.
3. Run Forward Pass.
"""

import sys
import os
import torch
import z3

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.architectures.pytorch_exporter import PyTorchDAG
from formal_nas.logic.temporal import Always, Eventually, Implies, IsOp, Next

def run_demo():
    print("=== PyTorch Export Demo ===")
    
    # 1. Synthesize Graph
    solver = z3.Solver()
    encoding = DAGEncoding(solver, max_nodes=8, input_channels=3)
    
    # Force a Residual Block (Op 3 = ADD)
    # Temporal Rule: Eventually(ADD)
    # And "No Dead Ends" ensures it connects back to Output
    solver.add(Eventually(IsOp(3)).encode(solver, encoding, 0))
    
    # Force a Conv (Op 1)
    solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
    
    print("Synthesizing Architecture...")
    if solver.check() == z3.sat:
        model = solver.model()
        arch_dict = encoding.decode_architecture(model)
        
        print(f"\nFound Architecture with {len(arch_dict)} nodes.")
        for node in arch_dict:
            print(node)
            
        # 2. Convert to PyTorch
        print("\nConverting to PyTorch...")
        try:
            torch_model = PyTorchDAG(arch_dict, input_channels=3, num_classes=10)
            print(torch_model)
            
            # 3. Test Forward Pass
            print("\nTesting Forward Pass...")
            dummy_input = torch.randn(2, 3, 32, 32) # Batch size 2
            output = torch_model(dummy_input)
            
            print(f"Input Shape: {dummy_input.shape}")
            print(f"Output Shape: {output.shape}")
            print("✅ Success! Model is runnable.")
            
        except Exception as e:
            print(f"❌ Conversion Failed: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("❌ Could not find solution.")

if __name__ == "__main__":
    run_demo()
