"""
Shape Inference Verification Tests.

Verifies that the Z3 symbolic shape logic exactly matches PyTorch's runtime behavior.
Correct-by-construction Guarantee: The constraints must match reality.
"""

import sys
import os
import z3
import torch
import unittest

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.architectures.pytorch_exporter import PyTorchDAG
from formal_nas.logic.temporal import Eventually, IsOp

class TestShapeInference(unittest.TestCase):
    
    def test_symbolic_shapes_match_pytorch(self):
        """
        Verify Z3-predicted shapes equals actual PyTorch tensor shapes.
        """
        solver = z3.Solver()
        encoding = DAGEncoding(solver, max_nodes=6, input_channels=3)
        
        # Enforce diversity (Conv+Pool+Add)
        solver.add(Eventually(IsOp(3)).encode(solver, encoding, 0)) # Add
        solver.add(Eventually(IsOp(2)).encode(solver, encoding, 0)) # Pool
        
        print("\n[Shape Test] Synthesizing diverse graph...")
        if solver.check() != z3.sat:
            self.fail("Could not synthesize graph for testing.")
            
        model = solver.model()
        arch_dict = encoding.decode_architecture(model)
        
        # 1. Extract Predicted Shapes
        predicted_shapes = {}
        for node in arch_dict:
            # node['shape'] is (C, H, W)
            predicted_shapes[node['id']] = node['shape']
            
        print("  Predicted Shapes extracted.")
        
        # 2. Extract Actual Shapes (PyTorch)
        pytorch_model = PyTorchDAG(arch_dict, input_channels=3)
        
        # Hook to capture shapes
        actual_shapes = {}
        def get_activation(node_id):
            def hook(model, input, output):
                # Output is (B, C, H, W)
                shape = output.shape
                # Store (C, H, W)
                actual_shapes[node_id] = (shape[1], shape[2], shape[3])
            return hook
            
        # Register hooks
        for node in arch_dict:
            node_id = node['id']
            # We need to access the submodule.
            # PyTorchDAG stores them in self.layers (ModuleDict)
            if str(node_id) in pytorch_model.layers:
                layer = pytorch_model.layers[str(node_id)]
                layer.register_forward_hook(get_activation(node_id))
                
        # Run Forward
        x = torch.randn(1, 3, 32, 32)
        _ = pytorch_model(x)
        
        print("  Actual Shapes captured.")
        
        # 3. Compare
        for node in arch_dict:
            nid = node['id']
            if nid not in actual_shapes: continue # Input node?
            
            pred = predicted_shapes[nid]
            act = actual_shapes[nid]
            
            # Allow minor mismatch? No. Must be exact.
            self.assertEqual(pred, act, f"Node {nid} Shape Mismatch! Z3={pred}, Torch={act}")
            
        print("✓ Z3 Shapes match PyTorch Shapes exactly.")

if __name__ == "__main__":
    unittest.main()
