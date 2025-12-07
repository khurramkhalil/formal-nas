"""
PyTorch Export & Gradient Flow Tests.

Verifies that synthesized models are valid `nn.Module`s with valid gradients.
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

class TestPyTorchExport(unittest.TestCase):
    
    def test_gradient_flow_in_residual_connections(self):
        """
        Ensure Add operations propagate gradients correctly (i.e., graph is connected).
        """
        solver = z3.Solver()
        encoding = DAGEncoding(solver, max_nodes=6, input_channels=3)
        
        # Enforce Residual (Add)
        solver.add(Eventually(IsOp(3)).encode(solver, encoding, 0))
        
        print("\n[Gradient Test] Synthesizing residual graph...")
        if solver.check() != z3.sat:
            self.fail("Could not synthesize residual graph.")
            
        model = solver.model()
        arch = encoding.decode_architecture(model)
        
        # Build Model
        net = PyTorchDAG(arch, input_channels=3, num_classes=10)
        net.train()
        
        # Backward Pass
        x = torch.randn(2, 3, 32, 32) # Batch 2
        out = net(x)
        loss = out.sum()
        loss.backward()
        
        # Check Gradients
        print("  Checking gradients...")
        has_grads = False
        for name, param in net.named_parameters():
            if param.requires_grad:
                has_grads = True
                self.assertIsNotNone(param.grad, f"Param {name} has no gradient! Graph broken?")
                self.assertFalse(torch.isnan(param.grad).any(), f"Param {name} has NaN gradient!")
                
        self.assertTrue(has_grads, "Model has no parameters?")
        print("✓ Gradients flow through all parameters.")

if __name__ == "__main__":
    unittest.main()
