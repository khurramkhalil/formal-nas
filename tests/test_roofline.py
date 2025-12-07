"""
Test Roofline Model Constraints.

Verifies that the SMT solver respects 'min_intensity' (Ops/Byte).
"""

import sys
import os
import z3
import unittest

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.logic.temporal import Always, Eventually, IsOp

class TestRoofline(unittest.TestCase):
    def test_arithmetic_intensity(self):
        """
        Test that higher intensity requirements force the solver to change strategy or fail.
        """
        solver = z3.Solver()
        
        # Scenario 1: No constraint
        # Should be easy
        limits_easy = {"luts": 10000, "dsp": 100, "bram": 10000, "min_intensity": 0}
        encoding = DAGEncoding(solver, max_nodes=5, input_channels=3, resource_limits=limits_easy)
        # Force at least one Conv
        solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
        
        print("\nChecking Easy (Intensity 0)...")
        self.assertEqual(solver.check(), z3.sat)
        
        # Scenario 2: Impossible Intensity
        # Conv2D intensity is roughly (K*K*C_in*C_out*H*W) / (W_params + In + Out)
        # For small layers, intensity is low (e.g. < 100).
        # Asking for 1000 should be impossible.
        
        solver.reset()
        limits_impossible = {"luts": 10000, "dsp": 100, "bram": 10000, "min_intensity": 1000}
        encoding = DAGEncoding(solver, max_nodes=5, input_channels=3, resource_limits=limits_impossible)
        solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0)) # Must have ops
        
        print("Checking Impossible (Intensity 1000)...")
        # should come back unsat
        self.assertEqual(solver.check(), z3.unsat)
        print("✓ Impossible constraint correctly rejected.")

if __name__ == "__main__":
    unittest.main()
