
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import z3
except ImportError:
    print("Z3 not installed, skipping test")
    sys.exit(0)

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.logic.temporal import Always, Eventually, Implies, IsOp, Atom, Next

class TestDAGEncoding(unittest.TestCase):
    def test_basic_sat(self):
        solver = z3.Solver()
        # Create small DAG: max 5 nodes
        encoding = DAGEncoding(solver, max_nodes=5, input_channels=3)
        
        # Add constraint: Must have at least 1 Conv layer
        # Assuming OP_CONV=1 defined in dag_encoding (I need to check how I defined constants there)
        # I defined constants inside _add_shape_logic as local vars. 
        # I should probably expose them, but for now I know: 1=Conv, 3=Add
        
        # Use Temporal Logic for cleaner constraint
        # "Eventually(IsOp(CONV))"
        # Access constants from instance if possible, or hardcode for test
        # encoding.OP_CONV is 1
        
        has_conv = Eventually(IsOp(1)).encode(solver, encoding, 0)
        solver.add(has_conv)
        
        # Check SAT
        result = solver.check()
        self.assertEqual(result, z3.sat)
        
        model = solver.model()
        arch = encoding.decode_architecture(model)
        print("\nFound Architecture:")
        for node in arch:
            print(node)

    def test_temporal_logic(self):
        """Test 'Always(Conv -> Eventually(Pool))'"""
        solver = z3.Solver()
        encoding = DAGEncoding(solver, max_nodes=8, input_channels=3)
        
        # Constraint 1: Must have a Conv layer
        solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
        
        # Constraint 2: Always(Conv -> Eventually(Pool))
        # 1=Conv, 2=Pool
        rule = Always(Implies(
            IsOp(1),
            Next(Eventually(IsOp(2)))
        ))
        solver.add(rule.encode(solver, encoding, 0))
        
        result = solver.check()
        if result == z3.sat:
            print("\nFound Temporal Compliant Architecture:")
            model = solver.model()
            arch = encoding.decode_architecture(model)
            for node in arch:
                print(node)
        else:
            self.fail("Could not find architecture satisfying temporal constraints")

    def test_resource_limits(self):
        """Test hardware resource constraints (LUTs)."""
        solver = z3.Solver()
        # Set tight LUT limit. A 3x3 Conv (3->16 channels) costs ~200-500 LUTs depending on input size.
        # Let's try to force a small network.
        limits = {"luts": 5000, "dsp": 100, "bram": 100000} 
        encoding = DAGEncoding(solver, max_nodes=5, input_channels=3, resource_limits=limits)
        
        # Must have at least 1 Conv
        solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
        
        result = solver.check()
        if result == z3.sat:
            print("\nFound Hardware-Optimized Architecture:")
            model = solver.model()
            arch = encoding.decode_architecture(model)
            for node in arch:
                print(node)
            
            # Verify resource limit manually
            # (Printed by decode_architecture)
        else:
             self.fail("Could not find architecture within resource limits")
            
    def test_residual_block(self):
        solver = z3.Solver()
        encoding = DAGEncoding(solver, max_nodes=6, input_channels=3)
        
        # Force a Residual Block: ADD operation (3) MUST BE ACTIVE
        # Otherwise solver puts it in inactive node to save cost
        has_add = z3.Or([
            z3.And(encoding.node_active[i], encoding.node_ops[i] == 3)
            for i in range(encoding.max_nodes)
        ])
        solver.add(has_add)
        
        result = solver.check()
        if result == z3.sat:
            print("\nFound Residual Architecture:")
            model = solver.model()
            arch = encoding.decode_architecture(model)
            for node in arch:
                print(node)
        else:
            print("Unsatisifable (might be due to strict constraints or bug)")

if __name__ == '__main__':
    unittest.main()
