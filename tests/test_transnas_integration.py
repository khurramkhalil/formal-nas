
import unittest
import z3
import os
import sys

# Ensure imports work
# Ensure imports work
sys.path.append(os.getcwd())

# Ensure TransNASBench is in path BEFORE importing transnas
transnas_path = os.path.join(os.getcwd(), 'TransNASBench')
if os.path.exists(transnas_path) and transnas_path not in sys.path:
    sys.path.append(transnas_path)

try:
    from src.formal_nas.synthesis.cell_encoding import CellEncoding
    from src.formal_nas.benchmarks.transnas import TransNASBenchmark, get_benchmark
except ImportError:
    # Handle CI environment where src might need to be explicitly added
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from formal_nas.synthesis.cell_encoding import CellEncoding
    from formal_nas.benchmarks.transnas import TransNASBenchmark, get_benchmark

class TestTransNASIntegration(unittest.TestCase):
    
    def test_generated_strings_are_valid(self):
        """CRITICAL: Verify SMT output -> TransNAS query works"""
        print("\n=== Testing SMT -> TransNAS API Integration ===")
        
        # 1. Setup SMT
        solver = z3.Solver()
        encoding = CellEncoding(solver)
        
        # 2. Add constraint for a valid graph (CellEncoding does this implicitly)
        
        # 3. Solve
        print("Solving for valid architecture...")
        if solver.check() == z3.sat:
            model = solver.model()
            arch_str = encoding.decode_architecture(model)
            print(f"Generated String: {arch_str}")
            
            # 4. Verify String Format (Basic Regex Check)
            # Expect: 64-41414-X_XX_XXX
            self.assertTrue(arch_str.startswith("64-41414-"), f"Invalid Prefix: {arch_str}")
            
            # 5. Connect to Benchmark
            # Check if data exists locally, otherwise skip integration part
            data_file = "transnas-bench_v10141024.pth"
            if not os.path.exists(data_file):
                print(f"⚠️ {data_file} not found. Skipping API query check.")
                return

            print("Connecting to TransNAS Benchmark...")
            benchmark = get_benchmark(data_file)
            
            # 6. Query API
            # Try a robust task e.g. "class_scene"
            task = "class_scene"
            print(f"Querying for task: {task}")
            trace = benchmark.get_training_trace(task, arch_str)
            
            print(f"Trace Result: {trace}")
            
            # 7. Assert Validity
            self.assertIsNotNone(trace, "Trace should not be None")
            self.assertTrue(len(trace) > 0, "API returned empty trace! The string was rejected.")
            self.assertTrue(trace[0]['accuracy'] > 0, "Returned accuracy is 0.0 (Invalid)")
            
            print("✅ CHECK PASSED: SMT String accepted by API.")
            
        else:
            self.fail("SMT Solver returned UNSAT for basic encoding! Logic error.")

if __name__ == "__main__":
    unittest.main()
