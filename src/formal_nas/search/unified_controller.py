"""
Unified Search Controller (Spatiotemporal Formal NAS).

The "Brain" that connects:
- Spatial Synthesis (Z3)
- Temporal Evaluation (TransNAS / P-STL)
- Feedback Refinement (CEGIS)

Algorithm:
1. SMT generates candidate 'A'.
2. TransNAS provides training trace for 'A'.
3. P-STL synthesizes performance bounds (theta) for 'A'.
4. If bounds < Goal:
     Use FeedbackEngine to constrain SMT (e.g. "TotalLUTs < X").
     Loop.
5. Else:
     Return 'A' (Verified Solution).
"""

import sys
import os
import z3
import wandb
import csv

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from formal_nas.synthesis.cell_encoding import CellEncoding
from formal_nas.synthesis.feedback import FeedbackEngine
from formal_nas.benchmarks.transnas import get_benchmark as get_transnas
from formal_nas.benchmarks.nasbench201 import get_benchmark as get_nas201
from formal_nas.stl.parametric import PSTLContext
from formal_nas.logic.temporal import Eventually, IsOp
# from formal_nas.synthesis.dag_encoding import DAGEncoding # Deprecated for benchmark alignment

class UnifiedController:
    
    def __init__(self, use_wandb=False, log_file=None, benchmark_type="transnas", project_name="formal-nas-adaptive-benchmark"):
        self.use_wandb = use_wandb
        self.log_file = log_file
        self.benchmark_type = benchmark_type
        
        # Initialize CSV
        if self.log_file:
            with open(self.log_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "arch_id", "luts", "accuracy", "is_pareto"])
        
        if self.use_wandb:
            wandb.init(project=project_name, name=f"search_{self.benchmark_type}")
            
        # 1. Init SMT Solver
        # Enable Parallel Solving (Utilization of multiple cores)
        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads', 4)
        
        self.solver = z3.Solver()
        self.solver.set("timeout", 30000) # 30 Second Timeout to prevent hangs
        
        # USE CELL ENCODING (BIT-EXACT ALIGNMENT)
        self.encoding = CellEncoding(
            self.solver, 
            resource_limits={"luts": 100000},
            benchmark_type=self.benchmark_type
        ) 
        
        # 2. Init Feedback Engine
        self.feedback = FeedbackEngine(self.encoding)
        
        # 3. Init Benchmark
        if self.benchmark_type == "transnas":
            self.benchmark = get_transnas()
        elif self.benchmark_type == "nas201":
            self.benchmark = get_nas201()
        else:
            raise ValueError(f"Unknown benchmark type: {self.benchmark_type}")
        
        # 4. History Tracking
        self.history = []
        
    def run_search(self, max_iterations: int = 20, task_name: str = "class_scene"):
        print(f"=== Unified Search Started | Task: {task_name} | Adaptive Logic Enabled ===")
        
        best_arch = None
        best_acc = 0.0
        
        # Efficiency assumptions (Transients)
        efficiency_assumptions = []
        efficiency_patience = 3
        
        for i in range(max_iterations):
            print(f"\n[Iter {i}] Synthesizing...")
            
            # --- STEP 1: SPATIAL SYNTHESIS (SMT) ---
            # Use check(assumptions) to allow backtracking from efficiency constraints
            result = self.solver.check(efficiency_assumptions)
            
            if result != z3.sat:
                reason = "UNSAT" if result == z3.unsat else "TIMEOUT/UNKNOWN"
                if efficiency_assumptions:
                     print(f"  ⚠️ Efficiency Limit Reached ({reason}). Resetting constraints to explore new regions.")
                     efficiency_assumptions = []
                     efficiency_patience = 3 # Reset patience
                     
                     # Retry immediately without assumptions
                     if self.solver.check() != z3.sat:
                         print("  ⚠️ SMT Unsatisfiable even after reset! Search Space Exhausted.")
                         break
                else:
                    print(f"  ⚠️ SMT {reason}! Search Space Exhausted or Over-Constrained.")
                    break
                
            model = self.solver.model()
            
            # Use Cell Encoding Decode
            arch_str = self.encoding.decode_architecture(model)
            arch_dict = self.encoding.decode_to_dict(model) # Wrapper for compatibility
            
            # Arch ID IS the string now
            arch_id = arch_str
            
            # Estimate Spatial Costs
            est_luts = self._get_estimated_luts(model)
            
            # --- STEP 2: TEMPORAL EVALUATION (TransNAS) ---
            trace = self.benchmark.get_training_trace(task=task_name, arch_id=arch_id)
            if not trace:
                print("  Error: No trace found. Blocking this architecture to avoid loop.")
                # CRITICAL: We must block this architecture so the solver doesn't propose it again!
                # Treat as "accuracy" violation (invalid topology)
                self.feedback.generate_feedback(self.solver, arch_dict, "accuracy", 0.0)
                continue
                
            # --- STEP 3: LOGIC SYNTHESIS (P-STL) ---
            ctx = PSTLContext()
            ctx.register_upper_bound("peak_acc", "accuracy")
            params = ctx.synthesize(trace)
            learned_acc = params.get("peak_acc", 0.0)
            
            # --- STEP 4: VERIFICATION & FEEDBACK ---
            # --- STEP 4: ADAPTIVE PARETO FEEDBACK ---
            
            is_pareto = False
            if learned_acc > best_acc:
                best_acc = learned_acc
                best_arch = arch_dict
                is_pareto = True
                print(f"  🔥 NEW BEST ACCURACY: {best_acc:.2f}%")

            if self.use_wandb:
                 wd_log = {
                    "iteration": i,
                    "estimated_luts": est_luts,
                    "learned_acc": learned_acc,
                    "is_pareto": is_pareto
                }
                 wandb.log(wd_log)
            
            # CSV Logging
            if self.log_file:
                with open(self.log_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, arch_id, est_luts, learned_acc, is_pareto])
            
            # Record history
            self.history.append({
                "iter": i,
                "luts": est_luts,
                "accuracy": learned_acc,
                "pareto": is_pareto
            })

            
            # Feedback Strategy:
            # 1. ALWAYS block the exact current topology to force exploration (Tabu Search).
            # Note: generate_feedback adds this permanently to solver.
            _, _ = self.feedback.generate_feedback(self.solver, arch_dict, "accuracy", 0)
            
            # 2. Adaptive Efficiency Search
            # Safety: Ensure best_acc is meaningful (>10%) before optimizing resources.
            if learned_acc >= (best_acc - 2.0) and best_acc > 10.0:
                print(f"     [Adaptive] High Performance (>{best_acc-2.0:.1f}%) detected. Constraining LUTs < {est_luts} for efficiency.")
                
                # Get constraint but DO NOT add to solver here. Add to assumptions.
                _, eff_constraint = self.feedback.generate_feedback(self.solver, arch_dict, "energy", est_luts)
                if eff_constraint is not None:
                    efficiency_assumptions.append(eff_constraint)
                    efficiency_patience = 3 # Reset patience on success
            
            else:
                 # If we are under efficiency constraints but Accuracy dropped, we are "Digging in the wrong place".
                 if efficiency_assumptions:
                     efficiency_patience -= 1
                     print(f"     [Adaptive] Efficiency Drill yielding low accuracy. Patience: {efficiency_patience}")
                     
                     if efficiency_patience <= 0:
                         print("     [Adaptive] Abandoning Efficiency Drill. Resurfacing to Global Search.")
                         efficiency_assumptions = []
                         efficiency_patience = 3
                 else:
                     print(f"     [Adaptive] Exploring Search Space...")
                
        print(f"\n=== Search Complete ===")
        print(f"Best Accuracy Found: {best_acc:.2f}%")
        
        if self.use_wandb:
            self.log_final_report()
            
        return best_arch

    def log_final_report(self):
        """Generates rich visualizations for WandB."""
        # 1. Create a Table
        data = [[h['iter'], h['luts'], h['accuracy'], h['pareto']] for h in self.history]
        table = wandb.Table(data=data, columns=["iter", "luts", "accuracy", "pareto"])
        wandb.log({"search_trace_table": table})
        
        # 2. Pareto Plot
        # Custom Chart: accuracy vs luts
        wandb.log({
            "pareto_frontier": wandb.plot.scatter(
                table, "luts", "accuracy", title="Pareto Frontier: Accuracy vs Resources"
            )
        })
        print("  Pareto Plot logged to WandB.")
    # Note: _get_estimated_luts logic remains same as long as 'total_luts' attribute exists.
    def _get_estimated_luts(self, model):
        """Helper to extract symbolic resource variable value."""
        if hasattr(self.encoding, 'total_luts'):
            val = model.eval(self.encoding.total_luts)
            if hasattr(val, 'as_long'): return val.as_long()
        return 0
