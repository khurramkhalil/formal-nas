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

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.synthesis.feedback import FeedbackEngine
from formal_nas.benchmarks.transnas import get_benchmark
from formal_nas.stl.parametric import PSTLContext
from formal_nas.logic.temporal import Eventually, IsOp

class UnifiedController:
    
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="formal-nas-unified", name="unified_search_loop")
            
        # 1. Init SMT Solver
        self.solver = z3.Solver()
        self.solver.set("timeout", 30000) # 30 Second Timeout to prevent hangs
        
        # Loose initial limits, will be tightened by Feedback
        self.encoding = DAGEncoding(
            self.solver, 
            max_nodes=6, 
            input_channels=3, 
            resource_limits={"luts": 100000, "dsp": 5000, "bram": 5000}
        ) 
        
        # Enforce basic validity (At least one Conv)
        self.solver.add(Eventually(IsOp(1)).encode(self.solver, self.encoding, 0))
        
        # 2. Init Feedback Engine
        self.feedback = FeedbackEngine(self.encoding)
        
        # 3. Init Benchmark
        self.benchmark = get_benchmark()
        
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
            arch_dict = self.encoding.decode_architecture(model)
            arch_id = f"arch_{i}" # Mock ID for TransNAS
            
            # Estimate Spatial Costs (Logic from Encoding)
            est_luts = self._get_estimated_luts(model)
            
            # --- STEP 2: TEMPORAL EVALUATION (TransNAS) ---
            trace = self.benchmark.get_training_trace(task=task_name, arch_id=arch_id)
            if not trace:
                print("  Error: No trace found.")
                continue
                
            # --- STEP 3: LOGIC SYNTHESIS (P-STL) ---
            ctx = PSTLContext()
            ctx.register_upper_bound("peak_acc", "accuracy")
            params = ctx.synthesize(trace)
            learned_acc = params.get("peak_acc", 0.0)
            
            print(f"  Candidate: {arch_id} | LUTs: {est_luts} | Learned Acc: {learned_acc:.2f}%")
            
            if self.use_wandb:
                wandb.log({
                    "iteration": i,
                    "estimated_luts": est_luts,
                    "learned_acc": learned_acc
                })
            
            # --- STEP 4: VERIFICATION & FEEDBACK ---
            # --- STEP 4: ADAPTIVE PARETO FEEDBACK ---
            # Instead of a hard threshold (e.g. 90%), we determine "Success" adaptively.
            # WE use P-STL learned parameter 'learned_acc' to drive the search.
            
            is_pareto = False
            if learned_acc > best_acc:
                best_acc = learned_acc
                best_arch = arch_dict
                is_pareto = True
                print(f"  🔥 NEW BEST ACCURACY: {best_acc:.2f}%")
            
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
    def _get_estimated_luts(self, model):
        """Helper to extract symbolic resource variable value."""
        if hasattr(self.encoding, 'total_luts'):
            val = model.eval(self.encoding.total_luts)
            if hasattr(val, 'as_long'): return val.as_long()
        return 0
