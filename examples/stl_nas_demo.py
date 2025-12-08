"""
STL-NAS Demo Script.

Demonstrates the "Universal Gatekeeper" enforcing temporal constraints on a mock NAS process.
"""

import sys
import os
import random
import statistics
from typing import List, Dict

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.stl.monitor import Predicate, Always, And
from formal_nas.stl.gatekeeper import STLGatedSearch

def mock_evaluate_fitness(arch_id):
    """Generates random fitness for a mock architecture."""
    # Acc: 70-95%, Energy: 10-20 units, Params: 1-5M
    return {
        'id': arch_id,
        'accuracy': random.uniform(70, 95),
        'energy': random.uniform(10, 20),
        'params': random.uniform(1, 5)
    }

def run_demo():
    print("=== STL-NAS Logic Demo ===")
    
    # 1. Define Constraints
    # Energy must be < 15 (Mean of population)
    # Robustness = 15 - mean_energy
    phi_energy = Predicate("EnergyConstraint", lambda s: 15.0 - s['mean_energy'])
    
    # Accuracy Stability: Best Acc shouldn't drop significantly (mock check)
    # Actually, let's just enforce Energy for clarity.
    specs = [Always(phi_energy)]
    
    print("\n[Spec] Globally(Mean Energy < 15.0)")
    
    # 2. Setup Gatekeeper
    gatekeeper = STLGatedSearch(specs, robustness_threshold=0.0)
    
    # 3. Simulate Search (Evolutionary-ish)
    population = []
    
    # Initial Pop
    print("\n--- Initialization ---")
    for i in range(5):
        arch = mock_evaluate_fitness(f"init_{i}")
        population.append(arch)
    
    gatekeeper.update_history(population)
    curr_metrics = gatekeeper.signal_trace[-1]
    print(f"Init Metrics: Energy={curr_metrics['mean_energy']:.2f}, Acc={curr_metrics['best_acc']:.2f}")
    
    # 3. Search Loop
    print("\n--- Search Start (10 Iters) ---")
    for t in range(10):
        # Base Algo Proposes Candidate
        candidate = mock_evaluate_fitness(f"gen_{t}")
        
        # Verify
        is_valid, rho = gatekeeper.is_valid_candidate(population, candidate)
        
        status = "✅ ACCEPT" if is_valid else "❌ REJECT"
        print(f"Iter {t}: Candidate(E={candidate['energy']:.1f}) -> Robustness={rho:.2f} -> {status}")
        
        if is_valid:
            # Update Pop (Simple Replacement of worst energy for demo)
            # Find max energy
            worst_idx = max(range(len(population)), key=lambda i: population[i]['energy'])
            population[worst_idx] = candidate
            gatekeeper.update_history(population)
        else:
            # Reject and keep old pop (Signal repeats or we skip time step? 
            # In Alg 1, if no valid candidates, we might just keep old pop)
            pass

    print("\n--- Final Trajectory ---")
    valid_steps = 0
    violations = 0
    for i, frame in enumerate(gatekeeper.signal_trace):
        e = frame['mean_energy']
        status = "OK" if e < 15 else "VIOLATION"
        if e >= 15: violations += 1
        print(f"Time {i}: Mean Energy = {e:.2f} [{status}]")
        
    print(f"\nTotal Violations: {violations}")
    if violations == 0:
        print("🏆 STL-NAS Guarantee Held: 0 Violations.")
    else:
        print("⚠️  Guarantee Failed (Check Init population?)")

if __name__ == "__main__":
    run_demo()
