"""
STL-NAS Gatekeeper.

Implements the "Modular Gatekeeper" architecture from the STL-NAS paper (Algorithm 1).
It acts as a filter: Base NAS -> Gatekeeper -> Validated Candidates.
"""

from typing import List, Dict, Any, Tuple
import copy
import statistics
from .monitor import STLFormula

class STLGatedSearch:
    """
    Augments any NAS process with STL constraints.
    """
    
    def __init__(self, specs: List[STLFormula], robustness_threshold: float = 0.0):
        self.specs = specs
        self.rho_th = robustness_threshold
        # Search History: A list of Population Statistics (The "Signal")
        # Each element is a dict: {'mean_energy': ..., 'max_params': ..., 'best_acc': ...}
        self.signal_trace: List[Dict[str, float]] = []
        
        # We also track the raw populations if needed, but for STL we just need the signal.
        
    def _compute_population_metrics(self, population: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Convert a populations of architectures into a signal frame.
        Metrics:
        - best_acc (max)
        - mean_energy (mean)
        - max_params (max)
        """
        if not population:
            return {'best_acc': 0, 'mean_energy': 0, 'max_params': 0}
            
        accs = [p.get('accuracy', 0) for p in population]
        energies = [p.get('energy', 0) for p in population]
        params = [p.get('params', 0) for p in population]
        
        return {
            'best_acc': max(accs) if accs else 0,
            'mean_energy': statistics.mean(energies) if energies else 0,
            'max_params': max(params) if params else 0
        }

    def update_history(self, population: List[Dict[str, Any]]):
        """
        Commit a valid population to the history. (Phase 3 of Alg 1)
        """
        metrics = self._compute_population_metrics(population)
        self.signal_trace.append(metrics)

    def is_valid_candidate(self, 
                           current_population: List[Dict[str, Any]], 
                           candidate: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Phase 2 Verification:
        Check if adding 'candidate' to 'current_population' satisfies constraints.
        
        Returns: (is_valid, min_robustness)
        """
        # Form hypothetical population
        # Note: In evolutionary algs, we might replace someone or just add to pool.
        # Paper says: "P_test = P_{t-1} U {A}" (Line 10)
        # We assume addictive model for testing.
        test_pop = current_population + [candidate]
        
        # Compute hypothetical signal frame
        test_metrics = self._compute_population_metrics(test_pop)
        
        # Form hypothetical trace
        test_trace = self.signal_trace + [test_metrics]
        
        # Check robustness of all specs
        min_rho = float('inf')
        for spec in self.specs:
            # Evaluate at time 0 (Does the WHOLE trace satisfy spec?)
            # Or evaluate at current time t?
            # Standard STL semantics: G(phi) checked at 0 means "For all t in 0..T".
            # This ensures history is preserved and future is valid.
            rho = spec.robustness(test_trace, 0)
            if rho < min_rho:
                min_rho = rho
                
        is_valid = min_rho >= self.rho_th
        return is_valid, min_rho
