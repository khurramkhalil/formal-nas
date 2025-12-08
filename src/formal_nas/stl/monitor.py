"""
STL Robustness Monitor.

Implements Signal Temporal Logic (STL) quantitative semantics.
Ref: "Monitoring Temporal Properties of Continuous Signals", Maler & Nickovic (2004).

This engine computes the Robustness Degree (rho) of a signal trace against a specification.
Rho > 0: Satisfied
Rho < 0: Violated
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Optional, Tuple

# A Signal is a sequence of states (dicts) indexed by time 0..T
Signal = List[Dict[str, float]]

class STLFormula(ABC):
    """Abstract Base Class for STL Formulas."""
    
    @abstractmethod
    def robustness(self, signal: Signal, t: int) -> float:
        """
        Compute the robustness degree rho(phi, signal, t).
        """
        pass

class Predicate(STLFormula):
    """
    Atomic proposition: f(x) >= 0.
    In STL-NAS, we often express "val < limit", which is "limit - val >= 0".
    
    fn: lambda state: float. The value itself is the robustness.
    """
    def __init__(self, name: str, fn: Callable[[Dict[str, float]], float]):
        self.name = name
        self.fn = fn
        
    def robustness(self, signal: Signal, t: int) -> float:
        if t >= len(signal):
            return float('-inf') # Undefined future
        return self.fn(signal[t])
    
    def __repr__(self):
        return f"Pred({self.name})"

class Not(STLFormula):
    def __init__(self, phi: STLFormula):
        self.phi = phi
        
    def robustness(self, signal: Signal, t: int) -> float:
        return -self.phi.robustness(signal, t)

class And(STLFormula):
    def __init__(self, phi1: STLFormula, phi2: STLFormula):
        self.phi1 = phi1
        self.phi2 = phi2
        
    def robustness(self, signal: Signal, t: int) -> float:
        return min(self.phi1.robustness(signal, t), self.phi2.robustness(signal, t))

class Or(STLFormula):
    def __init__(self, phi1: STLFormula, phi2: STLFormula):
        self.phi1 = phi1
        self.phi2 = phi2
        
    def robustness(self, signal: Signal, t: int) -> float:
        return max(self.phi1.robustness(signal, t), self.phi2.robustness(signal, t))

class Globally(STLFormula):
    """
    G_[a, b] phi: phi holds for all t' in [t+a, t+b].
    Robustness = min(rho(phi, t')) for t' in [t+a, t+b]
    """
    def __init__(self, phi: STLFormula, interval: Tuple[int, int] = (0, float('inf'))):
        self.phi = phi
        self.a, self.b = interval
        
    def robustness(self, signal: Signal, t: int) -> float:
        # Determine valid range
        start = t + self.a
        end = min(t + self.b + 1, len(signal)) if self.b != float('inf') else len(signal)
        
        if start >= len(signal):
             # Vacuously true? Or failure? 
             # In bounded STL, future beyond trace is typically handled by padding or returning -inf.
             # For NAS monitoring, if we check 'history', we only care about 0..NOW.
             # If formula looks into future, we can't eval. 
             # Assuming we evaluate up to current time or 'b' is 0 for past.
             # Actually, STL-NAS checks the *whole trajectory* so far.
             return float('inf') 

        val = float('inf')
        for k in range(start, int(end)):
            val = min(val, self.phi.robustness(signal, k))
        return val

class Eventually(STLFormula):
    """
    F_[a, b] phi: phi holds at SOME t' in [t+a, t+b].
    Robustness = max(rho(phi, t'))
    """
    def __init__(self, phi: STLFormula, interval: Tuple[int, int] = (0, float('inf'))):
        self.phi = phi
        self.a, self.b = interval
        
    def robustness(self, signal: Signal, t: int) -> float:
        start = t + self.a
        end = min(t + self.b + 1, len(signal)) if self.b != float('inf') else len(signal)
        
        if start >= len(signal):
            return float('-inf')

        val = float('-inf')
        for k in range(start, int(end)):
            val = max(val, self.phi.robustness(signal, k))
        return val

# Common Factory Functions
def Always(phi, interval=(0, float('inf'))): return Globally(phi, interval)
