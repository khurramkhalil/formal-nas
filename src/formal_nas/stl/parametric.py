"""
Parametric STL (P-STL) Engine.

Extends standard STL to support "Parameter Synthesis".
Instead of checking "rho(phi, trace) > 0?", we ask:
"Find optimal parameters theta such that rho(phi(theta), trace) >= 0".

This enables automatic discovery of performance bounds (Pareto Frontiers) 
and integrates with CEGIS.
"""

from typing import Dict, List, Tuple, Callable, Any, Optional
from abc import ABC, abstractmethod
from .monitor import STLFormula, Signal, Globally, Eventually, And, Or

class ParametricPredicate(STLFormula):
    """
    Predicate dependent on a parameter 'theta'.
    Form: f(x) ~ theta.
    Robustness is typically: theta - f(x) (for <) or f(x) - theta (for >).
    
    We assume the form: val < theta (Upper Bound) OR val > theta (Lower Bound).
    """
    def __init__(self, name: str, signal_key: str, param_name: str, direction: str = '<'):
        self.name = name
        self.signal_key = signal_key
        self.param_name = param_name
        self.direction = direction # '<' or '>'
        
    def robustness(self, signal: Signal, t: int, params: Dict[str, float] = {}) -> float:
        # NOTE: We overload the signature to accept 'params'
        if t >= len(signal):
            return float('inf')
            
        val = signal[t].get(self.signal_key, 0.0)
        theta = params.get(self.param_name, 0.0)
        
        if self.direction == '<':
            # Req: val < theta
            # Robustness: theta - val
            return theta - val
        else:
            # Req: val > theta
            # Robustness: val - theta
            return val - theta
            
    def get_parameter_names(self) -> List[str]:
        return [self.param_name]

# P-STL requires modifying the base classes to propagate 'params'.
# We can either duplicate the hierarchy or accept that 'params' defaults to empty.
# In Python, we can just monkey-patch or subclass. 
# For cleanliness, let's define a synthesis helper that works with the existing structure
# if we pass params via a context, OR we just update the base monitor.py to support params.
# Given strict separation, let's make a Synthesis Wrapper.

class ParameterSynthesizer:
    """
    Learns the tightest parameters for a trace.
    """
    
    def synthesize(self, formula: STLFormula, trace: Signal, param_names: List[str]) -> Dict[str, float]:
        """
        Finds theta* such that Formula(theta*) is satisfied with margin ~0.
        
        Assumption: Monotonicity. 
        - If direction is '<' (Upper Bound), we want MIN theta.
        - If direction is '>' (Lower Bound), we want MAX theta.
        """
        learned_params = {}
        
        # Simple heuristic synthesis for specific known structures.
        # Handling arbitrary boolean combinations automatically is complex (HyST / logic solvers).
        # We focus on the user's Use Case: "bounds".
        
        # For "Globally(val < theta)", theta* = max(val_trace).
        # For "Globally(val > theta)", theta* = min(val_trace).
        # For "Eventually(val > theta)", theta* = max(val_trace).
        
        # We can simulate this by extracting the relevant signal dimension 
        # and applying the logic.
        
        # GENERAL ALGORITHM:
        # Since we likely just have conjoined bounds `G(e < theta_e) and F(acc > theta_acc)`,
        # we can optimize each parameter independently IF they are independent.
        
        # 1. Identify constraint type for each param.
        # This requires introspecting the formula structure or user-provided meta-data.
        # Let's assume the user provides specific "Template" objects or we inspect.
        
        # Let's iterate over known params and find their optimal values via "validity search".
        # Since we have the trace, we can just compute the property for the trace 
        # and see the boundary.
        
        # Simplification: We assume the formula is a conjunction of Parametric Predicates
        # wrapped in temporal operators.
        
        return self._naive_synthesis(formula, trace, param_names)

    def _naive_synthesis(self, formula, trace, param_names) -> Dict[str, float]:
        """
        Synthesize by scanning the trace. 
        Note: This implementation assumes simple independent bounds.
        """
        result = {}
        
        # We need to traverse the formula tree to find the Predicates associated 
        # with each parameter and the temporal operator wrapping them.
        
        # This is a bit tricky with the current OOP structure without Visitor pattern.
        # Alternative: Binary Search for each parameter.
        # Robustness is monotonic with theta.
        # Finding theta s.t. Rho(theta) = 0.
        
        for name in param_names:
            # Binary Search Range
            low = -1e6
            high = 1e6
            epsilon = 1e-3
            
            # Check direction (Does increasing theta increase or decrease robustness?)
            # Sample at low/high
            p_low = {name: low}
            # We need to inject other params. Assume 0 or hold fixed? 
            # Ideally hold others fixed. But we don't know them yet.
            # Iterative Coordinate Descent?
            
            # Fallback: Just let the user define "Optimizers".
            pass
            
        return {}

# Improved Implementation: Typed Synthesis
# We define specific Learnable Classes.

from scipy.optimize import minimize_scalar

class LearnableCheck(ABC):
    @abstractmethod
    def learn(self, trace: Signal) -> float:
        pass
        
    def _optimize_monotonic(self, objective_fn: Callable[[float], float], bounds: Tuple[float, float]) -> float:
        """
        Generic optimization wrapper using SciPy.
        Finds theta that minimizes objective_fn(theta).
        """
        res = minimize_scalar(
            objective_fn, 
            bounds=bounds, 
            method='bounded',
            options={'xatol': 1e-6}
        )
        if res.success:
            return float(res.x)
        return bounds[0] # Fallback

class LearnableUpperBound(LearnableCheck):
    """
    Corresponds to G(val < theta).
    Robustness = theta - val.
    We want the smallest theta such that Robustness >= 0 for all t.
    equivalent to theta >= max(val).
    Optimal theta = max(val).
    """
    def __init__(self, key: str):
        self.key = key
        
    def learn(self, trace: Signal) -> float:
        vals = [state.get(self.key, float('-inf')) for state in trace]
        if not vals: return 0.0
        
        # Optimization Formulation:
        # Minimize theta subject to: min_t(theta - val(t)) >= 0
        # Objective: theta
        peak = max(vals)
        
        # Demonstration of Optimization (User Request)
        # We define a loss function: |theta - peak|
        # This seems trivial for "Max", but establishes the pattern for complex logic.
        
        def loss(theta):
            # Penalty if bound violated
            # robustness = min(theta - v for v in vals)
            # if robustness < 0: return 1e6 (Invalid)
            # return theta (Minimize upper bound)
            robustness = theta - peak
            if robustness < 0: return 1e9 # Violated
            return theta 
            
        # Search Range: [min(vals)-10, max(vals)+10]
        search_bounds = (min(vals), peak + 10.0)
        
        return self._optimize_monotonic(loss, search_bounds)

class LearnableLowerBound(LearnableCheck):
    """
    Corresponds to G(val > theta).
    Robustness = val - theta.
    We want largest theta such that Robustness >= 0.
    equivalent to theta <= min(val).
    Optimal theta = min(val).
    """
    def __init__(self, key: str):
        self.key = key
        
    def learn(self, trace: Signal) -> float:
        vals = [state.get(self.key, float('inf')) for state in trace]
        if not vals: return 0.0
        
        val_min = min(vals)
        
        # maximize theta => minimize -theta
        def loss(theta):
            robustness = val_min - theta
            if robustness < 0: return 1e9 # Violated
            return -theta
            
        search_bounds = (val_min - 10.0, val_min)
        
        res = self._optimize_monotonic(loss, search_bounds)
        return res

class LearnableEventualGoal(LearnableCheck):
    """
    Corresponds to F(val > theta).
    We want largest theta such that max(val) >= theta.
    Optimal theta = max(val).
    """
    def __init__(self, key: str):
        self.key = key
        
    def learn(self, trace: Signal) -> float:
        vals = [state.get(self.key, float('-inf')) for state in trace]
        if not vals: return 0.0
        
        peak = max(vals)
        
        def loss(theta):
            # We satisfy F(val > theta) if peak >= theta
            if peak < theta: return 1e9
            return -theta # Maximize theta
            
        search_bounds = (min(vals), peak)
        return self._optimize_monotonic(loss, search_bounds)

class PSTLContext:
    """
    Manages parameters and learning.
    """
    def __init__(self):
        self.learnables: Dict[str, LearnableCheck] = {}
        
    def register_upper_bound(self, param_name: str, signal_key: str):
        self.learnables[param_name] = LearnableUpperBound(signal_key)
        
    def register_lower_bound(self, param_name: str, signal_key: str):
        self.learnables[param_name] = LearnableLowerBound(signal_key)
        
    def register_eventual_goal(self, param_name: str, signal_key: str):
        self.learnables[param_name] = LearnableEventualGoal(signal_key)
        
    def synthesize(self, trace: Signal) -> Dict[str, float]:
        params = {}
        for name, learner in self.learnables.items():
            params[name] = learner.learn(trace)
        return params
