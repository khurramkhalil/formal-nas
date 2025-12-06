"""
Temporal Logic (TWTL/LTL) Engine for Formal NAS.

This module allows specifying architecture constraints using Temporal Logic.
Example: "Always(Conv -> Eventually(Pool))"
This is compiled into SMT constraints over the sequence of DAG nodes.

Note: In a DAG, "Time" is approximated by the topological sort index (node ID).
"""

import z3
from abc import ABC, abstractmethod
from typing import List, Any
from ..synthesis.dag_encoding import DAGEncoding

class TemporalFormula(ABC):
    @abstractmethod
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        """Encode formula at a specific time step."""
        pass

class Atom(TemporalFormula):
    """Atomic predicate (e.g., "node is Conv")."""
    def __init__(self, check_fn):
        self.check_fn = check_fn
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        if current_step >= encoding.max_nodes:
            return False # out of bounds
        
        # Check predicate on current node
        # We need to construct a Z3 boolean expression for the predicate
        # The check_fn should take (encoding, step) and return z3.BoolRef
        return self.check_fn(encoding, current_step)

class And(TemporalFormula):
    def __init__(self, *formulas):
        self.formulas = formulas
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        return z3.And([f.encode(solver, encoding, current_step) for f in self.formulas])

class Or(TemporalFormula):
    def __init__(self, *formulas):
        self.formulas = formulas
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        return z3.Or([f.encode(solver, encoding, current_step) for f in self.formulas])

class Not(TemporalFormula):
    def __init__(self, formula):
        self.formula = formula
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        return z3.Not(self.formula.encode(solver, encoding, current_step))

class Next(TemporalFormula):
    def __init__(self, formula):
        self.formula = formula
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        if current_step + 1 >= encoding.max_nodes:
            return False
        return self.formula.encode(solver, encoding, current_step + 1)

class Always(TemporalFormula):
    """Globally (G) operator."""
    def __init__(self, formula):
        self.formula = formula
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        # Unroll: Formula(t) AND Formula(t+1) ... AND Formula(end)
        conjunctions = []
        for t in range(current_step, encoding.max_nodes):
            # Only enforce on ACTIVE nodes? Usually yes.
            # "Always(P)" means "For all active nodes, P holds"
            p_holds = self.formula.encode(solver, encoding, t)
            is_active = encoding.node_active[t]
            conjunctions.append(z3.Implies(is_active, p_holds))
        return z3.And(conjunctions)

class Eventually(TemporalFormula):
    """Future (F) operator."""
    def __init__(self, formula):
        self.formula = formula
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        # Unroll: Or(Formula(t), Formula(t+1), ...)
        disjunctions = []
        for t in range(current_step, encoding.max_nodes):
            p_holds = self.formula.encode(solver, encoding, t)
            is_active = encoding.node_active[t]
            disjunctions.append(z3.And(is_active, p_holds))
        return z3.Or(disjunctions)

class Implies(TemporalFormula):
    def __init__(self, p, q):
        self.p = p
        self.q = q
        
    def encode(self, solver: z3.Solver, encoding: DAGEncoding, current_step: int) -> z3.BoolRef:
        return z3.Implies(
            self.p.encode(solver, encoding, current_step),
            self.q.encode(solver, encoding, current_step)
        )

# Helper for common predicates
def IsOp(op_code):
    return Atom(lambda enc, t: enc.node_ops[t] == op_code)
