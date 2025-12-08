"""
Feedback Engine (CEGIS-STL).

Translates High-Level Temporal Violations (from STL) into Low-Level Spatial Constraints (for SMT).
This is the "Refinement" step in the Counter-Example Guided Inductive Synthesis loop.
"""

import z3
from typing import Dict, Any, List

class FeedbackEngine:
    """
    Generates Z3 constraints to guide the solver away from failing regions.
    """
    
    def __init__(self, encoding_instance):
        """
        :param encoding_instance: The DAGEncoding instance used by the solver.
                                  We need access to its Z3 variables (node_ops, etc).
        """
        self.encoding = encoding_instance
        
    def generate_feedback(self, 
                          solver: z3.Solver, 
                          arch_dict: List[Dict[str, Any]], 
                          violation_type: str, 
                          violation_value: float) -> tuple[str, Any]:
        """
        Applies feedback constraints.
        Returns: (Description, SymbolicConstraint or None)
        
        - If 'accuracy' (Topology Blocking): Adds PERMANENT constraint to solver. Returns None.
        - If 'energy' (Resource Limit): Returns symbolic constraint for assumption usage. Does NOT add to solver.
        """
        
        # 1. ALWAYS Block the current specific invalid architecture (Permanent Tabu)
        self._block_specific_architecture(solver, arch_dict)
        refinement_desc = "Blocked current architecture"
        constraint_expr = None
        
        # 2. Heuristic Refinement based on Violation Type
        if violation_type == "energy":
            if hasattr(self.encoding, 'total_luts'):
                # We want TotalLUTs < CurrentValue * 0.9
                limit = int(violation_value * 0.9) 
                # Create the symbolic expression but DO NOT add to solver
                constraint_expr = (self.encoding.total_luts < limit)
                refinement_desc += f", Constrained LUTs < {limit}"
                
        elif violation_type == "accuracy":
            # Already handled by blocking above
            pass
            
        return refinement_desc, constraint_expr

    def _block_specific_architecture(self, solver: z3.Solver, arch_dict: List[Dict[str, Any]]):
        """
        Adds a Not(And(Equality)) constraint to block this SPECIFIC configuration.
        Includes: Ops, Wiring (Edges), and Hyperparameters (Kernel, Stride, Channels).
        """
        
        constraints = []
        for node in arch_dict:
            nid = node['id']
            # Only constrain nodes within range
            if nid < self.encoding.max_nodes:
                # 1. Block Op Type
                constraints.append(self.encoding.node_ops[nid] == node['op'])
                
                # 2. Block Wiring (Edges)
                constraints.append(self.encoding.node_inputs1[nid] == node['in1'])
                constraints.append(self.encoding.node_inputs2[nid] == node['in2'])
                
                # 3. Block Hyperparameters
                # We need to access the symbolic variables list from encoding
                # Note: node['k'] corresponds to self.encoding.kernel_sizes[nid]
                constraints.append(self.encoding.kernel_sizes[nid] == node['k'])
                constraints.append(self.encoding.strides[nid] == node['s'])
                
                # Determine channel param. 
                # 'shape' tuple is (c, h, w). We need 'c'.
                c = node['shape'][0]
                constraints.append(self.encoding.out_channels_params[nid] == c)
        
        # Block: Not( And( All Constraints ) )
        # This means "Don't pick THIS exact combinaton again", but allow similar ones.
        solver.add(z3.Not(z3.And(constraints)))
