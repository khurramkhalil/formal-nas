"""
SMT Encoding for NAS-Bench-201 (TransNAS-Bench-101) Cell Search Space.

This encoding is SPECIFIC to the benchmark to ensure valid queries.
Structure:
- 4 Nodes (0, 1, 2, 3).
- Node 0: Input.
- Node 3: Output.
- Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
- Operations: 5 types on each edge.
"""

import z3
from typing import List, Dict, Any, Optional

class CellEncoding:
    def __init__(self, solver: z3.Solver, resource_limits: Optional[Dict[str, int]] = None):
        self.solver = solver
        self.resource_limits = resource_limits or {"luts": 100000, "dsp": 5000}
        
        # TransNAS-Bench-Micro Op Set (Derived from cell_ops.py)
        # 0: zero (none)
        # 1: identity (skip_connect)
        # 2: relu_conv_bn_1x1
        # 3: relu_conv_bn_3x3
        # Note: avg_pool_3x3 (4) is NOT supported in TransNAS-Micro.
        self.OPS = {
            "zero": 0,
            "identity": 1,
            "rcb_1x1": 2,
            "rcb_3x3": 3
        }
        self.OP_NAMES = ["zero", "identity", "rcb_1x1", "rcb_3x3"]
        
        # The 6 specific edges in the DAG
        self.edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        # Z3 Variables: One Int per Edge
        self.edge_ops = [z3.Int(f"edge_{u}_{v}") for u, v in self.edges]
        
        # Resource Variables
        self.total_luts = z3.Int("total_luts")
        
        self._init_constraints()
        self._add_resource_logic()
        self._add_connectivity_logic()
        
    def _init_constraints(self):
        """Basic Domain Constraints."""
        for op_var in self.edge_ops:
            # Op must be in [0, 3]
            self.solver.add(z3.And(op_var >= 0, op_var <= 3))

    def _add_connectivity_logic(self):
        """
        Enforce that Input (Node 0) reaches Output (Node 3).
        A path is active if ALL edges on it are NOT 'zero' (0).
        Paths in NAS-Bench-201 (0->1->2->3):
        1. 0->3 (edge_0_3, idx 2)
        2. 0->1->3 (edge_0_1, edge_1_3, idx 0, 4)
        3. 0->2->3 (edge_0_2, edge_2_3, idx 1, 5)
        4. 0->1->2->3 (edge_0_1, edge_1_2, edge_2_3, idx 0, 3, 5)
        """
        e_0_1 = self.edge_ops[0] != 0
        e_0_2 = self.edge_ops[1] != 0
        e_0_3 = self.edge_ops[2] != 0
        e_1_2 = self.edge_ops[3] != 0
        e_1_3 = self.edge_ops[4] != 0
        e_2_3 = self.edge_ops[5] != 0
        
        path1 = e_0_3
        path2 = z3.And(e_0_1, e_1_3)
        path3 = z3.And(e_0_2, e_2_3)
        path4 = z3.And(e_0_1, e_1_2, e_2_3)
        
        self.solver.add(z3.Or(path1, path2, path3, path4))

    def _add_resource_logic(self):
        """
        Approximate Resource Costs for the Operations.
        TransNAS ops: 0 (zero), 1 (skip), 2 (1x1), 3 (3x3).
        """
        costs = []
        for op_var in self.edge_ops:
            # Symbolic lookup for relative cost
            # zero=0
            # identity=0 (negligible)
            # rcb_1x1=300
            # rcb_3x3=1000
            c = z3.If(op_var == self.OPS["zero"], 0,
                z3.If(op_var == self.OPS["identity"], 0,
                z3.If(op_var == self.OPS["rcb_1x1"], 300,
                z3.If(op_var == self.OPS["rcb_3x3"], 1000, 0))))
            costs.append(c)
            
        self.solver.add(self.total_luts == z3.Sum(costs))
        
        if self.resource_limits:
             if "luts" in self.resource_limits:
                 self.solver.add(self.total_luts <= self.resource_limits["luts"])

    def decode_architecture(self, model: z3.ModelRef) -> str:
        """
        Converts Z3 model to TransNAS-Bench-101 Micro String Format.
        Format: '64-41414-{n1}_{n2}_{n3}'
        e.g., '64-41414-0_00_000'
        
        Mapping Edges to String Position:
        Node 1 (input from 0): edge(0,1) -> Index 0
        Node 2 (input from 0,1): edge(0,2), edge(1,2) -> Index 1, 3
        Node 3 (input from 0,1,2): edge(0,3), edge(1,3), edge(2,3) -> Index 2, 4, 5
        """
        vals = []
        for i in range(6):
            vals.append(model.eval(self.edge_ops[i], model_completion=True).as_long())
            
        # Construct Segments
        # Node 1: e0
        seg1 = f"{vals[0]}"
        # Node 2: e1, e3
        seg2 = f"{vals[1]}{vals[3]}"
        # Node 3: e2, e4, e5
        seg3 = f"{vals[2]}{vals[4]}{vals[5]}"
        
        return f"64-41414-{seg1}_{seg2}_{seg3}"

    def decode_to_dict(self, model: z3.ModelRef) -> List[Dict]:
        """
        Returns structured dict for Feedback Blocking.
        Format: [{'edge': i, 'op': val}, ...]
        """
        res = []
        for i, (u, v) in enumerate(self.edges):
             val = model.eval(self.edge_ops[i], model_completion=True).as_long()
             res.append({'edge': i, 'op': val})
        return res
