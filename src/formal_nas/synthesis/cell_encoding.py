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
    def __init__(self, solver: z3.Solver, resource_limits: Optional[Dict[str, int]] = None, benchmark_type: str = "transnas"):
        self.solver = solver
        self.benchmark_type = benchmark_type
        self.resource_limits = resource_limits or {"luts": 100000, "dsp": 5000}
        
        # Operation Sets
        if self.benchmark_type == "transnas":
            # TransNAS-Bench-Micro (4 Ops)
            self.OPS = {"zero": 0, "identity": 1, "rcb_1x1": 2, "rcb_3x3": 3}
            self.OP_NAMES = ["zero", "identity", "rcb_1x1", "rcb_3x3"]
            self.max_op_id = 3
            self.nas201_op_map = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3"]
        else:
            # Standard NAS-Bench-201 (5 Ops)
            # 0: none, 1: skip_connect, 2: nor_conv_1x1, 3: nor_conv_3x3, 4: avg_pool_3x3
            self.OPS = {"zero": 0, "identity": 1, "rcb_1x1": 2, "rcb_3x3": 3, "avg_pool": 4}
            self.OP_NAMES = ["zero", "identity", "rcb_1x1", "rcb_3x3", "avg_pool"]
            self.max_op_id = 4
            self.nas201_op_map = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
        
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
            # Op must be in valid range
            self.solver.add(z3.And(op_var >= 0, op_var <= self.max_op_id))

    def _add_connectivity_logic(self):
        """
        Enforce Input->Output connectivity.
        """
        # Logic remains same: path active if ops != zero
        # 'zero' is always ID 0
        path_conditions = [self.edge_ops[i] != 0 for i in range(6)]
        e_0_1, e_0_2, e_0_3, e_1_2, e_1_3, e_2_3 = path_conditions
        
        path1 = e_0_3
        path2 = z3.And(e_0_1, e_1_3)
        path3 = z3.And(e_0_2, e_2_3)
        path4 = z3.And(e_0_1, e_1_2, e_2_3)
        
        self.solver.add(z3.Or(path1, path2, path3, path4))

    def _add_resource_logic(self):
        """Approximate Resource Costs."""
        costs = []
        for op_var in self.edge_ops:
            # Shared Costs
            c = z3.If(op_var == 0, 0,
                z3.If(op_var == 1, 0,
                z3.If(op_var == 2, 300,
                z3.If(op_var == 3, 1000, 
                      0)))) # Default
            
            # NAS-Bench-201 Specific (AvgPool)
            if self.benchmark_type == "nas201":
                 # AvgPool cost ~ Identity (low logic, some buf)
                 c = z3.If(op_var == 4, 100, c)
                 
            costs.append(c)
            
        self.solver.add(self.total_luts == z3.Sum(costs))
        
        if self.resource_limits and "luts" in self.resource_limits:
             self.solver.add(self.total_luts <= self.resource_limits["luts"])

    def decode_architecture(self, model: z3.ModelRef) -> str:
        """
        Converts Z3 model to specific Benchmark String Format.
        """
        vals = []
        for i in range(6):
            vals.append(model.eval(self.edge_ops[i], model_completion=True).as_long())
            
        if self.benchmark_type == "transnas":
            # Format: '64-41414-{n1}_{n2}_{n3}'
            # Edges: e0, e1, e2, e3, e4, e5 corresponding to (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            # TransNAS Mapping:
            # Node 1: e(0,1) -> vals[0]
            # Node 2: e(0,2), e(1,2) -> vals[1], vals[3]
            # Node 3: e(0,3), e(1,3), e(2,3) -> vals[2], vals[4], vals[5]
            seg1 = f"{vals[0]}"
            seg2 = f"{vals[1]}{vals[3]}"
            seg3 = f"{vals[2]}{vals[4]}{vals[5]}"
            return f"64-41414-{seg1}_{seg2}_{seg3}"
            
        else:
            # Standard NAS-Bench-201 Format
            # |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
            op_strs = [self.nas201_op_map[v] for v in vals]
            # Node 1: |e(0,1)~0|
            n1 = f"|{op_strs[0]}~0|"
            # Node 2: |e(0,2)~0|e(1,2)~1|
            n2 = f"|{op_strs[1]}~0|{op_strs[3]}~1|"
            # Node 3: |e(0,3)~0|e(1,3)~1|e(2,3)~2|
            n3 = f"|{op_strs[2]}~0|{op_strs[4]}~1|{op_strs[5]}~2|"
            
            return f"{n1}+{n2}+{n3}"

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
