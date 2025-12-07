"""
SMT Encoding for Directed Acyclic Graph (DAG) Neural Architectures.

This module implements the core 'Novelty': Representing arbitrary neural graphs
in Z3 and enforcing Correct-by-Construction properties.

Encoding Strategy: Node-Centric
Instead of a full O(N^2) adjacency matrix, each node 'chooses' its inputs
from previous nodes. This enforces the DAG property (i < j) by design and
simplifies constraints for standard layers (usually 1 or 2 inputs).
"""

import z3
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ..architectures.layer_types import LayerType
from ..logic.shape_inference import (
    get_conv2d_output_shape,
    get_pool2d_output_shape,
    is_valid_conv_config
)
from ..hardware_models.symbolic import SymbolicFPGAModel

class DAGEncoding:
    """
    Manages SMT variables and constraints for a DAG architecture.
    """
    
    def __init__(
        self,
        solver: z3.Solver,
        max_nodes: int,
        input_channels: int,
        input_resolution: int = 32,
        resource_limits: Optional[Dict[str, int]] = None
    ):
        self.solver = solver
        self.max_nodes = max_nodes
        self.input_channels = input_channels
        self.input_resolution = input_resolution
        self.resource_limits = resource_limits or {"luts": 1000000, "dsp": 5000, "bram": 10000000}
        self.hw_model = SymbolicFPGAModel()
        
        # New: Arithmetic Intensity Threshold (Ops/Byte)
        # If set in resource_limits as 'min_intensity', we enforce it.
        self.min_intensity = self.resource_limits.get("min_intensity", 0)
        
        # SMT Variables
        self.node_active = []    # Bool: Is node i part of the active graph?
        self.node_ops = []       # Int: Operation type Enum
        self.node_inputs1 = []   # Int: Index of first input (< i)
        self.node_inputs2 = []   # Int: Index of second input (< i) (for Add/Concat)
        
        # Shape Variables (Symbolic)
        self.out_h = []
        self.out_w = []
        self.out_c = []
        
        # Layer Hyperparameters (Symbolic)
        self.kernel_sizes = []   # 3 or 5 usually
        self.strides = []        # 1 or 2
        self.out_channels_params = [] # e.g., 32, 64, 128
        
        # Resource Variables
        self.total_luts = z3.Int("total_luts")
        self.total_dsp = z3.Int("total_dsp")
        self.total_bram = z3.Int("total_bram")
        
        # Roofline Variables
        self.total_ops = z3.Int("total_ops")
        self.total_bytes = z3.Int("total_bytes")
        
        self._init_variables()
        self._add_structural_constraints()
        self._add_shape_and_resource_logic()

    def _init_variables(self):
        """Initialize Z3 variables for all nodes."""
        # Op Codes
        self.OP_CONV = 1
        self.OP_POOL = 2
        self.OP_ADD = 3
        self.OP_CONCAT = 4
        self.OP_OUTPUT = -2
        self.OP_INPUT = -1
        
        valid_ops = [self.OP_CONV, self.OP_POOL, self.OP_ADD, self.OP_CONCAT, self.OP_OUTPUT, self.OP_INPUT]
        
        for i in range(self.max_nodes):
            # Topology
            self.node_active.append(z3.Bool(f"node_{i}_active"))
            op_var = z3.Int(f"node_{i}_op")
            self.node_ops.append(op_var)
            
            # Domain Constraint: Op must be valid
            self.solver.add(z3.Or([op_var == v for v in valid_ops]))
            
            self.node_inputs1.append(z3.Int(f"node_{i}_in1"))
            self.node_inputs2.append(z3.Int(f"node_{i}_in2"))
            
            # Constrain Hyperparameters
            # Initialize lists first!
            self.kernel_sizes.append(z3.Int(f"node_{i}_k"))
            self.strides.append(z3.Int(f"node_{i}_s"))
            self.out_channels_params.append(z3.Int(f"node_{i}_ch_param"))
            
            k = self.kernel_sizes[i]
            s = self.strides[i]
            c = self.out_channels_params[i]
            
            self.solver.add(z3.Or(k == 1, k == 3, k == 5))
            self.solver.add(z3.Or(s == 1, s == 2))
            self.solver.add(z3.Or(c == 16, c == 32, c == 64))
            
            # Shapes
            self.out_h.append(z3.Int(f"node_{i}_h"))
            self.out_w.append(z3.Int(f"node_{i}_w"))
            self.out_c.append(z3.Int(f"node_{i}_c"))
            
            # Non-negative constraints
            self.solver.add(self.out_h[i] > 0)
            self.solver.add(self.out_w[i] > 0)
            self.solver.add(self.out_c[i] > 0)
    
    def _add_structural_constraints(self):
        """Enforce basic graph validity."""
        # Node 0 is always INPUT
        self.solver.add(self.node_active[0] == True)
        self.solver.add(self.node_ops[0] == -1) # -1 Reserved for INPUT
        self.solver.add(self.out_h[0] == self.input_resolution)
        self.solver.add(self.out_w[0] == self.input_resolution)
        self.solver.add(self.out_c[0] == self.input_channels)
        
        for i in range(1, self.max_nodes):
            # Input indices must be valid predecessors
            self.solver.add(self.node_inputs1[i] >= 0)
            self.solver.add(self.node_inputs1[i] < i)
            self.solver.add(self.node_inputs2[i] >= 0)
            self.solver.add(self.node_inputs2[i] < i)
            
            # If node is active, its inputs must be active
            # Note: This is a simplification. Usually we trace back from output.
            # But enforcing "active implies inputs active" ensures connectivity.
            input1_active = self._get_var_at_index(self.node_active, self.node_inputs1[i])
            input2_active = self._get_var_at_index(self.node_active, self.node_inputs2[i])
            
            self.solver.add(z3.Implies(self.node_active[i], input1_active))
            
            # Determine if node *needs* 2 inputs (Add/Concat)
            needs_two = z3.Or(self.node_ops[i] == self.OP_ADD, self.node_ops[i] == self.OP_CONCAT)
            self.solver.add(z3.Implies(z3.And(self.node_active[i], needs_two), input2_active))
            
            # Constraint: Only Node 0 is INPUT. Others cannot be OP_INPUT (-1)
            self.solver.add(self.node_ops[i] != self.OP_INPUT)
            
            # Constraint: Only Last Node is OUTPUT. Others cannot be OP_OUTPUT (-2)
            if i < self.max_nodes - 1:
                 self.solver.add(self.node_ops[i] != self.OP_OUTPUT)
        
        # Enforce Output Node (Last Node)
        last_idx = self.max_nodes - 1
        self.solver.add(self.node_active[last_idx] == True)
        self.solver.add(self.node_ops[last_idx] == -2) # -2 Reserved for OUTPUT
        
        # Enforce "No Dead Ends" (All active nodes must connect to something, except Output)
        for i in range(self.max_nodes - 1): # Skip last node
            # A node 'i' is used if some 'j > i' is active AND:
            # (j's input1 is i) OR (j needs 2 inputs AND j's input2 is i)
            # Note: All active nodes use input1. Only some use input2.
            
            is_used_list = []
            for j in range(i + 1, self.max_nodes):
                j_needs_two = z3.Or(self.node_ops[j] == self.OP_ADD, self.node_ops[j] == self.OP_CONCAT)
                
                # Check actual usage
                used_by_j = z3.And(
                    self.node_active[j],
                    z3.Or(
                        self.node_inputs1[j] == i,
                        z3.And(j_needs_two, self.node_inputs2[j] == i)
                    )
                )
                is_used_list.append(used_by_j)
            
            is_used = z3.Or(is_used_list)
            self.solver.add(z3.Implies(self.node_active[i], is_used))

    def _get_var_at_index(self, var_list, idx_var):
        """Symbolic array access: list[idx_var]."""
        # Since max_nodes is small (~10-20), simple If-Else ladder is fine and often faster than Z3 Arrays
        # for bounded integers.
        expr = var_list[0]
        for j in range(1, len(var_list)):
            expr = z3.If(idx_var == j, var_list[j], expr)
        return expr

    def _add_shape_and_resource_logic(self):
        """
        Symbolic Shape Inference AND Resource Cost Accumulation.
        """
        node_luts = []
        node_dsps = []
        node_brams = []
        node_ops = []
        node_bytes = []
        
        # Cost of Input Node is 0
        node_luts.append(z3.IntVal(0))
        node_dsps.append(z3.IntVal(0))
        node_brams.append(z3.IntVal(0))
        node_ops.append(z3.IntVal(0))
        node_bytes.append(z3.IntVal(0))
        
        for i in range(1, self.max_nodes):
            # Get properties of input 1
            in1_h = self._get_var_at_index(self.out_h, self.node_inputs1[i])
            in1_w = self._get_var_at_index(self.out_w, self.node_inputs1[i])
            in1_c = self._get_var_at_index(self.out_c, self.node_inputs1[i])
            
            # Get properties of input 2
            in2_h = self._get_var_at_index(self.out_h, self.node_inputs2[i])
            in2_w = self._get_var_at_index(self.out_w, self.node_inputs2[i])
            in2_c = self._get_var_at_index(self.out_c, self.node_inputs2[i])
            
            # Logic branch based on Operation Type
            
            # --- CASE: CONV2D ---
            # Output shape logic
            conv_h, conv_w = get_conv2d_output_shape(
                in1_h, in1_w, self.kernel_sizes[i], self.strides[i], padding=1
            )
            res_conv = self.hw_model.estimate_conv2d_symbolic(
                in_channels=in1_c,
                out_channels=self.out_channels_params[i],
                kernel_size=self.kernel_sizes[i],
                input_height=in1_h, # Use input or output dims? Conv cost depends on output MACs actually
                input_width=in1_w 
            )
            
            # --- CASE: POOLING ---
            pool_h, pool_w = get_pool2d_output_shape(
                in1_h, in1_w, 2, 2, padding=0 # Fixed 2x2 maxpool for now
            )
            res_pool = self.hw_model.estimate_pool2d_symbolic(in1_h, in1_w, in1_c)
            
            # --- CASE: ADD (Residual) ---
            res_add = self.hw_model.estimate_add_symbolic(in1_h, in1_w, in1_c)
            
            # --- CASE: CONCAT ---
            # Concat cost is usually negligible or handled by routing, assume 0 for now
            res_concat_luts = z3.IntVal(0)
            res_concat_dsps = z3.IntVal(0)
            res_concat_brams = z3.IntVal(0)

            # --- CASE: OUTPUT ---
            # Output node cost is usually negligible, assume 0 for now
            res_output_luts = z3.IntVal(0)
            res_output_dsps = z3.IntVal(0)
            res_output_brams = z3.IntVal(0)
            
            # --- Constraints & Cost Selection ---
            
            is_conv = (self.node_ops[i] == self.OP_CONV)
            is_pool = (self.node_ops[i] == self.OP_POOL)
            is_add = (self.node_ops[i] == self.OP_ADD)
            is_concat = (self.node_ops[i] == self.OP_CONCAT)
            is_output = (self.node_ops[i] == self.OP_OUTPUT)
            
            # Apply Shape Logic (Implies...)
            self.solver.add(z3.Implies(is_conv, z3.And(
                self.out_h[i] == conv_h,
                self.out_w[i] == conv_w,
                self.out_c[i] == self.out_channels_params[i],
                is_valid_conv_config(in1_h, in1_w, self.kernel_sizes[i], padding=1)
            )))
            
            # Validity: Kernel size constraints for Conv
            self.solver.add(z3.Implies(is_conv, z3.Or(
                self.kernel_sizes[i] == 3,
                self.kernel_sizes[i] == 1
            )))

            self.solver.add(z3.Implies(is_pool, z3.And(
                self.out_h[i] == pool_h,
                self.out_w[i] == pool_w,
                self.out_c[i] == in1_c
            )))
            self.solver.add(z3.Implies(is_add, z3.And(
                 in1_h == in2_h, in1_w == in2_w, in1_c == in2_c,
                 self.out_h[i] == in1_h, self.out_w[i] == in1_w, self.out_c[i] == in1_c,
                 self.node_inputs1[i] != self.node_inputs2[i] # No self-loops/dupe inputs
            )))
            self.solver.add(z3.Implies(is_concat, z3.And(
                 in1_h == in2_h, in1_w == in2_w,
                 self.out_h[i] == in1_h, self.out_w[i] == in1_w, self.out_c[i] == in1_c + in2_c
            )))
            self.solver.add(z3.Implies(is_output, z3.And(
                self.out_h[i] == in1_h, self.out_w[i] == in1_w, self.out_c[i] == in1_c
            )))
            
            # Resource Cost Selection (Multiplexer Logic)
            # If active: cost = op_cost. If inactive: cost = 0.
            
            current_lut = z3.If(is_conv, res_conv['luts'],
                          z3.If(is_pool, res_pool['luts'],
                          z3.If(is_add, res_add['luts'],
                          z3.If(is_concat, res_concat_luts,
                          z3.If(is_output, res_output_luts, z3.IntVal(0)))))) # Default to 0 if not active or unknown op
            
            current_dsp = z3.If(is_conv, res_conv['dsp'],
                          z3.If(is_pool, res_pool['dsp'],
                          z3.If(is_add, res_add['dsp'],
                          z3.If(is_concat, res_concat_dsps,
                          z3.If(is_output, res_output_dsps, z3.IntVal(0))))))

            current_bram = z3.If(is_conv, res_conv['bram'],
                           z3.If(is_pool, res_pool['bram'],
                           z3.If(is_add, res_add['bram'],
                           z3.If(is_concat, res_concat_brams,
                           z3.If(is_output, res_output_brams, z3.IntVal(0))))))

            current_ops = z3.If(is_conv, res_conv['ops'],
                            z3.If(is_pool, res_pool['ops'],
                            z3.If(is_add, res_add['ops'],
                            z3.If(is_concat, z3.IntVal(0), # Concat ops=0
                            z3.If(is_output, z3.IntVal(0), z3.IntVal(0))))))

            current_bytes = z3.If(is_conv, res_conv['bytes'],
                            z3.If(is_pool, res_pool['bytes'],
                            z3.If(is_add, res_add['bytes'],
                            z3.If(is_concat, z3.IntVal(0), # Concat bytes=0
                            z3.If(is_output, z3.IntVal(0), z3.IntVal(0))))))

            # Only count cost if node is active
            node_luts.append(z3.If(self.node_active[i], current_lut, z3.IntVal(0)))
            node_dsps.append(z3.If(self.node_active[i], current_dsp, z3.IntVal(0)))
            node_brams.append(z3.If(self.node_active[i], current_bram, z3.IntVal(0)))
            node_ops.append(z3.If(self.node_active[i], current_ops, z3.IntVal(0)))
            node_bytes.append(z3.If(self.node_active[i], current_bytes, z3.IntVal(0)))
            
        # Sum total resources
        self.solver.add(self.total_luts == z3.Sum(node_luts))
        self.solver.add(self.total_dsp == z3.Sum(node_dsps))
        self.solver.add(self.total_bram == z3.Sum(node_brams))
        
        # Enforce Limits
        self.solver.add(self.total_luts <= self.resource_limits['luts'])
        self.solver.add(self.total_dsp <= self.resource_limits['dsp'])
        self.solver.add(self.total_bram <= self.resource_limits['bram'])
        
        self.solver.add(self.total_ops == z3.Sum(node_ops))
        self.solver.add(self.total_bytes == z3.Sum(node_bytes))
        
        # Roofline Constraint: Ops >= Intensity * Bytes
        if self.min_intensity > 0:
             self.solver.add(self.total_ops >= self.min_intensity * self.total_bytes)
        
    def decode_architecture(self, model: z3.ModelRef) -> List[Dict[str, Any]]:
        """Convert SMT model logic back to Python dict representation."""
        arch = []
        for i in range(self.max_nodes):
            is_active = model.eval(self.node_active[i], model_completion=True)
            if z3.is_true(is_active):
                op = model.eval(self.node_ops[i], model_completion=True).as_long()
                in1 = model.eval(self.node_inputs1[i], model_completion=True).as_long()
                in2 = model.eval(self.node_inputs2[i], model_completion=True).as_long()
                
                # Params
                # Retrieve symbolic values if they exist (they are initialized for all i)
                k = model.eval(self.kernel_sizes[i], model_completion=True).as_long()
                s = model.eval(self.strides[i], model_completion=True).as_long()
                
                arch.append({
                    "id": i,
                    "op": op,
                    "in1": in1,
                    "in2": in2,
                    "k": k,
                    "s": s,
                    "shape": (
                        model.eval(self.out_c[i], model_completion=True).as_long(),
                        model.eval(self.out_h[i], model_completion=True).as_long(),
                        model.eval(self.out_w[i], model_completion=True).as_long()
                    )
                })
        
        # Add resource usage to output
        try:
             luts = model.eval(self.total_luts, model_completion=True).as_long()
             dsps = model.eval(self.total_dsp, model_completion=True).as_long()
             brams = model.eval(self.total_bram, model_completion=True).as_long()
             print(f"Estimated Resources: LUTs={luts}, DSPs={dsps}, BRAMs={brams}")
        except z3.Z3Exception:
             print("Could not evaluate all resource variables from the model.")
             
        return arch

