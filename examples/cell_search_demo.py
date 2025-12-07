"""
Formal NAS: Cell-Based Search Demo.

Scalability Solution:
Instead of synthesizing a 50-layer flat graph (too hard for SMT),
we synthesize a high-performance "Cell" (Subgraph) and stack it.

1. SMT: Synthesize a valid Cell topology (Input -> ... -> Output).
2. PyTorch: Instantiate the Cell class multiple times to build a specific Deep Network.
"""

import sys
import os
import torch
import torch.nn as nn
import z3

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.architectures.pytorch_exporter import PyTorchDAG

class SynthesizedCell(nn.Module):
    """
    A Cell is a reusable block. It shares the SAME topology but different weights.
    We reuse PyTorchDAG logic but adapted to be a sub-module.
    """
    def __init__(self, arch_dict, in_channels, out_channels):
        super().__init__()
        # We need to adapt the PyTorchDAG to strictly map in_channels -> out_channels
        # In this simple demo, we basically use PyTorchDAG as the cell body
        # For simplicity, we assume the SMT solver found a graph that respects the I/O shapes
        
        # Hack: PyTorchDAG currently assumes it's the whole net with a classifier.
        # We will reuse its .layers and .forward logic but strip the classifier.
        
        self.body = PyTorchDAG(arch_dict, in_channels, num_classes=10)
        
        # Remove classifier/pooling to keep spatial dims
        del self.body.classifier
        del self.body.global_pool
        
        # We need to ensure channel matching for stacking.
        # If the synthesized cell output C != requested out_channels, we project.
        # Find the actual output channels of the synthesized graph
        self.last_node_id = arch_dict[-1]['id']
        actual_out_c = self.body.node_shapes[self.last_node_id]
        
        if actual_out_c != out_channels:
            self.adapter = nn.Conv2d(actual_out_c, out_channels, 1)
        else:
            self.adapter = nn.Identity()
            
    def forward(self, x):
        # We need to access PyTorchDAG.forward but without the head.
        # Easier to just cut-and-paste logic or make PyTorchDAG more modular.
        # For this demo, let's just copy the DAG forward logic here using self.body.
        
        node_outputs = {}
        node_outputs[0] = x
        
        for node in self.body.architecture_dict:
            node_id = node['id']
            op_code = node['op']
            
            if op_code == -1: continue # Input
            
            input_idx = node['in1']
            input_tensor = node_outputs.get(input_idx)
            
            if op_code == 1: # CONV
                # The layers are registered in self.body.layers
                node_outputs[node_id] = self.body.layers[str(node_id)](input_tensor)
            elif op_code == 2: # POOL
                node_outputs[node_id] = self.body.layers[str(node_id)](input_tensor)
            elif op_code == 3: # ADD
                 in2_idx = node.get('in2', -1)
                 input2 = node_outputs.get(in2_idx, torch.zeros_like(input_tensor))
                 node_outputs[node_id] = input_tensor + input2
            elif op_code == 4: # CONCAT
                 in2_idx = node.get('in2', -1)
                 input2 = node_outputs.get(in2_idx)
                 node_outputs[node_id] = torch.cat([input_tensor, input2], dim=1)
            elif op_code == -2: # OUTPUT
                 node_outputs[node_id] = input_tensor
        
        # Final output
        out = node_outputs[self.last_node_id]
        return self.adapter(out)

class SuperNet(nn.Module):
    def __init__(self, cell_arch, num_cells=5, channels=16):
        super().__init__()
        self.cells = nn.ModuleList()
        
        curr_c = 3 # Input image
        for _ in range(num_cells):
            # Stack cells
            cell = SynthesizedCell(cell_arch, curr_c, channels)
            self.cells.append(cell)
            curr_c = channels 
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, 10)
        
    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def run_cell_search():
    print("=== Formal NAS: Cell-Based Search Demo ===")
    
    # 1. Synthesize the "Cell"
    print("Synthesizing Cell Topology...")
    solver = z3.Solver()
    # We enforce input/output channels to be compatible for stacking (or let adapter handle it)
    encoding = DAGEncoding(solver, max_nodes=6, input_channels=16) # Assume internal channels 16
    
    # Just find a valid graph
    if solver.check() == z3.sat:
        model = solver.model()
        cell_arch = encoding.decode_architecture(model)
        
        print(f"Found Cell with {len(cell_arch)} nodes.")
        
        # 2. Build SuperNet
        print("Constructing 5-Layer SuperNet using this Cell...")
        try:
            supernet = SuperNet(cell_arch, num_cells=5, channels=16)
            
            # 3. Test
            dummy_in = torch.randn(2, 3, 32, 32)
            out = supernet(dummy_in)
            print(f"SuperNet Input: {dummy_in.shape}")
            print(f"SuperNet Output: {out.shape}")
            print("✅ Success! Hierarchical Synthesis achieved.")
            
        except Exception as e:
            print(f"❌ Failed to build SuperNet: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Could not synthesize cell.")

if __name__ == "__main__":
    run_cell_search()
