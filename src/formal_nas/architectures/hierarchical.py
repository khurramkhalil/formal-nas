"""
Hierarchical Architecture Components.

Implements reusable "Cells" and "SuperNets" to enable scalable NAS.
Instead of synthesizing a 50-layer graph, we synthesize a Cell and stack it.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
from .pytorch_exporter import PyTorchDAG

class SynthesizedCell(nn.Module):
    """
    A specific instance of a synthesized graph, adapted to be a layer in a larger net.
    """
    def __init__(self, arch_dict: List[Dict[str, Any]], in_channels: int, out_channels: int):
        super().__init__()
        
        # We reuse the logic from PyTorchDAG but need to be careful about scope
        # PyTorchDAG creates a classifier. We don't want that for a Cell.
        # Ideally PyTorchDAG should possess a 'headless' mode.
        # For now, we instantiate and strip.
        self.body = PyTorchDAG(arch_dict, in_channels, num_classes=10) # classes ignored
        
        # Remove head
        del self.body.classifier
        del self.body.global_pool
        
        # Determine output channels of the DAG body
        self.last_node_id = arch_dict[-1]['id']
        actual_out_c = self.body.node_shapes[self.last_node_id]
        
        # Project if needed to match requested out_channels
        if actual_out_c != out_channels:
            self.adapter = nn.Conv2d(actual_out_c, out_channels, kernel_size=1, bias=False)
        else:
            self.adapter = nn.Identity()
            
    def forward(self, x):
        # We need to manually drive the body's layers since body.forward() expects a head
        # This duplicates logic but ensures correctness without modifying PyTorchDAG heavily yet
        node_outputs = {}
        node_outputs[0] = x
        
        for node in self.body.architecture_dict:
            node_id = node['id']
            op_code = node['op']
            if op_code == -1: continue
            
            # Layer lookup
            layer = self.body.layers.get(str(node_id))
            if not layer: continue
            
            in1 = node['in1']
            input_tensor = node_outputs.get(in1)
            
            if op_code in [1, 2]: # Conv, Pool
                node_outputs[node_id] = layer(input_tensor)
            elif op_code == 3: # Add
                in2 = node.get('in2', -1)
                t2 = node_outputs.get(in2, torch.zeros_like(input_tensor))
                # Auto-align channels if mismatched? (e.g. via 1x1 conv)
                # In strict form, SMT guarantees match. In loose form, we might crash.
                if input_tensor.shape[1] != t2.shape[1]: 
                    # Silent failure or error? Let PyTorch error for now.
                    pass
                node_outputs[node_id] = input_tensor + t2
            elif op_code == 4: # Concat
                in2 = node.get('in2', -1)
                t2 = node_outputs.get(in2)
                node_outputs[node_id] = torch.cat([input_tensor, t2], dim=1)
            elif op_code == -2: # Output
                node_outputs[node_id] = input_tensor
                
        out = node_outputs[self.last_node_id]
        return self.adapter(out)

class SuperNet(nn.Module):
    """
    A Deep Network constructed by stacking SynthesizedCells.
    """
    def __init__(self, cell_arch: List[Dict[str, Any]], num_cells: int = 8, 
                 initial_channels: int = 16, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cells = nn.ModuleList()
        c_curr = initial_channels
        
        for _ in range(num_cells):
            # We treat every cell as keeping same channel count for simplicity
            # In a real ResNet, we'd double channels occasionally.
            cell = SynthesizedCell(cell_arch, c_curr, c_curr)
            self.cells.append(cell)
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_curr, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
