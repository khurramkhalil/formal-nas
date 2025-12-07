"""
PyTorch Exporter for Formal NAS.

Converts a synthesized DAG architecture (dictionary format) into a runnable
PyTorch `nn.Module`.

Bridging the gap between Formal Synthesis (Z3) and Empirical Evaluation (Training).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any

class PyTorchDAG(nn.Module):
    """
    A PyTorch Module representing a synthesized DAG.
    """
    def __init__(self, architecture_dict: List[Dict[str, Any]], input_channels: int, num_classes: int = 10):
        super().__init__()
        self.architecture_dict = sorted(architecture_dict, key=lambda x: x['id'])
        self.layers = nn.ModuleDict()
        
        # Track channel counts for each node to correctly instantiate subsequent layers
        # node_id -> out_channels
        self.node_shapes = {} 
        self.node_shapes[0] = input_channels # Input node
        
        # Operation Mapping (Constants from DAGEncoding)
        # 1=CONV, 2=POOL, 3=ADD, 4=CONCAT. -1=INPUT, -2=OUTPUT
        
        for node in self.architecture_dict:
            node_id = node['id']
            op_code = node['op']
            input_idx = node['in1'] # Primary input
            
            # Skip Input Node (Logic handled in forward)
            if op_code == -1:
                self.node_shapes[node_id] = input_channels
                continue
                
            # Resolve Input Channels
            # Note: For Concat/Add, we might need to look at multiple inputs, 
            # but DAGEncoding simplifies this by ensuring shape consistency logic.
            # We trust the Z3 shape inference!
            
            current_in_channels = self.node_shapes.get(input_idx, input_channels)
            
            if op_code == 1: # CONV
                # Kernel, channel, stride from node dict
                out_c = node['shape'][0]
                kernel_size = node.get('k', 3)
                stride = node.get('s', 1)
                padding = kernel_size // 2 # Same padding
                
                # Create Layer
                self.layers[str(node_id)] = nn.Sequential(
                    nn.Conv2d(current_in_channels, out_c, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
                self.node_shapes[node_id] = out_c
                
            elif op_code == 2: # POOL
                # MaxPool 2x2. Fixed for now but could be parameter
                self.layers[str(node_id)] = nn.MaxPool2d(kernel_size=2, stride=2)
                self.node_shapes[node_id] = current_in_channels
                
            elif op_code == 3: # ADD
                # No parameters, just logic
                self.layers[str(node_id)] = nn.Identity() 
                self.node_shapes[node_id] = current_in_channels
                
            elif op_code == 4: # CONCAT
                self.layers[str(node_id)] = nn.Identity()
                # Get input2 channels
                in2_idx = node.get('in2', -1)
                in2_c = self.node_shapes.get(in2_idx, 0)
                self.node_shapes[node_id] = current_in_channels + in2_c 
            
            elif op_code == -2: # OUTPUT
                self.layers[str(node_id)] = nn.Identity()
                self.node_shapes[node_id] = current_in_channels

        # Classifier Head
        final_c = self.node_shapes[self.architecture_dict[-1]['id']]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(final_c, num_classes)

    def forward(self, x):
        node_outputs = {}
        node_outputs[0] = x # Input Node
        
        for node in self.architecture_dict:
            node_id = node['id']
            op_code = node['op']
            
            if op_code == -1:
                continue
            
            input_idx = node['in1']
            input_tensor = node_outputs[input_idx]
            
            if op_code == 1: # CONV
                out = self.layers[str(node_id)](input_tensor)
                node_outputs[node_id] = out
                
            elif op_code == 2: # POOL
                out = self.layers[str(node_id)](input_tensor)
                node_outputs[node_id] = out
            
            elif op_code == 3: # ADD
                # We need input 2!
                in2_idx = node.get('in2', -1)
                input2_tensor = node_outputs.get(in2_idx, torch.zeros_like(input_tensor)) # Fallback if missing? Should not happen if valid DAG
                
                # Check shapes match? Implicitly guaranteed by SMT, but PyTorch will error if not.
                # Just add.
                node_outputs[node_id] = input_tensor + input2_tensor 
            
            elif op_code == 4: # CONCAT
                in2_idx = node.get('in2', -1)
                input2_tensor = node_outputs.get(in2_idx)
                
                if input2_tensor is None:
                     raise ValueError(f"Missing input2 for Concat node {node_id}")
                     
                node_outputs[node_id] = torch.cat([input_tensor, input2_tensor], dim=1) # Channel concat
            
            elif op_code == -2: # OUTPUT
                node_outputs[node_id] = input_tensor
                
        # Final classification
        final_out = node_outputs[self.architecture_dict[-1]['id']]
        x = self.global_pool(final_out)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
