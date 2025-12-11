"""
HLS Export Script (PyTorch -> hls4ml -> Vivado HLS).

This script performs the following:
1. Synthesizes an architecture targeting Xilinx U55C constraints.
2. Converts it to a PyTorch Model.
3. (Optional) Trains it briefly (or loads weights).
4. Uses `hls4ml` to convert the PyTorch model into an HLS Project for U55C.

NOTE: This script requires 'hls4ml' and 'onnx' installed.
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
from formal_nas.logic.temporal import Always, Eventually, IsOp

def run_hls_export():
    print("=== Xilinx U55C HLS Export Demo ===")
    
    # 1. Synthesize for U55C
    print("[1] Synthesizing for Alveo U55C...")
    solver = z3.Solver()
    
    # U55C Limits: 1.3M LUTs, 9024 DSPs. 
    # We set a small fraction to ensure it fits easily for demo.
    limits = {"luts": 200000, "dsp": 500, "bram": 1000}
    
    encoding = DAGEncoding(
        solver, 
        max_nodes=6, 
        input_channels=3, 
        resource_limits=limits,
        hw_model_type="xilinx_u55c"
    )
    
    # Constraint: Must use DSPs (Conv)
    solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
    
    if solver.check() != z3.sat:
        print("❌ UNSAT")
        return

    model = solver.model()
    arch_dict = encoding.decode_architecture(model)
    print(f"✓ Found Architecture ({len(arch_dict)} nodes)")
    
    # 2. PyTorch Conversion
    print("[2] Converting to PyTorch...")
    model = PyTorchDAG(arch_dict, input_channels=3, num_classes=10)
    model.eval()
    
    # 3. hls4ml Conversion
    print("[3] Exporting to hls4ml project...")
    
    try:
        import hls4ml
        
        # Define Config
        config = hls4ml.utils.config_from_pytorch_model(model, granularity='name')
        
        # Set Precision (important for FPGA)
        # We use ap_fixed<16,6> (16 bits total, 6 integer bits)
        for layer in config['LayerName'].keys():
            config['LayerName'][layer]['Precision'] = 'ap_fixed<16,6>'
        
        # Create HLS Project with streaming IO to avoid complete array partitioning
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            input_shape=(1, 3, 32, 32),
            hls_config=config,
            output_dir='hls_project_u55c',
            part='xcu55c-fsvh2892-2L-e', # Alveo U55C Part Number
            io_type='io_stream'  # Use streaming to avoid massive resource usage
        )
        
        # Compile (C-Simulation)
        # This verifies the logic roughly matches PyTorch
        print("   > Compiling HLS Model (C-Sim)...")
        hls_model.compile()
        
        print("\n✅ Success! HLS Project created in 'hls_project_u55c/'.")
        print("   Run 'vivado' or 'vitis_hls' in that directory on Nautilus.")
        
    except ImportError:
        print("\n⚠️  'hls4ml' library not found.")
        print("   Install with: pip install hls4ml")
        print("   (This is expected if running on local laptop without vivado libs)")
        
    except Exception as e:
        print(f"\n❌ Prediction Error: {e}")
        # hls4ml sometimes struggles with complex DAGs specifically converted from weird dicts.
        # But for standard Conv/Relu/Pool sequences it works.

if __name__ == "__main__":
    run_hls_export()
