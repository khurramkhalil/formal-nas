
import os
import sys
import torch
import wandb
import hls4ml
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from formal_nas.architectures.nasbench_ops import TinyNetwork

def parse_ops(arch_str: str):
    """
    Decodes standard NAS-201 string: |op~0|+|op~0|op~1|+...
    to list of indices.
    """
    # Mapping
    # 0: none, 1: skip_connect, 2: nor_conv_1x1, 3: nor_conv_3x3, 4: avg_pool_3x3
    smap = {
        "none": 0, "skip_connect": 1, "nor_conv_1x1": 2, "nor_conv_3x3": 3, "avg_pool_3x3": 4
    }
    
    # 1. Normalize
    # Standard format: |op0~0|+|op1~0|op2~1|+|op3~0|op4~1|op5~2|
    # Split by '+'
    nodes = arch_str.split('+')
    # nodes[0] = |op0~0|
    # nodes[1] = |op1~0|op2~1|
    # nodes[2] = |op3~0|op4~1|op5~2|
    
    ops = []
    
    # Helper to extract op name from |name~idx|
    def extract(s):
        # s is like "nor_conv_3x3~0"
        return s.split('~')[0]

    # Node 1 (1 edge from 0)
    n1 = nodes[0].strip('|') # op~0
    ops.append(smap[extract(n1)])
    
    # Node 2 (2 edges from 0, 1)
    n2_parts = nodes[1].strip('|').split('|')
    ops.append(smap[extract(n2_parts[0])]) # from 0
    ops.append(smap[extract(n2_parts[1])]) # from 1
    
    # Node 3 (3 edges from 0, 1, 2)
    n3_parts = nodes[2].strip('|').split('|')
    ops.append(smap[extract(n3_parts[0])]) # from 0
    ops.append(smap[extract(n3_parts[1])]) # from 1
    ops.append(smap[extract(n3_parts[2])]) # from 2
    
    return ops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, help="NAS-Bench-201 String")
    parser.add_argument('--project-name', type=str, default="formal_nas_hls")
    parser.add_argument('--output-dir', type=str, default="hls_project")
    args = parser.parse_args()
    
    print(f"Generating HLS for Arch: {args.arch}")
    
    # 1. Instantiate Model
    op_indices = parse_ops(args.arch)
    model = TinyNetwork(op_indices)
    model.eval()
    
    # 2. HLS Configuration
    config = hls4ml.utils.config_from_pytorch_model(model, input_shape=(3, 32, 32), granularity='name')
    
    # U55C Settings (approximate part)
    # XCU55C-FSVH2892-2L-E
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    # CRITICAL FIX: Propagate settings to ALL layers
    # granularity='name' creates individual layer configs that ignore 'Model' global defaults
    print(f"DEBUG: Config Keys: {list(config.keys())}")
    
    # 1. Update Global Defaults
    config['Model']['ReuseFactor'] = 128
    config['Model']['Strategy'] = 'Resource'
    
    # 2. Update Per-Layer Configs
    if 'LayerName' in config:
        for layer_name, layer_config in config['LayerName'].items():
            if isinstance(layer_config, dict):
                layer_config['ReuseFactor'] = 128
                layer_config['Strategy'] = 'Resource'
    
    # 3. Convert
    print("Converting to HLS...")
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=(3, 32, 32),
        hls_config=config,
        output_dir=args.output_dir,
        part='xcu55c-fsvh2892-2L-e', # U55C Part
        project_name=args.project_name
    )
    
    # 4. Write
    hls_model.write()
    print(f"HLS Project written to {args.output_dir}")
    
    # --- AUTO-PATCH FOR VITIS 2023.1 ---
    print("Applying Vitis 2023.1 Compatibility Patches (Proven on NRP)...")
    
    # Patch 1: build_prj.tcl (Remove deprecated config_array_partition and skip csim/cosim)
    tcl_path = os.path.join(args.output_dir, "build_prj.tcl")
    if os.path.exists(tcl_path):
        with open(tcl_path, "r") as f:
            lines = f.readlines()
        with open(tcl_path, "w") as f:
            for line in lines:
                # Remove deprecated command
                if "config_array_partition" in line:
                    continue
                # Skip C-Sim (Linker errors due to missing crt1.o)
                # Skip Co-Sim (Requires C-Sim results)
                if "csim" in line and "opt(csim)" not in line: # Avoid disabling the opt check
                     if "csim_design" in line:
                         f.write("#csim_design\n")
                         continue
                # Set default opts to 0 just in case
                if "csim       1" in line:
                    f.write("    csim       0\n")
                    continue
                if "cosim      1" in line:
                    f.write("    cosim      0\n")
                    continue
                f.write(line)
                
    # Patch 2: Firmware Cleanup (Disable Strict Dataflow & Deprecated Pragmas)
    # Walk through all source files
    for root, dirs, files in os.walk(os.path.join(args.output_dir, "firmware")):
        for fname in files:
            if fname.endswith(".cpp") or fname.endswith(".h"):
                fpath = os.path.join(root, fname)
                with open(fpath, "r") as f:
                    content = f.read()
                
                # Case-Insensitive replacements via simple string ops (covering common cases)
                # 1. DATAFLOW (Strict check crash)
                content = content.replace("#pragma HLS DATAFLOW", "//#pragma HLS DATAFLOW")
                
                # 2. ALLOCATION (Unexpected argument crash)
                content = content.replace("#pragma HLS allocation", "//#pragma HLS allocation")
                
                # 3. RESOURCE (Deprecated warning)
                content = content.replace("#pragma HLS RESOURCE", "//#pragma HLS RESOURCE")
                content = content.replace("#pragma HLS Resource", "//#pragma HLS Resource") # Case variation
                
                # 4. INLINE region (Deprecated warning)
                content = content.replace("#pragma HLS INLINE", "//#pragma HLS INLINE")
                
                # 5. SPECIFIC FIX: nnet_pooling.h line 298 (pool_op argument error)
                if "nnet_pooling.h" in fname:
                     lines = content.split('\n')
                     # Remove the specific offending pragma if it exists (usually around line 298)
                     lines = [l for l in lines if "pool_op" not in l or "#pragma" not in l]
                     content = '\n'.join(lines)

                with open(fpath, "w") as f:
                    f.write(content)
                
    print("Patches Applied: ReuseFactor=128, CSIM=Off, CoSim=Off, Pragmas Stripped.")
    # -----------------------------------
    
    # 5. Zip
    zip_name = f"{args.project_name}.zip"
    os.system(f"zip -r {zip_name} {args.output_dir}")
    
    # 6. WandB Upload
    if os.environ.get("WANDB_API_KEY"):
        run = wandb.init(project="formal-nas-adaptive-benchmark", job_type="hls_export")
        artifact = wandb.Artifact('hls_project', type='code')
        artifact.add_file(zip_name)
        run.log_artifact(artifact)
        print("Logged artifact to WandB.")
    else:
        print("Skipped WandB upload (No Key).")

if __name__ == "__main__":
    main()
