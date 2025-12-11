
import argparse
import sys
import os
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from formal_nas.search.unified_controller import UnifiedController

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10-valid')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--dsp-limit', type=int, default=3, help="Max DSP/3x3 Conv Ops")
    parser.add_argument('--wandb-project', type=str, default='formal-nas-fpga-production', help="WandB Project Name")
    args = parser.parse_args()
    
    print(f"=== FPGA-Aware Search (DSP < {args.dsp_limit}) ===")
    
    # 1. Configure Controller with Hardware Constraints
    controller = UnifiedController(
        use_wandb=True, 
        benchmark_type="nas201",
        project_name=args.wandb_project
    )
    
    # Inject Limits directly into the solver via the encoding's logic
    # The encoding is already built, but we can add more constraints to the solver!
    if hasattr(controller.encoding, 'total_dsp'):
        print(f"Applying DSP Constraint: <= {args.dsp_limit}")
        controller.solver.add(controller.encoding.total_dsp <= args.dsp_limit)
    else:
        print("Warning: CellEncoding does not have 'total_dsp'. Limits ignored.")
        
    # 2. Run Search
    print("Starting Search...")
    best_arch = controller.run_search(max_iterations=args.iterations, task_name=args.dataset)
    
    if not best_arch:
        print("No valid architecture found!")
        return
        
    # 3. Decode Best Arch to String
    # best_arch is a Dictionary format from controller.run_search (my bad, check return type)
    # Controller returns `best_arch` which is the dict `decode_to_dict` format.
    # We need the String format for the export script.
    # We can reconstruct it or update controller to return string.
    # Let's just fetch it from history or reconstruct.
    # Actually, let's just make the Controller return the string for convenience, or re-encode.
    
    # Hack: The dict has {'edge': i, 'op': val}.
    # We can rebuild the values list.
    vals = [0]*6
    for item in best_arch:
        vals[item['edge']] = item['op']
    
    # Re-use encoding to get string (need to mock a model or just use the logic)
    op_map = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    op_strs = [op_map[v] for v in vals]
    n1 = f"|{op_strs[0]}~0|"
    n2 = f"|{op_strs[1]}~0|{op_strs[3]}~1|"
    n3 = f"|{op_strs[2]}~0|{op_strs[4]}~1|{op_strs[5]}~2|"
    arch_str = f"{n1}+{n2}+{n3}"
    
    print(f"Best Arch String: {arch_str}")
    
    # 4. Run Export
    print("Running HLS Export...")
    cmd = [
        "python", "scripts/generate_hls.py",
        "--arch", arch_str,
        "--project-name", "fpga_best_arch"
    ]
    subprocess.check_call(cmd)
    
if __name__ == "__main__":
    main()
