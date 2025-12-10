"""
Unified Search Experiment Runner.

Entry point for Kubernetes Jobs.
Runs the Spatiotemporal Formal NAS search with specified parameters.
"""

import sys
import os
import argparse
import wandb
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Ensure TransNASBench is in path
transnas_path = os.path.join(os.path.dirname(__file__), '../TransNASBench')
if os.path.exists(transnas_path) and transnas_path not in sys.path:
    sys.path.append(transnas_path)

from formal_nas.search.unified_controller import UnifiedController

def main():
    parser = argparse.ArgumentParser(description="Run Unified Formal NAS Search")
    parser.add_argument("--iterations", type=int, default=1000, help="Max search iterations")
    parser.add_argument("--wandb-project", type=str, default="formal-nas-autoamatic", help="WandB Project Name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for CSV results")
    args = parser.parse_args()
    
    # Load Env
    load_dotenv()
    
    # Run Full Benchmark Suite
    # Tasks: class_scene, class_object, room_layout, jigsaw, segmentsemantic, normal, autoencoder
    tasks = ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
    
    # Create Output Dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== Starting Unified Search Benchmark (7 Tasks) ===")
    print(f"Iterations: {args.iterations}")
    print(f"WandB: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"Output Dir: {args.output_dir}")

    
    # Set Group ID for WandB (One logical experiment)
    group_id = wandb.util.generate_id()
    
    for task in tasks:
        print(f"\n>>> Running Task: {task}")
        
        # WandB Setup
        if not args.no_wandb:
            os.environ["WANDB_PROJECT"] = args.wandb_project
            os.environ["WANDB_RUN_GROUP"] = f"benchmark_{group_id}"
            os.environ["WANDB_JOB_TYPE"] = "search"
            os.environ["WANDB_NAME"] = f"search_{task}"
        
        # Instantiate fresh controller for each task
        log_file = os.path.join(args.output_dir, f"results_{task}.csv")
        controller = UnifiedController(
            use_wandb=not args.no_wandb,
            log_file=log_file
        )
        
        # Run Search
        best_arch = controller.run_search(
            max_iterations=args.iterations, 
            task_name=task
        )
        
        if best_arch:
            print(f"SUCCESS ({task}): Found valid architecture.")
        else:
            print(f"FAILURE ({task}): No architecture met criteria.")
            
        # Ensure run is closed before next loop
        if wandb.run:
            wandb.finish()
            
    print("\n=== All Tasks Completed ===")
if __name__ == "__main__":
    main()
