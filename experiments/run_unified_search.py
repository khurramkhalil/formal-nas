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

from formal_nas.search.unified_controller import UnifiedController

def main():
    parser = argparse.ArgumentParser(description="Run Unified Formal NAS Search")
    parser.add_argument("--iterations", type=int, default=1000, help="Max search iterations")
    parser.add_argument("--wandb-project", type=str, default="formal-nas-autoamatic", help="WandB Project Name")
    args = parser.parse_args()
    
    # Load Env
    load_dotenv()
    
    # Run Full Benchmark Suite
    # Tasks: class_scene, class_object, room_layout, jigsaw, segmentsemantic, normal, autoencoder
    tasks = ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
    
    print(f"=== Starting Unified Search Benchmark (7 Tasks) ===")
    print(f"Iterations: {args.iterations}")

    
    # Set Group ID for WandB (One logical experiment)
    group_id = wandb.util.generate_id()
    
    for task in tasks:
        print(f"\n>>> Running Task: {task}")
        
        # Initialize WandB per task (using reinit=True)
        # We pass task info to controller via init if needed, 
        # but Controller calls wandb.init() inside __init__.
        # We need to make sure Controller re-inits correctly or we manage it here.
        
        # Refactored Approach: Controller handles init.
        # We set env var WANDB_RUN_GROUP to group them in UI.
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_GROUP"] = f"benchmark_{group_id}"
        os.environ["WANDB_JOB_TYPE"] = "search"
        os.environ["WANDB_NAME"] = f"search_{task}"
        
        # Instantiate fresh controller for each task
        controller = UnifiedController(use_wandb=True)
        
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
