
"""
NAS-Bench-201 Search Experiment Runner.

Runs the Spatiotemporal Formal NAS search on the standard NAS-Bench-201 dataset.
"""

import sys
import os
import argparse
import wandb
from dotenv import load_dotenv

# Ensure src is in path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"DEBUG: Added src path (Head): {src_path}")

# Ensure nas_201_api is in path if nested
nas201_path = os.path.join(os.path.dirname(__file__), '../NASBench201')
if os.path.exists(nas201_path) and nas201_path not in sys.path:
    sys.path.append(nas201_path)

from formal_nas.search.unified_controller import UnifiedController

def main():
    parser = argparse.ArgumentParser(description="Run NAS-Bench-201 Search")
    parser.add_argument("--iterations", type=int, default=1000, help="Max search iterations")
    parser.add_argument("--dataset", type=str, default="cifar10-valid", help="Target Dataset (cifar10-valid, cifar100, ImageNet16-120)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--output-dir", type=str, default="results_nas201", help="Directory for CSV results")
    args = parser.parse_args()
    
    # Load Env
    load_dotenv()
    
    # Create Output Dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== Starting NAS-Bench-201 Search ===")
    print(f"Dataset: {args.dataset}")
    print(f"Iterations: {args.iterations}")
    print(f"Mode: {'Standard (5 Ops)'}")

    # Set Env for WandB
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "formal-nas-201-bench"
        os.environ["WANDB_JOB_TYPE"] = "search"
        os.environ["WANDB_NAME"] = f"search_201_{args.dataset}"
        
    # Instantiate Controller
    log_file = os.path.join(args.output_dir, f"results_{args.dataset}.csv")
    controller = UnifiedController(
        use_wandb=not args.no_wandb,
        log_file=log_file,
        benchmark_type="nas201"
    )
    
    # Run Search
    best_arch = controller.run_search(
        max_iterations=args.iterations, 
        task_name=args.dataset
    )
    
    if best_arch:
        print(f"SUCCESS: Found valid architecture for {args.dataset}.")
    else:
        print(f"FAILURE: No architecture met criteria.")

if __name__ == "__main__":
    main()
