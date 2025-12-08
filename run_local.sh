#!/bin/bash
# Local Execution Script for Formal NAS Unified Search

# 1. Export Environment Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/TransNASBench
export WANDB_API_KEY="c7ea31e5793ba44d95d91840a10d709378eb2d9d"
export WANDB_PROJECT="formal-nas-adaptive-benchmark"
export WANDB_ENTITY="khurramkhalil"

# 2. Link Data File (if not already linked)
# TransNASBench API expects data in specific path or passed as arg.
# We will create a symlink to 'transnas-bench_v10141024.pth' in the current dir
if [ ! -f "transnas-bench_v10141024.pth" ]; then
    echo "Linking data file..."
    ln -s transnas_data/transnas-bench_v10141024.pth transnas-bench_v10141024.pth
fi

# 3. Running Search
echo "=== Running Unified Search Locally ==="
python experiments/run_unified_search.py
