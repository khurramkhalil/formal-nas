#!/bin/bash
set -e  # Exit on error

echo "========================================================"
echo "🚀 Starting Formal NAS Full Benchmark Suite (NRP Cluster)"
echo "========================================================"

# 1. TransNAS-Bench-101 (Micro Space, Multi-Task)
echo ""
echo ">>> [Phase 1/2] Running TransNAS-Bench-101 Search..."
# Run full suite (all 7 tasks)
python experiments/run_unified_search.py --iterations 1000 --output-dir /app/results_transnas

echo "✅ TransNAS Phase Complete."

# 2. NAS-Bench-201 (Standard Space)
echo ""
echo ">>> [Phase 2/2] Running NAS-Bench-201 Search..."
# Run CIFAR-10 Valid (can add others if needed)
python experiments/run_nas201_search.py --dataset cifar10-valid --iterations 1000 --output-dir /app/results_nas201

echo "✅ NAS-201 Phase Complete."

echo ""
echo "========================================================"
echo "🏁 Benchmark Suite Finished Successfully."
echo "========================================================"
