
import torch
import sys
import os

# Helper to load API
sys.path.append(os.getcwd())
try:
    from TransNASBench.api.api import TransNASBenchAPI
except ImportError:
    print("Could not import TransNASBenchAPI. Ensure PYTHONPATH is set.")
    sys.exit(1)

def main():
    data_path = "transnas-bench_v10141024.pth"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    api = TransNASBenchAPI(data_path)
    
    print("\nAnalyzing 'micro' search space keys...")
    micro_keys = api.all_arch_dict.get('micro', [])
    
    prefixes = set()
    for key in micro_keys:
        # Key format: '64-41414-0_00_000'
        # Split by last dash?
        # Actually, split by first dash? '64'
        # The user said "64-41414" is the prefix.
        if '-' in key:
            # Split into parts
            parts = key.split('-')
            # Assuming format prefix-suffix
            # '64-41414-0_00_000' -> parts: ['64', '41414', '0_00_000']
            if len(parts) >= 2:
                prefix = f"{parts[0]}-{parts[1]}"
                prefixes.add(prefix)
    
    print(f"Unique Prefixes Found: {len(prefixes)}")
    for p in prefixes:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
