
import sys
import os

# Assuming TransNASBench is cloned in CWD
if os.path.exists("TransNASBench"):
    sys.path.append("TransNASBench")

try:
    from api import TransNASBenchAPI
except ImportError:
    print("Could not import TransNASBenchAPI. Make sure TransNASBench repo is in CWD.")
    sys.exit(1)

data_path = "transnas_data/transnas-bench_v10141024.pth"
if not os.path.exists(data_path):
    # Try alternate locations
    if os.path.exists("transnas-bench_v10141024.pth"):
        data_path = "transnas-bench_v10141024.pth"
    else:
        print(f"Data file not found at {data_path}")
        sys.exit(1)

try:
    print(f"Loading API from {data_path}...")
    api = TransNASBenchAPI(data_path, verbose=False)
    print("API Loaded.")
    
    if 'micro' in api.all_arch_dict:
        keys = list(api.all_arch_dict['micro'])[:5]
        print("\n=== VALID KEYS (MICRO) ===")
        for k in keys:
            print(f"'{k}'")
            
        print("\n=== VALID KEYS (MACRO) ===")
        if 'macro' in api.all_arch_dict:
            keys_m = list(api.all_arch_dict['macro'])[:5]
            for k in keys_m:
                print(f"'{k}'")
    else:
        print("No 'micro' search space found?")
        
except Exception as e:
    print(f"Error: {e}")
