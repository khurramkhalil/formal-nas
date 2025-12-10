"""
TransNAS-Bench-101 Interface.

Provides access to the multi-task NAS benchmark for STL validation.
If the real benchmark API/Data is not found, falls back to a Mock Generator
that produces realistic training curves for testing.
"""

import os
import random
import math
from typing import Dict, List, Any, Optional

# Try to import real API if available

# Try to import real API if available
# We assume the user clones the repo to /formal-nas/TransNASBench or similar
# causing 'api' module to be available if PYTHONPATH is set.
try:
    # Attempt 1: Standard (if root is in path)
    from api import TransNASBenchAPI
except ImportError:
    try:
        # Attempt 2: Nested (if api is a directory with api.py without init)
        from api.api import TransNASBenchAPI
    except ImportError:
        TransNASBenchAPI = None

class TransNASBenchmark:
    """
    Unified Interface for TransNAS-Bench-101.
    """
    def __init__(self, data_path: Optional[str] = None):
        self.api = None
        self.use_mock = True
        
        # Default path checks (K8s -> Local CWD -> transnas_data)
        if data_path is None:
            if os.path.exists("/formal-nas/transnas-bench_v10141024.pth"):
                data_path = "/formal-nas/transnas-bench_v10141024.pth"
            elif os.path.exists("./transnas-bench_v10141024.pth"):
                 data_path = "./transnas-bench_v10141024.pth"
            elif os.path.exists("transnas_data/transnas-bench_v10141024.pth"):
                 data_path = "transnas_data/transnas-bench_v10141024.pth"
            
        if TransNASBenchAPI and data_path and os.path.exists(data_path):
            print(f"Loading TransNAS-Bench from {data_path}...")
            try:
                self.api = TransNASBenchAPI(data_path, verbose=False)
                self.use_mock = False
                print("✅ TransNAS-Bench API Loaded Successfully.")
                
                # Debug: Print examples of valid strings
                if 'micro' in self.api.all_arch_dict:
                    print(f"    [Debug] Valid Arch Examples: {self.api.all_arch_dict['micro'][:3]}")
            except Exception as e:
                print(f"❌ Failed to load API: {e}")
                self.use_mock = True
        else:
            if not TransNASBenchAPI:
                print("⚠️ TransNASBenchAPI module not found.")
            if not data_path or not os.path.exists(data_path):
                print(f"⚠️ Data file not found at {data_path}.")
            print("Using Mock Data Generator.")

    def get_training_trace(self, task: str, arch_id: str) -> List[Dict[str, float]]:
        """
        Returns a time-series trace of training metrics.
        """
        if self.use_mock:
             # Mock data disabled via user request.
             raise RuntimeError("TransNAS API not loaded and Mock Data is disabled.")
        
        # Real API Access
        try:
            # SMT 'arch_id' is now the Bit-Exact NAS-Bench-201 String
            # e.g., "|nor_conv_3x3~0|+|nor_conv_3x3~0|...|"
            # We query the API directly.
            real_arch_str = arch_id
            
            # Note: TransNASBenchAPI might expect different method to get index or result by string.
            # Usually: api.get_arch_result(arch_str) is not standard?
            # Standard is: index = api.query_index_by_arch(arch_str) -> then get_epoch_status
            
            # Let's inspect API usage. If 'get_epoch_status' accepts string, great.
            # Based on docs/code: api.get_epoch_status(arch_info, ...)
            # arch_info can be index (int) or string.
            
            trace = []
            
            found_data = False
            for epoch in [1, 10, 20]: # Sparse sampling
                try:
                    info = self.api.get_epoch_status(real_arch_str, task, epoch=epoch)
                    
                    # Extract Metrics (Same logic as before)
                    acc = 0.0
                    loss = info.get('train_loss', 0.0)
                    
                    if 'valid_top1' in info: acc = info['valid_top1']
                    elif 'valid_ssim' in info: acc = info['valid_ssim'] * 100 
                    elif 'valid_mIoU' in info: acc = info['valid_mIoU']
                    elif 'valid_cos_similarity' in info: acc = info['valid_cos_similarity'] * 100
                    elif 'test_top1' in info: acc = info['test_top1']
                    
                    # Regression Tasks
                    elif 'valid_neg_l1_loss' in info: 
                        val = info['valid_neg_l1_loss']
                        acc = 100.0 / (1.0 + abs(val))
                        
                    elif 'valid_l1_loss' in info:
                         val = info['valid_l1_loss']
                         acc = 100.0 / (1.0 + val)
                         
                    elif loss > 0 and acc == 0.0:
                        acc = 100.0 / (1.0 + loss)
                    
                    trace.append({
                        'epoch': epoch,
                        'accuracy': float(acc),
                        'loss': float(loss),
                        'energy': 100.0
                    })
                    if acc > 0: found_data = True
                except Exception as e:
                    # Debug: Print exception to understand why valid strings fail
                    print(f"    [Debug] API Query Failed for Epoch {epoch}: {e}")
                    continue
            
            # Strict Mode: return empty if synthesis generated invalid string (should not happen with correct encoding)
            if not found_data:
                print(f"⚠️ API returned no data for {task}/{real_arch_str}. Skipping.")
                return []
                
            return trace
            
        except Exception as e:
            print(f"Error querying TransNAS API: {e}")
            return []
            
    # Note: _generate_mock_trace deleted as per user request.

# Singleton or Factory
def get_benchmark(path=None):
    return TransNASBenchmark(path)
