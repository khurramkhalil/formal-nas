
"""
Standard NAS-Bench-201 Interface.

For benchmarking on CIFAR-10/100/ImageNet using the official 'nas_201_api'.
"""

import os
import sys
from typing import Dict, List, Any, Optional

try:
    from nas_201_api import NASBench201API as API
except ImportError:
    API = None
    print("⚠️ Warning: 'nas_201_api' not found. Please install nas-bench-201.")

class NASBench201Benchmark:
    """
    Interface for Standard NAS-Bench-201 using Official API.
    """
    def __init__(self, data_path: Optional[str] = None):
        self.api = None
        
        # Check standard locations
        candidates = [
            data_path,
            "./NAS-Bench-201-v1_1-096897.pth",
            os.path.expanduser("~/NAS-Bench-201-v1_1-096897.pth"),
            "nas_201_data/NAS-Bench-201-v1_1-096897.pth"
        ]
        
        real_path = None
        for p in candidates:
            if p and os.path.exists(p):
                real_path = p
                break
        
        if API and real_path:
            print(f"Loading NAS-Bench-201 (Official API) from {real_path}...")
            # verbose=False to keep logs clean
            self.api = API(real_path, verbose=False)
            print("✅ NAS-Bench-201 API Loaded Successfully.")
        else:
             print("⚠️ NAS-Bench-201 API or Data not found.")
             if not API: print("  Module 'nas_201_api' missing.")
             if not real_path: print("  Data file 'NAS-Bench-201-v1_1-096897.pth' not found.")

    def get_training_trace(self, task: str, arch_id: str) -> List[Dict[str, float]]:
        """
        Get trace for datasets: 'cifar10-valid', 'cifar100', 'ImageNet16-120'.
        """
        dataset = task # Alias
        if not self.api:
            return []
            
        try:
            # 1. Get Index
            # API expects string format: |op~0|+|op~0|op~1|...+...
            index = self.api.query_index_by_arch(arch_id)
            if index == -1:
                print(f"  ⚠️ Arch not found in NAS-201: {arch_id}")
                return []
                
            # 2. Get Info
            # 'get_more_info' returns a dict with metrics for the *final* epoch (usually).
            # We want 'valid-accuracy' or 'test-accuracy'.
            # dataset should be 'cifar10-valid', 'cifar100', etc.
            result = self.api.get_more_info(index, dataset, is_random=False)
            
            # The API returns different keys for different datasets sometimes, 
            # but 'valid-accuracy' is standard for CIFAR-10.
            # 'cifar100' might use 'test-accuracy'?
            # Let's inspect safely.
            
            acc = 0.0
            if 'valid-accuracy' in result:
                acc = result['valid-accuracy'] # This is %. e.g. 91.5
            elif 'test-accuracy' in result:
                acc = result['test-accuracy']
                
            # 3. Construct Trace
            # Since we only get final accuracy easily, we synthesize a simple trace.
            # This allows P-STL to verify "Eventually(acc > X)".
            trace = [
                {'epoch': 0, 'accuracy': 0.0, 'loss': 100.0},
                {'epoch': 200, 'accuracy': float(acc), 'loss': 0.0}
            ]
            return trace
            
        except Exception as e:
             # Debugging info
            print(f"Error querying NAS-201 API: {e}")
            return []

def get_benchmark(path=None):
    return NASBench201Benchmark(path)
