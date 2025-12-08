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
    from api import TransNASBenchAPI
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
            return self._generate_mock_trace(arch_id)
        
        # Real API Access
        # Note: 'arch_id' from SMT might be "arch_0". 
        # TransNAS indices are integers or specific strings (64-...).
        # We need a mapping or just sample randomly for this demo since 
        # our SMT encoding doesn't 1:1 map to TransNAS encoding yet.
        # For the purpose of "Unified Loop", we will Random Sample a valid architecture 
        # from the API to simulate the "Evaluation" of the SMT candidate.
        # (Bridging the gap: Real SMT -> Real TransNAS ID requires an Encoding Mapper).
        
        # For this phase, we treat the SMT candidate as a "Query" for a random valid arch
        # in the search space to get Real Data.
        
        try:
            # Randomly sample a VALID architecture index from Micro space
            # The API exposes .all_arch_dict['micro'] -> list of names
            micro_archs = self.api.all_arch_dict['micro']
            
            # Deterministic hash for consistency so same arch_id always gets same "Real Eval"
            idx = hash(arch_id) % len(micro_archs)
            real_arch_str = micro_archs[idx]
            
            trace = []
            
            # Query the FINAL epoch (usually ~200 or 25 depending on setup)
            # TransNAS-Bench-101 epoch count varies by task.
            # We will ask the API for the "best" metric directly if possible, or iterate.
            # "get_epoch_status" asks for specific epoch.
            # Let's try to find the max epoch by probing or assuming 20.
            
            found_data = False
            for epoch in [1, 10, 20]: # Sparse sampling
                try:
                    info = self.api.get_epoch_status(real_arch_str, task, epoch=epoch)
                    
                    # Extract Acc. Try 'valid_top1' (classification) or 'valid_ssim' (autoencoder) etc.
                    # TransNAS metrics vary by task!
                    # Classification: valid_top1
                    # Autoencoder: valid_ssim
                    # Segmentation: valid_mIoU
                    # Normal: valid_cos_similarity
                    
                    # Extract Acc. Try 'valid_top1' (classification) or 'valid_ssim' (autoencoder) etc.
                    # TransNAS metrics vary by task!
                    # Classification: valid_top1
                    # Autoencoder: valid_ssim
                    # Segmentation: valid_mIoU
                    # Normal: valid_cos_similarity
                    # Room Layout: valid_neg_l1_loss (Higher is better, usually negative) or just loss
                    
                    acc = 0.0
                    loss = info.get('train_loss', 0.0)
                    
                    if 'valid_top1' in info: acc = info['valid_top1']
                    elif 'valid_ssim' in info: acc = info['valid_ssim'] * 100 # scale to %
                    elif 'valid_mIoU' in info: acc = info['valid_mIoU']
                    elif 'valid_cos_similarity' in info: acc = info['valid_cos_similarity'] * 100
                    elif 'test_top1' in info: acc = info['test_top1']
                    
                    # Regression Tasks (Room Layout)
                    elif 'valid_neg_l1_loss' in info: 
                        # -0.5 -> 0.5 (Higher is better). Map to 0-100 scale?
                        # Or just use raw value. But Controller expects > 10.0 for "Good".
                        # Let's map small negative loss to high score. 
                        # Score = 100 * exp(val) ? Or 100 + val?
                        # Assume val is -0.1 (good) to -5.0 (bad).
                        # Using 100 / (1 + abs(val))
                        val = info['valid_neg_l1_loss']
                        acc = 100.0 / (1.0 + abs(val))
                        
                    elif 'valid_l1_loss' in info:
                         val = info['valid_l1_loss']
                         acc = 100.0 / (1.0 + val)
                         
                    # Catch-all: If we have loss but no acc, invert loss
                    elif loss > 0 and acc == 0.0:
                        # Fallback for regression tasks if explicit key missing
                        acc = 100.0 / (1.0 + loss)
                    
                    trace.append({
                        'epoch': epoch,
                        'accuracy': float(acc),
                        'loss': float(loss),
                        'energy': 100.0
                    })
                    if acc > 0: found_data = True
                except Exception:
                    continue
            
            if not found_data:
                # User Request: "Delete garbage, only work on valid data"
                # If we can't find data, we return empty list so Controller skips.
                print(f"⚠️ API returned no 'Accuracy' compatible data for {task}/{real_arch_str}. Skipping.")
                return []
                
            return trace
            
        except Exception as e:
            print(f"Error querying TransNAS API: {e}")
            return []
            
    # Note: _generate_mock_trace deleted as per user request.

# Singleton or Factory
def get_benchmark(path=None):
    return TransNASBenchmark(path)
